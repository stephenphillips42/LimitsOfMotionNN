-- Standard packages
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'csvigo'
local debugger = require('fb.debugger')

-- Custom packages
require 'dataset-kitti-flow'
paths.dofile('util.lua')

-- References:
-- [1] He et al 2015, Deep Residual Learning for Image Recognition
-- [2] He et al 2016, Identity Mappings in Deep Residual Networks
-- [3] He et al 2015, Delving Deeper Into Rectifiers


-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

modules = {}

-- imsize = kittiflow.imsize
function shrink(size,n)
   return math.ceil(size/(2^n))
end

-- initialization from MSRA, as described in [3]
function modules.MSRinit(net)
    local function initConv(name)
        for k,v in pairs(net:findModules(name)) do
            --debugger.enter()
            local n = v.kW*v.kH*v.nOutputPlane
            --v.weight:zero()
            v.weight:normal(0,math.sqrt(2/n))
            if cudnn.version >= 4000 then
               v.bias = nil
               v.gradBias = nil
            else
               v.bias:zero()
            end
        end
    end
    local function initBN(name)
        for k,v in pairs(net:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end
    local function initLinear(name)
        for k,v in pairs(net:findModules(name)) do
            --debugger.enter()
            local n = v.weight:size()[2]
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end

    initConv'nn.SpatialConvolutionMM'
    initLinear'nn.Linear'
    initBN'nn.SpatialBatchNormalization'
end

-- Helper functions
-- The shortcut layer is either identity or BN + 1x1 convolution, in line with [2]
-- From fb.resnet.torch
function modules.Shortcut(nInputPlane, nOutputPlane, stride)
   -- Return a shortcut block structured as:
   --    Identity if no change in dimensionality;
   --    Filter output replication if doubling number of filters;
   --    BatchNormalization+Conv if changing the number of filters in any other way.
   -- Striding > 1 can be used to downsample in any of the three cases.

   -- TODO: extend to 2^n

    if nInputPlane==nOutputPlane then
      if stride == 1 then
         return nn.Sequential():add(nn.Identity())
      else
         return nn.Sequential():add(nn.SpatialSubSampling(nOutputPlane,1,1,stride,stride))
      end
   -- elseif nOutputPlane==2*nInputPlane then
   --    if stride == 1 then
   --       return Sequentional():add(nn.Replicate(2,1,3))
   --    else
   --       return nn.Sequential()
   --          :add(nn.SpatialSubSampling(nInputPlane,1,1,stride,stride))
   --          :add(nn.Replicate(2,1,3)) -- replicate 2x along 1st non-batch dimension
   --    end
   else
      return nn.Sequential()
            :add(nn.SpatialBatchNormalization(nOutputPlane))
            :add(nn.SpatialConvolutionMM(nInputPlane, nOutputPlane, 1, 1, stride, stride))
    end
end

function modules.StandardActivation(nInput)
   return nn.Sequential()
      :add(nn.SpatialBatchNormalization(nInput))
      :add(nn.ReLU(true))
end

-- This creates a Conv -> BatchNormalization -> ReLU layer
function modules.StandardBlock(nInput, nOutput, stride, filterSize, padding)
    -- Default arguments
    stride = stride or 1
    filterSize = filterSize or 3
    padding = padding or (filterSize-(filterSize%2))/2
    standardActivation = modules.StandardActivation(nOutput)
    return nn.Sequential()
                :add(nn.SpatialConvolutionMM(
                        nInput,nOutput,filterSize,filterSize,stride,stride,padding,padding))
                :add(standardActivation)
end

-- Convenience function for parallel standard blocks
function modules.StandardBlockParallel(nInput, nOutput, stride, filterSize, padding)
    -- Default arguments
    stride = stride or 1
    filterSize = filterSize or 3
    padding = padding or (filterSize-(filterSize%2))/2
    standardBlock = modules.StandardBlock(nInput,nOutput,stride,filterSize,padding)
    return nn.ParallelTable()
                :add(standardBlock)
                :add(standardBlock:clone('weight','bias','gradWeight','gradBias'))
end


-- Creates bottleneck residual layer, with a shortcut. In [1], these are the
-- building blocks for the ImageNet architecture.
-- Structured as :
-- BN+ReLU+Conv(1x1,stride1or2,pad0), stride 2 if downsampling here
-- BN+ReLU+Conv(3x3,stride1,pad1)
-- BN+ReLU+Conv(1x1,stride1,pad0)
-- From fb.resnet.torch
function modules.BottleneckBlock(nInput, nInter1, nInter2, nOutput, stride, isInit)
    -- Default arguments
    stride = stride or 1
    isInit = isInit or false
    local seq = nn.Sequential()
    if not isInit then
      seq:add(nn.SpatialBatchNormalization(nInter1))
      seq:add(nn.ReLU(true))
    end
    -- Downsampling via stride > 1 happens in the first conv layer if present
    seq:add(nn.SpatialConvolutionMM(nInput,nInter1,1,1,stride,stride,0,0))
    seq:add(nn.SpatialBatchNormalization(nInter2))
    seq:add(nn.ReLU(true))
    seq:add(nn.SpatialConvolutionMM(nInter1,nInter2,3,3,1,1,1,1))
    seq:add(nn.SpatialBatchNormalization(nOutput))
    seq:add(nn.ReLU(true))
    seq:add(nn.SpatialConvolutionMM(nInter2,nOutput,1,1,1,1,0,0))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(seq)
            :add(modules.Shortcut(nInput, nOutput, stride)))
        :add(nn.CAddTable(true))
end


-- Creates a residual layer, with a shortcut. In [1], these are the building
-- blocks for the CIFAR architecture.
-- Structured as :
-- BN+ReLU+Conv(3x3,stride1or2,pad0), stride 2 if downsampling here
-- BN+ReLU+Conv(3x3,stride1,pad0)
-- From fb.resnet.torch
function modules.ResBuildingBlock(nInput, nInter, nOutput, stride, isInit)
    -- Default arguments
    stride = stride or 1
    isInit = isInit or false

    local seq = nn.Sequential()
    if not isInit then
      seq:add(nn.SpatialBatchNormalization(nInter))
      seq:add(nn.ReLU(true))
    end
    seq:add(nn.SpatialConvolutionMM(nInput,nInter,3,3,stride,stride,1,1))
    seq:add(nn.SpatialBatchNormalization(nInter))
    seq:add(nn.ReLU(true))
    seq:add(nn.SpatialConvolutionMM(nInter,nOutput,3,3,1,1,1,1))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(seq)
            :add(modules.Shortcut(nInput, nOutput, stride)))
        :add(nn.CAddTable(true))
end

function modules.ResBuildingBlockParallel(nInput, nInter, nOutput, stride,isInit)
    -- Default arguments
    stride = stride or 1
    isInit = isInit or false
    resBuildingBlock = modules.ResBuildingBlock(nInput,nInter,nOutput,stride,isInit)
    return nn.ParallelTable()
                :add(resBuildingBlock)
                :add(resBuildingBlock:clone('weight','bias','gradWeight','gradBias'))
end

-- This creates a standard tensor low rank factorization Block
function modules.FactorBlock(nInput, nInter, height, width, downsize, stride, filterSize, padding)
    return modules.FactorBlockGraph(nInput, nInter, height, width, downsize, stride, filterSize, padding)
end

-- This creates a standard tensor low rank factorization Block
function modules.FactorBlockGraph(nInput, nInter, height, width, downsize, stride, filterSize, padding)
    -- Default arguments
    nInter = nInter or nInput -- Intermediate size
    nOutput = nInput -- Output size is always input size
    downsize = downsize or false -- Don't downsize by default.
    stride = stride or 1 -- stride of pre-filters
    filterSize = filterSize or 3 -- size of pre-filters
    padding = padding or (filterSize-(filterSize%2))/2 -- padding used for pre-filters
    -- Build block -- assumes table of inputs
    factorBlock = nn.Sequential()
    factorBlock
        -- Filters
        -- Batchnorm internally breaks the structure of the factorization, so do it beforehand
        -- If interpreting the WHOLE tensor multiplication as the nonlinearity (bilinearity),
        -- batchnorm should come before it
        :add(nn.ParallelTable()
           :add(nn.SpatialBatchNormalization(nOutput))
           :add(nn.SpatialBatchNormalization(nOutput)))
        :add(nn.ParallelTable()
            :add(nn.SpatialConvolutionMM(nInput, nInter, filterSize, filterSize, stride, stride, padding, padding)) -- U
            :add(nn.SpatialConvolutionMM(nInput, nInter, filterSize, filterSize, stride, stride, padding, padding))) -- V
        -- Combine spatial features
        :add(nn.CMulTable())
        -- Matrix Multiplication as convolution (1x1 filter)
        :add(nn.SpatialConvolutionMM(nInter,nOutput, 1,1, 1,1, 0,0))
    -- Graph building
    start = nn.Identity()()
    leftSide = nn.SelectTable(1)(start)
    rightSide = nn.SelectTable(2)(start)
    factorBlock = factorBlock(start)
    -- Maybe add a nonlinearity here before adding back to shortcuts
    leftOutput = nn.CAddTable()({leftSide, factorBlock})
    rightOutput = nn.CAddTable()({rightSide, factorBlock})
    fullFactorBlock = nn.gModule({start},{leftOutput, rightOutput})
    return fullFactorBlock
end


function modules.OutputBlock(nInput,nTargets,height,width,opt,usePool)
    -- nInput is the number of input filters, nTargest is the output dimensionality
    -- usePool is used to turn on (true) or off (false) spatial average pooling before the loss
    usePool = usePool or true
   --  usePool = false


    local fullyConnected = nn.Sequential()

   -- Dropout 1: not currently used
   --  if not opt.conover then
   --   fullyConnected:add(nn.Dropout()) -- have also explored using SpatialDropout here
   --  end


   -- Should explore using (non spatial) dropout here, i.e. before the spatial pooling. Will make it less servere.

    if usePool then
      fullyConnected:add(nn.SpatialAveragePooling(width, height, 1, 1, 0, 0)) -- AVERAGE pooling down to 1x1 spatial
                  :add(nn.Reshape(nInput,true))
    else -- Keep spatial positions
      fullyConnected:add(nn.Reshape(nInput*height*width,true))
      nInput = nInput*height*width
    end

    -- Dropout 2
   --if not opt.conover then
   --   fullyConnected:add(nn.Dropout()) -- have also explored using SpatialDropout here
   --end


    if opt.traintype == 'heatmap' then
        fullyConnected:add(nn.ConcatTable()
                            :add(nn.Linear(nInput,kittiflow.azHist:size(1)))
                            :add(nn.Linear(nInput,kittiflow.elHist:size(1)))
                            :add(nn.Linear(nInput,kittiflow.rhoHist:size(1))))
        fullyConnected:add(nn.ParallelTable()
                            :add(nn.LogSoftMax())
                            :add(nn.LogSoftMax())
                            :add(nn.LogSoftMax()))
    elseif opt.traintype == 'regression' or opt.traintype== 'trans_only' then
      --   debugger.enter()
      --   fullyConnected:add(nn.Linear(nInput,4096))
      --   fullyConnected:add(nn.BatchNormalization(4096))
      --   fullyConnected:add(nn.ReLU(true))
      --   fullyConnected:add(nn.Linear(4096,4096))
      --   fullyConnected:add(nn.BatchNormalization(4096))
      --   fullyConnected:add(nn.ReLU(true))
      --   fullyConnected:add(nn.Linear(4096,nTargets))
        fullyConnected:add(nn.Linear(nInput,nTargets))

    else
        fullyConnected:add(nn.Linear(nInput,nTargets))
        fullyConnected:add(nn.LogSoftMax())
    end
    return fullyConnected
end


-- nSamples = 5
-- depth = 2
-- width = 5
-- height = 5
-- ex = nn.Sequential()
--         :add(modules.FactorBlock(nSamples,depth,width,height))
--         :add(nn.CAddTable())
--         :add(nn.SpatialAveragePooling(2,2,2,2,(width%2),(height%2)))

-- print(ex:forward({torch.randn(2,5,5,5),torch.randn(2,5,5,5)}))
