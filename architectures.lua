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

require 'custom-modules'

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

local function shrink(size,n)
   return math.ceil(size/(2^n))
end


architectures = {}

-- TODO: Make which model we use part of opt
function architectures.createModel(imsize,noutputs,opt)
    -- return architectures.createLargeStandardBlockModel(imsize,noutputs,opt)
    -- return architectures.createVGG16Model(imsize,noutputs,opt)
    --return architectures.StartModel(imsize,noutputs,opt)
    -- return architectures.ResBuildingBlockOnly(imsize,noutputs,opt)
    --return architectures.createFactorBlockModel(imsize,noutputs,opt)
    return architectures.createStandardBlockModel(imsize,noutputs,opt)
end


function architectures.StandardBlockOnly(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 1024, 2, 3, 1))
    model:add(modules.StandardActivation(1024)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],1), shrink(imsize[2],1),opt))
    return model
end

function architectures.FactorBlockOnly(imsize,noutputs,opt)
   model = nn.Sequential()
   -- Begin Parallel Tracks
   model:add(nn.SplitTable(2))
        :add(nn.ParallelTable()
                :add(nn.Reshape(1,imsize[1],imsize[2],true))
                :add(nn.Reshape(1,imsize[1],imsize[2],true)))
        :add(modules.FactorBlock(1,2,imsize[1],imsize[2]))
        :add(nn.JoinTable(2))
        :add(modules.OutputBlock(2,noutputs,imsize[1],imsize[2],opt))
    return model
end

function architectures.ResBuildingBlockOnly(imsize,noutputs,opt)
   model = nn.Sequential()
   -- Begin Parallel Tracks
   model:add(nn.SplitTable(2))
        :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
        :add(modules.ResBuildingBlockParallel(1,32,64,2,true))
        :add(nn.JoinTable(2))
        :add(modules.OutputBlock(128,noutputs,shrink(imsize[1],1),shrink(imsize[2],1),opt))
    return model
end

function architectures.createSmallStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],3), shrink(imsize[2],3),opt,false))
    return model
end

function architectures.createLargeStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    model:add(modules.StandardBlock(2, 64, 1, 3, 1))
    model:add(modules.StandardBlock(64, 64, 2, 3, 1)) -- Stride here
    model:add(modules.StandardBlock(64, 128, 1, 3, 1))
    model:add(modules.StandardBlock(128, 128, 2, 3, 1)) -- Stride here
    model:add(modules.StandardBlock(128, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 2, 3, 1)) -- Stride here
    model:add(modules.StandardBlock(256, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 2, 3, 1)) -- Stride here
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1)) -- Stride here
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt))
    return model
end

function architectures.createLargeStandardBlockModel10(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 1, 3, 1))
    model:add(modules.StandardBlock(64, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 1, 3, 1))
    model:add(modules.StandardBlock(128, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt))
    return model
end

function architectures.createStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt))
    return model
end

function architectures.createVGG16Model(imsize,noutputs,opt)
   -- TODO: Make sure the spatial dimensions at the end match up.
   debugger.enter()
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 1, 3, 1))
    model:add(modules.StandardBlock(64, 64, 1, 3, 1))

    model:add(nn.SpatialMaxPooling(imsize[1], imsize[2], 2, 2, 0, 0))

    model:add(modules.StandardBlock(64, 128, 1, 3, 1))
    model:add(modules.StandardBlock(128, 128, 1, 3, 1))

    model:add(nn.SpatialMaxPooling(shrink(imsize[1],1), shrink(imsize[2],1), 2, 2, 0, 0))

    model:add(modules.StandardBlock(128, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))

    model:add(nn.SpatialMaxPooling(shrink(imsize[1],2), shrink(imsize[2],2), 2, 2, 0, 0))

    model:add(modules.StandardBlock(256, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))

    model:add(nn.SpatialMaxPooling(shrink(imsize[1],3), shrink(imsize[2],3), 2, 2, 0, 0))

    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))

    model:add(nn.SpatialMaxPooling(shrink(imsize[1],4), shrink(imsize[2],4), 2, 2, 0, 0))
    model:add(nn.Reshape(512*shrink(imsize[1],5)*shrink(imsize[2],5),true))

    model:add(nn.Linear(512*shrink(imsize[1],5)*shrink(imsize[2],5,4096)))
    model:add(nn.Linear(4096,4096))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt))
    return model
end


function architectures.createStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.StandardActivation(1024)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt))
    return model
end

function architectures.ModelSimple(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(128,128,128,2))
    model:add(modules.StandardActivation(128)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(128,noutputs,shrink(imsize[1],2), shrink(imsize[2],2),opt,true))
    return model
end

function architectures.StartModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlock(64,64,64,1,true))
    model:add(modules.ResBuildingBlock(64,64,64,1))
    model:add(modules.ResBuildingBlock(64,64,64,1))
    model:add(modules.ResBuildingBlock(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,512,512,2)) -- Stride here
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end


function architectures.createResBuildingBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlockParallel(64,64,64,1,true))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(512,512,512,2))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end

function architectures.createFactorBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlockParallel(64,64,64,1,true))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.FactorBlock(256,256,shrink(imsize[1],3),shrink(imsize[2],3)))
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(512,512,512,2))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end


-- function architectures.createFactorBlockModel(imsize,noutputs,opt)
--     model = nn.Sequential()
--     -- Begin Parallel Tracks
--     model:add(nn.SplitTable(2))
--          :add(nn.ParallelTable()
--                  :add(nn.Reshape(1,imsize[1],imsize[2],true))
--                  :add(nn.Reshape(1,imsize[1],imsize[2],true)))
--     -- Initial Conv layer:
--     model:add(modules.StandardBlockParallel(1, 64, 1, 3, 1))
--     -- Bottleneck and Factor Block alternation
--     model:add(modules.ResBuildingBlockParallel(64,64,64,2,true)) -- Stride here
--     model:add(modules.FactorBlock(64,32,shrink(imsize[1],1),shrink(imsize[2],1)))
--     model:add(modules.ResBuildingBlockParallel(64,64,64)) -- No Stride here
--     model:add(modules.FactorBlock(64,32,shrink(imsize[1],1),shrink(imsize[2],1)))
--     model:add(modules.ResBuildingBlockParallel(64,64,128,2)) -- Stride here
--     model:add(modules.FactorBlock(128,64,shrink(imsize[1],2),shrink(imsize[2],2)))
--     model:add(modules.ResBuildingBlockParallel(128,128,256,2)) -- Stride here
--     model:add(modules.FactorBlock(256,128,shrink(imsize[1],3),shrink(imsize[2],3)))
--     model:add(nn.JoinTable(2))
--
--     --model:add(modules.ResBuildingBlock(256,256,256,1)) -- Stride here
--     --model:add(modules.ResBuildingBlock(256,256,512,2)) -- Stride here
--     --model:add(modules.ResBuildingBlock(512,512,512,1)) -- Stride here
--     --model:add(modules.ResBuildingBlock(256,256,256,1)) -- Stride here
--
--     model:add(modules.StandardActivation(shrink(imsize[1],3))) -- Final activation after final ResBlock, as in [2]
--     model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],3), shrink(imsize[2],3),opt))
--     return model
-- end



-- Deprecated. The output size here is probably wrong as well.
function createOldModel(imsize,noutputs,opt)
    local model = nn.Sequential()

    -- Split the input
    split_0 = nn.Sequential()
    split_0:add(nn.SplitTable(2))
    -- Add the graph
    iden = nn.Identity()()
    input_1, input_2 = split_0(iden):split(2)
    -- Reshaping for convolutional purposes
    reshape_0_1 = nn.Reshape(1,imsize[1],imsize[2],true)(input_1)
    reshape_0_2 = nn.Reshape(1,imsize[1],imsize[2],true)(input_2)

    ----------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------
    -- Main Convolutional part
    -- Layer 1 - Standard layer
    Layer_1_1 = modules.StandardBlock(1,64,2,7)
    Layer_1_2 = Layer_1_1:clone('weight','bias','gradWeight','gradBias')
    Layer_1_1 = Layer_1_1(reshape_0_1)
    Layer_1_2 = Layer_1_2(reshape_0_2)

    -- Layer 2 -  Convolution layer
                   -- :add(nn.SpatialMaxPooling(2, 2, 2, 2))
    Layer_2_1 = nn.Sequential()
                   :add(modules.BottleneckBlock(64,64,64,128,2))
    Layer_2_2 = Layer_2_1:clone('weight','bias','gradWeight','gradBias')
    Layer_2_1 = Layer_2_1(Layer_1_1)
    Layer_2_2 = Layer_2_2(Layer_1_2)

    -- Layer 3 - Convolution layer
    Layer_3_1 = modules.BottleneckBlock(128,64,64,256,2)
    Layer_3_2 = Layer_3_1:clone('weight','bias','gradWeight','gradBias')
    Layer_3_1 = Layer_3_1(Layer_2_1)
    Layer_3_2 = Layer_3_2(Layer_2_2)

    -- Layer 3 Sublayer 3 - Factoring layer
    -- Convlutional part (No ReLU or normalization, NOT tied)
    Layer_3_3 = nn.Sequential()
                    :add(modules.FactorBlock(128, 64,shrink(imsize[1],2), shrink(imsize[2],2), true))
                    :add(nn.CAddTable())
                    :add(nn.SpatialAveragePooling(2,2,2,2,shrink(imsize[1],2)%2,shrink(imsize[2],2)%2))({Layer_2_1,Layer_2_2})

    -- Layer 4 - Convolution layer
    Layer_4_1 = modules.BottleneckBlock(256,128,128,512,2)
    Layer_4_2 = Layer_4_1:clone('weight','bias','gradWeight','gradBias')
    Layer_4_1 = Layer_4_1(Layer_3_1)
    Layer_4_2 = Layer_4_2(Layer_3_2)
    -- Layer 4 Sublayer 3 - Factoring layer
    Layer_4_3 = nn.Sequential()
                    :add(modules.FactorBlock(256, 128, shrink(imsize[1],3), shrink(imsize[2],3), true))
                    :add(nn.CAddTable())
                    :add(nn.SpatialAveragePooling(2,2,2,2,shrink(imsize[1],3)%2,shrink(imsize[2],3)%2))({Layer_3_1,Layer_3_2})
    -- Layer 4 Sublayer 4 - Convolutional
    Layer_4_4 = modules.BottleneckBlock(128, 64, 64, 256, 2)(Layer_3_3)

    -- Layer 5 - Factoring layer
    Layer_5_1 = nn.Sequential()
                    :add(modules.FactorBlock(512, 256, shrink(imsize[1],4), shrink(imsize[2],4)))
                    :add(nn.CAddTable())({Layer_4_1,Layer_4_2})

    -- Layer 5 Sublayer 2 - Convolution layer
    Layer_5_2 = nn.Sequential()
                   :add(nn.JoinTable(2))
                   :add(modules.BottleneckBlock(256+256,256,256,512))({Layer_4_3,Layer_4_4})

    ----------------------------------------------------------------------------------
    -- Combining Layers
    -- Layer 6 - Convolution layer
    Layer_6_1 = nn.Sequential()
                   :add(nn.JoinTable(2))
                   :add(nn.SpatialMaxPooling(2, 2, 2, 2))
                   :add(modules.BottleneckBlock(512+512+512+512,512,512,2048,1))({
                         Layer_5_1,Layer_5_2,Layer_4_1,Layer_4_2
                      })

    -- Layer 7 - Fully connected layer
    local noutputs = #kittiflow.classes
    if opt.traintype == "regression" then
        noutputs = 6 -- 6 if using rotation, 3 if not
    end
    Layer_7 = nn.Sequential()
                :add(nn.Dropout()) -- For regularization (for robustness for spatial)
                :add(nn.SpatialAveragePooling(shrink(imsize[2],5), shrink(imsize[1],5), 1, 1, 0, 0)) -- AVERAGE pooling
                :add(nn.Reshape(2048,true))
                :add(nn.Linear(2048, noutputs))(Layer_6_1)

    g = nn.gModule({iden}, {Layer_7})
    model:add(g)

    -- if opt.traintype == "heatmap" then
    --     model:add(nn.SoftMax())
    -- end

    return model
end
