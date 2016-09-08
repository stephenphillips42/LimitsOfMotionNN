--[[
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

-- load torchnet:
require 'torch'
require 'nn'
require 'optim'
require 'paths'
local tnt = require 'torchnet'
matio = require 'matio'
debugger = require('fb.debugger')

torch.setdefaulttensortype('torch.FloatTensor')

require 'architectures'
require 'custom-modules'
paths.dofile('util.lua')

-- use GPU or not:
-- local cmd = torch.CmdLine()
-- cmd:option('-usegpu', false, 'use gpu for training')
----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")       subdirectory to save logs
   -l,--load          (default "")           If nonempty, load previously saved model
   -p,--noplot                               plot while training
   -o,--optimization  (default "ADAM")       optimization: SGD | ADAM
   -e,--maxEpochs     (default 15)           Maximum number of epochs before halting
   -r,--learningRate  (default 0.1)          Learning rate
   -t,--traintype     (default "regression") Use regression or datagen
   -b,--batchSize     (default 8)            batch size
   -u,--momentum      (default 0.9)          momentum, for SGD only
   -w,--weightDecay   (default 0)            L2 penalty on the weights
   -n,--noiseSigma    (default 0)            Sigma for Gaussian noise added to training images
   --nogpu                                   Run without the GPU (runs with by default)
   --conover                                 Run controlled overfitting (10 examples, train/test) (defaults to false)
   --createdata                              Create the dataset
   --topdir           (default "")           To specify this directory, use .
]]
-- TODO: This is getting too complicated, use torch parameters
-- TODO: Implement "traintype" i.e. translation and trans_rot
-- TODO: Implement "optimization"
-- TODO: Make dataset sizes parameters
-- TODO: Make imsize configurable

-----------------------------------------------------------
-- Parameters
-----------------------------------------------------------
-- Dataset
local noutputs = 3 
local imsize = {128, 128}
local dataset_size = {
   train = 10000,
   test = 10000,
   test_inter = 2000,
}

-- Logging
opt.plot = not opt.noplot
if opt.save == "logs" then -- change default
    opt.save = "/scratch/KITTINet/logs"
end
local trainBatchLogger = optim.Logger(paths.concat(opt.save,'batchTrain.log'))
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local trainLogInterval = 10

-- Flow making
local dataset_seed = {
   train = 10007,
   test = 31907,
   test_inter = 60317,
}
if opt.topdir == "" then
   opt.topdir = "/scratch/KITTINet/synthFlowDataset"
end


function saveNet(opt,model,optConfig,optState,model_id)
   if opt.save == "" then
      return
   end
   -- Save out the current model state.
   local filename = paths.concat(opt.save, model_id .. '.net')
   local optFilename = paths.concat(opt.save, model_id .. '_opt.t7')
   os.execute('mkdir -p ' .. sys.dirname(filename))

   print('Saving network to '..filename)
   -- Save out the copy from ,a single GPU: don't need the 4x rep
   collectgarbage()
   model:clearState() -- clear intermediary states before saving.
   saveDataParallel(filename,model)
   if not opt.nogpu then
      model:cuda()
   end
   -- Save optimization state as well
   print('Saving optimization state to '..optFilename)
   local optData = {['optState'] = optState, ['optConfig'] = optConfig}
   torch.save(optFilename, optData)
end


-----------------------------------------------------------
-- Dataset setup
-----------------------------------------------------------
-- Build x,y matrices
xPixel = torch.Tensor(imsize[1],imsize[2])
yPixel = torch.Tensor(imsize[1],imsize[2])
for i=1,imsize[1] do
   for j=1,imsize[2] do
      xPixel[{i,j}] = j
      yPixel[{i,j}] = i
   end
end


-- Build flow parameters
local flowParams = {
   useOmega = false,
   omegaNorm = 1,
   meanDepth = 2,
   minDepth = 0.2,
   imCenter = {imsize[1]/2,imsize[2]/2},
   focalLength = 100,
   infDepthRatio = 0.1,
   noiseSigma = opt.noiseSigma,
}

flowParams.xCal = (xPixel - flowParams.imCenter[2])/flowParams.focalLength
flowParams.yCal = (xPixel - flowParams.imCenter[2])/flowParams.focalLength


-- TODO: Figure out outlier generation function
-- TODO: Create sparse version
local function createFlow(mode,i)
   seed = dataset_seed[mode] + i - 1
   gen = torch.Generator()
   torch.manualSeed(gen, seed)
   flow = torch.Tensor(2,imsize[1],imsize[2])
   foe = torch.randn(gen,3)
   foe = foe:div(torch.norm(foe))
   omega = torch.zeros(3)
   if flowParams.useOmega then
      omega = flowParams.omegaNorm*torch.randn(gen,3)
   end
   invDepths = 
      torch.log(1-torch.rand(gen,imsize[1],imsize[2]))*flowParams.meanDepth
         + flowParams.minDepth
   invDepths[torch.lt(torch.rand(imsize[1],imsize[2]),flowParams.infDepthRatio)] = 0
   flow[{1,{},{}}] = invDepths:cmul(foe[3]*flowParams.xCal - foe[1])
                        + omega[1]*flowParams.xCal:cmul(flowParams.yCal)
                        - omega[2]*(1+flowParams.xCal:cmul(flowParams.xCal))
                        + omega[3]*(flowParams.yCal)
                        + torch.randn(gen,imsize[1],imsize[2])*flowParams.noiseSigma
   flow[{2,{},{}}] = invDepths:cmul(foe[3]*flowParams.xCal - foe[1])
                        + omega[2]*(1+flowParams.yCal:cmul(flowParams.yCal))
                        - omega[1]*flowParams.xCal:cmul(flowParams.yCal)
                        - omega[3]*(flowParams.xCal)
                        + torch.randn(gen,imsize[1],imsize[2])*flowParams.noiseSigma
   return {
      input = flow,
      target = foe,
   }
end


local function saveFlow(mode,i)
   matio.save(string.format('%s/%s/%06d.mat',opt.topdir,mode,i),createFlow(mode,i))
end

local function loadFlow(mode,i)
   -- ld = matio.load(string.format('%s/%s/%06d.mat',opt.topdir,mode,i))
   -- return {
   --    input = ld.flow,
   --    target = ld.foe,
   -- }
   return matio.load(string.format('%s/%s/%06d.mat',opt.topdir,mode,i))
end



-- function that sets of dataset iterator:
function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() 
         require 'torchnet'
         matio = require 'matio'
         debugger = require('fb.debugger')
         require 'dataset-kitti-flow' 
         dataset_size = dataset_size[mode]
         torch.setdefaulttensortype('torch.FloatTensor')
      end,
      closure = function()
         -- return batches of data:
         return tnt.ShuffleDataset{ dataset = tnt.BatchDataset{
            batchsize = 4*opt.batchSize,
            dataset = tnt.ListDataset {  -- replace this by your own dataset
               list = torch.range(1, dataset_size[mode]):long(),
               load = function(idx)
                  -- sample contains input and target
                  return loadFlow(mode,idx)
                  -- return createFlow(mode,idx)
               end,
            }
         },
      }
      end,
   }
end

-----------------------------------------------------------
-- CNN Setup
-----------------------------------------------------------
-- local net = architectures.createModel(imsize,noutputs,opt)
local net = nn.Sequential()
net:add(modules.StandardBlock(2, 128, 2, 3, 1))
net:add(modules.StandardBlock(128, 512, 2, 3, 1))
net:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],2), shrink(imsize[2],2),opt,false))

local criterion = nn.MSECriterion()

-----------------------------------------------------------
-- Training and Testing
-----------------------------------------------------------

-- set up training engine:
local engine = tnt.OptimEngine()
local meter  = tnt.AverageValueMeter()
local meter_test  = tnt.AverageValueMeter()
local best_value = 1/0
engine.hooks.onStartEpoch = function(state)
   meter:reset()
end

-- TODO: Make the percentage meter
engine.hooks.onForwardCriterion = function(state)
   if state.training then
      meter:add(state.criterion.output)
      if state.t % trainLogInterval == 0 then
         avg_value = meter:value()
         print(string.format('Average loss: %2.4f', avg_value))
         trainBatchLogger:add{['Raw loss % batch mean error (train set)'] = state.criterion.output}
         if opt.plot then
            trainBatchLogger:style{['Raw loss % batch mean error (train set)'] = '-'}
            trainBatchLogger:plot()
         end
      end
   else
      meter_test:add(state.criterion.output)
   end
end

engine.hooks.onEndEpoch = function (state)
   print("End Epoch " .. state.epoch)
   meter_test:reset()
   engine:test{
      network   = net,
      iterator  = getIterator('test_inter'),
      criterion = criterion,
   }
   print(string.format('Intermediate Test Loss: %2.4f', meter_test:value()))
   testLogger:add{['Raw loss % batch mean error (train set)'] = state.criterion.output}
   if opt.plot then
      testLogger:style{['Raw loss % batch mean error (train set)'] = '-'}
      testLogger:plot()
   end
   saveNet(opt,net,state.config,state.optim,'most_recent')
   if meter_test:value() < best_value then 
      saveNet(opt,net,state.config,state.optim,'best_value')
   end
end

-- set up GPU training:
if not opt.nogpu then
   -- copy model to GPU:
   require 'cunn'
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end


if not opt.createdata then
   mylr = opt.learningRate
   for outerit=1,10 do -- TODO: Make this
      -- train the model:
      engine:train{
         network     = net,
         iterator    = getIterator('train'),
         criterion   = criterion,
         optimMethod = optim.adam,
         config = {
            learningRate       = mylr,
         },
         maxepoch    = 5,
      }

      -- measure test loss and error:
      meter_test:reset()
      engine:test{
         network   = net,
         iterator  = getIterator('test'),
         criterion = criterion,
      }
      print(string.format('Test loss: %2.4f', meter_test:value()))
      testLogger:add{['Raw loss % batch mean error (train set)'] = meter_test:value()}
      if opt.plot then
         testLogger:style{['Raw loss % batch mean error (train set)'] = '-'}
         testLogger:plot()
      end
      -- TODO: Figure out way to reduce learning rate
      -- mylr = mylr*0.1
   end

else -- Specified that we need to make the dataset
   print(dataset_size)
   for mode,nFlow in pairs(dataset_size) do
      for i=1,nFlow do
         saveFlow(mode,i)
         if i % 100 == 0 then
            print(string.format("Creating Flow %d of mode %s",i,mode))
         end
      end
   end
end


