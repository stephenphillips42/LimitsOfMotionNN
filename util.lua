require 'cunn'
local ffi=require 'ffi'
-- Adapted from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/util.lua
local debugger= require 'fb.debugger'

function makeDataParallel(model, nGPU)
   print('converting module to nn.DataParallelTable')
   assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
   local model_single = model
   model = nn.DataParallelTable(1)
   for i=1, nGPU do
      cutorch.setDevice(i)
      model:add(model_single:clone():cuda(), i)
   end
   cutorch.setDevice(1)

   return model
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, model:get(1))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(module:get(1)) -- strip DPT
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model:float())
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function tableHash(tbl)
  s = ""
  for k,v in pairs(tbl) do
    if type(v) == 'number' then
      if v - floor(v) == 0 then
        s = s .. v
      else
        s = s .. floor(v) .. "_" .. string.sub(tostring(v - floor(v)),3)
      end
    elseif type(v) == 'string' then
      s = s .. v
    elseif type(v) == 'boolean' then
      if v then
        s = s .. 1
      else
        s = s .. 0
      end
    elseif type(v) == 'table' then
      s = s .. "__" .. tableHash(v) .. "__"
    elseif type(v) == 'userdata' then
      s = s .. 'userdata'
    end
    s = s .. "-"
  end
  return string.sub(s,0,#s-1)
end

