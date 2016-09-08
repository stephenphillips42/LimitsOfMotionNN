

require 'torch'
require 'nn'
require 'cudnn'
require 'image'
require 'paths'
paths.dofile('util.lua')

debugger = require('fb.debugger')


-- TODO: This is not super general yet - get it to be so
function visualizeFirstLayerColor(args)
    -- Defaults
    netfile = args.netfile or error("Need to specify which net file")
    side = args.side or 3
    padding = args.padding or 1
    savefile = args.savefile

    -- Load the net and weights
    net = torch.load(netfile):float()
    w = net:get(1):get(1).weight:float()

    -- Setup
    deftype = torch.Tensor():type()
    torch.setdefaulttensortype(w:type())
    minval = torch.min(w)
    maxval = torch.max(w)
    batchSize = w:size(1)

    -- Format them properly since torch can't seem to do it :/
    nChannels = w:size(2) / (side*side)
    if nChannels == 2 then
        -- print(w:size())
        -- return
        -- w = torch.cat({w,maxval*torch.ones(batchSize,side*side)},2);
        sel = torch.totable(torch.range(1,w:size(2),2))
        w = w:index(2,torch.LongTensor(sel))
        -- w = w[{{},{1,w:size(2)/2}}]
        -- w = w[{{},{w:size(2)/2+1,w:size(2)}}]
        wTable = nn.SplitTable(1):forward(
                        torch.reshape(w,batchSize,1,side,side))
        -- -- Build legend
        -- v1 = torch.reshape(torch.range(0,1,1.0/(side-1)),1,side)
        -- v2 = torch.ones(1,side)
        -- -- side = 2*padding+side
        -- wTable[#wTable+1] = torch.cat({
        --     torch.reshape(-(maxval-minval)*(v2:transpose(1,2)*v1)+maxval,1,side,side),
        --     torch.reshape(-(maxval-minval)*(v1:transpose(1,2)*v2)+maxval,1,side,side),
        --     maxval*torch.ones(1,side,side)
        --     },1)
    else
        wTable = nn.SplitTable(1):forward(
                        torch.reshape(w,batchSize,nChannels,side,side))
    end


    -- Display
    displayimage = image.toDisplayTensor(wTable,padding,12)
    displayimage = image.scale(displayimage, 15*displayimage:size(3), 15*displayimage:size(2),'simple')
    image.display{image=displayimage}
    if savefile then
        image.save(savefile, displayimage)
    end

    -- Cleanup
    torch.setdefaulttensortype(deftype)
end

visualizeFirstLayerColor{
    netfile='/scratch/checkpoints/compositional_motion/model_most_recent.net',
    padding=1,
    savefile='/scratch/checkpoints/compositional_motion/filters_viz.png',
    side=5}

-- -- Display images
-- -- TODO: Make file loading configurable
-- net = torch.load('/scratch/KITTINet/logs/most_recent.net')
-- w = net:get(1):get(1).weight
-- torch.setdefaulttensortype(w:type())
-- val = torch.max(w)
-- test1 = torch.cat({w,torch.zeros(w:size(1),9)},2);
-- test2 = (torch.reshape(test1,128,3,3,3))
-- test3 = nn.SplitTable(1):forward(test2)
-- test4 = {}
-- for i=1,#test3 do
--     pad1 = val*torch.ones(test3[i]:size(1),1,test3[i]:size(3))
--     tmp = torch.cat({pad1,test3[i],pad1},2)
--     pad2 = val*torch.ones(tmp:size(1),tmp:size(2),1)
--     test4[i] = torch.cat({pad2,tmp,pad2},3)
-- end

-- displayimage = image.toDisplayTensor(test4)
-- image.display{image=displayimage,zoom=5*3,padding=10}



