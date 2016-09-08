#!/usr/bin/blender --background --python
# Blender libraries
import bpy
from bpy import context
# Standard libraries
import math
from math import sin, cos, radians
import random as rand
import time
# System libraries
import sys
import os
#Output resolution (Stereoscopic images & depthmap)
bpy.context.scene.render.resolution_x = 150
bpy.context.scene.render.resolution_y = 100

# Total number of set of stereoscopic images and depth maps
total_scene_number = 100

# Constants
IMG_WIDTH  = 1024
IMG_HEIGHT = 1024
GROUND_PLANE_NAME = 'Plane'
GROUND_PLANE_MAT_NAME = 'Material'
GROUND_SCALE = 50

# Global variables
g_materials = []


##################### Helper Functions ########################
def CubeName(cubeNum):
    if cubeNum <= 0:
        return "Cube"
    else:
        return "Cube.{:03d}".format(cubeNum)

def MaterialName(cubeNum):
    return "Material.{:03d}".format(cubeNum+1)

def BlockSize():
    return min([max([(rand.random()+1)*0.1, rand.expovariate(1.0/1.5)]),8])

def CubeLocation():
    return (2*rand.random()-1)*GROUND_SCALE

def ConvertToBase256(x):
    x1 = x % 256
    x2 = (x // 256) % 256
    x3 = (x // 256**2) % 256
    return (x1, x2, x3)

# Credit to StackOverflow
def redirect_stdout():
    # Flush previous output
    sys.stdout.flush()
    # Generate the new standard out and close old one
    newstdout = os.dup(1)
    devnull = os.open('/dev/null', os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    # Make it so that python can still use "print"
    sys.stdout = os.fdopen(newstdout, 'w')

################ Object Generator Functions ###################
def CreateGroundPlane():
    bpy.ops.mesh.primitive_plane_add(
        location=(0,0,0))
    bpy.ops.transform.resize(value=(GROUND_SCALE, GROUND_SCALE, GROUND_SCALE))
    bpy.ops.material.new()
    ground_mat = bpy.data.materials[GROUND_PLANE_MAT_NAME]
    g_materials.append(ground_mat)
    # ground_mat.use_shadeless = True
    ground_mat.diffuse_color = (1.0, 1.0, 1.0)

    bpy.data.objects[GROUND_PLANE_NAME].data.materials.append(ground_mat)

def CreateGroundCube(cubeNum):
    bpy.ops.mesh.primitive_cube_add(
        location=(CubeLocation(), CubeLocation(), 0))
    bpy.ops.transform.resize(value=(BlockSize(), BlockSize(), 2*BlockSize()))
    bpy.ops.material.new()
    cube_mat = bpy.data.materials[MaterialName(cubeNum)]
    g_materials.append(cube_mat)
    # cube_mat.use_shadeless = True
    cube_mat.diffuse_color = (rand.random(), rand.random(), rand.random())

    bpy.data.objects[CubeName(cubeNum)].data.materials.append(cube_mat)
    

################ Generate Scene Function ###################
def GenerateScene(base_dir,imNum):
    # Clear data from previous scenes
    for material in bpy.data.materials:
        material.user_clear();
        bpy.data.materials.remove(material);

    for texture in bpy.data.textures:
        texture.user_clear();
        bpy.data.textures.remove(texture);

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Setup lighting
    light = bpy.data.objects['Lamp']
    light.data.energy = 2.00
    light.select = False

    # Setup camera
    camera = bpy.data.objects['Camera']
    camera.select = True

    camera.rotation_mode = 'XYZ'
    angle1 = (80.0*math.pi/180.0)*rand.random()
    angle2 =  (5.0*math.pi/180.0)*rand.gauss(0.0,1.0)
    angle3 =              math.pi*rand.random()
    camera.rotation_euler = (angle1, angle2, angle3)

    Cam_x =      0.75*rand.random()
    Cam_y =      0.75*rand.random()
    Cam_z = 0.5 + 10.0*rand.random()
    camera.location = (Cam_x,Cam_y,Cam_z)
    camera.data.lens = 15 # focal length
    print((180.0*angle1/math.pi, 180.0*angle2/math.pi, 180.0*angle3/math.pi),file=sys.stderr)
    print((Cam_x, Cam_y, Cam_z),file=sys.stderr)
    camera.select = False

    # Remove objects from previsous scenes
    for item in bpy.data.objects:  
        if item.type == "MESH":  
            bpy.data.objects[item.name].select = True
            bpy.ops.object.delete()
        
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)

    # Create new scene -- setup for depth map
    scene = bpy.context.scene
    # OK I was not familiar with this so here is the explaination
    # This is basically a graph with each node doing something
    # Render Layers (rl)
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    # Compositor node - just basically
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    # composite.location = (200,0)
    # Inverse Depth Nodes
    normalize = tree.nodes.new(type = "CompositorNodeNormalize")
    invert = tree.nodes.new(type = "CompositorNodeInvert")

    scene = bpy.context.scene

    # Setup the depthmap calculation using blender's mist function:
    scene.render.layers['RenderLayer'].use_pass_mist = True
    # The depthmap can be calculated as the distance between objects
    # and camera ('LINEAR'), or square/inverse square of the distance
    # ('QUADRATIC'/'INVERSE_QUADRATIC')
    scene.world.mist_settings.falloff = 'LINEAR'
    # Minimum/Maximum depth:
    scene.world.mist_settings.intensity = 0.0
    scene.world.mist_settings.start = 0.005
    scene.world.mist_settings.depth = 2*Cam_z/cos(angle1)

    #magnitude of the random variation of object placements:
    magn = 10;

    # Create objects
    CreateGroundPlane()

    nCubes = 250
    for i in range(nCubes):
        CreateGroundCube(i)

    # Creating the image outputs
    bpy.context.scene.render.resolution_x = IMG_WIDTH 
    bpy.context.scene.render.resolution_y = IMG_HEIGHT

    # Redirect I/O (since it gets verbose)
    redirect_stdout()

    # Render the scene normally
    for i in range(len(g_materials)):
        g_materials[i].use_shadeless = False
    links.new(rl.outputs['Image'],composite.inputs['Image'])
    scene.render.filepath = '{}/Image{:06d}.png'.format(base_dir,imNum)
    bpy.ops.render.render( write_still=True ) 

    # Create the labels
    for i in range(len(g_materials)):
        g_materials[i].use_shadeless = True
        (i1,i2,i3) = ConvertToBase256(i+1)
        g_materials[i].diffuse_color = (i1/256.0, i2/256.0, i3/256.0)
    links.new(rl.outputs['Image'],composite.inputs['Image'])
    scene.render.filepath = '{}/Labels{:06d}.png'.format(base_dir,imNum)
    bpy.ops.render.render( write_still=True ) 

    # Output the inverse depth map
    # Create 
    links.new(rl.outputs['Mist'],composite.inputs['Image'])
    scene.render.filepath = '{}/Depth{:06d}.png'.format(base_dir,imNum)
    bpy.ops.render.render( write_still=True ) 

    # Normalized Depth
    links.new(rl.outputs['Z'], invert.inputs['Color'])
    links.new(invert.outputs['Color'], normalize.inputs['Value'])
    scene.render.filepath = "{}/InvDepth{:06d}.png".format(base_dir,imNum)
    links.new(normalize.outputs['Value'], composite.inputs['Image'])
    bpy.ops.render.render( write_still=True ) 


if __name__ == '__main__':
    # Get Blender Script Arguments 
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    imNum = int(argv[0])
    # Go crazy and generate scene
    rand.seed(imNum*202549)
    # GenerateScene(imNum)
    GenerateScene("/scratch/tmp",13)
