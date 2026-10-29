"""Source-exact 3D cutaway of the MCL layer, with steady-flow direction overlays."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

OUT = Path(__file__).resolve().parent
DATA = json.loads((OUT / "flow_scene.json").read_text())
OUTPUT = OUT / "UNET_3D_FLOW_v2.png"
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
M = .01  # one source mm becomes .01 Blender units, isotropically
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def material(name, color, metal=0., rough=.3, transmission=0., emission=False):
    mat=bpy.data.materials.new(name); mat.use_nodes=True
    nodes=mat.node_tree.nodes; node=nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value=(*color,1)
    node.inputs["Metallic"].default_value=metal
    node.inputs["Roughness"].default_value=rough
    node.inputs["Transmission Weight"].default_value=transmission
    if transmission: node.inputs["IOR"].default_value=1.333
    if emission:
        node.inputs["Emission Color"].default_value=(*color,1)
        node.inputs["Emission Strength"].default_value=1
    return mat


copper=material("Copper",(.58,.235,.085),.82,.28)
water=material("Water_blue_display",(.008,.16,.30),.05,.22,.25)
flow=material("Flow_direction",(.55,.95,1),0,.2,emission=True)
dark=material("Silicon",(.025,.038,.055),.25,.32)
tim=material("TIM",(.5,.55,.58),0,.55)
paper=material("Background",(.91,.93,.95),0,.65)
ink=material("Labels",(.018,.035,.055),emission=True)
blue=material("Boundary_flow_arrows",(.015,.32,.48),emission=True)


def cube(name,center,extent,mat):
    bpy.ops.mesh.primitive_cube_add(size=1,location=Vector(center)*M)
    obj=bpy.context.object; obj.name=name
    obj.dimensions=Vector(extent)*M
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(mat)
    return obj


def volume(name, mask, zlo, zhi, mat):
    verts=[]; faces=[]; cache={}; dx=DATA["dx_mm"];dy=DATA["dy_mm"]
    def face(coords):
        ids=[]
        for xyz in coords:
            if xyz not in cache:
                cache[xyz]=len(verts);verts.append(tuple(v*M for v in xyz))
            ids.append(cache[xyz])
        faces.append(ids)
    for i in range(64):
        for j in range(64):
            if not mask[i][j]:continue
            x0,x1,y0,y1=i*dx,(i+1)*dx,j*dy,(j+1)*dy
            face([(x0,y0,zhi),(x1,y0,zhi),(x1,y1,zhi),(x0,y1,zhi)])
            face([(x0,y1,zlo),(x1,y1,zlo),(x1,y0,zlo),(x0,y0,zlo)])
            if i==0 or not mask[i-1][j]:
                face([(x0,y0,zlo),(x0,y0,zhi),(x0,y1,zhi),(x0,y1,zlo)])
            if i==63 or not mask[i+1][j]:
                face([(x1,y1,zlo),(x1,y1,zhi),(x1,y0,zhi),(x1,y0,zlo)])
            if j==0 or not mask[i][j-1]:
                face([(x1,y0,zlo),(x1,y0,zhi),(x0,y0,zhi),(x0,y0,zlo)])
            if j==63 or not mask[i][j+1]:
                face([(x0,y1,zlo),(x0,y1,zhi),(x1,y1,zhi),(x1,y1,zlo)])
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def tube(name, points, radius, mat):
    curve=bpy.data.curves.new(name,"CURVE");curve.dimensions="3D"
    curve.bevel_depth=radius*M;curve.bevel_resolution=3;curve.resolution_u=1
    spline=curve.splines.new("POLY");spline.points.add(len(points)-1)
    for item,p in zip(spline.points,points):item.co=(*[v*M for v in p],1)
    obj=bpy.data.objects.new(name,curve);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat);return obj


def arrow(point, direction, size=1.05, mat=flow):
    direction=Vector(direction).normalized()
    bpy.ops.mesh.primitive_cone_add(vertices=24,radius1=.34*size*M,radius2=0,
                                   depth=size*M,location=Vector(point)*M)
    obj=bpy.context.object;obj.name="Flow_direction_arrow"
    obj.rotation_euler=direction.to_track_quat("Z","Y").to_euler()
    obj.data.materials.append(mat)


# Actual four-layer stack; no invented package, roof, pump or round hoses.
for i,c in enumerate(DATA["chiplets"]):
    x=(c["x"]+c["width"]/2)*DATA["dx_mm"]
    y=(c["y"]+c["height"]/2)*DATA["dy_mm"]
    w=c["width"]*DATA["dx_mm"];h=c["height"]*DATA["dy_mm"]
    cube(f"Silicon_{i}",(x,y,.25),(w,h,.5),dark)
    cube(f"TIM_{i}",(x,y,.55),(w,h,.1),tim)
cube("Copper_base",(37.7,36,1.1),(75.4,72,1),copper)
mask=DATA["binary"]
solid=[[not value for value in row] for row in mask]
volume("Copper_channel_walls",solid,1.6,2.6,copper)
volume("Water_in_exact_cells",mask,1.6,2.6,water)
for i,path in enumerate(DATA["paths_mm"]):
    points=[(x,y,2.66) for x,y in path]
    tube(f"Depth_averaged_streamline_{i}",points,.052,flow)
    # All arrows follow the locally computed tangent, not an invented circulation.
    for k in range(100,len(points)-10,180):
        tangent=Vector(points[k+3])-Vector(points[k-3])
        arrow(points[k],tangent,.95)
for start,end in (((-12,36,2.7),(-2,36,2.7)),((77.4,36,2.7),(87.4,36,2.7))):
    tube("Port_direction",[start,end],.2,blue)
    arrow(end,(1,0,0),2,blue)

# Neutral studio ground is display context only.
cube("Studio_floor",(37.7,36,-.9),(450,450,.5),paper)
scene=bpy.context.scene
scene.render.engine="CYCLES"
scene.cycles.samples=128
# The local Blender build exited during OIDN postprocessing. Raw Cycles
# finished successfully with the identical saved geometry and lighting.
scene.cycles.use_denoising=False
scene.cycles.max_bounces=8
try:
    preferences=bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type="OPTIX";preferences.get_devices()
    for device in preferences.devices:device.use=device.type=="OPTIX"
    scene.cycles.device="GPU" if any(d.use for d in preferences.devices) else "CPU"
except Exception:
    scene.cycles.device="CPU"
scene.render.resolution_x=3000;scene.render.resolution_y=2000
scene.render.resolution_percentage=100
scene.render.image_settings.file_format="PNG"
scene.render.filepath=str(OUTPUT)
scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.7,.78,.9,1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value=.45
scene.view_settings.view_transform="AgX"
scene.view_settings.look="AgX - Medium High Contrast"
scene.unit_settings.system="METRIC";scene.unit_settings.scale_length=.1


def light(name,position,power,size,target=(37.7,36,0)):
    data=bpy.data.lights.new(name,"AREA");data.energy=power;data.shape="DISK";data.size=size*M
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
    obj.location=Vector(position)*M
    obj.rotation_euler=(Vector(target)*M-obj.location).to_track_quat("-Z","Y").to_euler()


light("Key_softbox",(20,-50,150),12,120)
light("Fill_softbox",(-90,85,80),6,100)
light("Copper_rim",(120,120,110),16,70)
camera_data=bpy.data.cameras.new("Camera")
camera=bpy.data.objects.new("Camera",camera_data);bpy.context.collection.objects.link(camera)
camera.location=Vector((113,-117,105))*M
target=Vector((37.7,36,1.3))*M
camera.rotation_euler=(target-camera.location).to_track_quat("-Z","Y").to_euler()
camera_data.type="ORTHO";camera_data.ortho_scale=1.28;camera_data.lens=55
camera_data.clip_start=.001;camera_data.clip_end=100
scene.camera=camera
font=bpy.data.fonts.load("C:/Windows/Fonts/arial.ttf")
bold=bpy.data.fonts.load("C:/Windows/Fonts/arialbd.ttf")


def overlay(text,x,y,size,mat=ink,strong=False):
    data=bpy.data.curves.new("Label","FONT");data.body=text;data.font=bold if strong else font
    data.size=size;data.align_x="LEFT";data.extrude=0
    obj=bpy.data.objects.new("Label",data);bpy.context.collection.objects.link(obj)
    obj.rotation_euler=camera.rotation_euler
    obj.location=camera.location+camera.rotation_euler.to_matrix()@Vector((x,y,-.3))
    obj.data.materials.append(mat)
    obj.visible_shadow=False;obj.visible_glossy=False


overlay("U-Net (наша модель)  |  каналы и поток",-.59,.365,.034,strong=True)
overlay("Точная конструкция из расчёта · пример №6",-.59,.323,.022)
overlay("Верх открыт для обзора. Линии — направление рассчитанного потока.",-.59,-.365,.019)
overlay("75,4 × 72 мм · глубина каналов 1 мм · усреднённая по глубине модель",-.59,-.397,.017)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"UNET_3D_FLOW_v2.blend"))
print("WAVEFORGE_RENDER_DEVICE",scene.cycles.device,flush=True)
bpy.ops.render.render(write_still=True)
print("WAVEFORGE_RENDER_COMPLETE",str(OUTPUT),flush=True)
