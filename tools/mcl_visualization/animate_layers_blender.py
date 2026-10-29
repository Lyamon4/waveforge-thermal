"""Reproducible exploded-view animation of sealed MCL geometry and steady flow.

Run with Blender 4.0 --background --factory-startup --python this.py -- preview
or -- render. No training data or physics parameters are modified.
"""
import bisect
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ANIMATION_001"
D = json.loads((OUT / "scene_data.json").read_text(encoding="utf-8"))
MODE = sys.argv[sys.argv.index("--")+1] if "--" in sys.argv else "preview"
M = .01
W, H, FPS, END = 1920, 1080, 24, 600
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 48
scene.eevee.use_gtao = True
scene.eevee.gtao_distance = .05
scene.eevee.gtao_factor = 1.1
scene.eevee.use_soft_shadows = True
scene.eevee.use_ssr = False
scene.render.resolution_x = W; scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.fps = FPS; scene.frame_start = 1; scene.frame_end = END
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.82,.88,.97,1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = .35
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.unit_settings.system = "METRIC"; scene.unit_settings.scale_length = .1


def rgb(hexcode):
    values = [int(hexcode[i:i+2],16)/255 for i in (1,3,5)]
    return tuple(v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values)


STOPS = [(0,rgb("#2165b5")),(.25,rgb("#42bce6")),
         (.5,rgb("#f4dd8b")),(.75,rgb("#ef8541")),(1,rgb("#c73730"))]


def temperature_color(fraction):
    t = max(0., min(1., fraction))
    for (a,ca),(b,cb) in zip(STOPS,STOPS[1:]):
        if t <= b:
            f = (t-a)/(b-a)
            return tuple(x*(1-f)+y*f for x,y in zip(ca,cb))
    return STOPS[-1][1]


def mat(name,color,metal=0.,rough=.4,flat=False):
    result = bpy.data.materials.new(name); result.use_nodes = True
    nodes = result.node_tree.nodes
    if flat:
        nodes.clear(); emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color,1)
        output = nodes.new("ShaderNodeOutputMaterial")
        result.node_tree.links.new(emission.outputs[0],output.inputs[0])
    else:
        shader = nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color,1)
        shader.inputs["Metallic"].default_value = metal
        shader.inputs["Roughness"].default_value = rough
    return result


copper = mat("Copper",(.6,.27,.11),.6,.36)
chip_side = mat("Silicon_edges",(.035,.05,.075),.2)
tim_mat = mat("TIM",(.62,.67,.71),.08)
floor_mat = mat("Studio",(.82,.85,.89),rough=.8)
ink = mat("Text",(.013,.026,.043),flat=True)
muted = mat("Secondary_text",(.11,.16,.21),flat=True)
cyan = mat("Flow_markers",(.52,.94,1),flat=True)
port_mat = mat("Port_arrows",(.02,.36,.62),flat=True)
heat_mat = mat("Heat_arrows",(.85,.32,.05),flat=True)
leader_mat = mat("Leader_lines",(.2,.28,.34),flat=True)
# Diagnostic overlays are unlit, matching the legend exactly. Copper and
# chip edges retain physical lighting so the assembly remains volumetric.
palette = [mat(f"Temperature_{i}",temperature_color(i/63),flat=True) for i in range(64)]
legend_palette = [mat(f"Legend_{i}",temperature_color(i/63),flat=True) for i in range(64)]


def empty(name,parent=None):
    obj = bpy.data.objects.new(name,None); scene.collection.objects.link(obj)
    obj.parent = parent; return obj


root = empty("Display_transform_only")
groups = [empty(name,root) for name in ("Chips","TIM_layer","Copper_base_layer","Water_channel_layer")]


def cube(name,center,extent,material,parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1,location=Vector(center)*M)
    obj=bpy.context.object; obj.name=name
    obj.dimensions=Vector(extent)*M
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(material); obj.parent=parent
    return obj


def cell_volume(name,mask,zlo,zhi,material,parent,field=None):
    vertices=[];faces=[];indices=[];cache={}
    def face(coords,index):
        ids=[]
        for xyz in coords:
            if xyz not in cache:
                cache[xyz]=len(vertices); vertices.append(tuple(x*M for x in xyz))
            ids.append(cache[xyz])
        faces.append(ids); indices.append(index)
    dx,dy=D["dx_mm"],D["dy_mm"]
    for i in range(64):
        for j in range(64):
            if not mask[i][j]: continue
            x0,x1,y0,y1=i*dx,(i+1)*dx,j*dy,(j+1)*dy
            index=1+max(0,min(63,round(field[i][j]/100*63))) if field is not None else 0
            face([(x0,y0,zhi),(x1,y0,zhi),(x1,y1,zhi),(x0,y1,zhi)],index)
            face([(x0,y1,zlo),(x1,y1,zlo),(x1,y0,zlo),(x0,y0,zlo)],0)
            if i==0 or not mask[i-1][j]:
                face([(x0,y0,zlo),(x0,y0,zhi),(x0,y1,zhi),(x0,y1,zlo)],0)
            if i==63 or not mask[i+1][j]:
                face([(x1,y1,zlo),(x1,y1,zhi),(x1,y0,zhi),(x1,y0,zlo)],0)
            if j==0 or not mask[i][j-1]:
                face([(x1,y0,zlo),(x1,y0,zhi),(x0,y0,zhi),(x0,y0,zlo)],0)
            if j==63 or not mask[i][j+1]:
                face([(x0,y1,zlo),(x0,y1,zhi),(x1,y1,zhi),(x1,y1,zlo)],0)
    mesh=bpy.data.meshes.new(name); mesh.from_pydata(vertices,[],faces); mesh.update()
    obj=bpy.data.objects.new(name,mesh); scene.collection.objects.link(obj)
    obj.parent=parent; obj.data.materials.append(material)
    if field is not None:
        for color in palette: obj.data.materials.append(color)
    for poly,index in zip(mesh.polygons,indices): poly.material_index=index
    return obj


for i,c in enumerate(D["chiplets"]):
    mask=[[c["x"]<=x<c["x"]+c["width"] and c["y"]<=y<c["y"]+c["height"]
           for y in range(64)] for x in range(64)]
    cell_volume(f"Chip_{i}",mask,0,.5,chip_side,groups[0],D["temperature_rise_K"][0])
    center=((c["x"]+c["width"]/2)*D["dx_mm"],(c["y"]+c["height"]/2)*D["dy_mm"],.55)
    cube(f"TIM_{i}",center,(c["width"]*D["dx_mm"],c["height"]*D["dy_mm"],.1),tim_mat,groups[1])
cube("Copper_base",(37.7,36,1.1),(75.4,72,1),copper,groups[2])
solid=[[not v for v in row] for row in D["binary"]]
cell_volume("Copper_walls",solid,1.6,2.6,copper,groups[3])
cell_volume("Water",D["binary"],1.6,2.6,palette[0],groups[3],D["temperature_rise_K"][3])


def curve(name,points,radius,material,parent=None,scaled=True):
    data=bpy.data.curves.new(name,"CURVE"); data.dimensions="3D"
    data.bevel_depth=radius*(M if scaled else 1); data.bevel_resolution=2
    sp=data.splines.new("POLY"); sp.points.add(len(points)-1)
    for p,xyz in zip(sp.points,points):
        p.co=(*[v*(M if scaled else 1) for v in xyz],1)
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
    obj.data.materials.append(material);obj.parent=parent
    return obj


def cone(name,point,direction,size,material,parent=None):
    bpy.ops.mesh.primitive_cone_add(vertices=16,radius1=.3*size*M,depth=size*M,
                                  location=Vector(point)*M)
    obj=bpy.context.object;obj.name=name;obj.parent=parent
    obj.rotation_euler=Vector(direction).to_track_quat("Z","Y").to_euler()
    obj.data.materials.append(material); return obj


for i,path in enumerate(D["paths"]):
    curve(f"Streamline_{i}",[(x,y,2.68) for x,y in path["xy_mm"]],.035,cyan,groups[3])
for a,b in (((-11,36,2.7),(-1.5,36,2.7)),((77,36,2.7),(86,36,2.7))):
    curve("Port_direction",[a,b],.18,port_mat,groups[3]);cone("Port_arrow",b,(1,0,0),2,port_mat,groups[3])
particles=[]
for i,path in enumerate(D["paths"]):
    for phase in (.12,.37,.62,.87):
        obj=cone(f"Water_marker_{i}_{phase}",(0,0,2.85),(1,0,0),1.35,cyan,groups[3])
        particles.append((obj,path,phase))

cube("Studio_floor",(37.7,36,-2),(550,550,.5),floor_mat)


def area_light(name,position,energy,size):
    data=bpy.data.lights.new(name,"AREA");data.energy=energy;data.size=size*M
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
    obj.location=Vector(position)*M
    obj.rotation_euler=(Vector((37.7,36,25))*M-obj.location).to_track_quat("-Z","Y").to_euler()


area_light("Key",(20,-80,180),12,140)
area_light("Fill",(-100,75,130),7,120)
area_light("Rim",(120,120,160),16,90)
camera_data=bpy.data.cameras.new("Camera")
camera=bpy.data.objects.new("Camera",camera_data);scene.collection.objects.link(camera)
camera.location=Vector((125,-155,175))*M
camera.rotation_euler=(Vector((37.7,36,33))*M-camera.location).to_track_quat("-Z","Y").to_euler()
camera_data.type="ORTHO";camera_data.ortho_scale=2.6;camera_data.shift_x=.16
camera_data.clip_start=.001;camera_data.clip_end=100;scene.camera=camera
font=bpy.data.fonts.load("C:/Windows/Fonts/arial.ttf")
bold=bpy.data.fonts.load("C:/Windows/Fonts/arialbd.ttf")
UX=2.6/W


def hud_point(x,y,z=-.4):
    # Lens shift moves the visible frame right in camera-local coordinates.
    return ((x/W-.5+camera_data.shift_x)*2.6,(.5-y/H)*2.6*H/W,z)


def text(body,x,y,size=26,material=ink,strong=False):
    data=bpy.data.curves.new("Label","FONT");data.body=body;data.font=bold if strong else font
    data.size=size*UX*1.4;data.align_x="LEFT"
    obj=bpy.data.objects.new("Label",data);scene.collection.objects.link(obj)
    obj.parent=camera;obj.location=hud_point(x,y);obj.data.materials.append(material)
    obj.visible_shadow=False
    return obj


def visible_between(obj,start,end):
    for frame,hidden in ((1,start>1),(max(1,start),False),(min(END,end+1),end<END)):
        obj.hide_render=hidden;obj.keyframe_insert("hide_render",frame=frame)
        obj.hide_viewport=hidden;obj.keyframe_insert("hide_viewport",frame=frame)


text("WaveForge  |  U-Net (наша модель)",65,66,38,strong=True)
text("Охлаждение по слоям",65,108,25,muted)
text("Нагрев относительно входа, °C",1300,887,22,muted)
for i,color in enumerate(legend_palette):
    x0,y0=hud_point(1300+i*8,908)[:2];x1,y1=hud_point(1300+(i+1)*8,928)[:2]
    mesh=bpy.data.meshes.new("Legend");mesh.from_pydata([(x0,y0,-.4),(x1,y0,-.4),(x1,y1,-.4),(x0,y1,-.4)],[],[(0,1,2,3)])
    obj=bpy.data.objects.new("Legend",mesh);scene.collection.objects.link(obj);obj.parent=camera;obj.data.materials.append(color)
for x,txt in ((1300,"0"),(1544,"50"),(1770,"100")):text(txt,x,956,21,muted)
factor=1/D["display_seconds_to_model_seconds"]
text(f"Поток замедлен в {factor:.1f} раза. Цвет — постоянная температура из расчёта, не остывание во времени.".replace("10.8","10,8"),65,1005,21,muted)
text("Слои разнесены для обзора. Верх открыт. Модель также учитывает отвод тепла с поверхности.",65,1040,19,muted)

caption_specs=[
    (1,100,"Вода входит слева",["Делится между каналами", "Забирает тепло и выходит справа"]),
    (101,445,"Конструкция по слоям",[]),
    (446,600,"Тепло уносит вода",["Показан установившийся режим", "Форма каналов взята из расчёта"]),
]
for start,end,heading,lines in caption_specs:
    obj=text(heading,1300,220,29,strong=True);visible_between(obj,start,end)
    for k,line in enumerate(lines):
        obj=text(line,1300,280+42*k,23,muted);visible_between(obj,start,end)

labels=[]
for layer,y,title,subtitle in [
    (3,333,"Каналы с водой","Вода принимает тепло"),
    (2,473,"Медное основание","Проводит тепло от чиплетов"),
    (1,613,"Термоинтерфейс","Передаёт тепло в медь"),
    (0,753,"Чиплеты процессора","8 вычислительных + 1 I/O"),
]:
    obj=text(title,1300,y,28,strong=True);visible_between(obj,101,445)
    obj=text(subtitle,1300,y+34,22,muted);visible_between(obj,101,445)
    line=curve("Layer_pointer",[(0,0,-.41)]*3,.7*UX,leader_mat,camera,scaled=False)
    visible_between(line,101,445);labels.append((line,layer,y-10))

port_labels=[text("Вход воды",0,0,23,port_mat,True),text("Выход воды",0,0,23,port_mat,True)]
heat_arrows=[]
for k in range(3):
    line=curve("Heat_transfer_explanation",[(12,-7,0),(12,-7,10)],.15,heat_mat,root)
    arrow=cone("Heat_transfer_explanation",(12,-7,10),(0,0,1),1.8,heat_mat,root)
    visible_between(line,130,420);visible_between(arrow,130,420)
    heat_arrows.append((line,arrow))


def smooth(t):
    t=max(0,min(1,t));return t*t*(3-2*t)


def separation(frame):
    if frame<49:return 0.
    if frame<169:return smooth((frame-48)/120)
    if frame<361:return 1.
    if frame<457:return 1-smooth((frame-360)/96)
    return 0.


def point_at_time(path,t):
    times=path["travel_s"];points=path["xy_mm"]
    k=min(len(times)-2,max(0,bisect.bisect_right(times,t)-1))
    f=(t-times[k])/(times[k+1]-times[k])
    a,b=points[k],points[k+1]
    return (a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1]),2.87),(b[0]-a[0],b[1]-a[1],0)


offsets=[0.,18.,42.,70.]
hot_chip=max(D["chiplets"],key=lambda c:max(
    D["temperature_rise_K"][0][x][y]
    for x in range(c["x"],c["x"]+c["width"])
    for y in range(c["y"],c["y"]+c["height"])))
cx=(hot_chip["x"]+hot_chip["width"]/2)*D["dx_mm"]
cy=(hot_chip["y"]+hot_chip["height"]/2)*D["dy_mm"]
water_anchor=D["paths"][0]["xy_mm"][len(D["paths"][0]["xy_mm"])//2]
anchors=[(cx,cy,.5),(cx,cy,.6),(72,6,1.6),(*water_anchor,2.6)]
for frame in range(1,END+1):
    scene.frame_set(frame);a=separation(frame);scale=1.23-.23*a
    root.scale=(scale,scale,scale)
    root.location=(37.7*M*(1-scale),36*M*(1-scale),32*M*(1-a))
    root.keyframe_insert("scale",frame=frame);root.keyframe_insert("location",frame=frame)
    for group,offset in zip(groups,offsets):
        group.location.z=offset*M*a;group.keyframe_insert("location",frame=frame)
    real_t=(frame-1)/FPS*D["display_seconds_to_model_seconds"]
    for obj,path,phase in particles:
        t=(real_t+phase*path["travel_s"][-1])%path["travel_s"][-1]
        pos,direction=point_at_time(path,t)
        obj.location=Vector(pos)*M
        obj.rotation_euler=Vector(direction).to_track_quat("Z","Y").to_euler()
        obj.keyframe_insert("location",frame=frame);obj.keyframe_insert("rotation_euler",frame=frame)
    bpy.context.view_layer.update()
    for line,layer,y in labels:
        world=groups[layer].matrix_world@(Vector(anchors[layer])*M)
        p=world_to_camera_view(scene,camera,world)
        points=[hud_point(p.x*W,(1-p.y)*H,-.41),hud_point(1210,y,-.41),hud_point(1275,y,-.41)]
        for cp,co in zip(line.data.splines[0].points,points):
            cp.co=(*co,1);cp.keyframe_insert("co",frame=frame)
    for obj,pos,dx in zip(port_labels,[(-11,36,3),(86,36,3)],[-80,-40]):
        world=groups[3].matrix_world@(Vector(pos)*M)
        p=world_to_camera_view(scene,camera,world)
        obj.location=hud_point(p.x*W+dx,(1-p.y)*H-22)
        obj.keyframe_insert("location",frame=frame)
    for k,(line,arrow) in enumerate(heat_arrows):
        z0=[.5,.6,1.6][k]+offsets[k]*a
        z1=[.5,.6,1.6][k+1] if k<2 else 2.1
        z1+=offsets[k+1]*a
        for cp,z in zip(line.data.splines[0].points,[z0+1,z1-1]):
            cp.co=(12*M,-7*M,z*M,1);cp.keyframe_insert("co",frame=frame)
        arrow.location=(12*M,-7*M,(z1-1)*M);arrow.keyframe_insert("location",frame=frame)

# Every moving property is baked. The saved scene needs no execution handlers.
for action in bpy.data.actions:
    for fc in action.fcurves:
        for key in fc.keyframe_points:
            key.interpolation="CONSTANT" if "hide_" in fc.data_path else "LINEAR"
scene.frame_set(240)
scene.render.filepath=str(OUT/"frames"/"frame_")
(OUT/"frames").mkdir(exist_ok=True)
blend=OUT/"WAVEFORGE_ANIMATION_v2.blend"
if not blend.exists():bpy.ops.wm.save_as_mainfile(filepath=str(blend))
if MODE=="preview":
    for frame in (1,120,240,420,540):
        scene.frame_set(frame);scene.render.filepath=str(OUT/f"preview_v2_{frame:04}.png")
        bpy.ops.render.render(write_still=True)
        print("PREVIEW_DONE",frame,flush=True)
elif MODE=="render":
    for frame in range(1,END+1):
        target=OUT/"frames"/f"frame_{frame:04}.png"
        if target.exists():continue
        scene.frame_set(frame);scene.render.filepath=str(target)
        bpy.ops.render.render(write_still=True)
        if frame%24==0:print("VIDEO_PROGRESS",frame,END,flush=True)
else:
    raise ValueError(MODE)
