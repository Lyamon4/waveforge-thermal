"""Restyle the native animation, retaining source geometry and baked flow.

Decorative package details are explicitly not included in the thermal model.
"""
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"ANIMATION_001"
OUT=ROOT/"ANIMATION_002_REALISTIC"
MODE=sys.argv[sys.argv.index("--")+1] if "--" in sys.argv else "preview"
OUT.mkdir(exist_ok=True)
D=json.loads((SOURCE/"scene_data.json").read_text(encoding="utf-8"))
assert hashlib.sha256(Path(D["source"]).read_bytes()).hexdigest()==D["source_sha256"]
bpy.ops.wm.open_mainfile(filepath=str(SOURCE/"WAVEFORGE_ANIMATION_v2.blend"))
scene=bpy.context.scene;camera=scene.camera;root=bpy.data.objects["Display_transform_only"]
groups=[bpy.data.objects[n] for n in ("Chips","TIM_layer","Copper_base_layer","Water_channel_layer")]
M=.01;W,H=1920,1080;UX=2.6/W;SWITCH=337
decor=[]


def srgb(code):
    v=[int(code[i:i+2],16)/255 for i in (1,3,5)]
    return tuple(x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in v)


def material(name,color,metal=0.,rough=.35):
    m=bpy.data.materials.new(name);m.use_nodes=True
    bs=m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value=(*color,1)
    bs.inputs["Metallic"].default_value=metal;bs.inputs["Roughness"].default_value=rough
    return m


def microfinish(m,scale,strength,distance):
    n=m.node_tree.nodes;l=m.node_tree.links;bs=n.get("Principled BSDF")
    tc=n.new("ShaderNodeTexCoord");noise=n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value=scale
    noise.inputs["Detail"].default_value=2
    bump=n.new("ShaderNodeBump");bump.inputs["Strength"].default_value=strength
    bump.inputs["Distance"].default_value=distance
    l.new(tc.outputs["Generated"],noise.inputs["Vector"])
    l.new(noise.outputs["Fac"],bump.inputs["Height"])
    l.new(bump.outputs["Normal"],bs.inputs["Normal"])


nickel=material("Illustrative_nickel_finish",srgb("#c4ced4"),.72,.24)
microfinish(nickel,210,.12,.00012)
edge_copper=material("Copper_cut_edges",(.45,.16,.065),.72,.3)
green=material("Package_green_soldermask",srgb("#087449"),.05,.38)
microfinish(green,180,.2,.00015)
board_edge=material("Laminate_edge",srgb("#1c4237"),0,.55)
trace_mat=material("Buried_trace_hint",srgb("#23965c"),.12,.4)
gold=material("Contact_gold",(.65,.4,.09),.75,.24)
solder=material("Solder_ends",(.55,.59,.63),.65,.25)
ceramic=material("Capacitor_ceramic",srgb("#69645b"),.05,.55)
resistor=material("Resistor_body",srgb("#252d30"),.1,.4)
silicon=material("Reflective_silicon",srgb("#182a38"),.58,.17)
marking=material("Laser_marking",srgb("#8caaa7"),.3,.4)
etch=material("Subtle_die_detail",srgb("#23394b"),.5,.23)
rubber=material("Seal_outline",srgb("#182221"),0,.6)
tim=bpy.data.materials["TIM"]
tim.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value=(.52,.55,.58,1)


def decorate(obj):
    obj["display_context_only"]=True;obj["included_in_thermal_solver"]=False
    decor.append(obj);return obj


def cube(name,center,size,mat,parent,bevel=0.):
    bpy.ops.mesh.primitive_cube_add(size=1,location=Vector(center)*M)
    obj=bpy.context.object;obj.name=name;obj.dimensions=Vector(size)*M
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(mat);obj.parent=parent
    if bevel:
        mod=obj.modifiers.new("Machined_edge","BEVEL");mod.width=bevel*M;mod.segments=3
        normal=obj.modifiers.new("Surface_normals","WEIGHTED_NORMAL")
    return decorate(obj)


class Boxes:
    """Batch non-physics details into one mesh per material instead of objects."""
    def __init__(self,name,mat,parent):
        self.name,self.mat,self.parent=name,mat,parent;self.vertices=[];self.faces=[]
    def add(self,center,size):
        x,y,z=center;a,b,c=[v/2 for v in size];start=len(self.vertices)
        pts=[(x-a,y-b,z-c),(x+a,y-b,z-c),(x+a,y+b,z-c),(x-a,y+b,z-c),
             (x-a,y-b,z+c),(x+a,y-b,z+c),(x+a,y+b,z+c),(x-a,y+b,z+c)]
        self.vertices.extend([tuple(v*M for v in p) for p in pts])
        self.faces.extend([tuple(start+i for i in f) for f in
            ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))])
    def finish(self):
        mesh=bpy.data.meshes.new(self.name);mesh.from_pydata(self.vertices,[],self.faces);mesh.update()
        obj=bpy.data.objects.new(self.name,mesh);scene.collection.objects.link(obj)
        obj.data.materials.append(self.mat);obj.parent=self.parent;return decorate(obj)


def polyline(name,points,radius,mat,parent):
    curve=bpy.data.curves.new(name,"CURVE");curve.dimensions="3D"
    curve.bevel_depth=radius*M;curve.bevel_resolution=1
    spline=curve.splines.new("POLY");spline.points.add(len(points)-1)
    for p,xyz in zip(spline.points,points):p.co=(*[v*M for v in xyz],1)
    obj=bpy.data.objects.new(name,curve);scene.collection.objects.link(obj)
    obj.data.materials.append(mat);obj.parent=parent;return decorate(obj)


font=bpy.data.fonts.get("Arial Regular") or bpy.data.fonts.load("C:/Windows/Fonts/arial.ttf")
bold=bpy.data.fonts.load("C:/Windows/Fonts/arialbd.ttf")


def engraved(body,center,size,parent):
    data=bpy.data.curves.new("Illustrative_marking","FONT");data.body=body;data.font=font
    data.size=size*M;data.align_x="CENTER"
    obj=bpy.data.objects.new("Illustrative_marking",data);scene.collection.objects.link(obj)
    obj.location=Vector(center)*M;obj.parent=parent;obj.data.materials.append(marking)
    return decorate(obj)


def visible(obj,start=1,end=600):
    obj.animation_data_clear()
    for f,hidden in ((1,start>1),(start,False),(min(600,end+1),end<600)):
        obj.hide_render=hidden;obj.keyframe_insert("hide_render",frame=f)
        obj.hide_viewport=hidden;obj.keyframe_insert("hide_viewport",frame=f)


# Cosmetic finish only. Original cell mesh, vertex coordinates and face indices
# of the hydraulic channels remain untouched.
for name in ("Copper_base","Copper_walls"):
    obj=bpy.data.objects[name];obj.data.materials.clear()
    obj.data.materials.append(nickel);obj.data.materials.append(edge_copper)
    for poly in obj.data.polygons:poly.material_index=0 if poly.normal.z>.9 else 1


def switch_material(name,color,real_color,metal,rough):
    m=material(name,real_color,metal,rough);nodes=m.node_tree.nodes;links=m.node_tree.links
    bs=nodes.get("Principled BSDF");out=nodes.get("Material Output")
    emission=nodes.new("ShaderNodeEmission");emission.name="Saved_temperature_color"
    emission.inputs["Color"].default_value=(*color,1)
    mix=nodes.new("ShaderNodeMixShader");mix.name="Physical_view_to_temperature"
    links.new(bs.outputs[0],mix.inputs[1]);links.new(emission.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs[0])
    for f,value in ((1,0.),(SWITCH-1,0.),(SWITCH,1.),(600,1.)):
        mix.inputs[0].default_value=value;mix.inputs[0].keyframe_insert("default_value",frame=f)
    return m


chip_mats=[];water_mats=[]
for i in range(64):
    color=tuple(bpy.data.materials[f"Temperature_{i}"].node_tree.nodes.get("Emission").inputs["Color"].default_value[:3])
    chip_mats.append(switch_material(f"Silicon_state_{i}",color,srgb("#182a38"),.58,.17))
    water_mats.append(switch_material(f"Water_state_{i}",color,srgb("#087eaa"),.18,.17))
for obj in [o for o in bpy.data.objects if o.name.startswith("Chip_")]:
    obj.data.materials[0]=silicon
    for i,m in enumerate(chip_mats,1):obj.data.materials[i]=m
water=bpy.data.objects["Water"]
water.data.materials[0]=water_mats[0]
for i,m in enumerate(water_mats,1):water.data.materials[i]=m

# Illustrative package: these details do NOT change conductivity, power, BCs,
# channel volume, or the existing steady-state solution.
pcb=cube("Illustrative_package_substrate",(37.7,36,-.85),(83.4,80,1.5),green,groups[0],.45)
cube("Illustrative_laminate_edge",(37.7,36,-1.65),(83.1,79.7,.15),board_edge,groups[0],.3)
contacts=Boxes("Illustrative_gold_contacts",gold,groups[0])
for k in range(33):
    x=1.1+k*2.25
    for y in (-2.3,74.3):contacts.add((x,y,-.075),(1.3,.8,.04))
for k in range(31):
    y=2+k*2.25
    for x in (-2.3,77.7):contacts.add((x,y,-.075),(.8,1.3,.04))
contacts.finish()
bottom=Boxes("Illustrative_bottom_land_array",gold,groups[0])
for i in range(31):
    for j in range(29):bottom.add((4+2.25*i,4+2.25*j,-1.755),(.95,.95,.04))
bottom.finish()

rects=[]
for index,c in enumerate(D["chiplets"]):
    x0,y0=c["x"]*D["dx_mm"],c["y"]*D["dy_mm"]
    w,h=c["width"]*D["dx_mm"],c["height"]*D["dy_mm"]
    rects.append((x0,y0,x0+w,y0+h))
    label="I/O" if c["width"]==12 else f"CCD {index+1:02}"
    obj=engraved(label,(x0+w/2,y0+h*.38,.515),1.1,groups[0]);visible(obj,1,SWITCH-1)
    detail=Boxes(f"Illustrative_die_surface_{index}",etch,groups[0])
    detail.add((x0+w*.22,y0+h*.76,.505),(w*.27,h*.28,.006))
    detail.add((x0+w*.58,y0+h*.76,.505),(w*.35,h*.28,.006))
    obj=detail.finish();visible(obj,1,SWITCH-1)
    for off in (-.4,0,.4):
        points=[(x0+w/2+off,y0+h/2,-.07),(x0+w/2+off,36+off,-.07),(37.7,36+off,-.07)]
        polyline("Illustrative_masked_trace",points,.032,trace_mat,groups[0])

rng=random.Random(221018)
positions=[]
for x in [2+2.2*i for i in range(33)]:
    for y in [2+2.2*j for j in range(31)]:
        if not any(a-1.4<x<c+1.4 and b-1.4<y<d+1.4 for a,b,c,d in rects):positions.append((x,y))
rng.shuffle(positions)
caps=Boxes("Illustrative_SMD_capacitors",ceramic,groups[0])
res=Boxes("Illustrative_SMD_resistors",resistor,groups[0])
ends=Boxes("Illustrative_SMD_solder",solder,groups[0])
pads=Boxes("Illustrative_SMD_pads",gold,groups[0])
for k,(x,y) in enumerate(positions[:150]):
    target=res if k%5==0 else caps
    target.add((x,y,.13),(1.1,.58,.36))
    for sign in (-1,1):
        ends.add((x+sign*.59,y,.13),(.24,.62,.36))
        pads.add((x+sign*.59,y,-.06),(.4,.78,.03))
for group in (caps,res,ends,pads):group.finish()
engraved("WAVEFORGE  /  CONCEPT",(37.7,-1.5,-.06),1.15,groups[0])

# A metal perimeter / sealing outline visually suggests a cold-plate housing;
# ports stay open. No fictitious hoses, pump, or refined microgrooves are added.
for y in (-1.3,73.3):
    cube("Illustrative_metal_frame",(37.7,y,2.15),(79,2.2,1.2),nickel,groups[3],.25)
for x in (-1.3,76.7):
    for y in (11.8,60.2):
        cube("Illustrative_metal_frame",(x,y,2.15),(2.2,25,1.2),nickel,groups[3],.25)
for x in (-1.3,76.7):
    for y in (-1.3,73.3):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.65*M,depth=.28*M,location=Vector((x,y,2.85))*M)
        obj=bpy.context.object;obj.name="Illustrative_fastener";obj.parent=groups[3];obj.data.materials.append(solder);decorate(obj)
        cube("Illustrative_fastener_slot",(x,y,3.01),(.75,.13,.02),rubber,groups[3])
engraved("WAVEFORGE",(37.7,35.2,2.605),2.4,groups[3])
engraved("MICROCHANNEL CONCEPT",(37.7,32.6,2.605),.9,groups[3])

# Reposition callouts to make room for the substrate. No crop / tiny-font trick.
def hud(x,y,z=-.4):
    return ((x/W-.5+camera.data.shift_x)*2.6,(.5-y/H)*2.6*H/W,z)


def text(body,x,y,size=24,strong=False):
    data=bpy.data.curves.new("New_label","FONT");data.body=body;data.font=bold if strong else font
    data.size=size*UX*1.4
    obj=bpy.data.objects.new("New_label",data);scene.collection.objects.link(obj)
    obj.parent=camera;obj.location=hud(x,y);obj.data.materials.append(bpy.data.materials["Text"])
    return obj


for obj in list(bpy.data.objects):
    if obj.type!="FONT" or obj.parent!=camera:continue
    body=obj.data.body
    if body.startswith("Поток замедлен") or body.startswith("Слои разнесены"):
        bpy.data.objects.remove(obj,do_unlink=True)
        continue
    elif body=="Охлаждение по слоям":obj.data.body="Сборка, слои и поток воды"
    elif body=="Конструкция по слоям":obj.data.body="Устройство охлаждения"
    elif body=="Медное основание":obj.data.body="Медное основание"
    elif body=="Проводит тепло от чиплетов":obj.data.body="Серебристая отделка снаружи"
    elif body=="8 вычислительных + 1 I/O":obj.data.body="8 чиплетов + блок ввода-вывода"
    if body in ("Нагрев относительно входа, °C","0","50","100"):
        visible(obj,SWITCH,600)
for obj in bpy.data.objects:
    if obj.name.startswith("Legend"):visible(obj,SWITCH,600)

physical=text("РЕАЛИСТИЧНЫЙ ВИД",1300,162,19,True);visible(physical,1,SWITCH-1)
thermal=text("КАРТА ТЕМПЕРАТУРЫ",1300,162,19,True);visible(thermal,SWITCH,600)
physical_note=text("Подложка и детали корпуса — оформление концепта. Каналы и движение воды — из нашего расчёта.",65,1005,20)
visible(physical_note,1,SWITCH-1)
thermal_note=text("Цвет — постоянная температура из расчёта, не остывание во времени. Поток замедлен в 10,8 раза.",65,1005,20)
visible(thermal_note,SWITCH,600)
text("Не точная копия AMD. Дополнительные детали не входят в физическую модель. Верх открыт для обзора.",65,1040,18)
substrate_title=text("Подложка и компоненты",1300,804,28,True);visible(substrate_title,101,445)
substrate_sub=text("Основа корпуса процессора",1300,838,22);visible(substrate_sub,101,445)
layer_labels={
    "Каналы с водой":(286,True),"Вода принимает тепло":(320,False),
    "Медное основание":(416,True),"Серебристая отделка снаружи":(450,False),
    "Термоинтерфейс":(546,True),"Передаёт тепло в медь":(580,False),
    "Чиплеты процессора":(676,True),"8 чиплетов + блок ввода-вывода":(710,False),
}
for obj in bpy.data.objects:
    if obj.type=="FONT" and obj.parent==camera and obj.data.body in layer_labels:
        y,_=layer_labels[obj.data.body];obj.location=hud(1300,y)

ptr=polyline("Substrate_pointer",[(0,0,-41)]*3,.095,bpy.data.materials["Leader_lines"],camera)
visible(ptr,101,445)
lines=sorted([o for o in bpy.data.objects if o.name.startswith("Layer_pointer")],key=lambda o:o.name)
line_y=[276,406,536,666]
for frame in range(1,601):
    scene.frame_set(frame);bpy.context.view_layer.update()
    for line,y in zip(lines,line_y):
        # Preserve the existing geometry-bound starting endpoint.
        for cp,co in zip(line.data.splines[0].points[1:],[hud(1210,y,-.41),hud(1275,y,-.41)]):
            cp.co=(*co,1);cp.keyframe_insert("co",frame=frame)
    p=world_to_camera_view(scene,camera,groups[0].matrix_world@(Vector((76,-1.4,-.06))*M))
    for cp,co in zip(ptr.data.splines[0].points,[hud(p.x*W,(1-p.y)*H,-.41),hud(1210,794,-.41),hud(1275,794,-.41)]):
        cp.co=(*co,1);cp.keyframe_insert("co",frame=frame)

# Better reflected studio lighting and modest roughness improve material cues.
scene.eevee.use_ssr=True
scene.eevee.taa_render_samples=64
scene.eevee.gtao_factor=1.25
scene.world.node_tree.nodes["Background"].inputs[1].default_value=.5
for light in bpy.data.lights:
    if light.name=="Key":light.energy=18
    if light.name=="Fill":light.energy=9
    if light.name=="Rim":light.energy=23
for action in bpy.data.actions:
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            if "hide_" in curve.data_path or "default_value" in curve.data_path:
                point.interpolation="CONSTANT"
scene.frame_set(240)
(OUT/"frames").mkdir(exist_ok=True)
if not (OUT/"scene_data.json").exists():shutil.copy2(SOURCE/"scene_data.json",OUT/"scene_data.json")
blend=OUT/"WaveForge_REALISTIC.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
manifest={"source_sha256":D["source_sha256"],"unchanged_channel_cells":1024,
          "decoration_objects":len(decor),"decorative_package_dimensions_mm":[83.4,80,1.5],
          "temperature_mode_starts_frame":SWITCH,"temperature_mode_starts_seconds":14,
          "additional_parts_in_solver":False,"metal_finish_changes_physics":False,
          "actual_AMD_CAD":False,"new_physics_solve":False,
          "reference_for_general_chiplet_organization":"https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58015-epyc-9004-tg-architecture-overview.pdf"}
with (OUT/"VISUAL_SCOPE.json").open("w",encoding="utf-8") as handle:json.dump(manifest,handle,indent=2)
if MODE=="preview":
    for frame in (1,120,240,360,540):
        scene.frame_set(frame);scene.render.filepath=str(OUT/f"preview_{frame:04}.png")
        bpy.ops.render.render(write_still=True);print("PREVIEW_DONE",frame,flush=True)
elif MODE=="render":
    for frame in range(1,601):
        target=OUT/"frames"/f"frame_{frame:04}.png"
        if target.exists():continue
        scene.frame_set(frame);scene.render.filepath=str(target);bpy.ops.render.render(write_still=True)
        if frame%24==0:print("VIDEO_PROGRESS",frame,600,flush=True)
else:raise ValueError(MODE)
