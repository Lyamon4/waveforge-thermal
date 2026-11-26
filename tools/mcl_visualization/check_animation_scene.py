"""Read-only checks against the baked Blender scene and the sealed data."""
import hashlib
import json
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

OUT=Path(bpy.data.filepath).resolve().parent
data=json.loads((OUT/"scene_data.json").read_text(encoding="utf-8"))
assert hashlib.sha256(Path(data["source"]).read_bytes()).hexdigest()==data["source_sha256"]
scene=bpy.context.scene
markers=[o for o in bpy.data.objects if o.name.startswith("Water_marker_")]
assert len(markers)==32
assert (scene.render.resolution_x,scene.render.resolution_y,scene.render.fps)==(1920,1080,24)
assert scene.frame_end-scene.frame_start+1==600
positions_checked=0
label_bounds=[]
for frame in range(1,601):
    scene.frame_set(frame)
    for obj in markers:
        x,y=obj.location.x/.01,obj.location.y/.01
        ix,iy=int(x/data["dx_mm"]),int(y/data["dy_mm"])
        assert 0<=ix<64 and 0<=iy<64 and data["binary"][ix][iy],(frame,obj.name,x,y)
        positions_checked+=1
    if frame in (1,48,100,120,168,240,360,420,445,456,540,600):
        bpy.context.view_layer.update()
        for obj in bpy.data.objects:
            if obj.type!="FONT" or obj.hide_render:continue
            coords=[world_to_camera_view(scene,scene.camera,obj.matrix_world@Vector(p)) for p in obj.bound_box]
            xmin,xmax=min(p.x for p in coords),max(p.x for p in coords)
            ymin,ymax=min(p.y for p in coords),max(p.y for p in coords)
            assert 0<=xmin<xmax<=1 and 0<=ymin<ymax<=1,(frame,obj.data.body,[xmin,xmax,ymin,ymax])
            label_bounds.append([frame,obj.data.body,[xmin,xmax,ymin,ymax]])
water=bpy.data.objects["Water"]
top_faces=[p for p in water.data.polygons if p.normal.z>.9]
assert len(top_faces)==1024
for face in top_faces:
    ix=int(face.center.x/.01/data["dx_mm"])
    iy=int(face.center.y/.01/data["dy_mm"])
    expected=1+max(0,min(63,round(data["temperature_rise_K"][3][ix][iy]/100*63)))
    assert data["binary"][ix][iy] and face.material_index==expected
result={"status":"PASS","frames":600,"duration_s":25,"resolution":[1920,1080],
        "marker_positions_inside_actual_water_cells":positions_checked,
        "source_sha256_verified":True,"water_top_cells_verified":len(top_faces),
        "temperature_color_bindings_verified":1024,
        "visible_label_bounds_checked":len(label_bounds),
        "new_physics_solve":False,"closed_test_workloads_read":False}
with (OUT/"SCENE_CHECK.json").open("w",encoding="utf-8") as handle:
    json.dump(result,handle,indent=2)
print(json.dumps(result,indent=2),flush=True)
