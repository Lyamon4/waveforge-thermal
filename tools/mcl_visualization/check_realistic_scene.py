"""Verify the visual restyle has not changed any source-derived geometry/flow."""
import hashlib
import json
from pathlib import Path
import runpy

import bpy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"ANIMATION_002_REALISTIC"
original=ROOT/"ANIMATION_001"/"WAVEFORGE_ANIMATION_v2.blend"
updated=OUT/"WaveForge_REALISTIC.blend"


def fingerprint(obj):
    payload={"vertices":[list(v.co) for v in obj.data.vertices],
             "faces":[list(p.vertices) for p in obj.data.polygons],
             "parent":obj.parent.name if obj.parent else None,
             "local_transform":[list(row) for row in obj.matrix_basis]}
    if obj.name=="Water" or obj.name.startswith("Chip_"):
        payload["temperature_face_indices"]=[p.material_index for p in obj.data.polygons]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def curves(obj):
    action=obj.animation_data.action if obj.animation_data else None
    if action is None:return []
    return sorted((c.data_path,c.array_index,
        [(list(k.co),k.interpolation) for k in c.keyframe_points]) for c in action.fcurves)


def source_snapshot(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    mesh_names=["Water","Copper_base","Copper_walls"]
    mesh_names += sorted(o.name for o in bpy.data.objects if o.name.startswith(("Chip_","TIM_")) and o.type=="MESH")
    movement_names=["Display_transform_only","Chips","TIM_layer","Copper_base_layer","Water_channel_layer"]
    movement_names += sorted(o.name for o in bpy.data.objects if o.name.startswith("Water_marker_"))
    return {"geometry":{n:fingerprint(bpy.data.objects[n]) for n in mesh_names},
            "movement":{n:curves(bpy.data.objects[n]) for n in movement_names}}


before=source_snapshot(original)
source_colors=[bpy.data.materials[f"Temperature_{i}"].node_tree.nodes["Emission"].inputs["Color"].default_value[:] for i in range(64)]
after=source_snapshot(updated)
assert before==after,"Original mesh or baked motion changed"
assert (OUT/"scene_data.json").read_bytes()==(original.parent/"scene_data.json").read_bytes()
runpy.run_path(str(ROOT/"check_animation_scene.py"),run_name="__main__")

checked=0
for frame,expected in ((1,0),(240,0),(336,0),(337,1),(360,1),(600,1)):
    bpy.context.scene.frame_set(frame)
    for prefix in ("Silicon_state_","Water_state_"):
        for i in range(64):
            mat=bpy.data.materials[f"{prefix}{i}"]
            nodes=mat.node_tree.nodes
            assert nodes["Physical_view_to_temperature"].inputs[0].default_value==expected
            saved=source_colors[i]
            shown=nodes["Saved_temperature_color"].inputs["Color"].default_value[:]
            assert saved==shown
            checked+=1
scope=json.loads((OUT/"VISUAL_SCOPE.json").read_text(encoding="utf-8"))
decor=[o for o in bpy.data.objects if o.get("display_context_only")]
assert len(decor)==scope["decoration_objects"]
assert all(not o.get("included_in_thermal_solver") for o in decor)
result={"status":"PASS","unchanged_source_meshes":len(before["geometry"]),
        "unchanged_baked_motion_objects":len(before["movement"]),
        "temperature_switch_and_color_checks":checked,
        "source_scene_data_byte_identical":True,"new_physics":False,
        "decorations_explicitly_excluded_from_solver":len(decor)}
(OUT/"RESTYLE_CHECK.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
