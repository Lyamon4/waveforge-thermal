"""Render one source-exact static figure; never modify sealed run artifacts."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "waveforge-thermal/.worktrees/multitask-generative-nca/artifacts/mcl_stage3_20261018/run_002"
SELECTION = Path("artifacts/inputs/development_tasks.json")
PNG = ROOT / "ЛУЧШАЯ_КОНСТРУКЦИЯ_UNET.png"
METHOD = "Vascular-UNet-S31-R25"
COPPER, COPPER_SIDE = "#C98C58", "#98633D"
WATER, WATER_SIDE = "#43ACCD", "#217E9D"
INK, MUTED, SILICON, TIM = "#1D2A38", "#586572", "#3C4C60", "#D1D8DF"
DX, DY = 75.4 / 64, 72.0 / 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    manifest = json.loads((RUN / "manifest.json").read_text())
    receipt = json.loads((RUN / "completion.json").read_text())
    if not receipt["completed"] or sha(SELECTION) != manifest["selection_sha256"]:
        raise ValueError("incomplete run or changed selection")
    rows = json.loads((RUN / "summary.json").read_text())
    classics = {
        "Pixel-Adam", "Pixel-MMA", "Vascular-Adam", "Vascular-MMA",
        "Vascular-Adam600-Halo", "Vascular-Adam600-Decoupled",
    }
    gains = {}
    for row in rows:
        if row["method"] != METHOD or not row["accepted"]:
            continue
        best = min(
            r["peak_rise_k"] for r in rows
            if r["task"] == row["task"] and r["method"] in classics and r["accepted"]
        )
        gains[row["task"]] = 100 * (best - row["peak_rise_k"]) / best
    index = max(gains, key=gains.get)
    source = RUN / f"task_{index:02d}" / METHOD / "selected.npz"
    for path in (RUN / "summary.json", source):
        if sha(path) != receipt["files"][str(path.relative_to(RUN))]:
            raise ValueError(f"source seal mismatch: {path}")
    with np.load(source, allow_pickle=False) as fields:
        design = fields["binary"].copy()
    if design.dtype != bool or design.shape != (64, 64) or design.sum() != 1024:
        raise ValueError("wrong binary geometry or budget")
    if label(design)[1] != 1 or not design[0, 24:40].any() or not design[-1, 24:40].any():
        raise ValueError("not a connected inlet-outlet geometry")
    task = json.loads(SELECTION.read_text())["tasks"][index]
    if len(task["chiplets"]) != 9 or not np.allclose(np.sum(task["powers_w"], axis=1), 360):
        raise ValueError("unexpected chiplet task")
    return design, task, index, gains[index], source


def project(x, y, z):
    """Parallel illustrative projection, physical x/y in mm; exploded z display."""
    return np.stack((0.94 * np.asarray(x) - 0.46 * np.asarray(y),
                     0.15 * np.asarray(x) + 0.27 * np.asarray(y) + np.asarray(z)), -1)


def polygon(ax, xyz, color, edge=None, width=0.5):
    points = np.asarray(xyz)
    ax.add_patch(Polygon(project(*points.T), closed=True, facecolor=color,
                         edgecolor=edge or color, linewidth=width, joinstyle="miter"))


def box(ax, x, y, width, height, z, thickness, top, side):
    x1, y1, z1 = x + width, y + height, z + thickness
    polygon(ax, [(x,y,z),(x1,y,z),(x1,y,z1),(x,y,z1)], side)
    polygon(ax, [(x1,y,z),(x1,y1,z),(x1,y1,z1),(x1,y,z1)], side)
    polygon(ax, [(x,y,z1),(x1,y,z1),(x1,y1,z1),(x,y1,z1)], top)


def channel_layer(ax, design):
    z, thickness = 100.0, 4.0
    # Separate outer side faces preserve the actual material at the boundary.
    for i in range(64):
        x = i * DX
        color = WATER_SIDE if design[i, 0] else COPPER_SIDE
        polygon(ax, [(x,0,z),(x+DX,0,z),(x+DX,0,z+thickness),(x,0,z+thickness)], color)
    for j in range(64):
        y = j * DY
        color = WATER_SIDE if design[-1, j] else COPPER_SIDE
        polygon(ax, [(75.4,y,z),(75.4,y+DY,z),(75.4,y+DY,z+thickness),(75.4,y,z+thickness)], color)
    faces, colors = [], []
    for i in range(64):
        for j in range(64):
            x, y = i * DX, j * DY
            xyz = np.array([(x,y,z+thickness),(x+DX,y,z+thickness),
                            (x+DX,y+DY,z+thickness),(x,y+DY,z+thickness)])
            faces.append(project(*xyz.T))
            colors.append(WATER if design[i, j] else COPPER)
    ax.add_collection(PolyCollection(faces, facecolors=colors, edgecolors="none",
                                    antialiaseds=False))
    outline = np.array([(0,0,104),(75.4,0,104),(75.4,72,104),(0,72,104),(0,0,104)])
    xy = project(*outline.T)
    ax.plot(xy[:,0], xy[:,1], color=COPPER_SIDE, linewidth=0.7)


def render():
    design, task, index, gain, source = load()
    if PNG.exists():
        raise FileExistsError(PNG)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":16,
                         "text.color":INK, "axes.labelcolor":INK})
    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    fig.text(.055,.94,"WaveForge · U-Net (наша модель)",fontsize=29,weight="bold")
    fig.text(.055,.891,"Теплоотводящая конструкция с водяными каналами",fontsize=20,color=MUTED)
    fig.text(.055,.816,"01  Слои конструкции",fontsize=19,weight="bold")
    fig.text(.59,.816,"02  Точная карта каналов",fontsize=19,weight="bold")

    ax = fig.add_axes([.025,.20,.53,.58])
    ax.set_aspect("equal"); ax.set_xlim(-46,108); ax.set_ylim(-8,140); ax.axis("off")
    # Actual chip footprints, silicon+TIM; heights enlarged fourfold, layers separated.
    for c in sorted(task["chiplets"], key=lambda c:c["x"]+c["y"], reverse=True):
        x,y,w,h = c["x"]*DX,c["y"]*DY,c["width"]*DX,c["height"]*DY
        box(ax,x,y,w,h,0,2,SILICON,"#293746")
        box(ax,x,y,w,h,2,.4,TIM,"#A0ADB9")
    box(ax,0,0,75.4,72,48,4,COPPER,COPPER_SIDE)
    channel_layer(ax,design)
    for x,y in ((0,0),(75.4,0),(0,72),(75.4,72)):
        for lo,hi in ((3,46),(54,98)):
            xy=project([x,x],[y,y],[lo,hi])
            ax.plot(xy[:,0],xy[:,1],color="#B8C2CB",lw=.7,ls=(0,(3,4)),zorder=0)
    for text,target,position in (
        ("Вода + медь\nслой 1,0 мм",(75.4,0,104),(84,121)),
        ("Медное\nоснование\n1,0 мм",(75.4,0,52),(84,65)),
        ("8 чиплетов\n+ I/O",(62,15,1.2),(84,16)),
    ):
        ax.annotate(text,xy=project(*target),xytext=position,fontsize=13.5,
                    va="center",ha="left",color=INK,
                    arrowprops={"arrowstyle":"-","color":MUTED,"lw":.9})
    ax.text(-34,-5,"Светлый слой на чиплетах — термоинтерфейс",fontsize=11.8,color=MUTED)

    plan=fig.add_axes([.575,.25,.40,.52])
    rgba=np.empty((64,64,4))
    from matplotlib.colors import to_rgba
    rgba[:]=to_rgba(COPPER); rgba[design.T]=to_rgba(WATER)
    plan.imshow(rgba,origin="lower",extent=(0,75.4,0,72),interpolation="nearest")
    plan.add_patch(Rectangle((0,0),75.4,72,fill=False,edgecolor=COPPER_SIDE,lw=1.2))
    for c in task["chiplets"]:
        x,y,w,h=c["x"]*DX,c["y"]*DY,c["width"]*DX,c["height"]*DY
        plan.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="white",lw=2.8))
        plan.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=INK,lw=1.1,ls=(0,(4,2))))
    for x in (0,75.4):
        plan.plot([x,x],[24*DY,40*DY],color=WATER_SIDE,lw=3.5)
    plan.annotate("",xy=(0,36),xytext=(-11,36),
                  arrowprops={"arrowstyle":"-|>","color":WATER_SIDE,"lw":2,"mutation_scale":18})
    plan.annotate("",xy=(86.4,36),xytext=(75.4,36),
                  arrowprops={"arrowstyle":"-|>","color":WATER_SIDE,"lw":2,"mutation_scale":18})
    plan.text(-5.5,42,"Вход",ha="center",fontsize=14,color=WATER_SIDE)
    plan.text(80.9,42,"Выход",ha="center",fontsize=14,color=WATER_SIDE)
    plan.text(37.7,-7,"75,4 × 72 мм  ·  каналы занимают 25% площади",
              ha="center",fontsize=14,color=MUTED)
    plan.set_xlim(-13,90);plan.set_ylim(-11,75);plan.set_aspect("equal");plan.axis("off")

    fig.legend(handles=[Patch(facecolor=COPPER,label="Медь"),
                        Patch(facecolor=WATER,label="Каналы воды"),
                        Line2D([0],[0],color=INK,ls="--",lw=1.5,label="Чиплеты под пластиной")],
               loc="lower center",bbox_to_anchor=(.53,.15),ncol=3,frameon=False,
               fontsize=17,handlelength=2,columnspacing=2.0)
    fig.text(.055,.09,f"Пример №{index} из Stage 3.1 · геометрия из сохранённого результата 64 × 64",
             fontsize=14,color=MUTED)
    fig.text(.055,.05,"Слои разнесены, толщина увеличена для наглядности. Расчётная модель, не чертёж для производства.",
             fontsize=12.5,color=MUTED)
    fig.savefig(PNG,dpi=260,facecolor="white")
    plt.close(fig)
    provenance={"source":str(source),"source_sha256":sha(source),
                "selection_sha256":sha(SELECTION),"task_index":index,
                "selection_rule":"maximum gain versus best valid classical per-task within run_002",
                "gain_percent":gain,"candidate_type":METHOD,"channels_cells":int(design.sum()),
                "pixel_grid":[64,64],"plot_xy_mm":[75.4,72.0],
                "isometric_vertical_scale":4,"layers_exploded":True,
                "generated_by":"deterministic Matplotlib geometry renderer; no image synthesis",
                "no_new_physics_solve":True,"closed_workloads_read":False,
                "image_sha256":sha(PNG)}
    with (ROOT/"SOURCE.json").open("x",encoding="utf-8") as f:
        json.dump(provenance,f,indent=2,ensure_ascii=False)
    print(json.dumps(provenance,ensure_ascii=False))
    print(PNG)


if __name__=="__main__":
    render()
