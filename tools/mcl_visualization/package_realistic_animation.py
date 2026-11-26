"""Package and fully decode-check the native Blender restyle, no new physics."""
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"ANIMATION_002_REALISTIC"
frames=OUT/"frames"
for name in ("SCENE_CHECK.json","RESTYLE_CHECK.json"):
    assert json.loads((OUT/name).read_text(encoding="utf-8"))["status"]=="PASS"
expected=[frames/f"frame_{i:04}.png" for i in range(1,601)]
assert all(p.is_file() for p in expected),"Render incomplete"
assert len(list(frames.glob("frame_*.png")))==600
for path in expected:
    with Image.open(path) as im:
        assert im.size==(1920,1080)
        im.verify()

video=OUT/"WaveForge_Реалистичная_сборка.mp4"
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-hide_banner","-loglevel","error","-n",
    "-framerate","24","-start_number","1","-i",str(frames/"frame_%04d.png"),
    "-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p",
    "-movflags","+faststart",str(video)],check=True)
reader=imageio_ffmpeg.read_frames(str(video),pix_fmt="rgb24")
metadata=next(reader)
assert metadata["size"]==(1920,1080) and metadata["fps"]==24
samples={}
count=0
for count,raw in enumerate(reader,1):
    if count in (1,120,240,360,540,600):
        samples[count]=Image.frombytes("RGB",(1920,1080),raw)
assert count==600,count

posters={1:"ПРОЦЕССОР_В_СБОРЕ.png",240:"ПРОЦЕССОР_ПО_СЛОЯМ.png",360:"КАРТА_ТЕМПЕРАТУРЫ.png"}
for frame,name in posters.items():
    with Image.open(expected[frame-1]) as im:im.save(OUT/name)
contact=Image.new("RGB",(1440,540),"white")
for i,img in enumerate(samples.values()):
    contact.paste(img.resize((480,270)),((i%3)*480,(i//3)*270))
contact.save(OUT/"VIDEO_CONTACT_SHEET.jpg",quality=94)
with Image.open(expected[239]) as first,Image.open(expected[263]) as second:
    diff=ImageChops.difference(first.convert("RGB"),second.convert("RGB"))
    assert diff.getbbox() is not None,"Missing visible flow motion"
    motion_box=diff.getbbox()
with Image.open(expected[335]) as first,Image.open(expected[336]) as second:
    assert ImageChops.difference(first.convert("RGB"),second.convert("RGB")).getbbox() is not None

html='''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WaveForge — реалистичная сборка</title>
<style>body{margin:0;background:#111823;color:#edf3f8;font-family:Arial,sans-serif}
main{max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:28px}
video{width:100%;border-radius:8px;background:#000}p{font-size:18px;line-height:1.5;color:#bcc9d5}
a{color:#79cbed}.photos{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
img{width:100%;border-radius:6px}@media(max-width:800px){.photos{grid-template-columns:1fr}}
</style><main><h1>WaveForge · U-Net (наша модель)</h1>
<video controls autoplay muted loop playsinline poster="ПРОЦЕССОР_ПО_СЛОЯМ.png">
<source src="WaveForge_Реалистичная_сборка.mp4" type="video/mp4"></video>
<p>25 секунд: сборка → разбор по слоям → карта температуры → сборка.
Вода движется по каналам нашей конструкции; направление взято из сохранённого расчёта.</p>
<p>Зелёная подложка, мелкие компоненты и отделка корпуса добавлены для наглядности.
Это инженерный концепт, не точная CAD-модель AMD. Дополнительные детали не входят в расчёт.
С 14-й секунды цвет показывает постоянную температуру относительно воды на входе — не остывание во времени.</p>
<div class="photos">
<a href="ПРОЦЕССОР_В_СБОРЕ.png"><img src="ПРОЦЕССОР_В_СБОРЕ.png" alt="Собранный процессор">Сборка</a>
<a href="ПРОЦЕССОР_ПО_СЛОЯМ.png"><img src="ПРОЦЕССОР_ПО_СЛОЯМ.png" alt="Слои процессора">Слои и компоненты</a>
<a href="КАРТА_ТЕМПЕРАТУРЫ.png"><img src="КАРТА_ТЕМПЕРАТУРЫ.png" alt="Карта расчётной температуры">Температура</a>
</div></main></html>'''
with (OUT/"СМОТРЕТЬ.html").open("x",encoding="utf-8") as handle:handle.write(html)
report={"status":"PASS","video":str(video),"frames_decoded":count,
    "duration_s":count/metadata["fps"],"fps":metadata["fps"],"resolution":metadata["size"],
    "video_bytes":video.stat().st_size,"sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
    "all_600_source_pngs_verified":True,"flow_motion_detected":True,
    "flow_motion_pixel_bbox":motion_box,"temperature_view_switch_detected":True,
    "codec":"H.264","pixel_format":"yuv420p","posters":list(posters.values())}
with (OUT/"VIDEO_CHECK.json").open("x",encoding="utf-8") as handle:
    json.dump(report,handle,indent=2,ensure_ascii=False)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
