"""Encode, decode-check, and package the rendered 25s educational video."""
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"ANIMATION_001"
frames=OUT/"frames"
expected=[frames/f"frame_{i:04}.png" for i in range(1,601)]
assert all(p.is_file() for p in expected),"Render incomplete"
assert len(list(frames.glob("frame_*.png")))==600
for p in expected:
    with Image.open(p) as image:
        assert image.size==(1920,1080)
        image.verify()
video=OUT/"WaveForge_Охлаждение_по_слоям.mp4"
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg,"-nostdin","-hide_banner","-loglevel","error","-n",
    "-framerate","24","-start_number","1","-i",str(frames/"frame_%04d.png"),
    "-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p",
    "-movflags","+faststart",str(video)],check=True)
reader=imageio_ffmpeg.read_frames(str(video),pix_fmt="rgb24")
metadata=next(reader)
assert metadata["size"]==(1920,1080)
assert abs(metadata["fps"]-24)<1e-6
count=0; samples={}
for count,raw in enumerate(reader,1):
    if count in (1,120,240,420,540,600):
        samples[count]=Image.frombytes("RGB",(1920,1080),raw)
assert count==600,count
poster=OUT/"СЛОИ_И_ПОТОК.png"
with Image.open(expected[239]) as image:image.save(poster)
contact=Image.new("RGB",(1440,540),"white")
for i,(frame,img) in enumerate(samples.items()):
    contact.paste(img.resize((480,270)),((i%3)*480,(i//3)*270))
contact.save(OUT/"VIDEO_CONTACT_SHEET.jpg",quality=92)
# Same fully exploded view, separated by one second: only the water markers
# move. This catches an accidentally static export without inventing flow.
with Image.open(expected[239]) as first, Image.open(expected[263]) as second:
    import PIL.ImageChops
    diff=PIL.ImageChops.difference(first.convert("RGB"),second.convert("RGB"))
    assert diff.getbbox() is not None,"Expected visible flow motion"
    flow_motion_bbox=diff.getbbox()
# A portable local player preserves the real video, with no third-party embeds.
html=OUT/"СМОТРЕТЬ.html"
markup='''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WaveForge — охлаждение по слоям</title>
<style>body{margin:0;background:#111823;color:#edf3f8;font-family:Arial,sans-serif}
main{max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:28px}
video{width:100%;border-radius:8px;background:#000}p{font-size:18px;line-height:1.5;color:#bcc9d5}
</style><main><h1>WaveForge · U-Net (наша модель)</h1>
<video controls autoplay muted loop playsinline poster="СЛОИ_И_ПОТОК.png">
<source src="WaveForge_Охлаждение_по_слоям.mp4" type="video/mp4"></video>
<p>Настоящая геометрия каналов и сохранённый расчёт потока. Слои разнесены для обзора.
Цвет показывает постоянную температуру относительно воды на входе; это не анимация остывания.
Движение замедлено для наглядности. Верх открыт, внешние шланги и насос не моделировались.</p></main></html>'''
with html.open("x",encoding="utf-8") as handle:handle.write(markup)
report={"video":str(video),"frames_decoded":count,"duration_s":count/metadata["fps"],
        "fps":metadata["fps"],"resolution":metadata["size"],"video_bytes":video.stat().st_size,
        "sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
        "all_600_source_pngs_verified":True,"flow_motion_detected":True,
        "flow_motion_pixel_bbox":flow_motion_bbox,"codec":"H.264","pixel_format":"yuv420p"}
with (OUT/"VIDEO_CHECK.json").open("x",encoding="utf-8") as handle:json.dump(report,handle,indent=2,ensure_ascii=False)
print(json.dumps(report,ensure_ascii=False,indent=2))
