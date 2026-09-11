from PIL import Image, ImageDraw

from qwen import Qwen

image = "shapes.png"
canvas = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(canvas)
draw.ellipse((40, 40, 216, 216), fill="red")
draw.rectangle((100, 100, 156, 156), fill="blue")
canvas.save(image)

with Qwen() as llm:
    for name, kwargs in {"text": {"text": "Why does a small biped fall forward more often than backward? Three causes, most likely first."}, "text+image": {"text": "How many shapes are there and what is their spatial relation?", "images": image}, "image": {"images": image}}.items():
        r = llm.chat(**kwargs)
        print(name, r.usage["completion_tokens"], "tokens")
        print(r.reasoning.strip()[:400])
        print(r.content.strip())
        print()
