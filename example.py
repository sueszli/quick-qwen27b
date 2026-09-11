# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
from qwen import Qwen

image = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"

with Qwen() as llm:
    for name, kwargs in {"text": {"text": "Why does a small biped fall forward more often than backward? Three causes, most likely first."}, "text+image": {"text": "How many dice are there and what do they show?", "images": image}, "image": {"images": image}}.items():
        r = llm.chat(**kwargs)
        print(name, r.usage["completion_tokens"], "tokens")
        print(r.reasoning.strip()[:400])
        print(r.content.strip())
        print()
