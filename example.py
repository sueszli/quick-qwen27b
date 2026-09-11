# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
from qwen import Qwen

with Qwen() as llm:
    r = llm.chat("Why does a small biped fall forward more often than backward? Three causes, most likely first.")
    print(f"{r.reasoning.strip()[:400]}\n{r.content.strip()}\n")

    r = llm.chat("How many dice are there and what do they show?", images="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png")
    print(f"{r.reasoning.strip()[:400]}\n{r.content.strip()}\n")
