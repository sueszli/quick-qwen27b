# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
import time

from qwen import Qwen

DIM, RESET, CLEAR_LINE = "\033[2m", "\033[0m", "\r\033[K"


def show(llm: Qwen, **kwargs) -> None:
    start, tokens = time.time(), 0
    print(f"{DIM}processing prompt...{RESET}", end="", flush=True)
    for kind, value in llm.stream(**kwargs):
        if tokens == 0:
            print(CLEAR_LINE, end="")
        tokens += 1
        if kind == "reasoning":
            print(f"{DIM}{value}{RESET}", end="", flush=True)
        if kind == "content":
            print(value, end="", flush=True)
    print(f"\n{DIM}{tokens} tokens, {time.time() - start:.0f}s{RESET}\n")


with Qwen() as llm:
    show(llm, text="Why does a small biped fall forward more often than backward? Three causes, most likely first.")
    show(llm, text="How many dice are there and what do they show?", images="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png")
