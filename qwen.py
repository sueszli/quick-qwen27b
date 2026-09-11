from __future__ import annotations

import atexit
import base64
import json
import mimetypes
import os
import socket
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

SEED = 41
WEIGHTS_DIR = Path(os.environ.get("QWEN_WEIGHTS_DIR", Path(__file__).resolve().parent / "weights"))
LLAMA_CPP_TAG = "b10908"
LLAMA_CPP_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/llama-{LLAMA_CPP_TAG}-bin-ubuntu-vulkan-x64.tar.gz"

REPO = "unsloth/Qwen3.5-27B-GGUF"
MODEL_FILE = "Qwen3.5-27B-UD-Q5_K_XL.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"
CTX = 65536

THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
INSTRUCT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}


def server_binary() -> Path:
    root = WEIGHTS_DIR / f"llama.cpp-{LLAMA_CPP_TAG}"
    binary = root / "llama-server"
    if not binary.exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "llama.tar.gz"
        urllib.request.urlretrieve(LLAMA_CPP_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter=lambda m, _: m.replace(name=m.name.split("/", 1)[1]) if "/" in m.name else None)
        archive.unlink()
    return binary


def model_files(repo: str = REPO, model: str = MODEL_FILE, mmproj: str = MMPROJ_FILE) -> tuple[Path, Path]:
    local = WEIGHTS_DIR / repo.split("/")[1]
    return tuple(Path(hf_hub_download(repo, f, local_dir=local, cache_dir=WEIGHTS_DIR / "hf")) for f in (model, mmproj))


def _image_part(image: str | Path) -> dict:
    if str(image).startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": str(image)}}
    mime = mimetypes.guess_type(str(image))[0] or "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(Path(image).read_bytes()).decode()}"}}


@dataclass
class Response:
    content: str
    reasoning: str
    usage: dict


@dataclass
class Qwen:
    repo: str = REPO
    model: str = MODEL_FILE
    mmproj: str = MMPROJ_FILE
    ctx: int = CTX
    seed: int = SEED
    port: int = 0
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        model, mmproj = model_files(self.repo, self.model, self.mmproj)
        if self.port == 0:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
        cmd = [str(server_binary()), "-m", str(model), "--mmproj", str(mmproj), "-ngl", "99", "-c", str(self.ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(self.seed), "--reasoning-format", "deepseek", "--host", "127.0.0.1", "--port", str(self.port)]
        log = WEIGHTS_DIR / "llama-server.log"
        with log.open("w") as f:
            self._proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        for _ in range(600):
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {log}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except OSError:
                time.sleep(1)
        raise TimeoutError(f"llama-server did not come up, see {log}")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def chat(self, text: str | None = None, images: list[str | Path] | str | Path = (), think: bool = True, max_tokens: int = 32768, system: str | None = None) -> Response:
        images = [images] if isinstance(images, (str, Path)) else list(images)
        content = [_image_part(i) for i in images] + ([{"type": "text", "text": text}] if text else [])
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = {"messages": messages, "seed": self.seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}, **(THINKING if think else INSTRUCT)}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            data = json.load(r)
        message = data["choices"][0]["message"]
        return Response(message.get("content") or "", message.get("reasoning_content") or "", data["usage"])
