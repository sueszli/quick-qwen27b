from __future__ import annotations

import atexit
import base64
import importlib.util
import json
import mimetypes
import os
import random
import socket
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

WEIGHTS_DIR = Path(os.environ.get("QWEN_WEIGHTS_DIR", Path(__file__).resolve().parent / "weights"))


def set_seed(seed: int) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np := importlib.util.find_spec("numpy") and importlib.import_module("numpy"):
        np.random.seed(seed)
    if torch := importlib.util.find_spec("torch") and importlib.import_module("torch"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return seed


def server_binary(tag: str = "b10908") -> Path:
    root = WEIGHTS_DIR / f"llama.cpp-{tag}"
    binary = root / "llama-server"
    if not binary.exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "llama.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter=lambda m, _: m.replace(name=m.name.split("/", 1)[1]) if "/" in m.name else None)
        archive.unlink()
    return binary


def model_files(repo: str, model: str, mmproj: str) -> tuple[Path, Path]:
    local = WEIGHTS_DIR / repo.split("/")[1]
    return tuple(Path(hf_hub_download(repo, f, local_dir=local, cache_dir=WEIGHTS_DIR / "hf")) for f in (model, mmproj))


def _image_part(image: str | Path) -> dict:
    if str(image).startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": str(image)}}
    mime = mimetypes.guess_type(str(image))[0] or "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(Path(image).read_bytes()).decode()}"}}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        return True
    except OSError:
        return False


@dataclass
class Response:
    content: str
    reasoning: str
    usage: dict


@dataclass
class Qwen:
    repo: str = "unsloth/Qwen3.5-27B-GGUF"
    model: str = "Qwen3.5-27B-UD-Q5_K_XL.gguf"
    mmproj: str = "mmproj-F16.gguf"
    ctx: int = 65536
    seed: int = 41
    port: int = 0
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        set_seed(self.seed)
        model, mmproj = model_files(self.repo, self.model, self.mmproj)
        self.port = self.port or _free_port()
        cmd = [str(server_binary()), "-m", str(model), "--mmproj", str(mmproj), "-ngl", "99", "-c", str(self.ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(self.seed), "--reasoning-format", "deepseek", "--host", "127.0.0.1", "--port", str(self.port)]
        log = WEIGHTS_DIR / "llama-server.log"
        with log.open("w") as f:
            self._proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        for _ in range(600):
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {log}")
            if _healthy(self.port):
                return
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
        body = {"messages": messages, "seed": self.seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, **({"temperature": 1.0, "top_p": 0.95} if think else {"temperature": 0.7, "top_p": 0.8})}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            data = json.load(r)
        message = data["choices"][0]["message"]
        return Response(message.get("content") or "", message.get("reasoning_content") or "", data["usage"])
