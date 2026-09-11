# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub"]
# ///
from __future__ import annotations

import atexit
import base64
import importlib.util
import json
import mimetypes
import os
import random
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self


#
# utils
#


def set_storage(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "tmp").mkdir(exist_ok=True)
    os.environ["HF_HOME"] = os.environ["HF_HUB_CACHE"] = os.environ["TRANSFORMERS_CACHE"] = str(weights_dir / "hf")
    os.environ["HF_XET_CACHE"] = str(weights_dir / "hf" / "xet")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["LLAMA_CACHE"] = str(weights_dir / "llama")
    os.environ["TORCH_HOME"] = str(weights_dir / "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(weights_dir / "torch" / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(weights_dir / "torch" / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(weights_dir / "cuda")
    os.environ["XDG_CACHE_HOME"] = os.environ["XDG_DATA_HOME"] = os.environ["XDG_CONFIG_HOME"] = str(weights_dir / "cache")
    os.environ["TMPDIR"] = str(weights_dir / "tmp")
    return weights_dir


def set_seed(seed: int = 41) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    if importlib.util.find_spec("numpy"):
        importlib.import_module("numpy").random.seed(seed)
    if importlib.util.find_spec("torch"):
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


#
# setup
#


def download_llama_server(weights_dir: Path, tag: str = "b10908") -> Path:
    root = weights_dir / f"llama.cpp-{tag}"
    if not (root / "llama-server").exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", root / "llama.tar.gz")
        with tarfile.open(root / "llama.tar.gz") as tar:
            tar.extractall(root, filter=lambda m, _: m.replace(name=m.name.split("/", 1)[1]) if "/" in m.name else None)
        (root / "llama.tar.gz").unlink()
    assert os.access(root / "llama-server", os.X_OK), f"{root / 'llama-server'} is missing or not executable"
    return root / "llama-server"


def download_gguf(weights_dir: Path, repo: str, filename: str) -> Path:
    path = Path(importlib.import_module("huggingface_hub").hf_hub_download(repo, filename, local_dir=weights_dir / repo.split("/")[1], cache_dir=weights_dir / "hf"))
    assert path.is_relative_to(weights_dir), f"{path} escaped {weights_dir}"
    return path


def start_llama_server(weights_dir: Path, repo: str, model: str, mmproj: str, ctx: int, seed: int, port: int) -> subprocess.Popen:
    cmd = [str(download_llama_server(weights_dir)), "-m", str(download_gguf(weights_dir, repo, model)), "--mmproj", str(download_gguf(weights_dir, repo, mmproj)), "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(seed), "--reasoning-format", "deepseek", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=(weights_dir / "llama-server.log").open("w"), stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    for _ in range(600):
        assert proc.poll() is None, f"llama-server exited, see {weights_dir / 'llama-server.log'}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except OSError:
            time.sleep(1)
    raise AssertionError(f"llama-server not healthy after 600s, see {weights_dir / 'llama-server.log'}")


#
# inference
#


def image_part(image: str | Path) -> dict:
    if str(image).startswith("data:"):
        return {"type": "image_url", "image_url": {"url": str(image)}}
    if str(image).startswith(("http://", "https://")):
        data = urllib.request.urlopen(urllib.request.Request(str(image), headers={"user-agent": "quick-qwen27b"})).read()
    else:
        assert Path(image).is_file(), f"{image} is not a file"
        data = Path(image).read_bytes()
    return {"type": "image_url", "image_url": {"url": f"data:{mimetypes.guess_type(str(image))[0] or 'image/png'};base64,{base64.b64encode(data).decode()}"}}


@dataclass
class Response:
    content: str
    reasoning: str
    usage: dict


class Qwen:
    def __init__(self, weights_dir: str | Path | None = None, repo: str = "unsloth/Qwen3.5-27B-GGUF", model: str = "Qwen3.5-27B-UD-Q5_K_XL.gguf", mmproj: str = "mmproj-F16.gguf", ctx: int = 65536, seed: int = 41, port: int = 8080):
        weights_dir = set_storage(Path(weights_dir or Path(__file__).resolve().parent / "weights"))
        self.seed = set_seed(seed)
        self.port = port
        self.proc = start_llama_server(weights_dir, repo, model, mmproj, ctx, seed, port)

    def chat(self, text: str | None = None, images: list[str | Path] | str | Path | None = None, think: bool = True, max_tokens: int = 32768, system: str | None = None) -> Response:
        images = [images] if isinstance(images, (str, Path)) else list(images or [])
        assert text or images, "need text or at least one image"
        content = [image_part(i) for i in images] + ([{"type": "text", "text": text}] if text else [])
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        sampling = {"temperature": 1.0, "top_p": 0.95} if think else {"temperature": 0.7, "top_p": 0.8}
        body = {"messages": messages, "seed": self.seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, **sampling}
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=24 * 3600) as response:
            message = json.load(response)
        assert message.get("choices"), f"no choices in response: {message}"
        return Response(message["choices"][0]["message"].get("content") or "", message["choices"][0]["message"].get("reasoning_content") or "", message["usage"])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
