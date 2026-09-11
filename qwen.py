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
import socket
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self


def set_storage(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(weights_dir / "hf")
    os.environ["HF_HUB_CACHE"] = str(weights_dir / "hf")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_XET_CACHE"] = str(weights_dir / "hf" / "xet")
    os.environ["TRANSFORMERS_CACHE"] = str(weights_dir / "hf")
    os.environ["LLAMA_CACHE"] = str(weights_dir / "llama")
    os.environ["TORCH_HOME"] = str(weights_dir / "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(weights_dir / "torch" / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(weights_dir / "torch" / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(weights_dir / "cuda")
    os.environ["XDG_CACHE_HOME"] = str(weights_dir / "cache")
    os.environ["XDG_DATA_HOME"] = str(weights_dir / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(weights_dir / "cache")
    os.environ["TMPDIR"] = str(weights_dir / "tmp")
    (weights_dir / "tmp").mkdir(exist_ok=True)
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


class Download:
    @staticmethod
    def llama_server(weights_dir: Path, tag: str = "b10908") -> Path:
        root = weights_dir / f"llama.cpp-{tag}"
        if (root / "llama-server").exists():
            return root / "llama-server"
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", root / "llama.tar.gz")
        with tarfile.open(root / "llama.tar.gz") as tar:
            tar.extractall(root, filter=Download._strip_top_dir)
        (root / "llama.tar.gz").unlink()
        return root / "llama-server"

    @staticmethod
    def _strip_top_dir(member: tarfile.TarInfo, _: str) -> tarfile.TarInfo | None:
        return member.replace(name=member.name.split("/", 1)[1]) if "/" in member.name else None

    @staticmethod
    def gguf(weights_dir: Path, repo: str, filename: str) -> Path:
        hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download
        return Path(hf_hub_download(repo, filename, local_dir=weights_dir / repo.split("/")[1], cache_dir=weights_dir / "hf"))

    @staticmethod
    def _image_bytes(image: str | Path) -> bytes:
        if not str(image).startswith(("http://", "https://")):
            return Path(image).read_bytes()
        return urllib.request.urlopen(urllib.request.Request(str(image), headers={"user-agent": "quick-qwen27b"})).read()


class Server:
    @staticmethod
    def free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _is_healthy(port: int) -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return True
        except OSError:
            return False

    @staticmethod
    def wait_until_healthy(proc: subprocess.Popen, port: int, log: Path, timeout_s: int = 600) -> None:
        for _ in range(timeout_s):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {log}")
            if Server._is_healthy(port):
                return
            time.sleep(1)
        raise TimeoutError(f"llama-server did not come up, see {log}")

    @staticmethod
    def post_json(port: int, path: str, body: dict) -> dict:
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=24 * 3600) as response:
            return json.load(response)


class Message:
    @staticmethod
    def _image_part(image: str | Path) -> dict:
        if str(image).startswith("data:"):
            return {"type": "image_url", "image_url": {"url": str(image)}}
        mime = mimetypes.guess_type(str(image))[0] or "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(Download._image_bytes(image)).decode()}"}}

    @staticmethod
    def user(text: str | None, images: list[str | Path]) -> dict:
        return {"role": "user", "content": [Message._image_part(i) for i in images] + ([{"type": "text", "text": text}] if text else [])}

    @staticmethod
    def sampling(think: bool) -> dict:
        return {"top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, **({"temperature": 1.0, "top_p": 0.95} if think else {"temperature": 0.7, "top_p": 0.8})}


@dataclass
class Response:
    content: str
    reasoning: str
    usage: dict


class Qwen:
    def __init__(self, weights_dir: str | Path | None = None, repo: str = "unsloth/Qwen3.5-27B-GGUF", model: str = "Qwen3.5-27B-UD-Q5_K_XL.gguf", mmproj: str = "mmproj-F16.gguf", ctx: int = 65536, seed: int = 41, port: int = 0):
        weights_dir = set_storage(Path(weights_dir or Path(__file__).resolve().parent / "weights"))
        set_seed(seed)
        self.seed = seed
        self.port = port or Server.free_port()
        self.log = weights_dir / "llama-server.log"
        cmd = [str(Download.llama_server(weights_dir)), "-m", str(Download.gguf(weights_dir, repo, model)), "--mmproj", str(Download.gguf(weights_dir, repo, mmproj)), "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--seed", str(seed), "--reasoning-format", "deepseek", "--host", "127.0.0.1", "--port", str(self.port)]
        self.proc = subprocess.Popen(cmd, stdout=self.log.open("w"), stderr=subprocess.STDOUT)
        atexit.register(self.close)
        Server.wait_until_healthy(self.proc, self.port, self.log)

    def chat(self, text: str | None = None, images: list[str | Path] | str | Path | None = None, think: bool = True, max_tokens: int = 32768, system: str | None = None) -> Response:
        messages = ([{"role": "system", "content": system}] if system else []) + [Message.user(text, [images] if isinstance(images, (str, Path)) else list(images or []))]
        data = Server.post_json(self.port, "/v1/chat/completions", {"messages": messages, "seed": self.seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}, **Message.sampling(think)})
        return Response(data["choices"][0]["message"].get("content") or "", data["choices"][0]["message"].get("reasoning_content") or "", data["usage"])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
