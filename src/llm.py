"""Local LLM server process management and inference client for llama.cpp.
Handles starting, health-checking, querying, switching, and terminating local GGUF models.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import time
import json
import urllib.request
import urllib.error

from src.data_loader import get_project_root


class LocalLLMManager:
    """Manages local llama-server.exe processes and HTTP API communication."""

    MODELS = {
        "llama": "models/llm/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "qwen": "models/llm/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 99,
        server_timeout: int = 60
    ):
        """Initialize LocalLLMManager.

        Parameters
        ----------
        host : str
            Server bind address. Default is 127.0.0.1.
        port : int
            Server port. Default is 8080.
        n_gpu_layers : int
            Number of layers to offload to GPU via CUDA (-ngl). Default is 99.
        server_timeout : int
            Maximum seconds to wait for server health check. Default is 60.
        """
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.server_timeout = server_timeout
        self.base_url = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        self.log_file = get_project_root() / "results" / "llama_server.log"

    def get_server_executable(self) -> Path:
        """Return path to llama-server.exe."""
        server_path = get_project_root() / "runtime" / "llama-cuda" / "llama-server.exe"
        if not server_path.exists():
            raise FileNotFoundError(f"llama-server executable not found at {server_path}")
        return server_path

    def get_model_path(self, model_key: str) -> Path:
        """Return path to GGUF model file for given key ('llama' or 'qwen')."""
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(self.MODELS.keys())}")
        model_path = get_project_root() / self.MODELS[model_key]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return model_path

    def is_server_ready(self, timeout: float = 3.0) -> bool:
        """Check if server /health endpoint returns HTTP 200."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start_server(self, model_key: str = "llama") -> bool:
        """Start llama-server with the specified model if not already running."""
        # If server is already running with the requested model, return True
        if self.current_model == model_key and self.is_server_ready():
            return True

        # Stop any existing server before starting a new one
        self.stop_server()

        server_exe = self.get_server_executable()
        model_path = self.get_model_path(model_key)

        cmd = [
            str(server_exe),
            "-m", str(model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers)
        ]

        # Ensure directory for logs exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(self.log_file, "w", encoding="utf-8")

        self.process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT
        )

        # Poll health endpoint until ready or timeout
        start_time = time.time()
        while time.time() - start_time < self.server_timeout:
            if self.is_server_ready(timeout=2.0):
                self.current_model = model_key
                return True
            time.sleep(1.0)

        # Timed out
        self.stop_server()
        raise TimeoutError(
            f"Timed out waiting {self.server_timeout}s for llama-server ({model_key}) to become ready. "
            f"Check log file at {self.log_file}"
        )

    def stop_server(self) -> None:
        """Terminate the server process cleanly."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                self.current_model = None

    def switch_model(self, target_model: str) -> bool:
        """Safely transition from current model to target model."""
        if self.current_model == target_model and self.is_server_ready():
            return True
        return self.start_server(target_model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        request_timeout: int = 120
    ) -> Dict[str, Any]:
        """Send completion request to llama-server and record latency metrics.

        Parameters
        ----------
        prompt : str
            Full formatted prompt string.
        max_tokens : int, optional
            Maximum tokens to generate (n_predict). Default is 256.
        temperature : float, optional
            Sampling temperature. Default is 0.2.
        request_timeout : int, optional
            HTTP request timeout in seconds. Default is 120.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'content': generated text string
            - 'tokens_predicted': count of generated tokens
            - 'latency_seconds': total elapsed wall time
            - 'tokens_per_second': throughput rate
            - 'model': active model key
        """
        if not self.is_server_ready():
            raise ConnectionError(f"Server is not active or ready at {self.base_url}")

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature
        }

        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start_time = time.time()
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start_time

        content = result.get("content", "").strip()
        tokens_predicted = result.get("tokens_predicted", len(content.split()))
        tps = (tokens_predicted / elapsed) if elapsed > 0 else 0.0

        return {
            "content": content,
            "tokens_predicted": tokens_predicted,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_second": round(tps, 2),
            "model": self.current_model
        }
