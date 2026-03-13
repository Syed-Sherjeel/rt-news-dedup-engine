import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    # bge-small-en-v1.5: 33M params, 384-dim, MTEB 62.0 vs MiniLM's 56.3
    # same dim as MiniLM so buffer matmul cost is identical
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = _get_device()
        print(f"[embedder] loading {model_name} on {self.device} ...")
        self.model = SentenceTransformer(model_name, device=self.device)

        # fp16 on GPU — halves memory, ~20% faster, no quality loss for similarity
        if self.device in ("cuda", "mps"):
            self.model = self.model.half()

        self._warmup()
        print("[embedder] ready")

    def _warmup(self) -> None:
        # two passes: first allocates GPU kernels, second is representative
        for _ in range(2):
            self.model.encode(["warm up"], convert_to_numpy=True, normalize_embeddings=True)

    def encode(self, text: str) -> np.ndarray:
        t1 = time.perf_counter()
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(time.perf_counter()-t1)
        return vec[0].astype(np.float32)

    def benchmark(self, n: int = 50) -> None:
        sample = "Bitcoin whales accumulating BTC at current prices signal bullish momentum"
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.encode(sample)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        print(
            f"[benchmark] n={n}  device={self.device} | "
            f"p50={times[n//2]:.1f}ms  "
            f"p95={times[int(n*0.95)]:.1f}ms  "
            f"p99={times[int(n*0.99)]:.1f}ms"
        )