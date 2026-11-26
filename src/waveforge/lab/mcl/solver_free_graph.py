"""Replay fixed neural work on CUDA without changing channel decoding."""

import threading
import time

import torch


class GraphedGenerator:
    def __init__(self, model):
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("CUDA graph generation requires a CUDA model")
        self.lock = threading.Lock()
        started = time.perf_counter()
        self.raw = torch.zeros((1, 7, 64, 64), device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                model(self.raw)
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.parameters = model(self.raw)["parameters"]
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.perf_counter() - started

    def generate(self, raw):
        with self.lock, torch.inference_mode():
            self.raw.copy_(torch.from_numpy(raw[None]))
            self.graph.replay()
            return self.parameters[0].cpu().numpy()
