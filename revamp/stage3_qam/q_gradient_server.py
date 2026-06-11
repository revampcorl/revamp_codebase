"""Local PyTorch Q-gradient server for OpenPI QAM.

The distributed JAX trainer keeps OpenPI on JAX devices and asks this local
server for dQ/da. This keeps PyTorch/Wan memory and JAX/OpenPI memory in
separate processes, which is much more predictable than one mixed process.

Protocol:
  POST /grad with a compressed npz body containing:
    state          float32 [B, S]
    cam_left_cond  float32 [B, 3, 4, H, W] raw[t-3..t]
    cam_right_cond float32 [B, 3, 4, H, W] raw[t-3..t]
    cam_high_cond  float32 [B, 3, 4, H, W] raw[t-3..t]
    action         float32 [B, 8, 12]

The server pads the 4-frame condition segment to a 12-frame VAE-safe segment
by repeating the current frame. This preserves the Stage 3 design invariant
that Q sees only s_t while avoiding short temporal inputs in the video VAE.

Response is compressed npz:
    grad           float32 [B, 8, 12]
    q_values       float32 [B]
    q              float32 scalar mean Q
    decode_s       float32 scalar
    wait_s         float32 scalar queue wait before model execution
    compute_s      float32 scalar model execution time
    encode_s       float32 scalar
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf

from revamp.common.constants import NUM_RAW_FRAMES_PER_SAMPLE
from revamp.common.world_model import WorldModel


def _load_stage2_weights(model: WorldModel, checkpoint: str) -> None:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        model_bin = ckpt_path / "pytorch_model.bin"
        model_safe = ckpt_path / "model.safetensors"
        if model_bin.exists():
            state_dict = torch.load(model_bin, map_location="cpu")
        elif model_safe.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(model_safe), device="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_path}")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("world_model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[q_gradient_server] checkpoint load warning: "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            flush=True,
        )
        if missing:
            print(f"[q_gradient_server] first missing keys: {missing[:12]}", flush=True)
        if unexpected:
            print(f"[q_gradient_server] first unexpected keys: {unexpected[:12]}", flush=True)
        critical_prefixes = (
            "q_ensemble.",
            "state_proj.",
            "action_chunk_embedder.",
            "action_context_mlp.",
            "action_time_mlp.",
            "state_context_mlp.",
            "state_time_mlp.",
        )
        critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
        if critical_missing:
            raise RuntimeError(
                "Stage2 Q checkpoint is missing critical Q/action/state keys; "
                f"first keys: {critical_missing[:12]}"
            )


def _npz_bytes(payload: dict[str, np.ndarray], *, compressed: bool = True) -> bytes:
    bio = BytesIO()
    if compressed:
        np.savez_compressed(bio, **payload)
    else:
        np.savez(bio, **payload)
    return bio.getvalue()


class QGradientOracle:
    def __init__(self, config_path: str, device: str, *, compile_q: bool = False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        self.model = WorldModel(self.config).to(self.device)
        _load_stage2_weights(self.model, self.config.training.stage2_q_checkpoint)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        t5 = torch.load(self.config.dataset.tasks[0].t5_embedding_path, map_location=self.device).to(self.model.dtype)
        self.t5 = t5.unsqueeze(0) if t5.dim() == 2 else t5
        self.lambda_q = float(self.config.training.get("lambda_q", 1.0))
        self.detach_dit = bool(self.config.training.get("q_detach_dit", True))
        if compile_q:
            self.model.forward_q = torch.compile(  # type: ignore[method-assign]
                self.model.forward_q,
                mode="reduce-overhead",
                fullgraph=False,
            )

    @staticmethod
    def _vae_safe_cond_segment(cam: torch.Tensor) -> torch.Tensor:
        """Pad cond-only raw frames to the VAE's standard 12-frame length.

        The trainer sends raw[t-3..t]. Repeating raw[t] avoids leaking future
        demo frames into Q(s_t, a) while keeping the video VAE on its normal
        temporal input length.
        """
        if cam.dim() == 4:
            cam = cam.unsqueeze(2)
        if cam.dim() != 5:
            raise ValueError(f"camera condition must be [B,3,T,H,W] or [B,3,H,W], got {tuple(cam.shape)}")
        t_raw = int(cam.shape[2])
        if t_raw <= 0:
            raise ValueError(f"camera condition has empty temporal dimension: {tuple(cam.shape)}")
        target_t = int(NUM_RAW_FRAMES_PER_SAMPLE)
        if t_raw == target_t:
            return cam
        if t_raw > target_t:
            return cam[:, :, :target_t]
        pad = cam[:, :, -1:].expand(-1, -1, target_t - t_raw, -1, -1)
        return torch.cat([cam, pad], dim=2)

    def grad(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        state = torch.as_tensor(arrays["state"], device=self.device).float()
        cam_left = self._vae_safe_cond_segment(torch.as_tensor(arrays["cam_left_cond"], device=self.device).float())
        cam_right = self._vae_safe_cond_segment(torch.as_tensor(arrays["cam_right_cond"], device=self.device).float())
        cam_high = self._vae_safe_cond_segment(torch.as_tensor(arrays["cam_high_cond"], device=self.device).float())
        action = torch.as_tensor(arrays["action"], device=self.device).float().detach().requires_grad_(True)
        t5_batch = self.t5.expand(state.shape[0], -1, -1)

        with torch.enable_grad():
            q1, q2 = self.model.forward_q(
                state,
                cam_left,
                cam_right,
                cam_high,
                action,
                t5_batch,
                detach_dit=self.detach_dit,
            )
            q = torch.min(q1, q2)
            grad = torch.autograd.grad(q.sum() / self.lambda_q, action)[0]
        q_values = q.detach().cpu().numpy().astype(np.float32)
        return grad.detach().cpu().numpy().astype(np.float32), q_values, float(q.detach().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--response-compression",
        choices=("compressed", "uncompressed"),
        default="uncompressed",
        help="Compress the small grad response. Uncompressed is usually faster on localhost.",
    )
    parser.add_argument(
        "--compile-q",
        action="store_true",
        help="Wrap WorldModel.forward_q with torch.compile(reduce-overhead).",
    )
    args = parser.parse_args()

    oracle = QGradientOracle(args.config, args.device, compile_q=args.compile_q)
    oracle_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path != "/grad":
                self.send_error(404)
                return
            try:
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                t0 = time.time()
                data = np.load(BytesIO(raw))
                arrays = {k: data[k] for k in data.files}
                decode_s = time.time() - t0
                wait_t0 = time.time()
                # ThreadingHTTPServer can receive requests from multiple
                # trainers. A single PyTorch/Wan model instance is not safe to
                # run through concurrent forward/backward calls, so serialize
                # oracle access and let clients queue at the HTTP boundary.
                with oracle_lock:
                    wait_s = time.time() - wait_t0
                    compute_t0 = time.time()
                    grad, q_values, q = oracle.grad(arrays)
                    payload = {
                        "grad": grad,
                        "q_values": q_values,
                        "q": np.asarray(q, dtype=np.float32),
                    }
                    compute_s = time.time() - compute_t0
                t0 = time.time()
                body = _npz_bytes(
                    {
                        **payload,
                        "decode_s": np.asarray(decode_s, dtype=np.float32),
                        "wait_s": np.asarray(wait_s, dtype=np.float32),
                        "compute_s": np.asarray(compute_s, dtype=np.float32),
                        "encode_s": np.asarray(0.0, dtype=np.float32),
                    },
                    compressed=(args.response_compression == "compressed"),
                )
                encode_s = time.time() - t0
                # Re-encode once with the measured encode time. The response
                # is tiny compared with the request, and this makes profiling
                # accurate without changing the training math.
                body = _npz_bytes(
                    {
                        **payload,
                        "decode_s": np.asarray(decode_s, dtype=np.float32),
                        "wait_s": np.asarray(wait_s, dtype=np.float32),
                        "compute_s": np.asarray(compute_s, dtype=np.float32),
                        "encode_s": np.asarray(encode_s, dtype=np.float32),
                    },
                    compressed=(args.response_compression == "compressed"),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:  # pragma: no cover - server diagnostics
                traceback.print_exc()
                self.send_error(500, f"{type(exc).__name__}: {exc}")

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[q_gradient_server] listening on {args.host}:{args.port}, "
        f"device={args.device}, response_compression={args.response_compression}, "
        f"compile_q={args.compile_q}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
