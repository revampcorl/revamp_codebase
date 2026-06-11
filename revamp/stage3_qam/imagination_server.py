"""Local world-model imagination server for two-chunk OpenPI QAM.

The JAX/OpenPI trainer sends a real batch plus pi0 A1 action chunk. This
server runs the PyTorch Stage-1 world model with no gradients and returns the
predicted future RGB frames for constructing the A2 imagined observation.
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

from revamp.common.world_model import WorldModel


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        model_safe = path / "model.safetensors"
        model_bin = path / "pytorch_model.bin"
        if model_safe.exists():
            return model_safe
        if model_bin.exists():
            return model_bin
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    return path


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("world_model", ckpt)


def _npz_bytes(payload: dict[str, np.ndarray], *, compressed: bool = True) -> bytes:
    bio = BytesIO()
    if compressed:
        np.savez_compressed(bio, **payload)
    else:
        np.savez(bio, **payload)
    return bio.getvalue()


class WorldModelImagOracle:
    def __init__(self, config_path: str, checkpoint: str, device: str):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        self.model = WorldModel(self.config).to(self.device)
        ckpt = _checkpoint_file(checkpoint)
        missing, unexpected = self.model.load_state_dict(_load_state_dict(ckpt), strict=False)
        if missing or unexpected:
            print(
                "[world_model_imag_server] checkpoint load warning: "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(f"[world_model_imag_server] first missing keys: {missing[:12]}", flush=True)
            if unexpected:
                print(f"[world_model_imag_server] first unexpected keys: {unexpected[:12]}", flush=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        t5 = torch.load(self.config.dataset.tasks[0].t5_embedding_path, map_location=self.device).to(self.model.dtype)
        self.t5 = t5.unsqueeze(0) if t5.dim() == 2 else t5

    def predict(self, arrays: dict[str, np.ndarray], num_inference_steps: int) -> dict[str, np.ndarray]:
        state = torch.as_tensor(arrays["state"], device=self.device).float()
        cam_left = torch.as_tensor(arrays["cam_left_segment"], device=self.device).float()
        cam_right = torch.as_tensor(arrays["cam_right_segment"], device=self.device).float()
        cam_high = torch.as_tensor(arrays["cam_high_segment"], device=self.device).float()
        action = torch.as_tensor(arrays["action"], device=self.device).float()
        if action.dim() == 3:
            action_chunks = action.unsqueeze(1)
        elif action.dim() == 4:
            action_chunks = action
        else:
            raise ValueError(f"action must be [B,H,D] or [B,N,H,D], got {tuple(action.shape)}")

        if "states" in arrays:
            state_chunks = torch.as_tensor(arrays["states"], device=self.device).float()
            if state_chunks.dim() != 3:
                raise ValueError(f"states must be [B,N,S], got {tuple(state_chunks.shape)}")
            if state_chunks.shape[:2] != action_chunks.shape[:2]:
                raise ValueError(
                    f"states shape {tuple(state_chunks.shape)} does not match "
                    f"action chunks shape {tuple(action_chunks.shape)}"
                )
        else:
            state_chunks = None

        batch_size, num_chunks = action_chunks.shape[:2]
        t5_batch = self.t5.expand(batch_size, -1, -1)

        return_all_rgb = bool(
            np.asarray(arrays.get("return_all_rgb", np.asarray(True))).reshape(-1)[0]
        )

        with torch.no_grad():
            cond_latent = self.model.build_full_latent_sequence(
                cam_left, cam_right, cam_high,
            )[:, :, :1]

            rgb_by_cam = {
                "next_cam_left": [],
                "next_cam_right": [],
                "next_cam_high": [],
            }
            current_state = state
            final_pred = None
            for chunk_idx in range(num_chunks):
                if state_chunks is not None:
                    current_state = state_chunks[:, chunk_idx]
                decode_rgb = return_all_rgb or chunk_idx == num_chunks - 1
                pred = self.model.predict_chunk_from_cond_latent(
                    cond_latent=cond_latent,
                    state=current_state,
                    action_chunk=action_chunks[:, chunk_idx],
                    t5_embeddings=t5_batch,
                    num_inference_steps=int(num_inference_steps),
                    decode_rgb=decode_rgb,
                )
                final_pred = pred
                target_latents = pred["next_cam_concat_latent"]
                cond_latent = target_latents[:, :, -1:]
                if decode_rgb:
                    for key in rgb_by_cam:
                        rgb_by_cam[key].append(pred[key])
                if state_chunks is None:
                    next_state = pred["next_state"].float()
                    if next_state.shape[-1] != current_state.shape[-1] and chunk_idx + 1 < num_chunks:
                        raise ValueError(
                            "Multi-chunk latent autoregressive prediction without explicit "
                            "states requires action_dim == state_dim; pass states=[B,N,S] "
                            "for each chunk instead."
                        )
                    current_state = next_state

        if final_pred is None:
            raise RuntimeError("world-model imagination received zero action chunks")

        out = {}
        for key in ("next_cam_left", "next_cam_right", "next_cam_high"):
            if return_all_rgb:
                frames = torch.cat(rgb_by_cam[key], dim=2)
            else:
                frames = final_pred[key]
            frames = frames.detach().float()
            if float(frames.min().item()) < -0.05:
                frames = (frames + 1.0) * 0.5
            frames = frames.clamp(0.0, 1.0)
            out[key] = frames.cpu().numpy().astype(np.float32)
        out["num_pred_frames"] = np.asarray(out["next_cam_left"].shape[2], dtype=np.int32)
        out["num_pred_chunks"] = np.asarray(num_chunks, dtype=np.int32)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage-1 world-model config.")
    parser.add_argument("--checkpoint", required=True, help="Stage-1 world-model checkpoint dir or file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--response-compression",
        choices=("compressed", "uncompressed"),
        default="uncompressed",
    )
    args = parser.parse_args()

    oracle = WorldModelImagOracle(args.config, args.checkpoint, args.device)
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
            if self.path != "/predict_chunk":
                self.send_error(404)
                return
            try:
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                t0 = time.time()
                data = np.load(BytesIO(raw))
                arrays = {k: data[k] for k in data.files}
                decode_s = time.time() - t0
                steps = int(arrays.get("num_inference_steps", np.asarray(args.num_inference_steps)).reshape(-1)[0])

                wait_t0 = time.time()
                with oracle_lock:
                    wait_s = time.time() - wait_t0
                    compute_t0 = time.time()
                    pred = oracle.predict(arrays, steps)
                    compute_s = time.time() - compute_t0

                all_frames = np.concatenate(
                    [pred["next_cam_left"], pred["next_cam_right"], pred["next_cam_high"]],
                    axis=-1,
                )
                payload = {
                    **pred,
                    "pred_pixel_mean": np.asarray(float(all_frames.mean()), dtype=np.float32),
                    "pred_pixel_std": np.asarray(float(all_frames.std()), dtype=np.float32),
                    "pred_pixel_min": np.asarray(float(all_frames.min()), dtype=np.float32),
                    "pred_pixel_max": np.asarray(float(all_frames.max()), dtype=np.float32),
                    "decode_s": np.asarray(decode_s, dtype=np.float32),
                    "wait_s": np.asarray(wait_s, dtype=np.float32),
                    "compute_s": np.asarray(compute_s, dtype=np.float32),
                    "encode_s": np.asarray(0.0, dtype=np.float32),
                }
                t0 = time.time()
                body = _npz_bytes(payload, compressed=(args.response_compression == "compressed"))
                encode_s = time.time() - t0
                payload["encode_s"] = np.asarray(encode_s, dtype=np.float32)
                body = _npz_bytes(payload, compressed=(args.response_compression == "compressed"))

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
        f"[world_model_imag_server] listening on {args.host}:{args.port}, "
        f"device={args.device}, num_inference_steps={args.num_inference_steps}, "
        f"response_compression={args.response_compression}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
