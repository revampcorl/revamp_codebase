"""Contact-aware ViVa / LeRobot dataset adapter used by the world model.

This module is adapted from the ViVa dataset utilities and extended for the
REVAMP TurnOnSinkFaucet release with contact sidecars and release-package
path handling. See NOTICE for the license boundary.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

CAMERA_KEY_ALIASES = {
    "cam_high": (
        "observation.images.cam_high",
        "observation.images.robot0_agentview_left",
    ),
    "cam_left_wrist": (
        "observation.images.cam_left_wrist",
        "observation.images.robot0_eye_in_hand",
    ),
    "cam_right_wrist": (
        "observation.images.cam_right_wrist",
        "observation.images.robot0_agentview_right",
    ),
}

DEFAULT_LEFT_CONTACT_KEY = "observation.contact.left"
DEFAULT_RIGHT_CONTACT_KEY = "observation.contact.right"
DEFAULT_CONTACT_SIDECAR_NAME = "contact_features.npy"


def _to_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def _as_flat_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float().reshape(-1)
    return torch.tensor(np.asarray(value), dtype=torch.float32).reshape(-1)


class VivaDataset(Dataset):
    """Load RoboCasa LeRobot rows with normalized state and camera tensors.

    Contact sidecars are stored as ``[T, 2]`` left/right contact values. For the
    released world-model configs they are collapsed to one binary contact bit,
    giving ``state_dim = 16 proprio + 1 contact = 17``.
    """

    def __init__(
        self,
        data_paths: Optional[List[str]] = None,
        video_height: int = 384,
        video_width: int = 320,
        state_stats: Optional[Dict] = None,
        skip_video_decoding: bool = False,
        max_samples: Optional[int] = None,
        future_offset: int = 75,
        task_configs: Optional[List[Dict]] = None,
        state_dim: Optional[int] = None,
        use_contact: bool = False,
        left_contact_key: str = DEFAULT_LEFT_CONTACT_KEY,
        right_contact_key: str = DEFAULT_RIGHT_CONTACT_KEY,
        require_contact: bool = True,
        contact_sidecar_name: str = DEFAULT_CONTACT_SIDECAR_NAME,
    ):
        self.video_height = int(video_height)
        self.video_width = int(video_width)
        self.max_samples = max_samples
        self.future_offset = int(future_offset)
        self.configured_state_dim = int(state_dim) if state_dim is not None else None
        self.use_contact = bool(use_contact)
        self.left_contact_key = left_contact_key
        self.right_contact_key = right_contact_key
        self.require_contact = bool(require_contact)
        self.contact_sidecar_name = contact_sidecar_name
        self._camera_log_once = set()
        self._contact_log_once = set()
        self._contact_feature_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        self._subdataset_to_task_id = None
        if task_configs is not None:
            all_data_paths = []
            self._subdataset_to_task_id = {}
            sub_ds_counter = 0
            for task_idx, task_config in enumerate(task_configs):
                for data_path in task_config["data_paths"]:
                    all_data_paths.append(data_path)
                    self._subdataset_to_task_id[sub_ds_counter] = task_idx
                    sub_ds_counter += 1
            data_paths = all_data_paths
        elif data_paths is None:
            raise ValueError("Either data_paths or task_configs must be provided.")

        from giga_datasets.datasets.dataset import load_dataset

        configs = [
            {
                "_class_name": "LeRobotDataset",
                "data_path": str(input_dir),
                "skip_video_decoding": skip_video_decoding,
            }
            for input_dir in data_paths
        ]
        self.lerobot_dataset = load_dataset(configs)

        if not hasattr(self.lerobot_dataset, "cumulative_sizes"):
            cumsum = []
            total = 0
            for dataset in self.lerobot_dataset.datasets:
                total += len(dataset)
                cumsum.append(total)
            self.lerobot_dataset.cumulative_sizes = cumsum

        self._subdataset_episode_lengths = []
        max_episode_length = 0
        for dataset in self.lerobot_dataset.datasets:
            sub_lengths = []
            meta = dataset.dataset.meta
            total_episodes = meta.info["total_episodes"]
            for episode_idx in range(total_episodes):
                length = int(meta.episodes[episode_idx]["length"])
                sub_lengths.append(length)
                max_episode_length = max(max_episode_length, length)
            self._subdataset_episode_lengths.append(sub_lengths)

        self.base_state_dim = self._infer_base_state_dim()
        inferred_state_dim = self.base_state_dim + (1 if self.use_contact else 0)
        if self.configured_state_dim is not None and self.configured_state_dim != inferred_state_dim:
            raise ValueError(
                f"Configured state_dim={self.configured_state_dim}, but dataset base "
                f"state_dim={self.base_state_dim} and use_contact={self.use_contact} "
                f"imply state_dim={inferred_state_dim}."
            )
        self.state_dim = inferred_state_dim

        if state_stats is not None:
            self.state_min = torch.tensor(state_stats["state_min"], dtype=torch.float32)
            self.state_max = torch.tensor(state_stats["state_max"], dtype=torch.float32)
            if self.state_min.numel() != self.state_dim or self.state_max.numel() != self.state_dim:
                raise ValueError(
                    f"state_stats dim mismatch: got {self.state_min.numel()}/"
                    f"{self.state_max.numel()}, expected {self.state_dim}"
                )
        else:
            base_min = torch.full((self.base_state_dim,), -3.0, dtype=torch.float32)
            base_max = torch.full((self.base_state_dim,), 3.0, dtype=torch.float32)
            if self.use_contact:
                self.state_min = torch.cat([base_min, torch.zeros(1)], dim=0)
                self.state_max = torch.cat([base_max, torch.ones(1)], dim=0)
            else:
                self.state_min = base_min
                self.state_max = base_max

        self.max_episode_length = max_episode_length
        self._total_samples = len(self.lerobot_dataset)
        self._sample_indices = None
        if self.max_samples is not None and self.max_samples > 0:
            self._effective_length = min(int(self.max_samples), self._total_samples)
            if self._effective_length < self._total_samples:
                rng = np.random.RandomState(42)
                self._sample_indices = rng.choice(
                    self._total_samples,
                    size=self._effective_length,
                    replace=False,
                )
        else:
            self._effective_length = self._total_samples

        logger.info(
            "Loaded VivaDataset: samples=%d, state_dim=%d, base_state_dim=%d, "
            "video=%dx%d, future_offset=%d, contact=%s",
            self._effective_length,
            self.state_dim,
            self.base_state_dim,
            self.video_height,
            self.video_width,
            self.future_offset,
            self.use_contact,
        )

    def __len__(self) -> int:
        return self._effective_length

    def _get_subdataset_index(self, global_idx: int) -> int:
        return bisect.bisect_right(self.lerobot_dataset.cumulative_sizes, global_idx)

    def _infer_base_state_dim(self) -> int:
        for sub_dataset in self.lerobot_dataset.datasets:
            meta_info = getattr(sub_dataset.dataset.meta, "info", {})
            features = meta_info.get("features", {}) if isinstance(meta_info, dict) else {}
            state_feature = features.get("observation.state")
            if isinstance(state_feature, dict):
                shape = state_feature.get("shape")
                if isinstance(shape, list) and shape:
                    return int(shape[0])

        if len(self.lerobot_dataset) > 0:
            sample = self.lerobot_dataset[0]
            state_raw = sample.get("observation.state", None)
            if state_raw is not None:
                return int(_as_flat_float_tensor(state_raw).numel())

        if self.configured_state_dim is not None:
            return self.configured_state_dim - (1 if self.use_contact else 0)
        raise ValueError("Could not infer observation.state dimension.")

    def _process_image(self, img) -> torch.Tensor:
        import torch.nn.functional as F

        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                image = img.float() if img.shape[0] == 3 else img.permute(2, 0, 1).float()
            else:
                image = img.float()
            if image.max() > 1.0:
                image = image / 255.0
        elif hasattr(img, "convert"):
            img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            image = torch.tensor(np.asarray(img), dtype=torch.float32)
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            if image.max() > 1.0:
                image = image / 255.0

        _, height, width = image.shape
        scale = min(self.video_height / height, self.video_width / width)
        new_h, new_w = int(height * scale), int(width * scale)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        pad_h, pad_w = self.video_height - new_h, self.video_width - new_w
        return F.pad(
            image,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=0,
        )

    def _compute_value(self, frame_idx: int, episode_length: int) -> float:
        if episode_length <= 1:
            return 0.5
        return (episode_length - frame_idx - 1) / (episode_length - 1)

    def _contact_label_path(self, sub_ds_idx: int, episode_index: int) -> Path:
        root = Path(self.lerobot_dataset.datasets[sub_ds_idx].dataset.root)
        return root / "extras" / f"episode_{episode_index:06d}" / self.contact_sidecar_name

    def _load_episode_contact_features(
        self,
        sub_ds_idx: int,
        episode_index: int,
        episode_length: int,
    ) -> Optional[np.ndarray]:
        key = (sub_ds_idx, episode_index)
        if key in self._contact_feature_cache:
            return self._contact_feature_cache[key]

        path = self._contact_label_path(sub_ds_idx, episode_index)
        if not path.exists():
            self._contact_feature_cache[key] = None
            return None

        try:
            features = np.load(path).astype(np.float32)
        except Exception as exc:
            logger.warning("Failed to load contact features %s: %s", path, exc)
            self._contact_feature_cache[key] = None
            return None

        if features.ndim != 2 or features.shape[1] < 2:
            logger.warning("Ignoring malformed contact features at %s: shape=%s", path, features.shape)
            self._contact_feature_cache[key] = None
            return None

        features = (features[:, :2] > 0.5).astype(np.float32)
        features = features.max(axis=1, keepdims=True)
        if len(features) == episode_length + 1:
            features = features[:episode_length]
        elif len(features) != episode_length:
            logger.warning(
                "Ignoring contact features with bad length at %s: got %d, expected %d or %d",
                path,
                len(features),
                episode_length,
                episode_length + 1,
            )
            self._contact_feature_cache[key] = None
            return None

        self._contact_feature_cache[key] = features
        return features

    def _load_contact(
        self,
        sample,
        sub_ds_idx: int,
        episode_index: int,
        frame_index: int,
        episode_length: int,
    ) -> torch.Tensor:
        left_value = sample.get(self.left_contact_key, None)
        right_value = sample.get(self.right_contact_key, None)
        if left_value is not None and right_value is not None:
            left = 1.0 if _to_python_float(left_value) > 0.5 else 0.0
            right = 1.0 if _to_python_float(right_value) > 0.5 else 0.0
            return torch.tensor([max(left, right)], dtype=torch.float32)

        sidecar = self._load_episode_contact_features(sub_ds_idx, episode_index, episode_length)
        if sidecar is not None and 0 <= frame_index < len(sidecar):
            return torch.tensor(sidecar[frame_index], dtype=torch.float32).reshape(1)

        root = str(self.lerobot_dataset.datasets[sub_ds_idx].dataset.root)
        if self.require_contact:
            raise FileNotFoundError(
                f"Could not resolve contact for root={root}, episode={episode_index}, "
                f"frame={frame_index}. Expected sample keys {self.left_contact_key!r}/"
                f"{self.right_contact_key!r} or extras/{self.contact_sidecar_name}."
            )
        if root not in self._contact_log_once:
            logger.warning("Missing contact for %s; using zero contact bit.", root)
            self._contact_log_once.add(root)
        return torch.zeros(1, dtype=torch.float32)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.clamp(state, self.state_min, self.state_max)
        state_normalized = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return state_normalized * 2 - 1

    def _load_state(
        self,
        sample,
        sub_ds_idx: int,
        episode_index: int,
        frame_index: int,
        episode_length: int,
    ) -> torch.Tensor:
        state_raw = sample.get("observation.state", None)
        if state_raw is not None:
            base_state = _as_flat_float_tensor(state_raw)
            if base_state.numel() != self.base_state_dim:
                raise ValueError(
                    f"State dim mismatch: got {base_state.numel()}, expected {self.base_state_dim}"
                )
        else:
            base_state = torch.zeros(self.base_state_dim, dtype=torch.float32)

        if self.use_contact:
            contact = self._load_contact(
                sample,
                sub_ds_idx,
                episode_index,
                frame_index,
                episode_length,
            )
            base_state = torch.cat([base_state, contact], dim=0)
        return self._normalize_state(base_state)

    def _get_camera_image(self, sample, logical_name: str):
        for index, key in enumerate(CAMERA_KEY_ALIASES[logical_name]):
            value = sample.get(key, None)
            if value is not None:
                if index > 0:
                    log_key = (logical_name, key)
                    if log_key not in self._camera_log_once:
                        logger.info("Camera alias fallback: %s <- %s", logical_name, key)
                        self._camera_log_once.add(log_key)
                return value
        log_key = (logical_name, "missing")
        if log_key not in self._camera_log_once:
            logger.warning("Missing camera for %s; tried %s", logical_name, CAMERA_KEY_ALIASES[logical_name])
            self._camera_log_once.add(log_key)
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            actual_idx = idx
            if self._sample_indices is not None:
                actual_idx = int(self._sample_indices[idx])

            sample = self.lerobot_dataset[actual_idx]
            sub_ds_idx = self._get_subdataset_index(actual_idx)
            frame_index = _to_python_int(sample["frame_index"])
            episode_index = _to_python_int(sample["episode_index"])
            episode_length = self._subdataset_episode_lengths[sub_ds_idx][episode_index]

            state_normalized = self._load_state(
                sample,
                sub_ds_idx,
                episode_index,
                frame_index,
                episode_length,
            )

            cam_high_raw = self._get_camera_image(sample, "cam_high")
            cam_left_raw = self._get_camera_image(sample, "cam_left_wrist")
            cam_right_raw = self._get_camera_image(sample, "cam_right_wrist")
            dummy_img = torch.zeros(3, self.video_height, self.video_width)
            cam_high = self._process_image(cam_high_raw) if cam_high_raw is not None else dummy_img.clone()
            cam_left_wrist = self._process_image(cam_left_raw) if cam_left_raw is not None else dummy_img.clone()
            cam_right_wrist = self._process_image(cam_right_raw) if cam_right_raw is not None else dummy_img.clone()

            future_frame_idx = min(frame_index + self.future_offset, episode_length - 1)
            delta = future_frame_idx - frame_index
            if delta > 0:
                future_actual_idx = actual_idx + delta
                future_sample = self.lerobot_dataset[future_actual_idx]
                future_state_normalized = self._load_state(
                    future_sample,
                    sub_ds_idx,
                    episode_index,
                    future_frame_idx,
                    episode_length,
                )
                future_cam_high_raw = self._get_camera_image(future_sample, "cam_high")
                future_cam_left_raw = self._get_camera_image(future_sample, "cam_left_wrist")
                future_cam_right_raw = self._get_camera_image(future_sample, "cam_right_wrist")
                future_cam_high = (
                    self._process_image(future_cam_high_raw)
                    if future_cam_high_raw is not None
                    else cam_high.clone()
                )
                future_cam_left_wrist = (
                    self._process_image(future_cam_left_raw)
                    if future_cam_left_raw is not None
                    else cam_left_wrist.clone()
                )
                future_cam_right_wrist = (
                    self._process_image(future_cam_right_raw)
                    if future_cam_right_raw is not None
                    else cam_right_wrist.clone()
                )
            else:
                future_state_normalized = state_normalized.clone()
                future_cam_high = cam_high.clone()
                future_cam_left_wrist = cam_left_wrist.clone()
                future_cam_right_wrist = cam_right_wrist.clone()

            value = self._compute_value(frame_index, episode_length)
            value_normalized = value * 2 - 1
            task_id = self._subdataset_to_task_id[sub_ds_idx] if self._subdataset_to_task_id else 0

            return {
                "cam_high": cam_high,
                "cam_left_wrist": cam_left_wrist,
                "cam_right_wrist": cam_right_wrist,
                "state": state_normalized,
                "future_state": future_state_normalized,
                "future_cam_high": future_cam_high,
                "future_cam_left_wrist": future_cam_left_wrist,
                "future_cam_right_wrist": future_cam_right_wrist,
                "value": torch.tensor(value, dtype=torch.float32),
                "value_normalized": torch.tensor(value_normalized, dtype=torch.float32),
                "frame_idx": frame_index,
                "episode_length": episode_length,
                "future_frame_idx": future_frame_idx,
                "task_id": task_id,
            }
        except Exception as exc:
            logger.warning("Error loading sample idx=%s: %s", idx, exc)
            dummy_img = torch.zeros(3, self.video_height, self.video_width)
            return {
                "cam_high": dummy_img.clone(),
                "cam_left_wrist": dummy_img.clone(),
                "cam_right_wrist": dummy_img.clone(),
                "state": torch.zeros(self.state_dim),
                "future_state": torch.zeros(self.state_dim),
                "future_cam_high": dummy_img.clone(),
                "future_cam_left_wrist": dummy_img.clone(),
                "future_cam_right_wrist": dummy_img.clone(),
                "value": torch.tensor(0.5),
                "value_normalized": torch.tensor(0.0),
                "frame_idx": 0,
                "episode_length": 1,
                "future_frame_idx": 0,
                "task_id": 0,
            }


def viva_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "cam_high": torch.stack([item["cam_high"] for item in batch]),
        "cam_left_wrist": torch.stack([item["cam_left_wrist"] for item in batch]),
        "cam_right_wrist": torch.stack([item["cam_right_wrist"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "future_state": torch.stack([item["future_state"] for item in batch]),
        "future_cam_high": torch.stack([item["future_cam_high"] for item in batch]),
        "future_cam_left_wrist": torch.stack([item["future_cam_left_wrist"] for item in batch]),
        "future_cam_right_wrist": torch.stack([item["future_cam_right_wrist"] for item in batch]),
        "value": torch.stack([item["value"] for item in batch]),
        "value_normalized": torch.stack([item["value_normalized"] for item in batch]),
        "frame_idx": torch.tensor([item["frame_idx"] for item in batch]),
        "episode_length": torch.tensor([item["episode_length"] for item in batch]),
        "future_frame_idx": torch.tensor([item["future_frame_idx"] for item in batch]),
        "task_id": torch.tensor([item["task_id"] for item in batch], dtype=torch.long),
    }
