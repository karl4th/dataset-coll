from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PreparedAudio:
    samples: object
    source_num_samples: int
    source_sample_rate: int
    audio_seconds: float
    model_num_samples: int


class MimiEncoder:
    def __init__(
        self,
        *,
        hf_repo: str,
        revision: str,
        device: str,
        semantic_codebook_index: int,
    ) -> None:
        import torch
        from moshi.models import loaders

        self.torch = torch
        self.device = torch.device(device)
        checkpoint = loaders.CheckpointInfo.from_hf_repo(hf_repo, revision=revision)
        self.weights_path = Path(checkpoint.mimi_weights)
        self.model = checkpoint.get_mimi(device=self.device)
        self.model.eval()
        self.sample_rate = int(self.model.sample_rate)
        self.frame_rate = float(self.model.frame_rate)
        self.frame_size = int(round(self.sample_rate / self.frame_rate))
        self.semantic_codebook_index = semantic_codebook_index

    def prepare(self, array: np.ndarray, sample_rate: int) -> PreparedAudio:
        import torchaudio.functional as AF

        torch = self.torch
        waveform = torch.as_tensor(np.asarray(array), dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            # Hugging Face audio is normally [channels, samples].
            if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
                waveform = waveform.transpose(0, 1)
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"unsupported audio rank: {waveform.ndim}")
        source_num_samples = int(waveform.shape[-1])
        if sample_rate != self.sample_rate:
            waveform = AF.resample(waveform, sample_rate, self.sample_rate)
        return PreparedAudio(
            samples=waveform.contiguous(),
            source_num_samples=source_num_samples,
            source_sample_rate=sample_rate,
            audio_seconds=source_num_samples / sample_rate,
            model_num_samples=int(waveform.shape[-1]),
        )

    def encode_batch(self, prepared: list[PreparedAudio]) -> list[list[int]]:
        torch = self.torch
        if not prepared:
            return []
        maximum = max(item.model_num_samples for item in prepared)
        batch = torch.zeros((len(prepared), 1, maximum), dtype=torch.float32)
        for index, item in enumerate(prepared):
            batch[index, :, : item.model_num_samples] = item.samples
        batch = batch.to(self.device, non_blocking=True)
        with torch.inference_mode():
            codes = self.model.encode(batch)
        if codes.ndim != 3:
            raise ValueError(f"unexpected Mimi output shape: {tuple(codes.shape)}")
        if self.semantic_codebook_index >= codes.shape[1]:
            raise ValueError("semantic codebook index is outside Mimi output")
        result: list[list[int]] = []
        semantic = codes[:, self.semantic_codebook_index].to("cpu")
        for index, item in enumerate(prepared):
            valid_frames = math.ceil(item.model_num_samples / self.frame_size)
            result.append(semantic[index, :valid_frames].to(torch.int32).tolist())
        return result

    def encode_one(self, prepared: PreparedAudio) -> list[int]:
        return self.encode_batch([prepared])[0]
