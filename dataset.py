"""Dataset and collate utilities for Vietnamese zero-shot TTS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class TTSSample:
    """One training sample with text plus prompt/target audio."""

    text: str
    prompt_audio_path: str
    target_audio_path: str


class ManifestTTSDataset(Dataset):
    """Dataset that reads JSONL manifests with text/prompt/target paths."""

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 22050,
        text_key: str = "text",
        prompt_audio_key: str = "prompt_audio_path",
        target_audio_key: str = "target_audio_path",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.text_key = text_key
        self.prompt_audio_key = prompt_audio_key
        self.target_audio_key = target_audio_key
        self.samples = self._load_manifest(self.manifest_path)

    def _load_manifest(self, manifest_path: Path) -> List[TTSSample]:
        samples: List[TTSSample] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                samples.append(
                    TTSSample(
                        text=str(item[self.text_key]),
                        prompt_audio_path=str(item[self.prompt_audio_key]),
                        target_audio_path=str(item[self.target_audio_key]),
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        # shape: [C, T] -> mono [T]
        waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        prompt_wav = self._load_waveform(sample.prompt_audio_path)
        target_wav = self._load_waveform(sample.target_audio_path)

        return {
            "text": sample.text,
            "prompt_wav": prompt_wav,
            "target_wav": target_wav,
        }


def tts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad waveforms and keep texts as a list.

    Returns:
        prompt_wav: [B, T_prompt_max]
        prompt_wav_lens: [B]
        target_wav: [B, T_target_max]
        target_wav_lens: [B]
    """

    texts = [item["text"] for item in batch]

    prompt_wavs = [item["prompt_wav"] for item in batch]
    target_wavs = [item["target_wav"] for item in batch]

    prompt_lens = torch.tensor([x.size(0) for x in prompt_wavs], dtype=torch.long)
    target_lens = torch.tensor([x.size(0) for x in target_wavs], dtype=torch.long)

    # shape: list[[T]] -> [B, T_max]
    prompt_padded = pad_sequence(prompt_wavs, batch_first=True, padding_value=0.0)
    target_padded = pad_sequence(target_wavs, batch_first=True, padding_value=0.0)

    return {
        "text": texts,
        "prompt_wav": prompt_padded,
        "prompt_wav_lens": prompt_lens,
        "target_wav": target_padded,
        "target_wav_lens": target_lens,
    }
