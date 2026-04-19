"""Frontend components: Mel extraction."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence

class VietnameseTokenizer:
    """
    Tokenizer tĩnh dựa trên bảng chữ cái tiếng Việt.
    -
    Static tokenizer based on the Vietnamese alphabet.
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        
        # Bảng chữ cái tiếng Việt đầy đủ (bao gồm cả dấu thanh) + các dấu câu cơ bản
        raw_chars = " !\"#$%&'(),-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz«»ÀÁÂÉÊÍÔÚÜÝàáâãäçèéêëìíîòóôõöøùúüýĂăćĐđęĩıłŌōœšũƠơƯưșạẢảẤấầẨẩẫẬậẮắẰằẳẵặẹẻẽẾếỀềểễệỉỊịỌọỏỐốỒồỔổỗộớỜờỞởỡỢợụỦủỨứỪừửữỰựỳỵỷỹ–…"
        # Chuẩn hóa: Chuyển về chữ thường -> Lọc các ký tự duy nhất -> Sắp xếp để đảm bảo ID nhất quán
        cleaned_chars = "".join(sorted(list(set(raw_chars.lower()))))

        self.vocab = [self.pad_token, self.unk_token] + list(cleaned_chars)
        self.symbol_to_id = {s: i for i, s in enumerate(self.vocab)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_batch(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Chuyển đổi text thành list các Tensor (bắt buộc cho pad_sequence)
        encoded_tensors = [
            torch.tensor([self.symbol_to_id.get(char, 1) for char in text.lower()], dtype=torch.long) 
            for text in texts
        ]
        
        # 2. Tính lengths trước khi padding
        lengths = torch.tensor([len(x) for x in encoded_tensors], dtype=torch.long)
        
        # 3. Sử dụng pad_sequence
        # batch_first=True để output có shape [Batch, Max_Len] giống code cũ của bạn
        ids = pad_sequence(
            encoded_tensors, 
            batch_first=True, 
            padding_value=self.pad_id # Giả định bạn đã định nghĩa self.pad_id = 0
        )
        
        return ids, lengths
    

class MelSpectrogramFrontend(nn.Module):
    """
    Frontend xử lý Batch Waveform chưa padding, thực hiện Padding và tính Log-Mel.
    Phù hợp cho các kiến trúc DiT/Flow Matching và Vocoder BigVGAN v2.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        fmin: float = 0.0,
        fmax: float = 12000.0,
        power: float = 1.0,
        mel_scale: str = "slaney",
        norm: str = "slaney",
        log_clip_min: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.log_clip_min = log_clip_min
        self.hop_length = hop_length
        
        # Cấu hình chuẩn xác cho BigVGAN v2
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=power,
            center=True,
            pad_mode="reflect", # Bắt buộc reflect để tránh nhiễu biên
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(
        self, 
        wav: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wav: List các Tensor 1D [T1, T2, T3...] (Waveform thô chưa pad).
        
        Returns:
            log_mel: [B, T_mel, n_mels] (Tensor đã được pad và mask).
            mel_lens: [B] (Tensor chứa độ dài thật của từng Spectrogram).
        """
        
        # 1. TỰ ĐỘNG PADDING WAVEFORM
        if isinstance(wav, list):
            # Lấy độ dài thật từng câu trước khi pad
            wav_lens = torch.tensor([w.shape[-1] for w in wav], device=wav[0].device)
            # Pad về độ dài lớn nhất trong batch [B, T_max]
            wav_padded = pad_sequence(wav, batch_first=True, padding_value=0.0)
        else:
            # Nếu truyền vào tensor [B, T], coi như đã pad hoặc là 1 câu duy nhất
            wav_padded = wav
            wav_lens = torch.full((wav.size(0),), wav.size(1), device=wav.device, dtype=torch.long)

        # 2. BIẾN ĐỔI MEL SPECTROGRAM [B, n_mels, T_mel]
        mel = self.mel_transform(wav_padded)
        
        # 3. TÍNH LOG VÀ TRANSPOSE [B, T_mel, n_mels]
        log_mel = torch.log(torch.clamp(mel, min=self.log_clip_min))
        log_mel = log_mel.transpose(1, 2).contiguous()

        # 4. QUY ĐỔI ĐỘ DÀI (Samples -> Mel Frames)
        # Công thức cho center=True: L_mel = (L_wav // hop_length) + 1
        mel_lens = torch.div(wav_lens, self.hop_length, rounding_mode='floor') + 1

        # 5. DỌN RÁC (MASKING) VÙNG PADDING
        max_mel_len = log_mel.size(1)
        # Tạo mask Boolean [B, T_mel]
        mask = torch.arange(max_mel_len, device=log_mel.device)[None, :] < mel_lens[:, None]
        mask = mask.unsqueeze(-1) # [B, T_mel, 1]

        # Ép vùng pad về giá trị im lặng tuyệt đối (log(1e-5))
        silence_value = torch.log(torch.tensor(self.log_clip_min, device=log_mel.device))
        log_mel = log_mel.masked_fill(~mask, silence_value)

        return log_mel, mel_lens