import os
import glob
import re

import librosa
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle
from sea_g2p import SEAPipeline, Normalizer, G2P
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F

class PhonemeProcessor:
    def __init__(self):
        print("🔊 Đang khởi tạo bộ chuyển đổi ngôn ngữ sea_g2p cho Tiếng Việt...")
        self.pipeline = SEAPipeline(lang="vi") # <-- Bỏ comment dòng này khi chạy thật
        self.normalizer = Normalizer(lang="vi")

    def process(self, text: str):
        """"
        Dùng pipeline của sea_g2p để chuyển văn bản thô sang phonemes.
        """
        if not text or not isinstance(text, str): 
            return ""
        try:
            phonemes = self.pipeline.run(text) 
            
            if isinstance(phonemes, list):
                return " ".join([str(p) for p in phonemes])
            return str(phonemes)
        except:
            raise ValueError(f"Lỗi khi chuyển đổi sang phonemes: {text}")

    def normalize(self, text: str):
        """
        Dùng lõi Normalizer của sea_g2p để chuẩn hóa văn bản.
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Dùng trực tiếp hàm normalize của sea_g2p
            # Nó sẽ xử lý cả ngày tháng, con số, tiền tệ, v.v. rất thông minh
            norm_text = self.normalizer.normalize(text)
            
            # (Tùy chọn) sea_g2p đôi khi bọc từ tiếng Anh trong thẻ <en>...</en>
            # Để tính WER sạch nhất, ta có thể dùng regex nhẹ để xóa các thẻ này đi
            norm_text = re.sub(r'<[^>]+>', '', norm_text)
            
            return norm_text.strip()
            
        except Exception as e:
            print(f"Lỗi Normalizer: {str(e)}")
            return text.lower().strip()

class VietnamesePhonemeTokenizer:
    def __init__(self, vocab_path=None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        special_tokens = [self.pad_token, self.unk_token]
        
        # Thực hiện đọc vocab từ file nếu có, nếu không thì tạo vocab giả để test
        if vocab_path and os.path.exists(vocab_path):
            print("Đang load tập vocab...")
            with open(vocab_path, "r", encoding="utf-8") as f:
                symbols = [line.strip("\n") for line in f.readlines()]
            self.vocab = special_tokens + symbols
            print(f"Đã load vocab với vocab_size là {len(self.vocab)}")
        else:
            self.vocab = special_tokens + list("abcdefghijklmnopqrstuvwxyz")

        self.symbol_to_id = {s: i for i, s in enumerate(self.vocab)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.vocab)}

    @property
    def pad_id(self): 
        return 0

    @property
    def unk_id(self): 
        return 1

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, phoneme_seq):
        """
        Nhận vào chuỗi phonemes và trả về Tensor IDs.
        Giữ nguyên case-sensitive của ký hiệu.
        """
        # Nếu phoneme_seq của bạn là list các ký hiệu ['v', 'i', 'e', 't'], 
        # hãy duyệt qua list. Nếu là string, duyệt qua từng char.
        ids = [self.symbol_to_id.get(s, self.unk_id) for s in phoneme_seq]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        return [self.id_to_symbol.get(i, self.unk_token) for i in ids if i != self.pad_id]

    def encode_batch(self, phoneme_seqs):
        return [self.encode(seq) for seq in phoneme_seqs]

class SpeechProcessor:
    def __init__(self, mel_cfg, device="cuda"):
        self.device = device
        self.target_sr = mel_cfg['sample_rate']
        self.hop_size = mel_cfg['hop_length']
        self.win_size = mel_cfg['win_length']
        self.n_fft = mel_cfg['n_fft']
        self.n_mels = mel_cfg['n_mels']
        
        mel_basis = librosa_mel_fn(
            sr=self.target_sr, n_fft=self.n_fft, n_mels=self.n_mels, 
            fmin=mel_cfg.get('fmin', 0), fmax=mel_cfg.get('fmax', 12000)
        )
        self.mel_basis = torch.from_numpy(mel_basis).float().to(device)
        self.window = torch.hann_window(self.win_size).to(device)


    def compute_mel(self, wav):
        x = wav.unsqueeze(0) 
        pad_size = (self.n_fft - self.hop_size) // 2
        x = F.pad(x.unsqueeze(1), (pad_size, pad_size), mode='reflect').squeeze(1)
        spec = torch.stft(
            x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, 
            window=self.window, center=False, pad_mode='reflect', 
            normalized=False, onesided=True, return_complex=True
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(self.mel_basis, spec)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.squeeze(0).transpose(0, 1)

def prepare_viflow_cache(config):
    """
    HÀM NÀY CHẠY Ở MAIN (TRƯỚC KHI SPAWN)
    Quét toàn bộ file H5 và tạo file cache Pickle duy nhất.
    """
    log_dir = config['train'].get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    cache_path = os.path.join(log_dir, "metadata_cache.pkl")
    
    # Nếu đã có cache rồi thì bỏ qua không quét lại
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"Cache đã tồn tại tại: {cache_path}. Bỏ qua bước quét H5.")
        return cache_path

    dataset_dirs = config['data']['dataset_dirs']
    val_csv_path = config['data'].get('val_sample_path')
    
    train_samples = []
    val_samples = []

    print(f"Khởi tạo Metadata... Đang quét file H5...")
    
    # Logic lấy ID validation
    val_id_set = set()
    if val_csv_path and os.path.exists(val_csv_path):
        df = pd.read_csv(val_csv_path)
        for col in ['_id', 'id', 'sample_id']:
            if col in df.columns:
                val_id_set = set(df[col].astype(str).tolist())
                break

    all_h5 = []
    for d in dataset_dirs:
        all_h5.extend(glob.glob(os.path.join(d, "**/*.h5"), recursive=True))

    for h5_path in tqdm(all_h5, desc="Scanning H5"):
        with h5py.File(h5_path, 'r') as f:
            for sid in f.keys():
                meta = {
                    'path': h5_path,
                    'id': sid,
                    'n_frames': f[sid].attrs.get('n_frames', 0)
                }
                if sid in val_id_set:
                    val_samples.append(meta)
                else:
                    train_samples.append(meta)
    
    with open(cache_path, 'wb') as f:
        pickle.dump((train_samples, val_samples), f)
        
    print(f"Đã tạo xong cache: {cache_path} (Train: {len(train_samples)}, Val: {len(val_samples)})")
    return cache_path


def load_viflow_metadata(config):
    """
    HÀM NÀY CHẠY Ở WORKER (SAU KHI SPAWN)
    Chỉ đơn giản là đọc file Pickle đã có sẵn.
    """
    cache_path = os.path.join(config['train'].get('log_dir', 'logs'), "metadata_cache.pkl")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Không tìm thấy file cache tại {cache_path}. Hãy chạy prepare_viflow_cache trước.")
        
    with open(cache_path, 'rb') as f:
        train_samples, val_samples = pickle.load(f)
        
    return train_samples, val_samples

class ViFlowH5Dataset(Dataset):
    def __init__(self, samples):
        # Tìm độ dài tối đa của chuỗi để cố định kích thước tensor
        max_path_len = max(len(s['path']) for s in samples)
        max_id_len = max(len(s['id']) for s in samples)
        
        # Tensors tự động được chia sẻ qua shared memory giữa các workers
        self.path_tensors = torch.ByteTensor([
            list(s['path'].encode('ascii').ljust(max_path_len)) for s in samples
        ])
        self.id_tensors = torch.ByteTensor([
            list(s['id'].encode('ascii').ljust(max_id_len)) for s in samples
        ])
        self.n_frames = torch.IntTensor([s['n_frames'] for s in samples])

    def __len__(self):
        return len(self.path_tensors)

    def get_n_frames(self, idx):
        return self.n_frames[idx]

    def __getitem__(self, idx):
        # Decode lại thành string
        h5_path = bytes(self.path_tensors[idx].tolist()).decode('ascii').strip()
        sample_id = bytes(self.id_tensors[idx].tolist()).decode('ascii').strip()

        try:
            # - rdcc_nbytes=0: Tắt hoàn toàn Chunk Cache để RAM không tăng dần theo thời gian.
            # - swmr=True: Cho phép đọc file an toàn trong môi trường đa tiến trình.
            with h5py.File(h5_path, 'r', 
                           libver='latest', 
                           swmr=True, 
                           rdcc_nbytes=0, 
                           rdcc_nslots=1) as f:
                if sample_id not in f:
                    raise KeyError(f"ID {sample_id} not found in {h5_path}")
                
                group = f[sample_id]
                
                # Đọc dữ liệu vào memory (numpy)
                mel = group['mel'][()]
                phonemes = group.attrs.get('phonemes', "")
                speaker = group.attrs.get('speaker', "")
                text = group.attrs.get('text', "")
                mel_len = self.n_frames[idx]

            # 4. CHUYỂN ĐỔI SANG TENSOR
            return {
                "mel": torch.from_numpy(mel).float(),
                "phonemes": str(phonemes), # Ép kiểu string để tránh giữ tham chiếu H5
                "text": str(text),
                "speaker": str(speaker),
                "mel_len": int(mel_len),
                "id": str(sample_id)
            }

        except Exception as e:
            # IN THÔNG TIN CHI TIẾT TRƯỚC KHI CHẾT
            print(f"\n" + "!"*30)
            print(f"LỖI DỮ LIỆU TẠI INDEX: {idx}")
            print(f"   * File: {h5_path}")
            print(f"   * Sample ID: {sample_id}")
            print(f"   * Thông báo lỗi: {str(e)}")
            print("!"*30 + "\n")
            
            # Đẩy lỗi lên để dừng toàn bộ quá trình training
            raise e


class ViFlowCollate:
    def __init__(self, tokenizer, mel_pad_value=0.0):
        self.tokenizer = tokenizer
        self.mel_pad_value = mel_pad_value

    def __call__(self, batch):
        phoneme_tokens = [self.tokenizer.encode(item['phonemes']) for item in batch]
        phn_lens = torch.tensor([len(t) for t in phoneme_tokens], dtype=torch.long)
        
        mels = [item['mel'] for item in batch]
        mel_lens = torch.tensor([item['mel_len'] for item in batch], dtype=torch.long)
        
        # Padding
        phn_padded = pad_sequence(phoneme_tokens, batch_first=True, padding_value=self.tokenizer.pad_id)
        mels_padded = pad_sequence(mels, batch_first=True, padding_value=self.mel_pad_value)
        
        # mel_mask: 1 (True) là DỮ LIỆU THỰC, 0 (False) là PADDING
        batch_size, max_mel_len, _ = mels_padded.shape
        indices = torch.arange(max_mel_len).unsqueeze(0).to(mel_lens.device)
        mel_mask = indices < mel_lens.unsqueeze(1) # Thay đổi từ >= sang <

        return {
            "mels": mels_padded.contiguous(),
            "mel_lens": mel_lens,
            "mel_mask": mel_mask, 
            "phonemes": phn_padded.contiguous(),
            "phn_lens": phn_lens,
            "ids": [item['id'] for item in batch]
        }
    
class ViFlowProcessor:
    def __init__(self, mel_cfg, vocab_path=None, device="cuda", trim_db=30):
        self.speech_processor = SpeechProcessor(mel_cfg, device=device)
        self.phoneme_processor = PhonemeProcessor()
        self.phoneme_tokenizer = VietnamesePhonemeTokenizer(vocab_path)
        self.trim_db = trim_db
        self.device = device

    def vocab_size(self):
        return self.phoneme_tokenizer.vocab_size

    def process_text(self, text_ref, text_gen):
        ref_phonemes = self.phoneme_processor.process(text_ref).strip()
        gen_phonemes = self.phoneme_processor.process(text_gen).strip()
        
        # Sửa lại độ dài cho chuẩn số token thực tế
        ref_len = max(1, len(ref_phonemes.split()))
        gen_len = max(1, len(gen_phonemes.split()))
        
        combined_phonemes = f"{ref_phonemes} {gen_phonemes}"
        return combined_phonemes, (ref_len, gen_len)

    def process_speech(self, wav):
        # Ép về mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze()
        
        # Đã vá lỗi hàm trim_silence bằng thư viện chuẩn librosa
        if self.trim_db > 0:
            wav_np = wav.cpu().numpy()
            wav_clean_np, _ = librosa.effects.trim(wav_np, top_db=self.trim_db)
            wav = torch.from_numpy(wav_clean_np).to(self.device)
        else:
            wav = wav.to(self.device)

        mel = self.speech_processor.compute_mel(wav)
        return mel
    
    def prepare_input(self, wav, text_ref, text_gen, speed=1.0):
        # Chuyển đổi wav sang mel spectrogram
        mel = self.process_speech(wav) 
        n_prompt, mel_bins = mel.shape
        mel = mel.unsqueeze(0)

        # Chuyển đổi văn bản sang phonemes và tính toán độ dài token
        combined_phonemes, (ref_token_len, gen_token_len) = self.process_text(text_ref, text_gen)

        # Tính toán số frame cần sinh ra dựa trên tốc độ và độ dài token
        generate_frames = int((n_prompt / ref_token_len) * (1 / speed) * gen_token_len)
        total_frames = n_prompt + generate_frames
        
        # Tạo các tensor đầu vào cho mô hình
        x0 = torch.randn((1, total_frames, mel_bins), device=self.device)
        
        cond = torch.zeros((1, total_frames, mel_bins), device=self.device)
        cond[0, :n_prompt, :] = mel[0]
        
        target_mask = torch.zeros((1, total_frames), device=self.device)
        target_mask[0, n_prompt:] = 1.0
        
        text_ids_raw = self.phoneme_tokenizer.encode(combined_phonemes)
        
        if isinstance(text_ids_raw, torch.Tensor):
            text_ids = text_ids_raw.clone().detach().to(dtype=torch.long, device=self.device).view(1, -1)
        else:
            text_ids = torch.tensor(text_ids_raw, dtype=torch.long, device=self.device).view(1, -1)
            
        current_seq_len = text_ids.size(1)
        if current_seq_len < total_frames:
            text_ids = F.pad(text_ids, (0, total_frames - current_seq_len), value=self.phoneme_tokenizer.pad_id)
        elif current_seq_len > total_frames:
            text_ids = text_ids[:, :total_frames]

        return {
            "x0": x0,
            "cond": cond,
            "target_mask": target_mask,
            "text_ids": text_ids,
            "n_prompt": n_prompt,
        }
    
    