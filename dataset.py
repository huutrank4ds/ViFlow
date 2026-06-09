import os
import glob
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle

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
        print(f"✅ Cache đã tồn tại tại: {cache_path}. Bỏ qua bước quét H5.")
        return cache_path

    dataset_dirs = config['data']['dataset_dirs']
    val_csv_path = config['data'].get('val_sample_path')
    
    train_samples = []
    val_samples = []

    print(f"🔍 Khởi tạo Metadata... Đang quét file H5 (chỉ thực hiện một lần)...")
    
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
        
    print(f"💾 Đã tạo xong cache: {cache_path} (Train: {len(train_samples)}, Val: {len(val_samples)})")
    return cache_path


def load_viflow_metadata(config):
    """
    HÀM NÀY CHẠY Ở WORKER (SAU KHI SPAWN)
    Chỉ đơn giản là đọc file Pickle đã có sẵn.
    """
    cache_path = os.path.join(config['train'].get('log_dir', 'logs'), "metadata_cache.pkl")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file cache tại {cache_path}. Hãy chạy prepare_viflow_cache trước.")
        
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
                mel_len = self.n_frames[idx]

            # 4. CHUYỂN ĐỔI SANG TENSOR
            return {
                "mel": torch.from_numpy(mel).float(),
                "phonemes": str(phonemes), # Ép kiểu string để tránh giữ tham chiếu H5
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
        phn_padded = pad_sequence(phoneme_tokens, batch_first=True, padding_value=self.tokenizer.get_pad_id)
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