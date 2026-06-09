import random
from torch.utils.data import Sampler
import torch.distributed as dist

class UniversalBucketBatchSampler(Sampler):
    def __init__(self, dataset, max_frames, num_buckets=32, shuffle=True, seed=35, rank=None, world_size=None):
        self.dataset = dataset
        self.max_frames = max_frames
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if rank is not None and world_size is not None:
            self.rank = rank
            self.num_replicas = world_size
        elif dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        # Chuẩn bị buckets cố định
        self.buckets = self._prepare_buckets()
        
        # Biến lưu trữ danh sách batch đã được chia cho Rank này
        self.batch_list = []

    def _prepare_buckets(self):
        # Lấy metadata (chỉ chạy 1 lần khi khởi tạo)
        indices_with_len = []
        for i in range(len(self.dataset)):
            indices_with_len.append((i, self.dataset.get_n_frames(i)))
        
        indices_with_len.sort(key=lambda x: x[1])
        num_samples = len(indices_with_len)
        samples_per_bucket = num_samples // self.num_buckets
        
        buckets = []
        for i in range(self.num_buckets):
            start = i * samples_per_bucket
            end = (i + 1) * samples_per_bucket if i < self.num_buckets - 1 else num_samples
            bucket_data = indices_with_len[start:end]
            if len(bucket_data) > 0:
                buckets.append([idx for idx, _ in bucket_data])
        return buckets

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_batches(self):
        """Hàm quan trọng nhất: Xây dựng batch đồng bộ trên mọi GPU"""
        g = random.Random(self.seed + self.epoch)
        
        # Shuffle buckets (giống hệt nhau trên mọi Rank nhờ cùng Seed)
        current_buckets = [list(b) for b in self.buckets]
        if self.shuffle:
            for b in current_buckets:
                g.shuffle(b)
            g.shuffle(current_buckets)
        
        all_indices = []
        for b in current_buckets:
            all_indices.extend(b)

        # Gom toàn bộ batch cho cả dataset (Global Batching)
        global_batches = []
        current_batch = []
        max_f_in_batch = 0
        
        for idx in all_indices:
            n_frames = self.dataset.get_n_frames(idx)
            potential_max = max(max_f_in_batch, n_frames)
            
            if potential_max * (len(current_batch) + 1) <= self.max_frames:
                current_batch.append(idx)
                max_f_in_batch = potential_max
            else:
                if len(current_batch) > 0:
                    global_batches.append(current_batch)
                current_batch = [idx]
                max_f_in_batch = n_frames
        
        if len(current_batch) > 0:
            global_batches.append(current_batch)

        # Chia các batch này cho các Rank (Sharding)
        rank_batches = global_batches[self.rank::self.num_replicas]
        
        # Đảm bảo số lượng batch bằng nhau tuyệt đối
        # Lấy số lượng batch tối thiểu mà một GPU có thể nhận được
        min_batches = len(global_batches) // self.num_replicas
        
        # Trả về danh sách batch cho Rank hiện tại
        return rank_batches[:min_batches]

    def __iter__(self):
        # Mỗi khi bắt đầu Epoch, xây dựng lại danh sách batch
        self.batch_list = self._build_batches()
        for b in self.batch_list:
            yield b

    def __len__(self):
        # Tính toán chính xác số lượng batch Rank này sẽ chạy
        if not self.batch_list:
            # Ước tính tạm thời nếu chưa build
            temp_batches = self._build_batches()
            return len(temp_batches)
        return len(self.batch_list)