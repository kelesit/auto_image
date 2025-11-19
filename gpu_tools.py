"""
GPU加速版本的向量计算工具
使用CuPy和PyTorch进行大规模向量相似度计算
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

# 尝试导入GPU加速库
try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cp_distance
    HAS_CUPY = True
    print("✅ CuPy可用，将使用GPU加速")
except ImportError:
    HAS_CUPY = False
    print("⚠️ CuPy不可用，尝试PyTorch...")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        print(f"✅ PyTorch GPU可用，设备: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ PyTorch GPU不可用")
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch不可用")

# Fallback到CPU版本
if not (HAS_CUPY or HAS_TORCH):
    print("❌ 无GPU加速库，将使用CPU版本")


class GPUVectorCalculator:
    """GPU加速的向量计算器"""
    
    def __init__(self, backend='auto', device_id=0):
        """
        初始化GPU计算器
        
        Args:
            backend: 'cupy', 'torch', 'auto'
            device_id: GPU设备ID
        """
        self.backend = backend
        self.device_id = device_id
        
        if backend == 'auto':
            if HAS_CUPY:
                self.backend = 'cupy'
                cp.cuda.Device(device_id).use()
            elif HAS_TORCH:
                self.backend = 'torch'
                self.device = torch.device(f'cuda:{device_id}')
            else:
                self.backend = 'cpu'
                warnings.warn("无GPU加速库，将使用CPU计算")
        
        print(f"使用后端: {self.backend}")
    
    def batch_cosine_similarity(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """
        批量计算余弦相似度
        
        Args:
            vectors_a: 形状 (N, D) 的向量矩阵
            vectors_b: 形状 (M, D) 的向量矩阵
            
        Returns:
            形状 (N, M) 的相似度矩阵
        """
        if self.backend == 'cupy':
            return self._cupy_cosine_similarity(vectors_a, vectors_b)
        elif self.backend == 'torch':
            return self._torch_cosine_similarity(vectors_a, vectors_b)
        else:
            return self._cpu_cosine_similarity(vectors_a, vectors_b)
    
    def _cupy_cosine_similarity(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """CuPy版本的余弦相似度计算"""
        # 转换到GPU
        a_gpu = cp.asarray(vectors_a, dtype=cp.float32)
        b_gpu = cp.asarray(vectors_b, dtype=cp.float32)
        
        # 归一化向量
        a_norm = a_gpu / cp.linalg.norm(a_gpu, axis=1, keepdims=True)
        b_norm = b_gpu / cp.linalg.norm(b_gpu, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarity = cp.dot(a_norm, b_norm.T)
        
        # 转回CPU
        return cp.asnumpy(similarity)
    
    def _torch_cosine_similarity(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """PyTorch版本的余弦相似度计算"""
        # 转换到GPU
        a_tensor = torch.from_numpy(vectors_a).float().to(self.device)
        b_tensor = torch.from_numpy(vectors_b).float().to(self.device)
        
        # 归一化
        a_norm = F.normalize(a_tensor, p=2, dim=1)
        b_norm = F.normalize(b_tensor, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.mm(a_norm, b_norm.t())
        
        # 转回CPU
        return similarity.cpu().numpy()
    
    def _cpu_cosine_similarity(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """CPU版本的余弦相似度计算"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(vectors_a, vectors_b)
    
    def find_similar_pairs_batch(self, vectors: np.ndarray, threshold: float = 0.95, 
                                batch_size: int = 1000) -> List[Tuple[int, int, float]]:
        """
        批量找出相似度超过阈值的向量对
        
        Args:
            vectors: 向量矩阵 (N, D)
            threshold: 相似度阈值
            batch_size: 批处理大小
            
        Returns:
            相似向量对列表 [(i, j, similarity), ...]
        """
        n_vectors = len(vectors)
        similar_pairs = []
        
        print(f"开始批量相似度检测，总向量数: {n_vectors}")
        
        # 分批处理避免内存溢出
        for i in range(0, n_vectors, batch_size):
            end_i = min(i + batch_size, n_vectors)
            batch_a = vectors[i:end_i]
            
            for j in range(i, n_vectors, batch_size):
                end_j = min(j + batch_size, n_vectors)
                batch_b = vectors[j:end_j]
                
                # 计算批次相似度
                similarities = self.batch_cosine_similarity(batch_a, batch_b)
                
                # 找出超过阈值的对
                if i == j:  # 同一批次，避免重复和自相似
                    similarities = np.triu(similarities, k=1)  # 只保留上三角
                
                high_sim_indices = np.where(similarities > threshold)
                
                for local_i, local_j in zip(high_sim_indices[0], high_sim_indices[1]):
                    global_i = i + local_i
                    global_j = j + local_j
                    sim_value = similarities[local_i, local_j]
                    similar_pairs.append((global_i, global_j, sim_value))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"已处理 {end_i}/{n_vectors} 个向量")
        
        print(f"找到 {len(similar_pairs)} 个相似对")
        return similar_pairs
    
    def farthest_first_sampling_gpu(self, vectors: np.ndarray, n_samples: int) -> List[int]:
        """
        GPU加速的最远优先采样
        
        Args:
            vectors: 向量矩阵 (N, D)
            n_samples: 采样数量
            
        Returns:
            选中的向量索引列表
        """
        if len(vectors) <= n_samples:
            return list(range(len(vectors)))
        
        print(f"开始GPU加速的最远优先采样: {len(vectors)} -> {n_samples}")
        
        if self.backend == 'cupy':
            return self._cupy_farthest_first_sampling(vectors, n_samples)
        elif self.backend == 'torch':
            return self._torch_farthest_first_sampling(vectors, n_samples)
        else:
            return self._cpu_farthest_first_sampling(vectors, n_samples)
    
    def _cupy_farthest_first_sampling(self, vectors: np.ndarray, n_samples: int) -> List[int]:
        """CuPy版本的最远优先采样"""
        vectors_gpu = cp.asarray(vectors, dtype=cp.float32)
        n_vectors = len(vectors_gpu)
        
        # 归一化向量
        vectors_norm = vectors_gpu / cp.linalg.norm(vectors_gpu, axis=1, keepdims=True)
        
        selected_indices = []
        
        # 随机选择第一个点
        cp.random.seed(42)
        first_idx = cp.random.randint(0, n_vectors)
        selected_indices.append(int(first_idx))
        
        for i in range(n_samples - 1):
            if (i + 1) % 100 == 0:
                print(f"采样进度: {i + 1}/{n_samples}")
            
            # 计算所有点到已选点的最小距离
            selected_vectors = vectors_norm[selected_indices]
            
            # 计算相似度矩阵
            similarities = cp.dot(vectors_norm, selected_vectors.T)
            
            # 找出每个点到已选点的最大相似度（最小距离）
            max_similarities = cp.max(similarities, axis=1)
            
            # 将已选点的相似度设为1（避免重复选择）
            max_similarities[selected_indices] = 1.0
            
            # 选择最小相似度（最大距离）的点
            farthest_idx = cp.argmin(max_similarities)
            selected_indices.append(int(farthest_idx))
        
        return selected_indices
    
    def _torch_farthest_first_sampling(self, vectors: np.ndarray, n_samples: int) -> List[int]:
        """PyTorch版本的最远优先采样"""
        vectors_tensor = torch.from_numpy(vectors).float().to(self.device)
        n_vectors = len(vectors_tensor)
        
        # 归一化向量
        vectors_norm = F.normalize(vectors_tensor, p=2, dim=1)
        
        selected_indices = []
        
        # 随机选择第一个点
        torch.manual_seed(42)
        first_idx = torch.randint(0, n_vectors, (1,)).item()
        selected_indices.append(first_idx)
        
        for i in range(n_samples - 1):
            if (i + 1) % 100 == 0:
                print(f"采样进度: {i + 1}/{n_samples}")
            
            # 计算所有点到已选点的最小距离
            selected_vectors = vectors_norm[selected_indices]
            
            # 计算相似度矩阵
            similarities = torch.mm(vectors_norm, selected_vectors.t())
            
            # 找出每个点到已选点的最大相似度
            max_similarities, _ = torch.max(similarities, dim=1)
            
            # 将已选点的相似度设为1
            max_similarities[selected_indices] = 1.0
            
            # 选择最小相似度的点
            farthest_idx = torch.argmin(max_similarities).item()
            selected_indices.append(farthest_idx)
        
        return selected_indices
    
    def _cpu_farthest_first_sampling(self, vectors: np.ndarray, n_samples: int) -> List[int]:
        """CPU版本的最远优先采样（备选方案）"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        selected_indices = []
        n_vectors = len(vectors)
        
        # 随机选择第一个点
        np.random.seed(42)
        first_idx = np.random.randint(0, n_vectors)
        selected_indices.append(first_idx)
        
        for i in range(n_samples - 1):
            if (i + 1) % 100 == 0:
                print(f"采样进度: {i + 1}/{n_samples}")
            
            max_min_distance = -1
            farthest_idx = -1
            
            selected_vectors = vectors[selected_indices]
            
            for j in range(n_vectors):
                if j in selected_indices:
                    continue
                
                # 计算到所有已选点的相似度
                similarities = cosine_similarity([vectors[j]], selected_vectors)[0]
                min_distance = 1 - np.max(similarities)  # 最小距离
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    farthest_idx = j
            
            if farthest_idx != -1:
                selected_indices.append(farthest_idx)
        
        return selected_indices


def gpu_accelerated_cleaning(vectors_dict: Dict[str, np.ndarray], 
                           category_df: pd.DataFrame, 
                           threshold: float = 0.95) -> pd.DataFrame:
    """
    GPU加速的相似产品清洗
    
    Args:
        vectors_dict: SPU向量字典
        category_df: 类别DataFrame
        threshold: 相似度阈值
        
    Returns:
        需要清洗掉的SPU DataFrame
    """
    print("开始GPU加速的相似产品清洗...")
    
    # 准备向量矩阵
    spu_list = category_df['SPU'].tolist()
    vectors = []
    valid_indices = []
    
    for i, spu in enumerate(spu_list):
        if str(spu) in vectors_dict:
            vectors.append(vectors_dict[str(spu)])
            valid_indices.append(i)
    
    if len(vectors) == 0:
        return pd.DataFrame()
    
    vectors = np.array(vectors)
    print(f"准备清洗 {len(vectors)} 个向量")
    
    # 初始化GPU计算器
    gpu_calc = GPUVectorCalculator()
    
    # 找出相似对
    similar_pairs = gpu_calc.find_similar_pairs_batch(vectors, threshold)
    
    # 统计每个向量被标记为相似的次数
    similarity_counts = {}
    for i, j, sim in similar_pairs:
        similarity_counts[i] = similarity_counts.get(i, 0) + 1
        similarity_counts[j] = similarity_counts.get(j, 0) + 1
    
    # 选择要清洗的向量（相似度高的优先清洗）
    to_remove = set()
    for i, j, sim in sorted(similar_pairs, key=lambda x: x[2], reverse=True):
        if i not in to_remove and j not in to_remove:
            # 优先清洗相似度更高的那个
            if similarity_counts.get(i, 0) >= similarity_counts.get(j, 0):
                to_remove.add(i)
            else:
                to_remove.add(j)
    
    # 转换回SPU索引
    removed_spu_indices = [valid_indices[i] for i in to_remove]
    # 计算每个被清洗向量的最大相似度及对应SPU
    max_similarities = []
    most_similar_spus = []
    for i in to_remove:
        # 找到所有与i相关的similar_pairs
        related = [(a, b, sim) for a, b, sim in similar_pairs if a == i or b == i]
        if related:
            # 找到最大相似度的那一对
            max_pair = max(related, key=lambda x: x[2])
            max_sim = max_pair[2]
            # 另一个spu的索引
            other_idx = max_pair[1] if max_pair[0] == i else max_pair[0]
            most_similar_spu = spu_list[other_idx]
        else:
            max_sim = None
            most_similar_spu = None
        max_similarities.append(max_sim)
        most_similar_spus.append(most_similar_spu)
    removed_df = category_df.iloc[removed_spu_indices].copy()
    removed_df['max_similarity'] = max_similarities
    removed_df['most_similar_spu'] = most_similar_spus
    print(f"标记清洗 {len(removed_df)} 个相似SPU")
    return removed_df


def gpu_accelerated_sampling(vectors_dict: Dict[str, np.ndarray], 
                           category_df: pd.DataFrame, 
                           n_samples: int) -> pd.DataFrame:
    """
    GPU加速的多样化采样
    
    Args:
        vectors_dict: SPU向量字典
        category_df: 类别DataFrame  
        n_samples: 采样数量
        
    Returns:
        采样后的DataFrame
    """
    if len(category_df) <= n_samples:
        return category_df
    
    print(f"开始GPU加速的多样化采样: {len(category_df)} -> {n_samples}")
    
    # 准备向量矩阵
    spu_list = category_df['SPU'].tolist()
    vectors = []
    valid_indices = []
    
    for i, spu in enumerate(spu_list):
        if str(spu) in vectors_dict:
            vectors.append(vectors_dict[str(spu)])
            valid_indices.append(i)
    
    if len(vectors) <= n_samples:
        return category_df.iloc[valid_indices]
    
    vectors = np.array(vectors)
    
    # GPU加速采样
    gpu_calc = GPUVectorCalculator()
    selected_vector_indices = gpu_calc.farthest_first_sampling_gpu(vectors, n_samples)
    
    # 转换回DataFrame索引
    selected_df_indices = [valid_indices[i] for i in selected_vector_indices]
    return category_df.iloc[selected_df_indices]


# 测试函数
def test_gpu_acceleration():
    """测试GPU加速性能"""
    import time
    
    print("GPU加速性能测试")
    print("=" * 50)
    
    # 生成测试数据
    n_vectors = 5000
    vector_dim = 768
    vectors = np.random.rand(n_vectors, vector_dim).astype(np.float32)
    
    gpu_calc = GPUVectorCalculator()
    
    # 测试相似度计算
    print(f"测试数据: {n_vectors} 个 {vector_dim} 维向量")
    
    start_time = time.time()
    similarities = gpu_calc.batch_cosine_similarity(vectors[:1000], vectors[:1000])
    gpu_time = time.time() - start_time
    
    print(f"GPU计算1000x1000相似度矩阵耗时: {gpu_time:.2f}秒")
    print(f"相似度矩阵形状: {similarities.shape}")
    
    # 测试采样
    start_time = time.time()
    selected_indices = gpu_calc.farthest_first_sampling_gpu(vectors[:2000], 100)
    sampling_time = time.time() - start_time
    
    print(f"GPU采样2000->100耗时: {sampling_time:.2f}秒")
    print(f"选中索引数: {len(selected_indices)}")


if __name__ == "__main__":
    test_gpu_acceleration()
