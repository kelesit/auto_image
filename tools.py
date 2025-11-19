import os
import numpy as np
import pandas as pd


def load_spu_vectors(data_dir='./data/preprocess'):
    """
    加载SPU向量数据
    返回: (metadata_df, vectors_dict)
    """
    import pickle
    
    # 方法1: 从pickle文件加载完整数据（推荐）
    pickle_path = os.path.join(data_dir, 'spu_data_complete.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data['metadata'], data['vectors']
    
    # 方法2: 分别加载元数据和向量文件
    metadata_path = os.path.join(data_dir, 'spu_metadata.csv')
    vectors_path = os.path.join(data_dir, 'spu_vectors.npz')
    
    if os.path.exists(metadata_path) and os.path.exists(vectors_path):
        metadata_df = pd.read_csv(metadata_path, dtype={'SPU': str})
        vectors_data = np.load(vectors_path)
        vectors_dict = {spu: vectors_data[spu] for spu in vectors_data.files}
        return metadata_df, vectors_dict
    
    raise FileNotFoundError("未找到向量数据文件")


def get_spu_vector(spu, vectors_dict):
    """
    获取指定SPU的向量
    """
    return vectors_dict.get(str(spu))


def calculate_vector_similarity(vector1, vector2, method='cosine'):
    """
    计算两个向量的相似度
    method: 'cosine', 'euclidean', 'dot'
    """
    if method == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif method == 'euclidean':
        return 1 / (1 + np.linalg.norm(vector1 - vector2))
    elif method == 'dot':
        return np.dot(vector1, vector2)
    else:
        raise ValueError("支持的方法: 'cosine', 'euclidean', 'dot'")


def find_similar_spus(target_spu, vectors_dict, metadata_df, top_k=10, method='cosine'):
    """
    找到与目标SPU最相似的SPU
    """
    target_vector = get_spu_vector(target_spu, vectors_dict)
    if target_vector is None:
        return None
    
    similarities = []
    for spu, vector in vectors_dict.items():
        if spu != str(target_spu):
            sim = calculate_vector_similarity(target_vector, vector, method)
            similarities.append((spu, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回top_k个结果
    result_spus = [spu for spu, sim in similarities[:top_k]]
    result_df = metadata_df[metadata_df['SPU'].isin(result_spus)].copy()
    result_df['similarity'] = [sim for spu, sim in similarities[:top_k]]
    
    return result_df.sort_values('similarity', ascending=False)



def find_similar_spus_above_threshold(target_spu, vectors_dict, category_df, threshold=0.9, method='cosine'):
    """
    找到与目标SPU相似度超过阈值的SPU（用于清洗相似产品）
    只在category_df指定的SPU范围内搜索
    返回相似度超过阈值的SPU DataFrame（不包括目标SPU本身）
    """
    target_vector = get_spu_vector(target_spu, vectors_dict)
    if target_vector is None:
        return pd.DataFrame()
    
    similar_spus = []
    
    # 只在category_df中的SPU中比较
    for _, row in category_df.iterrows():
        spu = row['SPU']
        if spu == str(target_spu):
            continue
            
        vector = get_spu_vector(spu, vectors_dict)
        if vector is not None:
            sim = calculate_vector_similarity(target_vector, vector, method)
            if sim > threshold:
                similar_spus.append({
                    'SPU': spu,
                    'similarity': sim,
                    'target_spu': target_spu
                })
    
    if similar_spus:
        # 返回相似的SPU信息，与原DataFrame结构保持一致
        similar_spu_ids = [item['SPU'] for item in similar_spus]
        result_df = category_df[category_df['SPU'].isin(similar_spu_ids)].copy()
        
        # 添加相似度信息
        similarity_dict = {item['SPU']: item['similarity'] for item in similar_spus}
        result_df['similarity_to_target'] = result_df['SPU'].map(similarity_dict)
        result_df['target_spu'] = target_spu
        
        return result_df
    
    return pd.DataFrame()


def sample_diverse_spus(category_df, vectors_dict, target_num, method='farthest_first'):
    """
    从category_df中采样出target_num个最具多样性的SPU
    
    Args:
        category_df: 包含SPU信息的DataFrame
        vectors_dict: SPU向量字典
        target_num: 目标采样数量
        method: 采样方法 ('farthest_first', 'random', 'kmeans')
    
    Returns:
        采样后的DataFrame
    """
    if len(category_df) <= target_num:
        return category_df
    
    if method == 'farthest_first':
        return _farthest_first_sampling(category_df, vectors_dict, target_num)
    elif method == 'random':
        return category_df.sample(n=target_num, random_state=42)
    elif method == 'kmeans':
        return _kmeans_sampling(category_df, vectors_dict, target_num)
    else:
        raise ValueError(f"不支持的采样方法: {method}")


def _farthest_first_sampling(category_df, vectors_dict, target_num):
    """
    最远优先采样：选择在向量空间中分布最分散的SPU
    """
    spu_list = category_df['SPU'].tolist()
    vectors = []
    valid_spus = []
    
    # 获取有效的向量
    for spu in spu_list:
        vector = get_spu_vector(spu, vectors_dict)
        if vector is not None:
            vectors.append(vector)
            valid_spus.append(spu)
    
    if len(valid_spus) <= target_num:
        return category_df[category_df['SPU'].isin(valid_spus)]
    
    vectors = np.array(vectors)
    selected_indices = []
    
    # 随机选择第一个点
    np.random.seed(42)
    first_idx = np.random.randint(0, len(vectors))
    selected_indices.append(first_idx)
    
    # 迭代选择最远的点
    for _ in range(target_num - 1):
        max_min_distance = -1
        farthest_idx = -1
        
        for i in range(len(vectors)):
            if i in selected_indices:
                continue
                
            # 计算当前点到所有已选点的最小距离
            min_distance = float('inf')
            for selected_idx in selected_indices:
                distance = 1 - calculate_vector_similarity(
                    vectors[i], vectors[selected_idx], 'cosine'
                )
                min_distance = min(min_distance, distance)
            
            # 选择最小距离最大的点
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = i
        
        if farthest_idx != -1:
            selected_indices.append(farthest_idx)
    
    # 返回选中的SPU
    selected_spus = [valid_spus[i] for i in selected_indices]
    return category_df[category_df['SPU'].isin(selected_spus)]


def _kmeans_sampling(category_df, vectors_dict, target_num):
    """
    K-means采样：使用聚类方法选择代表性的SPU
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("警告：sklearn未安装，改用最远优先采样")
        return _farthest_first_sampling(category_df, vectors_dict, target_num)
    
    spu_list = category_df['SPU'].tolist()
    vectors = []
    valid_spus = []
    
    # 获取有效的向量
    for spu in spu_list:
        vector = get_spu_vector(spu, vectors_dict)
        if vector is not None:
            vectors.append(vector)
            valid_spus.append(spu)
    
    if len(valid_spus) <= target_num:
        return category_df[category_df['SPU'].isin(valid_spus)]
    
    vectors = np.array(vectors)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=target_num, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # 从每个聚类中选择最接近中心的点
    selected_spus = []
    for cluster_id in range(target_num):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # 找到最接近聚类中心的点
        cluster_center = kmeans.cluster_centers_[cluster_id]
        min_distance = float('inf')
        closest_idx = -1
        
        for idx in cluster_indices:
            distance = np.linalg.norm(vectors[idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        
        if closest_idx != -1:
            selected_spus.append(valid_spus[closest_idx])
    
    return category_df[category_df['SPU'].isin(selected_spus)]


# 使用示例
def example_usage():
    """
    示例用法
    """
    # 加载数据
    metadata_df, vectors_dict = load_spu_vectors()

    # 找相似产品
    similar_products = find_similar_spus('12345', vectors_dict, metadata_df, top_k=5)

    # 计算相似度
    vector1 = get_spu_vector('12345', vectors_dict)
    vector2 = get_spu_vector('67890', vectors_dict)
    similarity = calculate_vector_similarity(vector1, vector2)
    print(f"SPU 12345 和 SPU 67890 的相似度: {similarity}")


if __name__ == "__main__":
    example_usage()