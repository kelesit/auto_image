"""
构造B图库
cluster mapping
{
    "90 - Coffee Tables": [
        "cluster_1": [...],  # 质心向量
        "cluster_2": [...],
        ...
    ]
    ...
}

### 方法：
1. 对每个品类下的所有图片向量进行聚类
   - 方法：K-Means
   - 聚类数目：N
        - N = min(50, max(10, int(sqrt(M/100))))
        - M: 该品类下的图片总数
   - 输出：每个品类下的聚类质心向量以及其对应的簇ID
   
"""
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from tools import load_spu_vectors, get_spu_vector, calculate_vector_similarity
import pickle

def build_cluster_mapping():
    data_dir = './data/preprocess'
    spu_csv_path = f'{data_dir}/spu_data_cleaned_gpu.csv'
    spu_df = pd.read_csv(spu_csv_path)
    categories = spu_df['产品分类'].unique().tolist()
    cluster_mapping = {}
    _, vectors_dict = load_spu_vectors(data_dir)
    for category in categories:
        category_df = spu_df[spu_df['产品分类'] == category]
        image_vectors = []
        for _, row in category_df.iterrows():
            spu = row['SPU']
            vector = get_spu_vector(spu, vectors_dict)
            if vector is not None:
                image_vectors.append(vector)
        image_vectors = np.array(image_vectors)
        M = len(image_vectors)
        if M < 10:
            continue  # 样本太少，跳过
        N = min(50, max(10, int(np.sqrt(M / 100))))
        kmeans = KMeans(n_clusters=N, random_state=42)
        kmeans.fit(image_vectors)
        cluster_centers = kmeans.cluster_centers_.tolist()
        cluster_mapping[category] = {f'cluster_{i+1}': center for i, center in enumerate(cluster_centers)}
    
    import json
    with open('./data/preprocess/cluster_mapping.json', 'w') as f:
        json.dump(cluster_mapping, f, indent=4)
    print("Cluster mapping saved to './data/preprocess/cluster_mapping.json'")

    return cluster_mapping



def load_cluster_mapping(json_path):
    import json
    with open(json_path, 'r') as f:
        cluster_mapping = json.load(f)
    return cluster_mapping


def build_es_docs():
    """
    构建最终的docs信息，包含每个图片的spu id, image_id, category id, cluster id, image_vector, used_num(初始为0)
    """
    data_dir = './data/preprocess'
    spu_csv_path = f'{data_dir}/spu_data_cleaned_gpu.csv'
    spu_df = pd.read_csv(spu_csv_path)
    cluster_mapping = load_cluster_mapping(f'{data_dir}/cluster_mapping.json')
    final_meta = []
    _, vectors_dict = load_spu_vectors(data_dir)
    for _, row in spu_df.iterrows():
        spu = row['SPU']
        image_id = row['主图ID']
        category = row['产品分类']
        vector = get_spu_vector(spu, vectors_dict)
        if vector is None or category not in cluster_mapping:
            continue
        cluster_centers = np.array(list(cluster_mapping[category].values()))
        dists = np.linalg.norm(vector - cluster_centers, axis=1)
        cluster_id = np.argmin(dists) + 1  # 簇ID从1开始
        final_meta.append({
            'spu_id': spu,
            'image_id': image_id,
            'category': category,
            'cluster_id': f'cluster_{cluster_id}',
            'image_vector': vector.tolist(),
            'used_num': 0
        })

    with open(f'{data_dir}/es_docs.pkl', 'wb') as f:
        pickle.dump(final_meta, f)
    return final_meta



if __name__ == "__main__":
    # cluster_mapping = build_cluster_mapping()
    final_meta = build_es_docs()


