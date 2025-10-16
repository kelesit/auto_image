import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any
import json
from collections import defaultdict
import numpy as np
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """加载 pickle 文件数据"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"成功加载数据，类型: {type(data)}")
        if hasattr(data, '__len__'):
            logger.info(f"数据长度: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None
    
"""
es_doc 示例
{
    'spu_id': 21242159, 
    'image_id': 5179354, 
    'category': '90 - Coffee Tables', 
    'cluster_id': 'cluster_4', 
    'image_vector': [0.015886696055531502, -0.03519846498966217, -0.012169115245342255, 0.0363895520567894, -0.08190244436264038, 0.005889222491532564, -0.031871918588876724, -0.03584397956728935, 0.026817139238119125, -0.06688567250967026, -0.028138739988207817, -0.11050479859113693, 0.018992990255355835, -0.004327915143221617, 0.023950541391968727, -0.03174896165728569, 0.013533389195799828, 0.01715991459786892, 0.03471502661705017, ...], 
}

cluster_mapping 示例
{
    "90 - Coffee Tables": {
        "cluster_1": [
            -0.01955372467637062,
            -0.026874225586652756,
            ...
        ]
        "cluster_2": [...],
        ...
    },
    "82 - Accent Chairs": {
        "cluster_1": [...],
        ...
    },
    ...
}


### **采样流程**

假设要个A生成n张新图：

1. 第一层：与A差异最大
   - 在每个簇里，检索找出与A差异大的top-K 候选
2. 第二层：冷门B优先
   - 候选池中每张图一个加权分：score=α⋅(1−similarity(A,B))+β⋅(1−use_count(B))
     - similarity(A,B) A与B的相似度
     - use_count(B): B 图已被使用次数归一化
     - α, β : 控制差异性与冷门优先的权重

3. 第三层： B之间差异最大
   1. 从排序好的额候选池里使用Max-Min（下一个选的 B **离所有已选 B 图都尽量远**）采样:
      1. 先选分数最高的B1
      2. 后续每次选与已选B集合中最接近的B 距离最大的那个
4. 输出n张B图


第一层 (选簇)：输入A图，输出与A图差异最大的 top-k 个候选簇。
第二层 (选图)：在这些候选簇中，为每个簇找出与A图差异最大的 top-p 个候选B图，汇集成一个总的候选池。
第三层 (精选)：对总候选池进行加权评分和排序，然后用Max-Min采样从中选出最终的 n 个B图，确保它们之间也互相差异。
"""

class BImageSampler:
    def __init__(self, data_dir: str, usage_file: str, cluster_mapping_file: str):
        """初始化 B 图采样器
        Args:
            data_dir (str): 存放按品类拆分的 es_docs_{category}.pkl 文件的目录.
            usage_file (str): 存放 B 图使用次数的 JSON 文件路径.
            cluster_mapping_file (str): 存放品类-簇中心向量映射的 JSON 文件路径.
        """

        self.data_dir = Path(data_dir)
        self.usage_file = Path(usage_file)
        self.cluster_mapping_file = Path(cluster_mapping_file)
        self.alpha = 0.7  # 差异性权重
        self.beta = 0.3   # 冷门优先权重

        self._category_data_cache = {}  # 缓存已加载的品类数据
        self.usage_counts = self._load_json_file(self.usage_file, "B图使用次数")
        self.cluster_mapping = self._load_json_file(self.cluster_mapping_file, "簇中心向量")

    def _load_json_file(self, file_path: Path, description: str) -> defaultdict:
        """通用加载JSON文件功能，返回defaultdict。"""
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"从 {file_path} 加载 {description} 记录。")
                    loaded_data = json.load(f)
                    return defaultdict(int, loaded_data)
            except (json.JSONDecodeError, TypeError):
                logger.error(f"无法解析JSON文件: {file_path}。将使用空记录。")
                return defaultdict(int)
        logger.info(f"未找到 {description} 文件 ({file_path})，将创建新的记录。")
        return defaultdict(int)
    
    def save_usage_counts(self):
        """将当前的使用次数保存到文件。"""
        logger.info(f"保存B图使用次数记录到 {self.usage_file}。")
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_file, 'w', encoding='utf-8') as f:
            json.dump(self.usage_counts, f, indent=4)
    
    def _load_category_data(self, category: str):
        """加载指定品类的B图数据。
        return:
        按簇ID分组的B图数据字典: 
        {"cluster_id": [doc1, doc2, ...], ... }
        """
        if category in self._category_data_cache:
            return self._category_data_cache[category]
        
        category_file_path = self.data_dir / 'es_docs' / f'{category}.pkl'

        if not category_file_path.exists():
            logger.warning(f"品类文件不存在: {category_file_path}")
            return []
        
        try:
            with open(category_file_path, 'rb') as f:
                data = pickle.load(f)
                # 将数据按cluster_id分组存入缓存，方便后续查找
                grouped_data = defaultdict(list)
                for doc in data:
                    grouped_data[doc['cluster_id']].append(doc)
                self._category_data_cache[category] = grouped_data
                logger.info(f"成功加载品类 '{category}' 的数据，簇数量: {len(grouped_data)}")
                return self._category_data_cache[category]
        except Exception as e:
            logger.error(f"加载品类数据失败: {e}")
            return []
        
    def sample_b_images_for_a(self, a_vector: np.ndarray, a_category: str, n_samples: int=2) -> List[Dict]:
        """为单个A图采样B图"""
        b_image_pool_by_cluster = self._load_category_data(a_category)
        if not b_image_pool_by_cluster:
            logger.warning(f"品类 '{a_category}' 无可用B图数据，无法采样。")
            return []
        
        # 第一步：选出top-k个与A差异最大的候选簇
        candidate_clusters = self._get_diverse_clusters(a_vector, a_category, top_k=5)

        # 第二层-步骤1：在每个候选簇中的B图进行加权评分，每个簇选出top-p个B图，得到总候选B图池
        candidate_b_images = self._get_scored_candidates_from_clusters(a_vector, candidate_clusters, b_image_pool_by_cluster, top_p=10)
        
        # 第三层-步骤：在评分后的候选池中进行Max-Min采样
        selected_b_images = self._max_min_sampling(candidate_b_images, n_samples)
        
        # 更新使用次数
        self._update_usage_counts(selected_b_images)

        # 清理临时数据并返回
        for item in selected_b_images:
            if 'image_vector_np' in item:
                del item['image_vector_np']
        return selected_b_images
    

    def _get_diverse_clusters(self, a_vector: np.ndarray, category: str, top_k: int = 5) -> List[str]:
        """第一层：通过比较簇中心，找到与A图最不相似的 top-k 个簇。"""
        category_clusters = self.cluster_mapping.get(category)
        if not category_clusters:
            logger.warning(f"在cluster_mapping中未找到品类 '{category}' 的簇信息。")
            return []

        cluster_diversities = []
        for cluster_id, centroid_vec in category_clusters.items():
            centroid_np = np.array(centroid_vec)
            similarity = np.dot(a_vector, centroid_np)
            diversity = 1 - similarity
            cluster_diversities.append((diversity, cluster_id))
        
        # 按差异性降序排序
        cluster_diversities.sort(key=lambda x: x[0], reverse=True)
        
        top_clusters = [cluster_id for _, cluster_id in cluster_diversities[:top_k]]
        logger.info(f"第一层：选出与A图差异最大的 {len(top_clusters)} 个候选簇。")
        return top_clusters
    
    def _get_scored_candidates_from_clusters(self, a_vector: np.ndarray, cluster_ids: List[str], b_image_pool: Dict[str, List], top_p: int) -> List[Dict]:
        """第二层：在每个候选簇中，对B图进行加权评分，并选出每个簇的top-p个，
        最终汇集成一个按分数排序的总候选池。"""
        final_candidates = []
        for cluster_id in cluster_ids:
            docs_in_cluster = b_image_pool.get(cluster_id, [])
            if not docs_in_cluster:
                continue

            # 1. 对簇内所有B图进行加权评分
            scored_cluster_candidates = []

            # 为了归一化，需要先获取当前簇内所有图片的使用次数
            usage_values = [self.usage_counts.get(str(doc.get('image_id')), 0) for doc in docs_in_cluster]
            max_usage = max(usage_values) if usage_values else 1
            min_usage = min(usage_values) if usage_values else 0
            usage_range = max_usage - min_usage if max_usage != min_usage else 1

            for i, doc in enumerate(docs_in_cluster):
                b_vector = doc['image_vector']
                similarity = np.dot(a_vector, b_vector)
                diversity = 1 - similarity

                use_count = usage_values[i]
                normalized_use = (use_count - min_usage) / usage_range if usage_range > 0 else 0
                coldness = 1 - normalized_use

                score = self.alpha * diversity + self.beta * coldness
                
                doc['score'] = score
                scored_cluster_candidates.append(doc)

            # 2. 按综合分数排序
            scored_cluster_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 3. 选出该簇的top-p个，加入最终候选池
            final_candidates.extend(scored_cluster_candidates[:top_p])

        logger.info(f"第二层：从候选簇中汇集得到 {len(final_candidates)} 个候选B图。")
        return final_candidates
    

    def _max_min_sampling(self, candidates: List[Dict], n_samples: int) -> List[Dict]:
        """第三层：在已评分的候选池中进行Max-Min采样。"""
        if not candidates:
            logger.warning("候选B图池为空，无法进行Max-Min采样。")
            return []
        
        if n_samples >= len(candidates):
            logger.info(f"请求的样本数 {n_samples} 大于等于候选池大小 {len(candidates)}，将返回所有候选。")
            return candidates
        
        # 预处理：将所有候选的向量转换为numpy数组，方便计算距离
        for doc in candidates:
            if 'image_vector_np' not in doc:
                doc['image_vector_np'] = np.array(doc['image_vector'])
        
        selected = []
        # 1. 先选分数最高的B图
        candidates.sort(key=lambda x: x['score'], reverse=True)
        first_choice = candidates[0]
        selected.append(first_choice)
        candidates.remove(first_choice)

        # 2. 迭代选择剩余的B图
        while len(selected) < n_samples and candidates:
            max_min_distance = -1
            next_choice = None

            for candidate in candidates:
                # 计算 candidate 与已选 B 图集合中最近的距离
                distances = [np.linalg.norm(candidate['image_vector_np'] - sel['image_vector_np']) for sel in selected]
                min_distance = min(distances) if distances else float('inf')

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_choice = candidate
            
            if next_choice:
                selected.append(next_choice)
                candidates.remove(next_choice)
            else:
                break  # 无法再选择更多

        logger.info(f"第三层：通过Max-Min采样选出 {len(selected)} 张B图。")
        return selected
    
    def _update_usage_counts(self, selected_b_images: List[Dict]):
        """更新所选B图的使用次数。"""
        for doc in selected_b_images:
            image_id = str(doc.get('image_id'))
            if image_id:
                self.usage_counts[image_id] += 1
        

if __name__ == "__main__":
    DATA_DIR = "/root/autodl-tmp/B_image_dataset/es_docs/"
    USAGE_FILE = "/root/auto_image/data/b_image_usage.json"
    CLUSTER_MAPPING_FILE = "/root/auto_image/data/cluster_mapping.json"

    # 确保示例数据目录存在
    Path(DATA_DIR).mkdir(exist_ok=True)
    # 初始化采样器
    sampler = BImageSampler(
        data_dir=DATA_DIR,
        usage_file=USAGE_FILE,
        cluster_mapping_file=CLUSTER_MAPPING_FILE
    )

    a_image_vector_file =  "/root/autodl-tmp/A_image_vectors/82 - Accent Chairs/210004124/2449377981.pt"
    a_vector = torch.load(a_image_vector_file).numpy()
    sampled_b_images = sampler.sample_b_images_for_a(a_vector, "82 - Accent Chairs", n_samples=2)
    for img in sampled_b_images:
        logger.info(f"采样到B图: ID={img['image_id']}, SPU={img['spu_id']}, 类别={img['category']}, 簇={img['cluster_id']}, 评分={img['score']:.4f}")

    # sampler.save_usage_counts()
    print(len(sampled_b_images))