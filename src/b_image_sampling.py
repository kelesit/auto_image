import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any
import json
from collections import defaultdict
import numpy as np
import torch
from abc import ABC, abstractmethod
import pandas as pd


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _cosine_similarity(v1, v2):
    """计算两个numpy向量的余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class BaseSamplingStrategy(ABC):
    """采样策略的抽象基类"""
    def __init__(self, sampler_instance):
        self.sampler = sampler_instance

    @abstractmethod
    def sample(self, a_vector: np.ndarray, a_category: str, n_samples: int, **kwargs) -> List[Dict]:
        """执行采样逻辑"""
        pass

    def _max_min_sampling(self, candidates: List[Dict], n_samples: int) -> List[Dict]:
        """在已评分的候选池中进行Max-Min采样，确保多样性。"""
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
        # 使用 list.remove() 可能会因为字典比较问题出错，改用索引或更安全的方式
        candidates = [c for c in candidates if c['image_id'] != first_choice['image_id']]

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
                candidates = [c for c in candidates if c['image_id'] != next_choice['image_id']]
            else:
                break  # 无法再选择更多

        logger.info(f"通过Max-Min采样选出 {len(selected)} 张B图。")
        return selected

class FarthestSampler(BaseSamplingStrategy):
    """
    策略一：与A图差异最大化的采样器（原始逻辑）。
    三层漏斗：选簇 -> 选图评分 -> Max-Min采样
    """
    def sample(self, a_vector: np.ndarray, a_category: str, n_samples: int, spu_list: List[str] = None, **kwargs) -> List[Dict]:
        b_image_pool_by_cluster = self.sampler._load_category_data(a_category, specific_spus=spu_list)
        if not b_image_pool_by_cluster:
            logger.warning(f"品类 '{a_category}' 无可用B图数据(或在指定SPU列表下)，无法使用 FarthestSampler 采样。")
            return []
        
        # 第一步：选出top-k个与A差异最大的候选簇
        candidate_clusters = self._get_diverse_clusters(a_vector, a_category, top_k=5)

        # 第二层：在每个候选簇中的B图进行加权评分，得到总候选B图池
        candidate_b_images = self._get_scored_candidates_from_clusters(a_vector, candidate_clusters, b_image_pool_by_cluster, top_p=10)
        
        # 第三层：在评分后的候选池中进行Max-Min采样
        selected_b_images = self._max_min_sampling(candidate_b_images, n_samples)
        
        return selected_b_images

    def _get_diverse_clusters(self, a_vector: np.ndarray, category: str, top_k: int = 5) -> List[str]:
        """通过比较簇中心，找到与A图最不相似的 top-k 个簇。"""
        category_clusters = self.sampler.cluster_mapping.get(category)
        if not category_clusters:
            logger.warning(f"在cluster_mapping中未找到品类 '{category}' 的簇信息。")
            return []

        cluster_diversities = []
        for cluster_id, centroid_vec in category_clusters.items():
            centroid_np = np.array(centroid_vec)
            similarity = _cosine_similarity(a_vector, centroid_np)
            diversity = 1 - similarity
            cluster_diversities.append((diversity, cluster_id))
        
        cluster_diversities.sort(key=lambda x: x[0], reverse=True)
        top_clusters = [cluster_id for _, cluster_id in cluster_diversities[:top_k]]
        logger.info(f"FarthestSampler: 选出与A图差异最大的 {len(top_clusters)} 个候选簇。")
        return top_clusters
    
    def _get_scored_candidates_from_clusters(self, a_vector: np.ndarray, cluster_ids: List[str], b_image_pool: Dict[str, List], top_p: int) -> List[Dict]:
        """在每个候选簇中，对B图进行加权评分，并选出每个簇的top-p个。"""
        final_candidates = []
        for cluster_id in cluster_ids:
            docs_in_cluster = b_image_pool.get(cluster_id, [])
            if not docs_in_cluster:
                continue

            scored_cluster_candidates = self._score_docs(a_vector, docs_in_cluster, is_farthest=True)
            scored_cluster_candidates.sort(key=lambda x: x['score'], reverse=True)
            final_candidates.extend(scored_cluster_candidates[:top_p])

        logger.info(f"FarthestSampler: 从候选簇中汇集得到 {len(final_candidates)} 个候选B图。")
        return final_candidates

    def _score_docs(self, a_vector: np.ndarray, docs: List[Dict], is_farthest: bool) -> List[Dict]:
        """对一组文档进行评分"""
        scored_docs = []
        usage_values = [self.sampler.usage_counts.get(str(doc.get('image_id')), 0) for doc in docs]
        max_usage = max(usage_values) if usage_values else 1
        min_usage = min(usage_values) if usage_values else 0
        usage_range = max_usage - min_usage if max_usage != min_usage else 1

        for i, doc in enumerate(docs):
            b_vector = np.array(doc['image_vector'])
            similarity = _cosine_similarity(a_vector, b_vector)
            
            use_count = usage_values[i]
            normalized_use = (use_count - min_usage) / usage_range if usage_range > 0 else 0
            coldness = 1 - normalized_use

            if is_farthest:
                # 差异性越大越好
                primary_metric = 1 - similarity
            else:
                # 相似性越大越好
                primary_metric = similarity

            score = self.sampler.alpha * primary_metric + self.sampler.beta * coldness
            doc['score'] = score
            scored_docs.append(doc)
        return scored_docs

class ClosestSampler(BaseSamplingStrategy):
    """
    策略二：在相似度阈值内，寻找与A图最近的B图。
    """
    def __init__(self, sampler_instance, similarity_threshold=0.95):
        super().__init__(sampler_instance)
        self.threshold = similarity_threshold

    def sample(self, a_vector: np.ndarray, a_category: str, n_samples: int, spu_list: List[str] = None, **kwargs) -> List[Dict]:
        b_image_pool_by_cluster = self.sampler._load_category_data(a_category, specific_spus=spu_list)
        if not b_image_pool_by_cluster:
            logger.warning(f"品类 '{a_category}' 无可用B图数据(或在指定SPU列表下)，无法使用 ClosestSampler 采样。")
            return []

        # 1. 汇集所有B图并过滤
        all_b_images = [doc for docs in b_image_pool_by_cluster.values() for doc in docs]
        
        candidate_b_images = []
        for doc in all_b_images:
            b_vector = np.array(doc['image_vector'])
            similarity = _cosine_similarity(a_vector, b_vector)
            if similarity < self.threshold:
                doc['similarity'] = similarity # 临时存储
                candidate_b_images.append(doc)
        
        if not candidate_b_images:
            logger.warning(f"ClosestSampler: 没有找到相似度低于 {self.threshold} 的B图。")
            return []
            
        logger.info(f"ClosestSampler: 找到 {len(candidate_b_images)} 张相似度低于阈值的B图。")

        # 2. 对候选B图进行评分（相似度越高，冷门度越高，分数越高）
        scored_candidates = self._score_docs(a_vector, candidate_b_images, is_farthest=False)

        # 3. Max-Min采样
        selected_b_images = self._max_min_sampling(scored_candidates, n_samples)
        
        return selected_b_images

    def _score_docs(self, a_vector: np.ndarray, docs: List[Dict], is_farthest: bool) -> List[Dict]:
        """对一组文档进行评分 (ClosestSampler 版本)"""
        scored_docs = []
        usage_values = [self.sampler.usage_counts.get(str(doc.get('image_id')), 0) for doc in docs]
        max_usage = max(usage_values) if usage_values else 1
        min_usage = min(usage_values) if usage_values else 0
        usage_range = max_usage - min_usage if max_usage != min_usage else 1

        for i, doc in enumerate(docs):
            # similarity 已经在上一步计算并存储
            similarity = doc.get('similarity', 0)
            
            use_count = usage_values[i]
            normalized_use = (use_count - min_usage) / usage_range if usage_range > 0 else 0
            coldness = 1 - normalized_use

            # 相似性越大越好
            primary_metric = similarity

            score = self.sampler.alpha * primary_metric + self.sampler.beta * coldness
            doc['score'] = score
            scored_docs.append(doc)
        return scored_docs


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
        self.alpha = 0.7  # 差异性/相似性权重
        self.beta = 0.3   # 冷门优先权重

        self._category_data_cache = {}  # 缓存已加载的品类数据
        self.usage_counts = self._load_json_file(self.usage_file, "B图使用次数")
        self.cluster_mapping = self._load_json_file(self.cluster_mapping_file, "簇中心向量")

        # 注册所有可用的采样策略
        self.strategies = {
            "farthest": FarthestSampler(self),
            "closest": ClosestSampler(self, similarity_threshold=0.95)
        }
        logger.info(f"B图采样器已初始化，支持的策略: {list(self.strategies.keys())}")

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
    
    def _load_category_data(self, category: str, specific_spus: List[str] = None) -> Dict[str, List[Dict]]:
        """加载指定品类的B图数据。
        可以限定在 specific_spus 列表中的 SPU。
        Args:
            category (str): 品类名称.
            specific_spus (List[str], optional): 仅加载这些 SPU 的数据.
        return:
        按簇ID分组的B图数据字典: 
        {"cluster_id": [doc1, doc2, ...], ... }
        """
        # 如果没有指定spu列表，并且缓存中存在，则直接返回缓存
        if not specific_spus and category in self._category_data_cache:
            logger.info(f"从缓存加载品类 '{category}' 的数据。")
            return self._category_data_cache[category]
        
        category_file_path = self.data_dir / 'es_docs' / f'{category}.pkl'

        if not category_file_path.exists():
            logger.warning(f"品类文件不存在: {category_file_path}")
            return {}
        
        try:
            with open(category_file_path, 'rb') as f:
                data = pickle.load(f)
            
            grouped_data = defaultdict(list)
            
            # 如果提供了 specific_spus，则只处理这些spu的数据
            if specific_spus:
                spu_set = set(map(str, specific_spus)) # 转换为字符串集合以便快速查找
                logger.info(f"为品类 '{category}' 加载指定的 {len(spu_set)} 个SPU的数据。")
                for doc in data:
                    if str(doc.get('spu_id')) in spu_set:
                        grouped_data[doc['cluster_id']].append(doc)
            else:
                # 否则，加载所有数据
                logger.info(f"首次加载品类 '{category}' 的所有数据。")
                for doc in data:
                    grouped_data[doc['cluster_id']].append(doc)
                # 只有在加载全部数据时才更新缓存
                self._category_data_cache[category] = grouped_data
            
            logger.info(f"成功加载品类 '{category}' 的数据，簇数量: {len(grouped_data)}")
            return grouped_data

        except Exception as e:
            logger.error(f"加载品类数据失败: {e}")
            return {}
        
    def sample_b_images_for_a(self, a_vector: np.ndarray, a_category: str, n_samples: int=2, strategy: str = "farthest", **strategy_kwargs) -> List[Dict]:
        """为单个A图使用指定策略采样B图"""
        if strategy not in self.strategies:
            logger.error(f"未知的采样策略: '{strategy}'。可用策略: {list(self.strategies.keys())}")
            return []

        logger.info(f"使用策略 '{strategy}' 为A图采样 {n_samples} 张B图...")
        
        # 获取策略实例并执行采样
        strategy_instance = self.strategies[strategy]
        selected_b_images = strategy_instance.sample(a_vector, a_category, n_samples, **strategy_kwargs)
        
        # 更新使用次数
        self._update_usage_counts(selected_b_images)

        # 清理临时数据并返回
        for item in selected_b_images:
            # 清理可能存在的临时键
            item.pop('image_vector_np', None)
            item.pop('similarity', None)
            # 附加策略信息
            item['sampling_strategy'] = strategy

        return selected_b_images
    
    def _update_usage_counts(self, selected_b_images: List[Dict]):
        """更新所选B图的使用次数。"""
        for doc in selected_b_images:
            image_id = str(doc.get('image_id'))
            if image_id:
                self.usage_counts[image_id] += 1
        


def get_saled_spus(category:str):
    """
    获取已售SPU列表
    """
    saled_df = pd.read_csv('/root/auto_image/data/litfad_orders.csv')
    category_saled_spus = saled_df[saled_df['产品分类'] == category]['SPU'].astype(str).unique().tolist()
    return category_saled_spus

    


if __name__ == "__main__":
    DATA_DIR = "/root/autodl-tmp/B_image_dataset"
    USAGE_FILE = "/root/auto_image/data/b_image_usage.json"
    CLUSTER_MAPPING_FILE = "/root/auto_image/data/b_image_cluster_mapping.json"

    # 确保示例数据目录存在
    Path(DATA_DIR).mkdir(exist_ok=True)
    # 初始化采样器
    sampler = BImageSampler(
        data_dir=DATA_DIR,
        usage_file=USAGE_FILE,
        cluster_mapping_file=CLUSTER_MAPPING_FILE
    )

    a_image_vector_file =  "/root/autodl-tmp/A_image_vectors/90 - Coffee Tables/21332728/2415504730.pt"
    a_vector = torch.load(a_image_vector_file).numpy()

    # --- 测试 ---
    print("\n--- 1. 测试 Farthest 策略 (无SPU限制) ---")
    sampled_farthest = sampler.sample_b_images_for_a(a_vector, "90 - Coffee Tables", n_samples=2, strategy="farthest")
    for img in sampled_farthest:
        logger.info(f"采样到B图: ID={img['image_id']}, SPU={img['spu_id']}, 策略={img.get('sampling_strategy', 'N/A')}, 评分={img['score']:.4f}")

    print("\n--- 2. 测试 Closest 策略 (无SPU限制) ---")
    sampled_closest = sampler.sample_b_images_for_a(a_vector, "90 - Coffee Tables", n_samples=2, strategy="closest")
    for img in sampled_closest:
        logger.info(f"采样到B图: ID={img['image_id']}, SPU={img['spu_id']}, 策略={img.get('sampling_strategy', 'N/A')}, 评分={img['score']:.4f}")

    print("\n--- 3. 测试 Farthest 策略 (有SPU限制) ---")
    specific_spus = ["21717479", "21684310", "21761485"]
    sampled_farthest_specific = sampler.sample_b_images_for_a(
        a_vector, "90 - Coffee Tables", n_samples=1, 
        strategy="farthest", spu_list=specific_spus
    )
    for img in sampled_farthest_specific:
        logger.info(f"采样到B图: ID={img['image_id']}, SPU={img['spu_id']}, 策略={img.get('sampling_strategy', 'N/A')}, 评分={img['score']:.4f}")

    # sampler.save_usage_counts()
    print(f"\nFarthest 采样数量: {len(sampled_farthest)}")
    print(f"Closest 采样数量: {len(sampled_closest)}")
    print(f"Farthest with SPU list 采样数量: {len(sampled_farthest_specific)}")