"""
B图采样算法实现
基于你的设计思路，实现高效的场景图采样
"""

import numpy as np
from elasticsearch import Elasticsearch
from typing import List, Dict, Tuple
import heapq


class BImageSampler:
    def __init__(self, es_client: Elasticsearch, index_name: str = "scene_images"):
        self.es = es_client
        self.index_name = index_name
        self.alpha = 0.7  # 差异性权重
        self.beta = 0.3   # 冷门优先权重
    
    def sample_b_images_for_a(self, a_vector: np.ndarray, a_category: str, n_samples: int = 10) -> List[Dict]:
        """
        为单个A图采样n个B图
        
        Args:
            a_vector: A图的向量
            a_category: A图的品类
            n_samples: 需要采样的B图数量
            
        Returns:
            采样得到的B图列表
        """
        # 第一层：从每个簇中找出与A差异大的候选
        candidates = self._get_diverse_candidates_from_clusters(a_vector, a_category)
        
        # 第二层：计算加权分数（差异性 + 冷门优先）
        scored_candidates = self._calculate_weighted_scores(a_vector, candidates)
        
        # 第三层：Max-Min采样确保B之间的多样性
        selected_b_images = self._max_min_sampling(scored_candidates, n_samples)
        
        # 更新使用次数
        self._update_usage_counts(selected_b_images)
        
        return selected_b_images
    
    def _get_diverse_candidates_from_clusters(self, a_vector: np.ndarray, a_category: str, top_k_per_cluster: int = 20) -> List[Dict]:
        """
        从每个簇中获取与A图差异最大的候选
        """
        # 获取该品类下的所有簇
        clusters = self._get_category_clusters(a_category)
        candidates = []
        
        for cluster_id in clusters:
            # 在每个簇中搜索与A向量最不相似的图片
            query = {
                "size": top_k_per_cluster,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"category_id": a_category}},
                            {"term": {"cluster_id": cluster_id}}
                        ]
                    }
                },
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": """
                        // 计算余弦距离（1-相似度）作为差异性分数
                        double dotProduct = 0.0;
                        double normA = 0.0;
                        double normB = 0.0;
                        
                        for (int i = 0; i < params.a_vector.length; i++) {
                            dotProduct += params.a_vector[i] * doc['image_vector'][i];
                            normA += params.a_vector[i] * params.a_vector[i];
                            normB += doc['image_vector'][i] * doc['image_vector'][i];
                        }
                        
                        double cosine_sim = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
                        return 1.0 - cosine_sim;  // 返回差异性分数
                        """,
                        "params": {"a_vector": a_vector.tolist()}
                    }
                }
            }
            
            response = self.es.search(index=self.index_name, body=query)
            candidates.extend([hit["_source"] for hit in response["hits"]["hits"]])
        
        return candidates
    
    def _calculate_weighted_scores(self, a_vector: np.ndarray, candidates: List[Dict]) -> List[Tuple[float, Dict]]:
        """
        计算候选图片的加权分数
        """
        scored_candidates = []
        
        # 获取使用次数的统计信息用于归一化
        use_counts = [c.get("used_num", 0) for c in candidates]
        max_use_count = max(use_counts) if use_counts else 1
        
        for candidate in candidates:
            b_vector = np.array(candidate["image_vector"])
            
            # 计算A-B差异性
            similarity_ab = np.dot(a_vector, b_vector) / (np.linalg.norm(a_vector) * np.linalg.norm(b_vector))
            diversity_score = 1 - similarity_ab
            
            # 计算冷门优先分数
            use_count = candidate.get("used_num", 0)
            cold_score = 1 - (use_count / max_use_count) if max_use_count > 0 else 1.0
            
            # 综合分数
            final_score = self.alpha * diversity_score + self.beta * cold_score
            
            scored_candidates.append((final_score, candidate))
        
        # 按分数降序排序
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates
    
    def _max_min_sampling(self, scored_candidates: List[Tuple[float, Dict]], n_samples: int) -> List[Dict]:
        """
        Max-Min采样：确保选中的B图之间差异最大
        """
        if len(scored_candidates) <= n_samples:
            return [candidate for _, candidate in scored_candidates]
        
        selected = []
        candidates_pool = [candidate for _, candidate in scored_candidates]
        
        # 选择分数最高的作为第一个
        selected.append(candidates_pool[0])
        candidates_pool.remove(candidates_pool[0])
        
        # 迭代选择与已选集合中最近点距离最大的点
        for _ in range(n_samples - 1):
            max_min_distance = -1
            best_candidate = None
            best_idx = -1
            
            for idx, candidate in enumerate(candidates_pool):
                candidate_vector = np.array(candidate["image_vector"])
                
                # 计算与所有已选点的最小距离
                min_distance = float('inf')
                for selected_candidate in selected:
                    selected_vector = np.array(selected_candidate["image_vector"])
                    # 使用余弦距离
                    similarity = np.dot(candidate_vector, selected_vector) / (
                        np.linalg.norm(candidate_vector) * np.linalg.norm(selected_vector)
                    )
                    distance = 1 - similarity
                    min_distance = min(min_distance, distance)
                
                # 选择最小距离最大的候选
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                candidates_pool.pop(best_idx)
        
        return selected
    
    def _get_category_clusters(self, category: str) -> List[str]:
        """
        获取指定品类下的所有簇ID
        """
        query = {
            "size": 0,
            "query": {"term": {"category_id": category}},
            "aggs": {
                "clusters": {
                    "terms": {
                        "field": "cluster_id",
                        "size": 1000
                    }
                }
            }
        }
        
        response = self.es.search(index=self.index_name, body=query)
        return [bucket["key"] for bucket in response["aggregations"]["clusters"]["buckets"]]
    
    def _update_usage_counts(self, selected_images: List[Dict]):
        """
        更新选中图片的使用次数
        """
        for image in selected_images:
            doc_id = image.get("image_id")
            if doc_id:
                self.es.update(
                    index=self.index_name,
                    id=doc_id,
                    body={
                        "script": {
                            "source": "ctx._source.used_num++",
                            "lang": "painless"
                        }
                    }
                )


# 使用示例
def example_usage():
    """
    使用示例
    """
    from elasticsearch import Elasticsearch
    
    # 初始化ES客户端
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    sampler = BImageSampler(es)
    
    # 为A图采样10个B图
    a_vector = np.random.rand(768)  # 示例A图向量
    a_category = "90 - Coffee Tables"
    
    b_images = sampler.sample_b_images_for_a(a_vector, a_category, n_samples=10)
    
    print(f"为A图采样得到 {len(b_images)} 个B图")
    for i, b_img in enumerate(b_images):
        print(f"B图{i+1}: {b_img['spu_id']}, 使用次数: {b_img.get('used_num', 0)}")


if __name__ == "__main__":
    example_usage()
