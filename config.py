from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32

    # 图像模型配置
    feature_dimension: int = 768  # VIT的特征维度
    # 模型配置
    model_path: str = "models/vit-base"
    model_name: str = "google/vit-base-patch16-224"

    # 文本模型配置
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    text_model_path: str = r'D:\work\auto_image\models\text_model'
    text_feature_dimension: int = 768  # 文本模型的特征维度

    # ES 配置
    es_host: str = "192.168.100.85"
    es_port: int = 9200
    es_user: str = "elastic"
    es_password: str = "123456"

    # 数据路径
    a_image_dataset_path: str = "/root/autodl-tmp/A_image_dataset"
    b_image_dataset_path: str = "/root/autodl-tmp/B_image_dataset"
    c_image_dataset_path: str = "/root/autodl-tmp/C_image_dataset"
    progress_file_path: str = "/root/auto_image/data/progress.json"
    metadata_dir: str = "/root/auto_image/data/A_image_dataset/metadata"
    vectors_path: str = "/root/autodl-tmp/A_image_vectors"
    cluster_mapping_file: str = "/root/auto_image/data/b_image_cluster_mapping.json"
    b_image_usage_file: str = "/root/auto_image/data/b_image_usage_counts.json"


    # ComfyUI 配置
    comfyui_server_address: str = "127.0.0.1:8188"
    comfyui_workflow_path: str = "换图小子.json"
    comfyui_node_mapping: Dict[str, str] = field(default_factory=lambda: {
        "a_image_node": "191",
        "b_image_node": "192",
        "prompt_node": "6",
        "prompt_node2": "198",
        "output_node": "136"
    })

    # 其他配置
    # 指定处理的品类列表，如果为 None 或空列表，则处理所有品类
    specified_categories: Optional[List[str]] = field(default_factory=lambda: ['90 - Coffee Tables'])
    num_b_images: int = 2  # 每张A图采样的B图数量

