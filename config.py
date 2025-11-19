from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32

    # 图像模型配置
    feature_dimension: int = 768  # VIT的特征维度
    # 模型配置
    model_path: str = "/root/autodl-tmp/models/vit-base"
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
    a_image_dataset_path: str = "/root/autodl-fs/baycheer/img_gen/A_image_data"
    b_image_dataset_path: str = "/root/autodl-fs/baycheer/img_gen/B_image_data"
    c_image_dataset_path: str = "/root/autodl-fs/baycheer/img_gen/C_image_data"

    progress_file_path: str = "/root/autodl-fs/baycheer/img_gen/progress_files/pic_boss1_progress.json"
    vectors_path: str = "/root/autodl-fs/baycheer/img_gen/A_image_vectors"

    cluster_mapping_file: str = "/root/auto_image/data/b_image_cluster_mapping.json"
    b_image_usage_file: str = "/root/auto_image/data/b_image_usage_counts.json"
    metadata_dir: str = "/root/auto_image/data/A_image_dataset/metadata"


    # ComfyUI 配置
    comfyui_server_address: str = "127.0.0.1:8188"
    comfyui_workflow_path: str = "换图老子.json"
    comfyui_node_mapping: Dict[str, str] = field(default_factory=lambda: {
        "a_image_node": "78",
        # "b_image_node": "192",
        "prompt_node": "111",
        # "prompt_node2": "198",
        "output_node": "60"
    })


    # 其他配置
    # 指定处理的品类列表，如果为 None 或空列表，则处理所有品类
    specified_categories: Optional[List[str]] = field(default_factory=lambda: ['90 - Coffee Tables'])
    num_b_images: int = 1  # 每张A图采样的B图数量

