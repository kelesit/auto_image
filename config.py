from dataclasses import dataclass


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32

    # 图像模型配置
    feature_dimension: int = 768  # VIT的特征维度
    model_name: str = "google/vit-base-patch16-224"
    model_path: str = r'D:\work\auto_image\models\vit-base'

    # 文本模型配置
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    text_model_path: str = r'D:\work\auto_image\models\text_model'
    text_feature_dimension: int = 768  # 文本模型的特征维度

    # ES 配置
    es_host: str = "192.168.100.85"
    es_port: int = 9200
    es_user: str = "elastic"
    es_password: str = "123456"

    