import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer, AutoModel
import torch.nn.functional as F
import requests
from io import BytesIO

class VITFeatureExtractor:
    """基于VIT的特征提取器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_processor, self.model = self._load_img_processor_and_model()


    def _load_img_processor_and_model(self):
        model_path = self.config["model_path"]
        model_name = self.config["model_name"]
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist.")
            print(f"trying to load {model_name} from HuggingFace Hub")

            os.makedirs(model_path, exist_ok=True)
            img_processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTModel.from_pretrained(model_name)

            img_processor.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print(f"Model {model_name} downloaded and saved to {model_path}.")
        else:
            img_processor = ViTImageProcessor.from_pretrained(model_path)
            model = ViTModel.from_pretrained(model_path)
            print(f"Model loaded from {model_path}.")

        model.to(self.device)
        model.eval()
        return img_processor, model


    @torch.no_grad()
    def extract_features(self, images, normalize=True):
        """
        提取图像特征
        
        Args:
            images: 单个PIL图像对象或PIL图像列表
            normalize: 是否对特征向量进行L2归一化，默认为True
        
        Returns:
            对于单个图像：返回特征向量 tensor，形状为 (feature_dim,)
            对于批量图像：返回特征向量 tensor，形状为 (batch_size, feature_dim)
        """
        batch_mode = True
        if not isinstance(images, list):
            images = [images]
            batch_mode = False

        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL Image, got {type(img)}")
            
        inputs = self.img_processor(images=images, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        
        output = self.model(**inputs)
        # 获取[CLS]令牌特征，它位于序列的第一个位置
        # last_hidden_state的形状为 (batch_size, sequence_length, hidden_size)
        features = output.last_hidden_state[:, 0, :]

        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)

        if not batch_mode:
            features = features.squeeze(0)

        return features.cpu()
    
    def extract_from_url(self, image_url, auth_params=None, timeout=10):
        """
        从URL直接获取图像并提取特征，无需保存到本地
        
        Args:
            image_url: 图像URL
            auth_params: 可选的认证参数字典
            timeout: 请求超时时间，默认10秒
            
        Returns:
            特征向量列表或None（如果提取失败）
        """
        try:
            if auth_params:
                response = requests.get(image_url, timeout=timeout, params=auth_params)
            else:
                response = requests.get(image_url, timeout=timeout)

            if response.status_code != 200:
                print(f"无法获取图像: {image_url}, 状态码: {response.status_code}")
                return None
            
            image = Image.open(BytesIO(response.content)).convert("RGB")
            feature_tensor = self.extract_features(image)
            return feature_tensor.tolist()
        
        except Exception as e:
            print(f"处理图像是出错: {image_url}, 错误: {e}")
            return None




if __name__ == "__main__":
    config = {
        "model_path": "/root/auto_image/models/vit-base",
        "model_name": "google/vit-base-patch16-224"
    }

    extractor = VITFeatureExtractor(config)

    # 测试单张图像
    img_path = "/root/auto_image/src/2415506610.jpg"  # 替换为你的图像路径
    image = Image.open(img_path).convert("RGB")
    features = extractor.extract_features(image)
    print("单张图像特征向量形状:", features.shape)

    # 测试批量图像
    img_paths = ["test_image1.jpg", "test_image2.jpg"]  # 替换为你的图像路径列表
    images = [Image.open(p).convert("RGB") for p in img_paths]
    batch_features = extractor.extract_features(images)
    print("批量图像特征向量形状:", batch_features.shape)

