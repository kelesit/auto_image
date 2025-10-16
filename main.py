"""
主程序流程：
1. 加载配置和初始化日志。
2. 加载或初始化进度跟踪文件。
3. 获取所有待处理的SPU。
4. 遍历每个SPU：
    a. 为SPU下的每张A图采样B图。
    b. 为B图生成提示词。
    c. 调用ComfyUI生成最终的C图。
    d. 更新进度文件。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import torch

from config import Config

from src.feature_extractor import VITFeatureExtractor
from src.b_image_sampling import BImageSampler
from src.tools import load_b_img_path
# from prompt_generator import generate_prompts_for_b_images
# from comfyui_runner import run_comfyui_generation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_progress(progress_file: Path) -> Dict:
    """加载进度文件，如果不存在则返回一个空的字典。"""
    if progress_file.exists():
        logging.info(f"从 {progress_file} 加载进度。")
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logging.info("未找到进度文件，将创建新的进度记录。")
        return {}

def save_progress(progress_data: Dict, progress_file: Path):
    """将进度数据保存到文件。"""
    logging.info(f"保存进度到 {progress_file}。")
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=4)

def get_spu_list(a_image_path: Path, specified_categories:Optional[List]=None) -> Dict[str, List[str]]:
    """
    扫描A图数据集路径，获取所有品类及其下的SPU ID。
    返回: {"category_name": ["spu_id_1", "spu_id_2", ...]}
    """
    spu_by_category = {}
    if specified_categories:
        for category in specified_categories:
            category_dir = a_image_path / category
            if category_dir.exists() and category_dir.is_dir():
                spu_ids = [spu_dir.name for spu_dir in category_dir.iterdir() if spu_dir.is_dir()]
                if spu_ids:
                    spu_by_category[category] = spu_ids
            else:
                logging.warning(f"指定的品类路径不存在或不是目录: {category_dir}")
        logging.info(f"发现 {len(spu_by_category)} 个指定品类。")
        return spu_by_category
    
    for category_dir in a_image_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            spu_ids = [spu_dir.name for spu_dir in category_dir.iterdir() if spu_dir.is_dir()]
            if spu_ids:
                spu_by_category[category_name] = spu_ids
    logging.info(f"发现 {len(spu_by_category)} 个品类。")
    return spu_by_category

def process_spu(spu_id: str, category_name: str, config: Config, progress: Dict, extractor: VITFeatureExtractor, sampler: BImageSampler):
    """处理单个SPU的所有A图。"""
    spu_progress = progress.setdefault(category_name, {}).setdefault(spu_id, {})
    spu_path = Path(config.a_image_dataset_path) / category_name / spu_id
    
    if not spu_path.exists():
        logging.warning(f"SPU路径不存在: {spu_path}")
        return

    a_image_files = [f for f in spu_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    for a_image_path in a_image_files:
        a_image_id = a_image_path.stem
        if spu_progress.get(a_image_id, {}).get("c_image_generated"):
            logging.info(f"跳过已完成的A图: {a_image_id}")
            continue
        
        logging.info(f"开始处理A图: {a_image_id} (SPU: {spu_id})")
        process_a_image(
            a_image_id, a_image_path, 
            spu_id, category_name, 
            config, spu_progress, 
            extractor, sampler)
        # 处理完一张A图后立即保存进度
        save_progress(progress, Path(config.progress_file_path))


def process_a_image(
        a_image_id: str, a_image_path: Path, 
        spu_id: str, category_name: str, 
        config: Config, spu_progress: Dict, 
        extractor: VITFeatureExtractor, sampler: BImageSampler):
    """处理单张A图的完整流程：获取A图特征向量 -> 采样B图 -> 生成Prompt -> 生成C图。"""
    a_image_progress = spu_progress.setdefault(a_image_id, {})
    a_image_progress["a_image_path"] = str(a_image_path)

    # 1. 加载或计算并保存A图特征向量
    if "a_image_vector_path" not in a_image_progress:
        logging.info(f"计算A图 {a_image_id} 的特征向量...")
        
        # 定义向量文件的保存路径
        vector_dir = Path(config.vectors_path) / category_name / spu_id
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_path = vector_dir / f"{a_image_id}.pt"

        try:
            image = Image.open(a_image_path).convert("RGB")
            vector = extractor.extract_features(image)
            torch.save(vector, vector_path)
            # 在进度文件中记录路径
            a_image_progress["a_image_vector_path"] = str(vector_path)
            logging.info(f"A图 {a_image_id} 特征向量已保存到: {vector_path}")
        except Exception as e:
            logging.error(f"计算或保存A图 {a_image_id} 特征向量时出错: {e}")
            return  None # 如果特征提取失败，则跳过后续步骤, 该SPU判断为未完成


    # 2. 采样B图
    if "b_images" not in a_image_progress or len(a_image_progress['b_images']):
        logging.info(f"为A图 {a_image_id} 采样B图...")
        a_vector = torch.load(a_image_progress["a_image_vector_path"], weights_only=True)
        sampled_b_images_docs = sampler.sample_b_images_for_a(a_vector, category_name, config.num_b_images)
        a_image_progress["b_images"] = []
        for b_img in sampled_b_images_docs:
            b_image_spu_id = b_img["spu_id"]
            b_image_path = load_b_img_path(b_image_spu_id, b_img_dir=Path(config.b_image_dataset_path) / category_name)
            if not b_image_path:
                logging.warning(f"SPU {spu_id} 的A图 {a_image_id} 采样的B图 SPU: {b_image_spu_id} 主图 下载失败。")
                return None # 如果B图下载失败，则跳过后续步骤, 该SPU判断为未完成
            b_img_dict = {
                "b_image_spu_id": b_image_spu_id,
                "b_image_path": b_image_path,
                "score": b_img["score"],
                "prompt": "",
                "c_image_path": ""
            }
            a_image_progress["b_images"].append(b_img_dict)
        
    # 3. 为B图生成Prompt
    if a_image_progress.get("b_images") and not a_image_progress.get("prompts_generated"):
        logging.info(f"为A图 {a_image_id} 的B图生成Prompt...")
        # generate_prompts_for_b_images(a_image_progress["b_images"]) # 示例调用
        # a_image_progress["prompts_generated"] = True
        pass # 在此处实现或调用Prompt生成逻辑

    # 4. 生成C图
    if a_image_progress.get("prompts_generated") and not a_image_progress.get("c_image_generated"):
        logging.info(f"为A图 {a_image_id} 生成C图...")
        # run_comfyui_generation(a_image_path, a_image_progress["b_images"]) # 示例调用
        # a_image_progress["c_image_generated"] = True
        pass # 在此处实现或调用ComfyUI生成逻辑

    logging.info(f"完成A图处理: {a_image_id}")
    # 保存采样器的使用计数
    sampler.save_usage_counts()
    return spu_progress



def main():
    """主函数， orchestrates the entire process."""
    cfg = Config()
    progress_file = Path(cfg.progress_file_path)

    # 初始化特征提取器
    logging.info("初始化特征提取器...")
    vit_config = {
        "model_path": cfg.model_path,
        "model_name": cfg.model_name
    }
    feature_extractor = VITFeatureExtractor(vit_config)
    logging.info("特征提取器初始化完成。")

    # 初始化B图采样器
    logging.info("初始化B图采样器...")
    b_img_sampler = BImageSampler(
        data_dir=Path(cfg.b_image_dataset_path),
        usage_file=Path(cfg.b_image_usage_file),
        cluster_mapping_file=Path(cfg.cluster_mapping_file),
    )
    logging.info("B图采样器初始化完成。")
    
    progress_data = load_progress(progress_file)
    
    spu_by_category = get_spu_list(Path(cfg.a_image_dataset_path), cfg.specified_categories)

    for category, spu_list in spu_by_category.items():
        logging.info(f"===== 开始处理品类: {category} =====")
        for spu_id in spu_list:
            if progress_data.get(category, {}).get(spu_id, {}).get("all_done"):
                logging.info(f"SPU {spu_id} 已全部处理完成，跳过。")
                continue
            
            logging.info(f"--- 处理SPU: {spu_id} ---")
            process_spu(spu_id, category, cfg, progress_data, feature_extractor, b_img_sampler)
            
            # 标记整个SPU已完成
            progress_data.setdefault(category, {}).setdefault(spu_id, {})["all_done"] = True
            save_progress(progress_data, progress_file)

        logging.info(f"===== 品类 {category} 处理完成 =====")

    logging.info("所有任务处理完成！")


if __name__ == "__main__":
    main()
