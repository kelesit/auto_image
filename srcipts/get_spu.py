import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
import pickle
from config import Config

from src.feature_extractor import VITFeatureExtractor
from src.b_image_sampling import BImageSampler, get_saled_spus
from src.tools import load_b_img_path
from src.prompt_generator import generate_prompt
from src.comfyui_runner import ComfyUIRunner

def get_spu_list(a_image_path: Path, specified_categories:Optional[List]=None, select_num:int=600) -> Dict[str, List[str]]:
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
                    spu_by_category[category] = spu_ids[:select_num]
            else:
                logging.warning(f"指定的品类路径不存在或不是目录: {category_dir}")
        logging.info(f"发现 {len(spu_by_category)} 个指定品类。")
    else:
        # 如果未指定类别，则扫描所有类别
        for category_dir in a_image_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                spu_ids = [spu_dir.name for spu_dir in category_dir.iterdir() if spu_dir.is_dir()]
                if spu_ids:
                    spu_by_category[category_name] = spu_ids[:select_num]
        logging.info(f"发现 {len(spu_by_category)} 个品类。")
    return spu_by_category



if __name__ == "__main__":
    cfg = Config()
    spu_by_category = get_spu_list(Path(cfg.a_image_dataset_path), cfg.specified_categories)
    with open("spu_by_category.json", "w", encoding="utf-8") as f:
        json.dump(spu_by_category, f, ensure_ascii=False, indent=4)
    for category, spu_ids in spu_by_category.items():
        logging.info(f"品类: {category}, SPU数量: {len(spu_ids)}")