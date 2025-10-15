"""
# A图集构建方案

每个spu关联一个主图，n个场景图，m个sku图

需要以结构化存储的方式保存这些图片，保存图片路径
{
    "spu_id": 123456,
    "main_image": "path/to/main_image.jpg",
    "scene_images": [
        "path/to/scene_image1.jpg",
        "path/to/scene_image2.jpg",
        ...
    ],
    "sku_images": [
        "path/to/sku_image1.jpg",
        "path/to/sku_image2.jpg",
        ...
    ]
}

文件夹结构：
A_image_dataset/
    ├── spu_123456/
    │   ├── image_id_1.jpg  # 主图
    │   ├── image_id_2.jpg  # 场景图1
    │   ├── image_id_3.jpg  # 场景图2
    │   ├── image_id_4.jpg  # SKU图1
    │   ├── image_id_5.jpg  # SKU图2
    │   └── ...
    ├── spu_789012/
    │   ├── image_id_1.jpg
    │   ├── image_id_2.jpg
    │   ├── image_id_3.jpg
    │   ├── image_id_4.jpg
    │   ├── image_id_5.jpg
    │   └── ...
    └── ...
每个spu的图片信息存储在一个JSON文件中，方便后续处理

"""

from pydantic import BaseModel, HttpUrl
from enum import Enum
from pathlib import Path
from typing import List, Optional
import requests
import pandas as pd
from tqdm import tqdm

#产品图片类型id映射关系
image_type_id_dict = {
    2524:'销售',
    2528:'销售SKU',
    25201:'卖点图',
    25200:'尺寸图',
    2529:'销售素材',
    2520:'厂商橱窗',
    2527:'厂商SKU',
    2526:'厂商素材',
    2522:'厂商已修',
    2525:'自拍已修',
    2521:'自拍原图',
    2523:'已删除',
    2550:'白底图',
    2551:'场景图',
    2552:'尺寸图',
    2553:'主视图',
    2554:'细节图',
    2555:'静物图',
}


def get_main_and_scene_images_ids(spu_id: int):
    """
    获得spu的主图和场景图片id列表
    {
        "spu_id": 123456,
        "main_image_id": "image_id_1",
        "scene_image_ids": ["image_id_2", "image_id_3"],
    }
    """
    DOWNLOAD_URL = 'https://erp.baycheer.com/api/fileAccess/getSpuImage'
    params = {
        "app_id": 392013,
        "app_token": 'URC2P3GKZGDPFAAX8M61LQ88NLRRO3T9',
        "spu_id": spu_id,
        "image_type_id": '2524'
    }
    scene_images_id_dict = {}
    scene_images_id_dict['spu_id'] = spu_id
    scene_images_id_dict['scene_image_ids'] = []
    response = requests.get(DOWNLOAD_URL, params=params)
    if response.status_code == 200:
        resp = response.json()
        if resp['code'] == 0:
            data = resp['data']
            for item in data:
                if item['is_cover_image'] == 1:
                    scene_images_id_dict['main_image_id'] = item['product_image_id']
                    continue
                if item['topic'] =='A':
                    scene_images_id_dict['scene_image_ids'].append(item['product_image_id'])
            return scene_images_id_dict
        else:
            print(f"Error in response for spu_id: {spu_id}, message: {resp['msg']}")
            return None

                
def get_sku_images_ids(spu_id: int):
    """
    获得spu的sku图片id列表
    {
        "spu_id": 123456,
        "sku_image_ids": ['image_id_1', 'image_id_2']
        }
    }
    """
    DOWNLOAD_URL = 'https://erp.baycheer.com/api/fileAccess/getSpuImage'
    params = {
        "app_id": 392013,
        "app_token": 'URC2P3GKZGDPFAAX8M61LQ88NLRRO3T9',
        "spu_id": spu_id,
        "image_type_id": '2528'
    }
    sku_images_id_dict = {}
    sku_images_id_dict['spu_id'] = spu_id
    sku_images_id_dict['sku_image_ids'] = []
    response = requests.get(DOWNLOAD_URL, params=params)
    if response.status_code == 200:
        resp = response.json()
        if resp['code'] == 0:
            data = resp['data']
            for item in data:
                sku_images_id_dict['sku_image_ids'].append(item['product_image_id'])
            return sku_images_id_dict
        else:
            print(f"Error in response for spu_id: {spu_id}, message: {resp['msg']}")
            return None

        
    

def get_images_id_with_spu_id(spu_id: int):
    """
    获得spu的所有图片id
    {
        "spu_id": 123456,
        "main_image_id": "image_id_1",
        "scene_image_ids": ["image_id_2", "image_id_3"],
        "sku_image_ids": ['image_id_4', 'image_id_5'...]
    }
    """
    main_and_scene_images_ids_dict = get_main_and_scene_images_ids(spu_id)
    sku_images_ids_dict = get_sku_images_ids(spu_id)
    if main_and_scene_images_ids_dict and sku_images_ids_dict:
        combined_dict = {
            "spu_id": spu_id,
            "main_image_id": main_and_scene_images_ids_dict.get("main_image_id"),
            "scene_image_ids": main_and_scene_images_ids_dict.get("scene_image_ids", []),
            "sku_image_ids": sku_images_ids_dict.get("sku_image_ids", [])
        }
        return combined_dict
    else:
        return None

    

def image_downloader(img_id, save_dir: Path) -> Optional[Path]:
    """下载图片并保存到指定目录，返回图片路径
    save_dir: Path - 图片保存目录 (如: Path("A_image_dataset/90 - Coffee Tables/spu_123456/"))
    """
    img_download_url = f"https://erp.baycheer.com/upload/product/{img_id}.jpg"
    
    # 判断保存目录是否存在，不存在则创建
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / f"{img_id}.jpg"


    response = requests.get(
        img_download_url, 
        params={
            'app_id': 392013,
            'app_token': 'URC2P3GKZGDPFAAX8M61LQ88NLRRO3T9'
        }
    )
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    else:
        print(f"Failed to download image for image_id: {img_id}")
        return None


def create_metadata_json(spu_file: Path, save_dir: Path):
    """
    读取spu_id列表文件，获取每个spu的图片id，并以品类为组，分别保存为JSON文件
    spu_file: Path - 包含spu_id列表的文件 (如: Path("D:\work\auto_image\data\litfad_final_selected_spus.csv"))
    save_dir: Path - JSON文件保存目录 (如: Path("A_image_dataset/metadata/"))
    """
    import json

    # 判断保存目录是否存在，不存在则创建
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    spu_df = pd.read_csv(spu_file)
    categories = spu_df['产品分类'].unique()
    for category in categories:
        category_spu_df = spu_df[spu_df['产品分类'] == category]
        category_spu_ids = category_spu_df['SPU'].tolist()
        meta_json_path = save_dir / f"images_metadata_{category}.json"
        images_id_dicts_list = []
        for spu_id in category_spu_ids:
            spu_id_int = int(spu_id)
            images_id_dict = get_images_id_with_spu_id(spu_id_int)
            if images_id_dict:
                images_id_dicts_list.append(images_id_dict)
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(images_id_dicts_list, f, ensure_ascii=False, indent=4)
        print(f"Saved metadata JSON for category: {category}, path: {meta_json_path}")
    print("All metadata JSON files created.")


def download_images_from_metadata_json(meta_json_path: Path, base_save_dir: Path):
    """
    从metadata JSON文件中读取图片id，下载图片并保存到指定目录
    meta_json_path: Path - 包含图片id的JSON文件 (如: Path("A_image_dataset/metadata/images_metadata_CategoryA.json"))
    base_save_dir: Path - 图片保存的基础目录 (如: Path("A_image_dataset/CategoryA/"))
    """
    import json

    with open(meta_json_path, "r", encoding="utf-8") as f:
        images_id_dicts_list = json.load(f)
    category = meta_json_path.stem.replace("images_metadata_", "")
    for images_id_dict in tqdm(images_id_dicts_list, desc=f"Downloading {category} images"):
        spu_id = images_id_dict['spu_id']
        spu_save_dir = base_save_dir / f"{spu_id}"
        
        # 下载主图
        main_image_id = images_id_dict.get('main_image_id')
        if main_image_id:
            image_downloader(main_image_id, spu_save_dir)
        
        # 下载场景图
        scene_image_ids = images_id_dict.get('scene_image_ids', [])
        for scene_image_id in scene_image_ids:
            image_downloader(scene_image_id, spu_save_dir)
        
        # 下载SKU图
        sku_image_ids = images_id_dict.get('sku_image_ids', [])
        for sku_image_id in sku_image_ids:
            image_downloader(sku_image_id, spu_save_dir)
    
    print(f"All {category} images downloaded and saved to {base_save_dir}")


def download_all_images(metadata_dir: Path, base_image_dir: Path):
    """
    下载metadata目录下所有JSON文件中的图片
    metadata_dir: Path - 包含多个metadata JSON文件的目录 (如: Path("A_image_dataset/metadata/"))
    base_image_dir: Path - 图片保存的基础目录 (如: Path("A_image_dataset/"))
    """
    json_files = list(metadata_dir.glob("images_metadata_*.json"))
    for json_file in json_files:
        category = json_file.stem.replace("images_metadata_", "")
        category_save_dir = base_image_dir / category
        download_images_from_metadata_json(json_file, category_save_dir)
    print("All images from all categories downloaded.")


if __name__ == "__main__":
    # # 构建metadata json文件
    # spu_file = Path(r"data/litfad_final_selected_spus.csv")
    # save_dir = Path(r"data/A_image_dataset/metadata")
    # create_metadata_json(spu_file, save_dir)
    
    # 下载所有图片
    metadata_dir = Path(r"data/A_image_dataset/metadata")
    base_image_dir = Path(r"data/A_image_dataset")
    download_all_images(metadata_dir, base_image_dir)