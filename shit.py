from pydantic import BaseModel, HttpUrl
from enum import Enum
from pathlib import Path
from typing import List, Optional
import requests

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

    

# def image_downloader(img_id):
#     img_download_url = f"https://erp.baycheer.com/upload/product/{img_id}.jpg"
#     temp_img_dir = Path("temp_img")
#     temp_img_dir.mkdir(exist_ok=True)
#     img_path = temp_img_dir / f"{img_id}.jpg"
#     response = requests.get(img_download_url, params=ErpAuthParams().model_dump())
#     if response.status_code == 200:
#         with open(img_path, "wb") as f:
#             f.write(response.content)
#         return img_path
#     else:
#         print(f"Failed to download image for image_id: {img_id}")
#         return None

if __name__ == "__main__":
    spu_id = 21772288
    # main_and_scene_images_ids_dict = get_main_and_scene_images_ids(spu_id)
    # print(main_and_scene_images_ids_dict)
    sku_images_ids_dict = get_sku_images_ids(spu_id)
    print(sku_images_ids_dict)