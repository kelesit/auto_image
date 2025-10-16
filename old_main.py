"""
每个SPU有多张A图（产品原图），
需要根据每张A图采样出若干张B图（背景图）， （根据A图特征向量进行相似度搜索，选出最不相似且相互不相似的B图s）
然后通过大模型对这若干张B图生成prompt，
最终通过comfyui生成带有产品的场景图（C图）。

因为spu数量众多，图片数量众多，每个环节的效果都需要做评估检测，所以每个环节的中间结果都需要保存。



所有图片本地存储路径：
A图：
/root/autodl-tmp/A_image_dataset/{category_name}/{spu_id}/{image_id}.jpg

A图特征向量：
/root/autodl-tmp/A_image_dataset/{category_name}/{spu_id}/vectors.npy

B图：
/root/autodl-tmp/B_image_dataset/{category_name}/{A_image_id}_{B_image_id}.jpg

C图：
/root/autodl-tmp/C_image_dataset/{category_name}/{spu_id}/{A_image_id}_{B_image_id}.jpg

过程中生成的记录文件，确保能够中断恢复：
{
    "spu_id": {
        "A_image_id": {
            "A_image_path": "",
            "A_image_vector": [],
            "B_image_list": [
                {
                    "B_image_id": "",
                    "B_image_path": "",
                    "similarity_to_A": 0.0,
                    "prompt": "",
                    "C_image_path": ""
                },
                ...
            ]
        },
        ...
    },
    ...
}

"""


def load_featur_extractor():
    pass


def load_images_and_vectors(spu_list, category_name):
    """获取该品类下指定SPU所有图片的本地存储地址和对应的向量
    return: List[Dict]  
    [
        {"spu_id_1": {
            "image_id_1": {"path": "xxx", "vector": [...]},
            "image_id_2": {"path": "xxx", "vector": [...]},
            ...
        }},
        {"spu_id_2": {
            "image_id_3": {"path": "xxx", "vector": [...]},
            ...
        }},
        ...
    ]
    """
    pass


def sample_background_imgs_for_spu(spu_dicts, num_each=2):
    """为每个SPU的所有图片采样出若干张背景图片
     return: Dict
     {
        "spu_id_1": {
            "image_id_1": ["bg_path_1", "bg_path_2"],
            "image_id_2": ["bg_path_3", "bg_path_4"],
            ...
        },
        "spu_id_2": {
            "image_id_3": ["bg_path_5", "bg_path_6"],
            ...
        },
        ...
    }
    """
    pass


def generate_image_4spus(spu_list, category_name, num_each=2):
    """为某品类的SPU下所有图片进行背景替换"""

    spu_dicts = load_images_and_vectors(spu_list, category_name)

    spu_dicts_with_b = sample_background_imgs_for_spu(spu_dicts, num_each=num_each)






def main():
    print("Hello from auto-image!")


if __name__ == "__main__":
    main()
