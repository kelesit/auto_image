# 说明文档

品类：
90 - Coffee Tables
114 - Benches
88 - TV Stands & Entertainment Centers
91 - End & Side Tables
83 - Sofa
176 - Room Dividers
92 - Cabinets & Chests
82 - Accent Chairs
218 - Plant Stands & Tables


## step 1: 图库构建

一、数据准备 (src/get_all_spu.py)
1. 获取所需品类的spu数据
2. 通过erp api获取spu对应的图片向量，如果有则存储，没有则标记
3. 过滤出带有向量的spu数据

二、数据清洗和筛选(src/clean_spu_with_gpu.py)
1. 根据品类分层
2. 每一层中遍历图片向量，筛选掉相似度过高的产品

三、计算簇心并构建es文档
1. 对每个品类下的所有图片向量进行聚类
   - 方法：K-Means
   - 聚类数目：N
        - N = min(50, max(10, int(sqrt(M/100))))
        - M: 该品类下的图片总数
   - 输出：每个品类下的聚类质心向量以及其对应的簇ID
cluster mapping
{
    "90 - Coffee Tables": [
        "cluster_1": [...],  # 质心向量
        "cluster_2": [...],
        ...
    ]
    ...
}

2. 构建最终的docs信息，包含每个图片的spu id, image_id, category id, cluster id, image_vector, used_num(初始为0)


四、将所有数据存入elasticsearch，供后续使用



## step 2: 挑选出7000个spu进行换图
