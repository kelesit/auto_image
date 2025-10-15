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



## step 2: 挑选出7000个spu
### 挑选方案
一、通过埋没因子+整体销量共同决定各个品类的挑选数量N

用现有数据估算各品类中"被埋没但有潜力"的SPU数量
1. 计算各品类的理想新品动销率a和埋没因子b
   - 理想新品动销率a：反映该品类的SPU中，经过一个季度的时间售卖后得到动销率
   - 埋没因子b：反映该品类的SPU中后期动销率不足的因子，即上一个季度上线的SPU，在下一个季度时由于新品推流期过后，导致实际动销率受到影响的因子


**计算埋没因子b**：
假设理想动销率为a， 埋没因子为b，(0 < b < 1)
埋没因子b：反映该品类的SPU中后期动销率不足的因子，即上一个季度上线的SPU，在下一个季度时由于新品推流期过后，导致实际动销率受到影响的因子
理想新品动销率a：反映该品类的SPU中，经过一个季度的时间售卖后得到动销率


以n个月为周期
理想新品动销率a = 过去n到2n个月上新的spu 销售数 / 过去n到2n个月上新的spu 数
过去n到2n个月动销SPU数 = 过去n到2n个月上新的spu 销售数 +（该品类24年至今上线的总SPU数 * 月均动销率)

即，
理想新品动销率a = (过去n到2n个月动销SPU数 -（该品类24年至今上线的总SPU数 * 月均动销率)) / 过去n到2n个月上线SPU数
第n季度实际动销数 = 第n-1季度上线SPU数 * a * 2 * b + 第n季度上线SPU数 * a * 1 

ps: b越小，埋没程度越高

选品权重计算
- **埋没程度权重**：
  埋没指数 = 1 - b
  埋没权重 = 第n-1季度上线SPU数 * 埋没指数
  含义：埋没指数越大，说明埋没程度越严重，该品类需要更多翻新机会
  
- **品类权重计算**：
  综合权重 = α * 埋没权重 + β * 近12个月销售SPU数
  其中：α = 0.7（重视埋没程度），β = 0.3（兼顾销量规模）

- **挑选数量分配**：
  品类挑选数量N = round(7000 * 品类综合权重 / 所有品类综合权重总和)


二、其次是各个品类中如何挑选具体的N个spu
    1. SPU质量检查
       - 确保SPU存储了主图向量。
       - 确保SPU在线可售。
       - 筛选掉上一个周期内上线的产品

    2. 每个品类下，以销售额为排序依据，从各个品类中挑选出对应数量的SPU



## step 3: 构建A图集
### A图集构建方案

每个spu关联一个主图，n个场景图，m个sku图
需要以结构化存储的方式保存这些图片，保存图片路径

每个品类的所有spu的图片信息存储在一个JSON文件中，方便后续处理
{
    "spu_id": 123456,
    "main_image": "image_id_1",
    "scene_images": [
        "image_id_2",
        "image_id_3",
        ...
    ],
    "sku_images": [
        "image_id_4",
        "image_id_5",
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

