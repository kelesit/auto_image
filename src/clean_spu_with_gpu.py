"""
二、数据清洗和筛选
1. 根据品类分层
2. 每一层中遍历图片向量，筛选掉相似度过高的产品，保留多样性

三、根据品类分层，分别进行kmeans聚类，记录下每个品类的N个簇心点，作为该品类的N中类别心产品
N设定为该品类spu数的10%或至少10个

"""
import pandas as pd
import os
import numpy as np
import time
from typing import Dict, List, Tuple

from tools import load_spu_vectors
from gpu_tools import GPUVectorCalculator, gpu_accelerated_cleaning, gpu_accelerated_sampling


# ---------- GPU加速版本的核心函数 ----------

def gpu_clean_spus_optimized(data_dir=None, filtered_path=None):
    """
    GPU优化版本的SPU清洗
    使用批量GPU计算大幅提升性能
    """
    print("🚀 开始GPU加速的数据清洗...")
    
    # 读取有向量的SPU数据
    
    filtered_df = pd.read_csv(filtered_path, dtype={'SPU': str})
    all_spu_num = len(filtered_df)
    
    # 加载向量数据
    _, vectors_dict = load_spu_vectors(data_dir)
    
    categories = filtered_df['产品分类'].unique()
    all_abolished_list = []
    
    print(f"发现 {len(categories)} 个不同的产品分类")
    print("初始化GPU计算器...")
    
    # 初始化GPU计算器
    gpu_calc = GPUVectorCalculator()
    
    total_start_time = time.time()
    
    for i, category in enumerate(categories):
        category_start_time = time.time()
        category_df = filtered_df[filtered_df['产品分类'] == category]
        category_spu_num = len(category_df)
        target_spu_num = max(10, int((category_spu_num / all_spu_num) * 5000))
        
        print(f"\n📊 处理分类 [{i+1}/{len(categories)}] '{category}'")
        print(f"   包含 {category_spu_num} 个SPU，目标筛选到 {target_spu_num} 个")
        
        if category_spu_num <= 50:
            print("   数量较少，使用传统方法...")
            # 对于小数据集，直接使用传统方法
            from tools import find_similar_spus_above_threshold
            similar_threshold = 0.95
            abolished_spus = set()
            
            for spu in category_df['SPU']:
                if spu in abolished_spus:
                    continue
                similar_df = find_similar_spus_above_threshold(spu, vectors_dict, category_df, threshold=similar_threshold)
                if len(similar_df) > 0:
                    all_abolished_list.append(similar_df)
                    abolished_spus.update(similar_df['SPU'].tolist())
        else:
            print("   🎯 使用GPU批量相似度计算...")
            # 使用GPU加速版本
            abolished_df = gpu_accelerated_cleaning(vectors_dict, category_df, threshold=0.95)
            if len(abolished_df) > 0:
                all_abolished_list.append(abolished_df)
        
        category_time = time.time() - category_start_time
        print(f"   ⏱️  分类处理完成，耗时: {category_time:.2f}秒")
    
    # 合并所有被筛选掉的SPU数据
    if all_abolished_list:
        abolished_df = pd.concat(all_abolished_list, ignore_index=True)
        abolished_df = abolished_df.drop_duplicates(subset=['SPU'])
    else:
        abolished_df = pd.DataFrame(columns=filtered_df.columns)
    
    cleaned_df = filtered_df[~filtered_df['SPU'].isin(abolished_df['SPU'])].copy()
    
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 === GPU加速清洗阶段完成 ===")
    print(f"⏰ 总耗时: {total_time:.2f}秒")
    print(f"📈 原始SPU数: {all_spu_num}")
    print(f"✅ 清洗后剩余SPU数: {len(cleaned_df)}")
    print(f"🗑️  清洗掉的SPU数: {len(abolished_df)}")
    print(f"📊 清洗效率: {all_spu_num/total_time:.0f} SPU/秒")
    
    # 保存清洗后的数据
    cleaned_df.to_csv(os.path.join(data_dir, 'spu_data_cleaned_gpu.csv'), index=False, encoding='utf-8-sig')
    abolished_df.to_csv(os.path.join(data_dir, 'spu_data_abolished_gpu.csv'), index=False, encoding='utf-8-sig')

    return cleaned_df

# ---------- 主执行流程 ----------

def run_gpu_spu_cleaner_pipeline(data_dir, filtered_path):
    """运行完整的GPU加速SPU清洗流水线"""
    print("启动GPU加速的完整数据处理流水线 ")
    print("=" * 60)
    
    pipeline_start_time = time.time()
    
    try:
        # GPU加速的数据清洗
        print("\n🔥 spu 数据清洗")
        cleaned_df = gpu_clean_spus_optimized(data_dir=data_dir, filtered_path=filtered_path)
        
        pipeline_time = time.time() - pipeline_start_time
        
        print(f"\n🎉🎉🎉 GPU加速流水线执行完成！🎉🎉🎉")
        print("=" * 60)
        print(f"⏰ 总执行时间: {pipeline_time:.2f}秒 ({pipeline_time/60:.1f}分钟)")
        
        if cleaned_df is not None:
            print("📁 输出文件:")
            print("  - spu_data_cleaned_gpu.csv: GPU清洗后的数据")
            print("  - spu_data_abolished_gpu.csv: 被清洗掉的数据")
            
    except Exception as e:
        print(f"❌ 流水线执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行完整的GPU加速SPU清洗流水线
    data_dir = './data/preprocess'
    filtered_path = os.path.join(data_dir, 'spu_data_with_vectors.csv')
    run_gpu_spu_cleaner_pipeline(data_dir, filtered_path)
