"""
预先准备工作
一、数据准备
1. 获取所需品类的spu数据
2. 通过erp api获取spu对应的图片向量，如果有则存储，没有则标记
3. 过滤出带有向量的spu数据

二、数据清洗和筛选
1. 根据品类分层
2. 每一层中遍历图片向量，筛选掉相似度过高的产品，保留多样性
3. 使用使用minmax算法，确保每个品类至少有N个产品，N由该品类的spu数占比决定，总共需要有5000个产品


三、根据品类分层，分别进行kmeans聚类，记录下每个品类的N个簇心点，作为该品类的N中类别心产品
N设定为该品类spu数的10%或至少10个

将所有数据存入elasticsearch，供后续使用

"""

from sqlalchemy import create_engine
import pandas as pd
import os
import asyncio
import aiohttp
import numpy as np

from tools import load_spu_vectors, get_spu_vector, calculate_vector_similarity, find_similar_spus, find_similar_spus_above_threshold
from gpu_tools import gpu_accelerated_cleaning, gpu_accelerated_sampling, GPUVectorCalculator

# 连接数据库
def get_db_engine(user, password, host, port, db):
    db_url = f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4'
    engine = create_engine(db_url, pool_size=20, max_overflow=0)
    return engine

# 获取指定品类的spu数据
def fetch_spu_data(engine, selected_categories):
    placeholders = ','.join(['%s'] * len(selected_categories))
    query = f"""
    SELECT SPU, 产品分类, 主图ID
    FROM stg_bayshop_litfad_spu
    WHERE 产品分类 IN ({placeholders})
    AND 主图ID IS NOT NULL
    AND 状态 = '正常销售'
    """
    df = pd.read_sql(query, engine, params=tuple(selected_categories))
    return df


def step1_fetch_and_save_spu_data():
    # 数据库连接参数
    db_params = {
        'user': 'zhenggantian',
        'password': '123456',
        'host': '192.168.100.33',
        'port': 3306,
        'db': 'ods'
    }

    # 需要获取的品类
    selected_categories = [
        '90 - Coffee Tables',
        '114 - Benches',
        '88 - TV Stands & Entertainment Centers',
        '91 - End & Side Tables',
        '83 - Sofa',
        '176 - Room Dividers',
        '92 - Cabinets & Chests',
        '82 - Accent Chairs',
        '218 - Plant Stands & Tables',
        '105 - Hall Trees & Coat Racks',
        '104 - Console Tables',
        '85 - Bookcases'
    ]

    # 创建数据库引擎
    engine = get_db_engine(**db_params)

    # 获取spu数据
    spu_data = fetch_spu_data(engine, selected_categories)

    # 保存到本地CSV文件
    output_dir = './data/preprocess'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'spu_data.csv')
    spu_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"SPU数据已保存到 {output_path}")



# ---------- step 2 ----------

async def get_batch_spu_image_vectors(session, spu_list):
    """
    批量获取多个SPU的图片向量
    params:
        session: aiohttp session
        spu_list: SPU列表
    """
    try:
        url = "https://erp.baycheer.com/api/ImageAi/getSpuImageVector"
        # 将SPU列表转换为逗号分隔的字符串
        spu_ids = ','.join(map(str, spu_list))
        
        # 使用form-data格式发送请求
        form_data = aiohttp.FormData()
        form_data.add_field('app_id', '392028')
        form_data.add_field('app_token', 'WRFDN0O98N7QP91HMTVHL9HE58G32UCD')
        form_data.add_field('spu_id', spu_ids)
        
        async with session.post(url, data=form_data) as response:
            if response.status == 200:
                result = await response.json()
                
                if result.get('code') == 0:
                    data = result.get('data', {})
                    success_data = data.get('success', {})
                    error_data = data.get('error', [])
                    
                    results = []
                    
                    # 处理成功的SPU
                    for spu_id, vector_info in success_data.items():
                        results.append({
                            'SPU': spu_id,
                            'vector': vector_info.get('image_vector'),
                            'status': 'success'
                        })
                    
                    # 处理失败的SPU
                    for spu_id in error_data:
                        results.append({
                            'SPU': spu_id,
                            'vector': None,
                            'status': 'no_vector'
                        })
                    
                    # 处理请求中但没有在响应中的SPU
                    response_spus = set(success_data.keys()) | set(error_data)
                    for spu in spu_list:
                        if str(spu) not in response_spus:
                            results.append({
                                'SPU': str(spu),
                                'vector': None,
                                'status': 'missing_in_response'
                            })
                    
                    return results
                else:
                    # API返回错误码
                    return [{
                        'SPU': str(spu),
                        'vector': None,
                        'status': f'api_error: {result.get("msg", "unknown")}'
                    } for spu in spu_list]
            else:
                return [{
                    'SPU': str(spu),
                    'vector': None,
                    'status': f'http_error_{response.status}'
                } for spu in spu_list]
                
    except Exception as e:
        return [{
            'SPU': str(spu),
            'vector': None,
            'status': f'error_{str(e)}'
        } for spu in spu_list]



async def fetch_all_vectors(spu_list, batch_size=50, delay=0.1, checkpoint_interval=100):
    """
    批量获取所有SPU的图片向量，支持分批处理和断点续传
    """
    import time
    
    # 检查是否有断点文件
    checkpoint_file = './data/preprocess/checkpoint.json'
    start_batch = 0
    all_results = []
    
    if os.path.exists(checkpoint_file):
        import json
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_batch = checkpoint.get('last_batch', 0)
            print(f"从断点继续，起始批次: {start_batch + 1}")
    
    # 配置session超时
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        total_batches = len(spu_list) // batch_size + (1 if len(spu_list) % batch_size else 0)
        
        # 分批处理SPU列表
        for i in range(start_batch * batch_size, len(spu_list), batch_size):
            batch_num = i // batch_size
            batch = spu_list[i:i + batch_size]
            
            print(f"处理批次 {batch_num + 1}/{total_batches}: SPU {i+1}-{min(i+batch_size, len(spu_list))}")
            
            try:
                # 获取当前批次的结果
                start_time = time.time()
                batch_results = await get_batch_spu_image_vectors(session, batch)
                elapsed = time.time() - start_time
                
                all_results.extend(batch_results)
                
                # 统计当前批次成功率
                success_count = sum(1 for r in batch_results if r.get('status') == 'success')
                print(f"批次 {batch_num + 1} 完成，耗时: {elapsed:.2f}s, 成功: {success_count}/{len(batch)}")
                
                # 保存断点
                if (batch_num + 1) % checkpoint_interval == 0:
                    save_checkpoint(batch_num, all_results, checkpoint_file)
                    print(f"已保存断点: 批次 {batch_num + 1}")
                
                # 添加延迟，避免对服务器造成压力
                if delay > 0:
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                print(f"批次 {batch_num + 1} 失败: {e}")
                # 保存当前进度
                save_checkpoint(batch_num - 1, all_results, checkpoint_file)
                raise
        
        # 处理完成，删除断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
        return all_results


def save_checkpoint(last_batch, results, checkpoint_file):
    """保存处理断点"""
    import json
    import time
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    
    checkpoint = {
        'last_batch': last_batch,
        'timestamp': time.time(),
        'processed_count': len(results)
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)
    
    # 同时保存已处理的结果
    if results:
        temp_results_file = './data/preprocess/temp_results.json'
        with open(temp_results_file, 'w') as f:
            json.dump(results, f)


def step2_fetch_and_save_image_vectors(batch_size=50, delay=0.1):
    """
    获取并保存所有SPU的图片向量
    """
    spu_file_path = './data/preprocess/spu_data.csv'
    
    # 读取SPU数据
    spu_df = pd.read_csv(spu_file_path, dtype={'SPU': str})
    spu_list = spu_df['SPU'].tolist()
    
    print(f"开始获取 {len(spu_list)} 个SPU的图片向量...")
    print(f"预估批次数: {len(spu_list) // batch_size + 1}")
    print(f"预估数据量: {len(spu_list) * 768 * 4 / (1024**2):.1f}MB")
    
    # 检查是否存在临时结果文件
    temp_results_file = './data/preprocess/temp_results.json'
    if os.path.exists(temp_results_file):
        print("发现临时结果文件，是否从断点继续？")
        response = input("输入 'y' 继续，其他键重新开始: ")
        if response.lower() != 'y':
            os.remove(temp_results_file)
            checkpoint_file = './data/preprocess/checkpoint.json'
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
    
    # 运行异步任务
    results = asyncio.run(fetch_all_vectors(spu_list, batch_size=batch_size, delay=delay))
    
    # 处理结果
    success_count = 0
    vector_data = []
    
    for result in results:
        if isinstance(result, dict):
            vector_data.append(result)
            if result['status'] == 'success':
                success_count += 1
        else:
            print(f"异常结果: {result}")
    
    # 分离向量数据和元数据
    metadata = []
    vectors = {}
    
    for result in vector_data:
        spu = result['SPU']
        status = result['status']
        vector = result.get('vector')
        
        metadata.append({
            'SPU': spu,
            'status': status,
            'has_vector': status == 'success'
        })
        
        if status == 'success' and vector:
            vectors[spu] = np.array(vector, dtype=np.float32)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(metadata)
    
    # 合并原始数据和元数据
    final_df = spu_df.merge(results_df, on='SPU', how='left')
    
    # 保存结果
    output_dir = './data/preprocess'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存元数据到CSV
    metadata_path = os.path.join(output_dir, 'spu_metadata.csv')
    final_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
    
    # 保存向量到NPZ文件（压缩的numpy格式）
    if vectors:
        vectors_path = os.path.join(output_dir, 'spu_vectors.npz')
        np.savez_compressed(vectors_path, **vectors)
        print(f"向量数据已保存到: {vectors_path}")
    
    # 也可以保存为pickle格式（包含完整信息）
    import pickle
    full_data = {
        'metadata': final_df,
        'vectors': vectors,
        'vector_dimension': len(next(iter(vectors.values()))) if vectors else 0
    }
    pickle_path = os.path.join(output_dir, 'spu_data_complete.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(full_data, f)
    
    print(f"处理完成！")
    print(f"成功获取向量: {success_count} 个")
    print(f"总计处理: {len(vector_data)} 个")
    print(f"元数据已保存到: {metadata_path}")
    print(f"完整数据已保存到: {pickle_path}")
    if vectors:
        print(f"向量维度: {len(next(iter(vectors.values())))}")
    
    return final_df, vectors
    
    
def step3_filter_spus_by_vector_existence():
    """
    过滤出有向量的SPU数据
    """
    data_dir = './data/preprocess'
    metadata_path = os.path.join(data_dir, 'spu_metadata.csv')
    # 读取元数据
    metadata_df = pd.read_csv(metadata_path, dtype={'SPU': str})
    
    # 过滤出有向量的SPU
    filtered_df = metadata_df[metadata_df['has_vector'] == True].copy()
    
    print(f"总SPU数: {len(metadata_df)}, 有向量的SPU数: {len(filtered_df)}")
    
    # 保存过滤后的数据
    filtered_path = os.path.join(data_dir, 'spu_data_with_vectors.csv')
    filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
    print(f"有向量的SPU数据已保存到: {filtered_path}")
    
    return filtered_df


"""
第二部分
数据清洗和筛选
1. 根据品类分层
2. 每一层中遍历图片向量，筛选掉相似度过高的产品，保留多样性
3. 使用使用minmax算法，确保每个品类至少有N个产品，N由该品类的spu数占比决定，总共需要有5000个产品
"""

def clean_spus():
    # 读取有向量的SPU数据
    data_dir = './data/preprocess'
    filtered_path = os.path.join(data_dir, 'spu_data_with_vectors.csv')
    filtered_df = pd.read_csv(filtered_path, dtype={'SPU': str})
    all_spu_num = len(filtered_df)
    
    # 加载向量数据
    _, vectors_dict = load_spu_vectors(data_dir)

    categories = filtered_df['产品分类'].unique()
    abolished_list = []  # 用列表收集被筛选掉的SPU数据
    print(f"发现 {len(categories)} 个不同的产品分类")
    
    for category in categories:
        category_df = filtered_df[filtered_df['产品分类'] == category]
        category_spu_num = len(category_df)
        target_spu_num = max(10, int((category_spu_num / all_spu_num) * 5000))
        print(f"处理分类 '{category}'，包含 {category_spu_num} 个SPU，目标筛选到 {target_spu_num} 个")
        print(f" step 1: 清洗相似度过高的SPU")
        
        similar_threshold = 0.95
        abolished_spus = set()  # 使用集合存储已被筛选的SPU，提高查找效率
        processed_count = 0
        
        for spu in category_df['SPU']:
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"   处理进度: {processed_count}/{category_spu_num}, 已清洗: {len(abolished_spus)}")
                
            if spu in abolished_spus:
                continue  # 已经被清洗掉，跳过
            
            similar_df = find_similar_spus_above_threshold(spu, vectors_dict, category_df, threshold=similar_threshold)
            if len(similar_df) > 0:
                abolished_list.append(similar_df)
                # 将相似的SPU加入到已筛选集合中
                abolished_spus.update(similar_df['SPU'].tolist())
                print(f"   发现 {len(similar_df)} 个与SPU {spu} 相似度>{similar_threshold}的产品")

        
        print(f" 分类 '{category}' 清洗完成，清洗掉 {len(abolished_spus)} 个相似SPU")

    # 合并所有被筛选掉的SPU数据
    if abolished_list:
        abolished_df = pd.concat(abolished_list, ignore_index=True)
        # 以SPU为唯一键，去重
        abolished_df = abolished_df.drop_duplicates(subset=['SPU'])
    else:
        abolished_df = pd.DataFrame(columns=filtered_df.columns)
    
    cleaned_df = filtered_df[~filtered_df['SPU'].isin(abolished_df['SPU'])].copy()
    print(f"\n=== 清洗阶段完成 ===")
    print(f"原始SPU数: {all_spu_num}")
    print(f"清洗后剩余SPU数: {len(cleaned_df)}")
    print(f"清洗掉的SPU数: {len(abolished_df) if len(abolished_df) > 0 else 0}")
    
    # 保存清洗后的数据
    cleaned_df.to_csv(os.path.join(data_dir, 'spu_data_cleaned.csv'), index=False, encoding='utf-8-sig')
    
    return cleaned_df


def select_diverse_spus(target_total=5000):
    """
    从清洗后的数据中选择5000个多样化的SPU
    按品类比例分配，并使用多样化采样算法
    """
    from tools import sample_diverse_spus
    
    data_dir = './data/preprocess'
    cleaned_path = os.path.join(data_dir, 'spu_data_cleaned.csv')
    
    if not os.path.exists(cleaned_path):
        print("未找到清洗后的数据文件，请先运行clean_spus()")
        return None
    
    cleaned_df = pd.read_csv(cleaned_path, dtype={'SPU': str})
    _, vectors_dict = load_spu_vectors(data_dir)
    
    print(f"\n=== 多样化选择阶段 ===")
    print(f"从 {len(cleaned_df)} 个清洗后的SPU中选择 {target_total} 个")
    
    categories = cleaned_df['产品分类'].unique()
    selected_list = []
    
    for category in categories:
        category_df = cleaned_df[cleaned_df['产品分类'] == category]
        category_count = len(category_df)
        
        # 按比例分配目标数量，但至少保证每个类别有10个
        target_count = max(10, int((category_count / len(cleaned_df)) * target_total))
        
        # 如果某个类别的SPU数量不足目标数量，就全部选择
        actual_target = min(target_count, category_count)
        
        print(f"分类 '{category}': {category_count} -> {actual_target} 个SPU")
        
        # 使用多样化采样
        selected_category_df = sample_diverse_spus(
            category_df, vectors_dict, actual_target, method='farthest_first'
        )
        
        selected_list.append(selected_category_df)
    
    # 合并所有选中的SPU
    selected_df = pd.concat(selected_list, ignore_index=True)
    
    # 如果总数超过目标，进行最终筛选
    if len(selected_df) > target_total:
        print(f"当前选中 {len(selected_df)} 个，需要进一步筛选到 {target_total} 个")
        selected_df = sample_diverse_spus(
            selected_df, vectors_dict, target_total, method='farthest_first'
        )
    
    print(f"\n=== 最终选择结果 ===")
    print(f"最终选中SPU数: {len(selected_df)}")
    
    # 按类别统计
    category_stats = selected_df['产品分类'].value_counts()
    for category, count in category_stats.items():
        print(f"  {category}: {count} 个")
    
    # 保存最终结果
    final_path = os.path.join(data_dir, 'spu_data_final_5000.csv')
    selected_df.to_csv(final_path, index=False, encoding='utf-8-sig')
    print(f"最终结果已保存到: {final_path}")
    
    return selected_df

if __name__ == "__main__":
    # 第一部分：数据准备
    # step1_fetch_and_save_spu_data()
    # step2_fetch_and_save_image_vectors()
    # step3_filter_spus_by_vector_existence()

    # 第二部分：数据清洗和筛选
    print("开始数据清洗...")
    cleaned_df = clean_spus()
    
    print("\n开始多样化选择...")
    selected_df = select_diverse_spus(target_total=5000)
    
    if selected_df is not None:
        print(f"\n✅ 完成！已成功选择 {len(selected_df)} 个多样化的SPU")
        print("输出文件:")
        print("  - spu_data_cleaned.csv: 清洗后的数据")
        print("  - spu_data_final_5000.csv: 最终选择的5000个SPU")



