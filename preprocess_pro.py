"""
预先准备工作
1. 获取所需品类的spu数据
2. 通过erp api获取spu对应的图片向量，如果有则存储，没有则标记
"""

from sqlalchemy import create_engine
import pandas as pd
import os
import asyncio
import aiohttp
import numpy as np

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



async def fetch_and_save_vectors_streaming(spu_list, spu_df, batch_size=50, delay=0.1, save_interval=10):
    """
    流式获取并保存SPU图片向量，边请求边存储
    """
    import time
    
    # 检查是否有断点文件
    checkpoint_file = './data/preprocess/checkpoint.json'
    start_batch = 0
    
    # 初始化存储文件
    output_dir = './data/preprocess'
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储文件路径
    vectors_path = os.path.join(output_dir, 'spu_vectors.npz')
    metadata_path = os.path.join(output_dir, 'spu_metadata.csv')
    
    # 用于累积数据的变量
    all_vectors = {}
    all_metadata = []
    total_success = 0
    total_processed = 0
    
    # 检查断点
    if os.path.exists(checkpoint_file):
        import json
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_batch = checkpoint.get('last_batch', 0) + 1  # 从下一批开始
            total_success = checkpoint.get('total_success', 0)
            total_processed = checkpoint.get('total_processed', 0)
            print(f"从断点继续，起始批次: {start_batch + 1}")
            
            # 加载已有数据
            if os.path.exists(vectors_path):
                existing_vectors = np.load(vectors_path)
                all_vectors = {spu: existing_vectors[spu] for spu in existing_vectors.files}
                print(f"加载已有向量数据: {len(all_vectors)} 个")
            
            if os.path.exists(metadata_path):
                existing_metadata = pd.read_csv(metadata_path, dtype={'SPU': str})
                all_metadata = existing_metadata.to_dict('records')
                print(f"加载已有元数据: {len(all_metadata)} 个")
    
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
                
                # 处理批次结果
                batch_success = process_and_accumulate_batch(batch_results, all_vectors, all_metadata)
                total_success += batch_success
                total_processed += len(batch_results)
                
                print(f"批次 {batch_num + 1} 完成，耗时: {elapsed:.2f}s, 成功: {batch_success}/{len(batch)}")
                print(f"累计进度: {total_processed}/{len(spu_list)}, 成功率: {total_success/total_processed*100:.1f}%")
                
                # 定期保存数据
                if (batch_num + 1) % save_interval == 0:
                    save_current_data(all_vectors, all_metadata, spu_df, output_dir)
                    save_checkpoint_streaming(batch_num, total_success, total_processed, checkpoint_file)
                    print(f"已保存数据，批次: {batch_num + 1}")
                
                # 添加延迟
                if delay > 0:
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                print(f"批次 {batch_num + 1} 失败: {e}")
                # 保存当前数据
                save_current_data(all_vectors, all_metadata, spu_df, output_dir)
                save_checkpoint_streaming(batch_num - 1, total_success, total_processed, checkpoint_file)
                raise
        
        # 最终保存
        save_current_data(all_vectors, all_metadata, spu_df, output_dir)
        
        # 处理完成，删除断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"全部处理完成！总成功: {total_success}/{total_processed}")
        
        return total_success, total_processed


def process_and_accumulate_batch(batch_results, all_vectors, all_metadata):
    """处理批次结果并累积到全局数据中"""
    batch_success = 0
    
    for result in batch_results:
        spu = result['SPU']
        status = result['status']
        vector = result.get('vector')
        
        # 添加到元数据
        all_metadata.append({
            'SPU': spu,
            'status': status,
            'has_vector': status == 'success'
        })
        
        # 如果有向量，添加到向量字典
        if status == 'success' and vector:
            all_vectors[spu] = np.array(vector, dtype=np.float32)
            batch_success += 1
    
    return batch_success


def save_current_data(all_vectors, all_metadata, spu_df, output_dir):
    """保存当前累积的数据"""
    # 保存向量数据
    if all_vectors:
        vectors_path = os.path.join(output_dir, 'spu_vectors.npz')
        np.savez_compressed(vectors_path, **all_vectors)
    
    # 保存元数据
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        # 与原始SPU数据合并
        final_df = spu_df.merge(metadata_df, on='SPU', how='left')
        
        metadata_path = os.path.join(output_dir, 'spu_metadata.csv')
        final_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
    
    # 保存完整数据到pickle
    if all_vectors and all_metadata:
        import pickle
        full_data = {
            'metadata': final_df,
            'vectors': all_vectors,
            'vector_dimension': 768,
            'total_vectors': len(all_vectors)
        }
        pickle_path = os.path.join(output_dir, 'spu_data_complete.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(full_data, f)


def save_checkpoint_streaming(last_batch, total_success, total_processed, checkpoint_file):
    """保存流式处理的断点"""
    import json
    import time
    
    checkpoint = {
        'last_batch': last_batch,
        'total_success': total_success,
        'total_processed': total_processed,
        'timestamp': time.time()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)





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
    
    # 运行流式异步任务
    success_count, total_processed = asyncio.run(
        fetch_and_save_vectors_streaming(
            spu_list, 
            spu_df, 
            batch_size=batch_size, 
            delay=delay, 
            save_interval=10  # 每10个批次保存一次
        )
    )
    
    print(f"流式处理完成！")
    print(f"成功获取向量: {success_count} 个")
    print(f"总计处理: {total_processed} 个")
    print(f"成功率: {success_count/total_processed*100:.1f}%")
    
    # 加载最终结果
    try:
        metadata_df, vectors_dict = load_spu_vectors()
        print(f"最终数据: 元数据 {len(metadata_df)} 条，向量 {len(vectors_dict)} 个")
        return metadata_df, vectors_dict
    except FileNotFoundError:
        print("警告: 未找到最终数据文件")
        return None, None
    
    


# ---------- 工具函数 ----------

def load_spu_vectors(data_dir='./data/preprocess'):
    """
    加载SPU向量数据
    返回: (metadata_df, vectors_dict)
    """
    import pickle
    
    # 方法1: 从pickle文件加载完整数据（推荐）
    pickle_path = os.path.join(data_dir, 'spu_data_complete.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data['metadata'], data['vectors']
    
    # 方法2: 分别加载元数据和向量文件
    metadata_path = os.path.join(data_dir, 'spu_metadata.csv')
    vectors_path = os.path.join(data_dir, 'spu_vectors.npz')
    
    if os.path.exists(metadata_path) and os.path.exists(vectors_path):
        metadata_df = pd.read_csv(metadata_path, dtype={'SPU': str})
        vectors_data = np.load(vectors_path)
        vectors_dict = {spu: vectors_data[spu] for spu in vectors_data.files}
        return metadata_df, vectors_dict
    
    raise FileNotFoundError("未找到向量数据文件")


def get_spu_vector(spu, vectors_dict):
    """
    获取指定SPU的向量
    """
    return vectors_dict.get(str(spu))


def calculate_vector_similarity(vector1, vector2, method='cosine'):
    """
    计算两个向量的相似度
    method: 'cosine', 'euclidean', 'dot'
    """
    if method == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif method == 'euclidean':
        return 1 / (1 + np.linalg.norm(vector1 - vector2))
    elif method == 'dot':
        return np.dot(vector1, vector2)
    else:
        raise ValueError("支持的方法: 'cosine', 'euclidean', 'dot'")


def find_similar_spus(target_spu, vectors_dict, metadata_df, top_k=10, method='cosine'):
    """
    找到与目标SPU最相似的SPU
    """
    target_vector = get_spu_vector(target_spu, vectors_dict)
    if target_vector is None:
        return None
    
    similarities = []
    for spu, vector in vectors_dict.items():
        if spu != str(target_spu):
            sim = calculate_vector_similarity(target_vector, vector, method)
            similarities.append((spu, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回top_k个结果
    result_spus = [spu for spu, sim in similarities[:top_k]]
    result_df = metadata_df[metadata_df['SPU'].isin(result_spus)].copy()
    result_df['similarity'] = [sim for spu, sim in similarities[:top_k]]
    
    return result_df.sort_values('similarity', ascending=False)

# 使用示例
def example_usage():
    """
    示例用法
    """
    # 加载数据
    metadata_df, vectors_dict = load_spu_vectors()

    # 找相似产品
    similar_products = find_similar_spus('12345', vectors_dict, metadata_df, top_k=5)

    # 计算相似度
    vector1 = get_spu_vector('12345', vectors_dict)
    vector2 = get_spu_vector('67890', vectors_dict)
    similarity = calculate_vector_similarity(vector1, vector2)
    print(f"SPU 12345 和 SPU 67890 的相似度: {similarity}")

# ---------- step 2 ----------



if __name__ == "__main__":
    step1_fetch_and_save_spu_data()
    step2_fetch_and_save_image_vectors()