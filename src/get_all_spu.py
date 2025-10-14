"""
一、数据准备
1. 获取所需品类的spu数据
2. 通过erp api获取spu对应的图片向量，如果有则存储，没有则标记
3. 过滤出带有向量的spu数据
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
    SELECT SPU, 产品分类, 主图ID, 最低价格
    FROM stg_bayshop_litfad_spu
    WHERE 产品分类 IN ({placeholders})
    AND 主图ID IS NOT NULL
    AND 状态 = '正常销售'
    """
    df = pd.read_sql(query, engine, params=tuple(selected_categories))
    return df


def step1_fetch_and_save_spu_data(output_dir='./data/preprocess', output_file='spu_data.csv', selected_categories=None):
    # 数据库连接参数
    db_params = {
        'user': 'zhenggantian',
        'password': '123456',
        'host': '192.168.100.33',
        'port': 3306,
        'db': 'ods'
    }

    if selected_categories is None:
        raise ValueError("请提供所需的产品分类列表")
    
    print(f"目标产品分类: {selected_categories}")

    # 创建数据库引擎
    engine = get_db_engine(**db_params)

    # 获取spu数据
    spu_data = fetch_spu_data(engine, selected_categories)

    # 保存到本地CSV文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
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


def step2_fetch_and_save_image_vectors(
        spu_file_name='spu_data.csv', 
        output_dir='./data/preprocess',
        batch_size=50, delay=0.1):
    """
    获取并保存所有SPU的图片向量
    """
    
    # 读取SPU数据
    spu_file_path = os.path.join(output_dir, spu_file_name)
    if not os.path.exists(spu_file_path):
        raise FileNotFoundError(f"未找到SPU数据文件: {spu_file_path}")
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
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存元数据到CSV
    metadata_path = os.path.join(output_dir, 'spu_metadata.csv')
    final_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
    
    # # 保存向量到NPZ文件（压缩的numpy格式）
    # if vectors:
    #     vectors_path = os.path.join(output_dir, 'spu_vectors.npz')
    #     np.savez_compressed(vectors_path, **vectors)
    #     print(f"向量数据已保存到: {vectors_path}")
    
    # 保存为pickle格式）
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
    
    
def step3_filter_spus_by_vector_existence(data_dir='./data/preprocess', metadata_file_name='spu_metadata.csv', output_file_name='spu_data_with_vectors.csv'):
    """
    过滤出有向量的SPU数据
    """
    metadata_path = os.path.join(data_dir, metadata_file_name)
    # 读取元数据
    metadata_df = pd.read_csv(metadata_path, dtype={'SPU': str})
    
    # 过滤出有向量的SPU
    filtered_df = metadata_df[metadata_df['has_vector'] == True].copy()
    
    print(f"总SPU数: {len(metadata_df)}, 有向量的SPU数: {len(filtered_df)}")
    
    # 保存过滤后的数据
    filtered_path = os.path.join(data_dir, output_file_name)
    filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
    print(f"有向量的SPU数据已保存到: {filtered_path}")
    
    return filtered_df


def get_all_spu_with_categories(data_dir, selected_categories=None):

    step1_fetch_and_save_spu_data(
        output_dir=data_dir, output_file='spu_data.csv')
    
    step2_fetch_and_save_image_vectors(
        spu_file_name='spu_data.csv', 
        output_dir=data_dir, batch_size=50, delay=0.1)
    
    step3_filter_spus_by_vector_existence(
        data_dir=data_dir, 
        metadata_file_name='spu_metadata.csv', 
        output_file_name='spu_data_with_vectors.csv'
    )
    

if __name__ == "__main__":
    # 第一部分：数据准备
    data_dir = './data/preprocess'
    os.makedirs(data_dir, exist_ok=True)
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
    get_all_spu_with_categories(data_dir=data_dir, selected_categories=selected_categories)