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



async def fetch_all_vectors(spu_list, batch_size=50):
    """
    批量获取所有SPU的图片向量，支持分批处理
    """
    # 使用form-data时不需要设置Content-Type，aiohttp会自动处理
    async with aiohttp.ClientSession() as session:
        all_results = []
        
        # 分批处理SPU列表
        for i in range(0, len(spu_list), batch_size):
            batch = spu_list[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}: SPU {i+1}-{min(i+batch_size, len(spu_list))}")
            
            # 获取当前批次的结果
            batch_results = await get_batch_spu_image_vectors(session, batch)
            all_results.extend(batch_results)
        
        return all_results


def step2_fetch_and_save_image_vectors():
    """
    获取并保存所有SPU的图片向量
    """
    spu_file_path = './data/preprocess/spu_data.csv'
    
    # 读取SPU数据
    spu_df = pd.read_csv(spu_file_path, dtype={'SPU': str})
    spu_list = spu_df['SPU'].tolist()
    
    print(f"开始获取 {len(spu_list)} 个SPU的图片向量...")
    
    # 运行异步任务
    results = asyncio.run(fetch_all_vectors(spu_list))
    
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
    # step1_fetch_and_save_spu_data()
    step2_fetch_and_save_image_vectors()