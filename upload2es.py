#!/usr/bin/env python3
"""
上传 es_docs.pkl 数据到 Elasticsearch
"""

import pickle
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging
from urllib3.exceptions import InsecureRequestWarning
import warnings
# 关闭 InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Elasticsearch 配置
ES_HOST = "192.168.100.85"
ES_PORT = 9200
ES_SCHEME = "https"
INDEX_NAME = "auto_image_docs"
# 如果需要身份验证，请设置用户名和密码
ES_USERNAME = "elastic"  # 设置为实际用户名，如 "elastic"
ES_PASSWORD = "123456"  # 设置为实际密码

def load_data(file_path):
    """加载 pickle 文件数据"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"成功加载数据，类型: {type(data)}")
        if hasattr(data, '__len__'):
            logger.info(f"数据长度: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None

def prepare_documents(data):
    """准备文档数据用于上传到 ES"""
    documents = []
    
    if isinstance(data, list):
        for i, item in enumerate(data):
            doc = {
                "_index": INDEX_NAME,
                "_id": i,  # 使用索引作为文档ID
                "_source": item if isinstance(item, dict) else {"content": item}
            }
            documents.append(doc)
    elif isinstance(data, dict):
        # 如果是字典，将其作为单个文档
        doc = {
            "_index": INDEX_NAME,
            "_id": 1,
            "_source": data
        }
        documents.append(doc)
    else:
        # 其他类型，包装成文档
        doc = {
            "_index": INDEX_NAME,
            "_id": 1,
            "_source": {"content": str(data)}
        }
        documents.append(doc)
    
    logger.info(f"准备了 {len(documents)} 个文档")
    return documents

def create_es_client():
    """创建 Elasticsearch 客户端"""
    
    # 尝试多种连接方式
    config = {
            "hosts": [f"https://{ES_HOST}:{ES_PORT}"],
            "basic_auth": (ES_USERNAME, ES_PASSWORD),
            "verify_certs": False  # 如果使用自签名证书，可能需要设置为 False
        }

    try:
    
        es = Elasticsearch(**config)
        
        # 测试连接
        if es.ping():
            logger.info(f"成功连接到 Elasticsearch")
            return es
        else:
            logger.warning(f"配置  ping 失败")
            
    except Exception as e:
        logger.warning(f"配置  连接异常: {str(e)[:100]}...")


    return None

def create_index_if_not_exists(es, index_name):
    """如果索引不存在则创建"""
    try:
        if not es.indices.exists(index=index_name):
            # 创建索引配置，适配我们的数据结构
            index_config = {
                "mappings": {
                    "properties": {
                        "spu_id": {"type": "long"},
                        "image_id": {"type": "long"},
                        "category": {"type": "keyword"},
                        "cluster_id": {"type": "keyword"},
                        "image_vector": {"type": "dense_vector", "dims": 768},  # 假设向量维度为768
                        "used_num": {"type": "integer"},
                        "upload_timestamp": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            es.indices.create(index=index_name, body=index_config)
            logger.info(f"创建索引: {index_name}")
        else:
            logger.info(f"索引已存在: {index_name}")
    except Exception as e:
        logger.error(f"创建索引失败: {e}")

def upload_documents(es, documents):
    """批量上传文档到 ES"""
    try:
        # 添加时间戳
        for doc in documents:
            if isinstance(doc["_source"], dict):
                doc["_source"]["upload_timestamp"] = datetime.now().isoformat()
        
        # 批量上传
        success, failed = bulk(es, documents, chunk_size=100, request_timeout=60)
        logger.info(f"成功上传 {success} 个文档")
        
        if failed:
            logger.warning(f"失败 {len(failed)} 个文档")
            for fail in failed[:5]:  # 只显示前5个失败的
                logger.warning(f"失败文档: {fail}")
        
        return success, failed
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        return 0, []

def main():
    """主函数"""
    logger.info("开始上传数据到 Elasticsearch")
    
    # 1. 加载数据
    data = load_data('data/es_docs.pkl')
    if data is None:
        logger.error("数据加载失败，退出")
        return
    
    # 2. 创建 ES 客户端
    es = create_es_client()
    if es is None:
        logger.error("ES 连接失败，退出")
        return
    
    # 3. 创建索引
    create_index_if_not_exists(es, INDEX_NAME)
    
    # 4. 准备文档
    documents = prepare_documents(data)
    if not documents:
        logger.error("没有文档需要上传")
        return
    
    # 5. 上传文档
    success, failed = upload_documents(es, documents)
    
    logger.info(f"上传完成！成功: {success}, 失败: {len(failed)}")

if __name__ == "__main__":
    main()