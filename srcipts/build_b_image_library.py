import pickle
import logging
from pathlib import Path
from typing import Dict
import json
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
def split_es_doc_by_category():
    """
    按品类拆分 ES 文档列表。
    返回: { "category_name": [doc1, doc2, ...], ... }
    """
    es_docs_file_path = '/root/auto_image/data/preprocess/es_docs.pkl'
    save_dir = '/root/autodl-tmp/B_image_dataset/es_docs/'


    es_docs = load_data(es_docs_file_path)
    # 将传入的字符串路径转换为 Path 对象
    save_dir_path = Path(save_dir)

    category_dict = {}
    for doc in es_docs:
        category = doc.get('category', '未知品-未知品类')
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(doc)
    
    # 拆分完成后，一次性保存每个品类的文件
    for category, docs in category_dict.items():
        # 使用 Path 对象创建目录
        save_dir_path.mkdir(parents=True, exist_ok=True)
        # 使用 Path 对象构建文件路径
        category_file = save_dir_path / f'{category}.pkl'
        with open(category_file, 'wb') as f:
            pickle.dump(docs, f)
        logger.info(f"已保存品类 '{category}' 的文档到 {category_file}")


    logger.info(f"按品类拆分完成，共 {len(category_dict)} 个品类。")
    return category_dict