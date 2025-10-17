"""
主程序流程：
1. 加载配置和初始化日志。
2. 加载或初始化进度跟踪文件。
3. 获取所有待处理的SPU。
4. 遍历每个SPU：
    a. 为SPU下的每张A图采样B图。
    b. 为B图生成提示词。
    c. 调用ComfyUI生成最终的C图。
    d. 更新进度文件。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import torch

from config import Config

from src.feature_extractor import VITFeatureExtractor
from src.b_image_sampling import BImageSampler
from src.tools import load_b_img_path
from src.prompt_generator import generate_prompt
from src.comfyui_runner import ComfyUIRunner


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_progress(progress_file: Path) -> Dict:
    """加载进度文件，如果不存在则返回一个空的字典。"""
    if progress_file.exists():
        logging.info(f"从 {progress_file} 加载进度。")
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logging.info("未找到进度文件，将创建新的进度记录。")
        return {}

def save_progress(progress_data: Dict, progress_file: Path):
    """将进度数据保存到文件。"""
    logging.info(f"保存进度到 {progress_file}。")
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=4)

def get_spu_list(a_image_path: Path, specified_categories:Optional[List]=None) -> Dict[str, List[str]]:
    """
    扫描A图数据集路径，获取所有品类及其下的SPU ID。
    返回: {"category_name": ["spu_id_1", "spu_id_2", ...]}
    """
    spu_by_category = {}
    if specified_categories:
        for category in specified_categories:
            category_dir = a_image_path / category
            if category_dir.exists() and category_dir.is_dir():
                spu_ids = [spu_dir.name for spu_dir in category_dir.iterdir() if spu_dir.is_dir()]
                if spu_ids:
                    spu_by_category[category] = spu_ids
            else:
                logging.warning(f"指定的品类路径不存在或不是目录: {category_dir}")
        logging.info(f"发现 {len(spu_by_category)} 个指定品类。")
        return spu_by_category
    
    for category_dir in a_image_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            spu_ids = [spu_dir.name for spu_dir in category_dir.iterdir() if spu_dir.is_dir()]
            if spu_ids:
                spu_by_category[category_name] = spu_ids
    logging.info(f"发现 {len(spu_by_category)} 个品类。")
    return spu_by_category

def process_spu(spu_id: str, category_name: str, config: Config, progress: Dict, extractor: VITFeatureExtractor, sampler: BImageSampler, comfy_runner: ComfyUIRunner):
    """处理单个SPU的所有A图。"""
    spu_progress = progress.setdefault(category_name, {}).setdefault(spu_id, {})
    spu_path = Path(config.a_image_dataset_path) / category_name / spu_id
    
    if not spu_path.exists():
        logging.warning(f"SPU路径不存在: {spu_path}")
        return

    a_image_files = [f for f in spu_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    for a_image_path in a_image_files:
        a_image_id = a_image_path.stem
        if spu_progress.get(a_image_id, {}).get("c_image_generated"):
            logging.info(f"跳过已完成的A图: {a_image_id}")
            continue
        
        logging.info(f"开始处理A图: {a_image_id} (SPU: {spu_id})")
        process_a_image(
            a_image_id, a_image_path, 
            spu_id, category_name, 
            config, spu_progress, 
            extractor, sampler, comfy_runner)
        # 处理完一张A图后立即保存进度
        save_progress(progress, Path(config.progress_file_path))


def process_a_image(
        a_image_id: str, a_image_path: Path, 
        spu_id: str, category_name: str, 
        config: Config, spu_progress: Dict, 
        extractor: VITFeatureExtractor, sampler: BImageSampler, runner: ComfyUIRunner):
    """处理单张A图的完整流程：获取A图特征向量 -> 采样B图 -> 生成Prompt -> 生成C图。"""
    a_image_progress = spu_progress.setdefault(a_image_id, {})
    a_image_progress["a_image_path"] = str(a_image_path)

    # 1. 加载或计算并保存A图特征向量
    if "a_image_vector_path" not in a_image_progress:
        logging.info(f"计算A图 {a_image_id} 的特征向量...")
        
        # 定义向量文件的保存路径
        vector_dir = Path(config.vectors_path) / category_name / spu_id
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_path = vector_dir / f"{a_image_id}.pt"

        try:
            image = Image.open(a_image_path).convert("RGB")
            vector = extractor.extract_features(image)
            torch.save(vector, vector_path)
            # 在进度文件中记录路径
            a_image_progress["a_image_vector_path"] = str(vector_path)
            logging.info(f"A图 {a_image_id} 特征向量已保存到: {vector_path}")
        except Exception as e:
            logging.error(f"计算或保存A图 {a_image_id} 特征向量时出错: {e}")
            return  None # 如果特征提取失败，则跳过后续步骤, 该SPU判断为未完成


    # 2. 采样B图
    if "b_images" not in a_image_progress or len(a_image_progress['b_images']):
        logging.info(f"为A图 {a_image_id} 采样B图...")
        a_vector = torch.load(a_image_progress["a_image_vector_path"], weights_only=True)
        sampled_b_images_docs = sampler.sample_b_images_for_a(a_vector, category_name, config.num_b_images)
        a_image_progress["b_images"] = []
        for b_img in sampled_b_images_docs:
            b_image_spu_id = b_img["spu_id"]
            b_image_path = load_b_img_path(b_image_spu_id, b_img_dir=Path(config.b_image_dataset_path) / category_name)
            if not b_image_path:
                logging.warning(f"SPU {spu_id} 的A图 {a_image_id} 采样的B图 SPU: {b_image_spu_id} 主图 下载失败。")
                return None # 如果B图下载失败，则跳过后续步骤, 该SPU判断为未完成
            b_img_dict = {
                "b_image_spu_id": b_image_spu_id,
                "b_image_path": b_image_path,
                "score": b_img["score"],
                "prompt": "",
                "c_image_path": ""
            }
            a_image_progress["b_images"].append(b_img_dict)
        
    # 3. 为B图生成Prompt
    if a_image_progress.get("b_images") and not a_image_progress.get("prompts_generated"):
        logging.info(f"为A图 {a_image_id} 的B图生成Prompt...")
        for b_img in a_image_progress["b_images"]:
            if not b_img.get("prompt"):
                try:
                    b_img["prompt"] = generate_prompt(b_img["b_image_path"])
                except Exception as e:
                    logging.error(f"为B图 {b_img['b_image_spu_id']} 生成Prompt时出错: {e}")
                    return None # 如果Prompt生成失败，则跳过后续步骤, 该SPU判断为未完成
        a_image_progress["prompts_generated"] = True


    # 4. 生成C图
    if a_image_progress.get("prompts_generated") and not a_image_progress.get("c_image_generated"):
        logging.info(f"为A图 {a_image_id} 生成C图...")
        if runner.is_server_running() is False:
            logging.error("ComfyUI服务器未运行，无法生成C图。")
            return None # 如果ComfyUI未运行，则跳过后续步骤, 该SPU判断为未完成
        
        all_c_images_generated = True
        for i, b_img_info in enumerate(a_image_progress["b_images"]):
            if b_img_info.get("c_image_path"):
                continue # 如果这张C图已经生成，则跳过

            c_image_dir = Path(config.c_image_dataset_path) / category_name / spu_id
            c_image_filename = f"{a_image_id}_{i}" # 为每张C图生成唯一文件名
            text_prompt2 = f"remove the {category_name.split(' - ')[1]}, only keep the background"
            c_image_path = runner.generate_image(
                a_image_path=a_image_progress["a_image_path"],
                b_image_path=b_img_info["b_image_path"],
                prompt_text=b_img_info["prompt"],
                prompt_text2=text_prompt2,
                output_dir=c_image_dir,
                output_filename=c_image_filename
            )

            if c_image_path:
                b_img_info["c_image_path"] = c_image_path
            else:
                logging.error(f"为A图 {a_image_id} 和B图 {b_img_info['b_image_spu_id']} 生成C图失败。")
                all_c_images_generated = False
                break # 一旦有C图生成失败，就中断当前A图的处理

        if all_c_images_generated:
            a_image_progress["c_image_generated"] = True

    logging.info(f"完成A图处理: {a_image_id}")
    # 保存采样器的使用计数
    sampler.save_usage_counts()
    return spu_progress



def main():
    """主函数， orchestrates the entire process."""
    cfg = Config()
    progress_file = Path(cfg.progress_file_path)

    # 初始化特征提取器
    logging.info("初始化特征提取器...")
    vit_config = {
        "model_path": cfg.model_path,
        "model_name": cfg.model_name
    }
    feature_extractor = VITFeatureExtractor(vit_config)
    logging.info("特征提取器初始化完成。")

    # 初始化B图采样器
    logging.info("初始化B图采样器...")
    b_img_sampler = BImageSampler(
        data_dir=Path(cfg.b_image_dataset_path),
        usage_file=Path(cfg.b_image_usage_file),
        cluster_mapping_file=Path(cfg.cluster_mapping_file),
    )
    logging.info("B图采样器初始化完成。")

    # 初始化ComfyUI运行器
    logging.info("初始化ComfyUI运行器...")
    comfy_runner = ComfyUIRunner(
        server_address=cfg.comfyui_server_address,
        workflow_path=cfg.comfyui_workflow_path,
        node_mapping=cfg.comfyui_node_mapping
    )
    logging.info("ComfyUI运行器初始化完成。")

    # 在开始任务前，进行初始检查
    if not comfy_runner.is_server_running():
        logging.error("ComfyUI 服务器未运行。请先启动 ComfyUI。程序即将退出。")
        return # 直接退出
    logging.info("ComfyUI运行器初始化完成，服务器在线。")

    progress_data = load_progress(progress_file)
    
    spu_by_category = get_spu_list(Path(cfg.a_image_dataset_path), cfg.specified_categories)

    # 原本要7000个，现在只要3000个SPU(所有品类加起来)所以要等比例缩减
    total_spu_count = sum(len(spu_list) for spu_list in spu_by_category.values())
    if total_spu_count > 3000:
        reduction_ratio = 3000 / total_spu_count
        logging.info(f"当前总SPU数为 {total_spu_count}，超过3000，按比例缩减至3000。缩减比例: {reduction_ratio:.4f}")
        for category in spu_by_category:
            original_count = len(spu_by_category[category])
            new_count = max(1, int(original_count * reduction_ratio))  # 确保至少保留1个SPU
            spu_by_category[category] = spu_by_category[category][:new_count]
            logging.info(f"品类 {category} 从 {original_count} 个SPU 缩减到 {new_count} 个SPU。")

    try:
        for category, spu_list in spu_by_category.items():
            logging.info(f"===== 开始处理品类: {category} =====")
            
            # spu_list = spu_list[:5]  # 测试时只处理前10个SPU，正式运行时可移除该行
            for spu_id in spu_list:
                if not comfy_runner.is_server_running():
                    logging.error("检测到 ComfyUI 服务器连接中断。程序将终止。")
                    # 跳出外层循环，以便执行 finally 块
                    raise ConnectionError("ComfyUI 服务器不在线，请先启动ComfyUI服务器。启动命令示例: cd /root/autodl-tmp/ComfyUI && python main.py --listen")
                
                if progress_data.get(category, {}).get(spu_id, {}).get("all_done"):
                    logging.info(f"SPU {spu_id} 已全部处理完成，跳过。")
                    continue
                
                logging.info(f"--- 处理SPU: {spu_id} ---")
                process_spu(spu_id, category, cfg, progress_data, feature_extractor, b_img_sampler, comfy_runner)
                
                # 标记整个SPU已完成
                progress_data.setdefault(category, {}).setdefault(spu_id, {})["all_done"] = True
                save_progress(progress_data, progress_file)

            logging.info(f"===== 品类 {category} 处理完成 =====")

        logging.info("所有任务处理完成！")

    except ConnectionError as e:
        # 捕获我们主动抛出的连接错误
        logging.info(f"因服务器连接问题停止处理: {e}")
    except Exception as e:
        # 捕获其他意外错误
        logging.error(f"处理过程中发生意外错误: {e}", exc_info=True)
    finally:
        # 确保无论程序是正常结束还是因错误中断，都会保存B图使用次数
        logging.info("正在保存B图使用次数...")
        b_img_sampler.save_usage_counts()
        logging.info("B图使用次数已保存。")

    logging.info("所有任务处理完成或已终止。")

if __name__ == "__main__":
    main()
