#!/usr/bin/env python3
"""
GPU加速版本启动脚本
快速开始GPU加速的图像处理流水线
"""

import sys
import os

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    required_files = [
        'gpu_tools.py',
        'preprocess_gpu.py',
        'tools.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 文件检查通过")
    return True

def check_gpu_libraries():
    """检查GPU库可用性"""
    print("\n🔍 检查GPU加速库...")
    
    gpu_available = False
    
    # 检查CuPy
    try:
        import cupy as cp
        print("✅ CuPy可用")
        gpu_available = True
    except ImportError:
        print("⚠️ CuPy不可用")
    
    # 检查PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch GPU可用 - {torch.cuda.get_device_name()}")
            gpu_available = True
        else:
            print("⚠️ PyTorch GPU不可用")
    except ImportError:
        print("⚠️ PyTorch不可用")
    
    if not gpu_available:
        print("\n⚠️ 未检测到GPU加速库，将使用CPU模式")
        print("💡 运行 python setup_gpu.py 安装GPU加速环境")
    else:
        print("\n🚀 GPU加速环境就绪！")
    
    return gpu_available

def show_menu():
    """显示操作菜单"""
    print("\n" + "="*60)
    print("🚀 GPU加速图像处理流水线")
    print("="*60)
    print("请选择操作:")
    print("1. 🏃‍♂️ 运行完整GPU加速流水线")
    print("2. 🧪 GPU性能基准测试")
    print("3. 🔧 安装GPU环境")
    print("4. 📊 仅运行数据清洗")
    print("5. 🎯 仅运行多样化选择")
    print("6. 📈 向量分布分析")
    print("0. 退出")
    print("="*60)

def run_full_pipeline():
    """运行完整流水线"""
    print("\n🚀 启动完整GPU加速流水线...")
    try:
        from preprocess_gpu import run_gpu_accelerated_pipeline
        run_gpu_accelerated_pipeline()
    except Exception as e:
        print(f"❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()

def run_benchmark():
    """运行性能测试"""
    print("\n🏃‍♂️ 启动GPU性能基准测试...")
    try:
        from preprocess_gpu import gpu_performance_benchmark
        gpu_performance_benchmark()
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")

def install_gpu_environment():
    """安装GPU环境"""
    print("\n🔧 启动GPU环境安装...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "setup_gpu.py"], capture_output=False)
        if result.returncode == 0:
            print("✅ GPU环境安装完成")
        else:
            print("❌ GPU环境安装失败")
    except Exception as e:
        print(f"❌ 安装失败: {e}")

def run_cleaning_only():
    """仅运行数据清洗"""
    print("\n🧹 启动GPU加速数据清洗...")
    try:
        from preprocess_gpu import gpu_clean_spus_optimized
        gpu_clean_spus_optimized()
    except Exception as e:
        print(f"❌ 清洗失败: {e}")

def run_selection_only():
    """仅运行多样化选择"""
    print("\n🎯 启动GPU加速多样化选择...")
    try:
        from preprocess_gpu import gpu_select_diverse_spus_optimized
        gpu_select_diverse_spus_optimized()
    except Exception as e:
        print(f"❌ 选择失败: {e}")

def run_analysis_only():
    """仅运行向量分析"""
    print("\n📈 启动向量分布分析...")
    try:
        from preprocess_gpu import batch_similarity_analysis
        from tools import load_spu_vectors
        
        data_dir = './data/preprocess'
        _, vectors_dict = load_spu_vectors(data_dir)
        batch_similarity_analysis(vectors_dict, sample_size=2000)
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    """主函数"""
    print("🎯 GPU加速图像处理系统启动器")
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请确保所有必要文件存在")
        return
    
    # GPU检查
    gpu_available = check_gpu_libraries()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选择 (0-6): ").strip()
            
            if choice == '0':
                print("👋 再见！")
                break
            elif choice == '1':
                run_full_pipeline()
            elif choice == '2':
                run_benchmark()
            elif choice == '3':
                install_gpu_environment()
            elif choice == '4':
                run_cleaning_only()
            elif choice == '5':
                run_selection_only()
            elif choice == '6':
                run_analysis_only()
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 执行出错: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
