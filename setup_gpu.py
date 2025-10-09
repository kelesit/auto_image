"""
GPU加速环境安装和配置脚本
自动检测GPU环境并安装相应的加速库
"""

import subprocess
import sys
import platform
import os

def run_command(command, description=""):
    """运行命令并显示结果"""
    if description:
        print(f"🔧 {description}")
    
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ 失败")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def check_gpu_availability():
    """检查GPU可用性"""
    print("🔍 检查GPU环境...")
    
    # 检查NVIDIA GPU
    nvidia_gpu = run_command("nvidia-smi", "检查NVIDIA GPU")
    
    # 检查CUDA
    cuda_available = run_command("nvcc --version", "检查CUDA")
    
    return nvidia_gpu, cuda_available

def install_cupy():
    """安装CuPy"""
    print("\n📦 安装CuPy (NVIDIA GPU加速库)...")
    
    # 检查CUDA版本
    try:
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if "release 11" in result.stdout:
            cuda_version = "cu11x"
        elif "release 12" in result.stdout:
            cuda_version = "cu12x"
        else:
            cuda_version = "cu11x"  # 默认
            
        print(f"检测到CUDA版本: {cuda_version}")
        
        # 安装对应版本的CuPy
        if cuda_version == "cu12x":
            command = "pip install cupy-cuda12x"
        else:
            command = "pip install cupy-cuda11x"
            
        return run_command(command, f"安装CuPy ({cuda_version})")
        
    except Exception as e:
        print(f"安装CuPy失败: {e}")
        return False

def install_pytorch_gpu():
    """安装PyTorch GPU版本"""
    print("\n📦 安装PyTorch GPU版本...")
    
    system = platform.system().lower()
    
    if system == "windows":
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif system == "linux":
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:  # macOS
        print("⚠️ macOS不支持CUDA，安装CPU版本的PyTorch")
        command = "pip install torch torchvision torchaudio"
    
    return run_command(command, "安装PyTorch GPU版本")

def install_other_dependencies():
    """安装其他依赖"""
    print("\n📦 安装其他依赖包...")
    
    dependencies = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "aiohttp",
        "sqlalchemy",
        "pymysql"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"安装 {dep}")

def test_gpu_installation():
    """测试GPU安装"""
    print("\n🧪 测试GPU加速库...")
    
    # 测试CuPy
    try:
        import cupy as cp
        print("✅ CuPy导入成功")
        
        # 简单测试
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        print(f"CuPy计算测试: {cp.asnumpy(c)}")
        
    except ImportError:
        print("❌ CuPy不可用")
    except Exception as e:
        print(f"❌ CuPy测试失败: {e}")
    
    # 测试PyTorch
    try:
        import torch
        print("✅ PyTorch导入成功")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU设备数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # 简单测试
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            print(f"PyTorch GPU计算测试: {y.cpu().numpy()}")
        
    except ImportError:
        print("❌ PyTorch不可用")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")

def create_performance_test_script():
    """创建性能测试脚本"""
    test_script = """
import time
import numpy as np

def cpu_vs_gpu_benchmark():
    print("🏃‍♂️ CPU vs GPU 性能对比测试")
    print("=" * 50)
    
    # 测试数据
    size = 2000
    dim = 768
    data = np.random.rand(size, dim).astype(np.float32)
    
    # CPU测试
    print(f"🖥️  CPU计算 {size}x{size} 相似度矩阵...")
    start_time = time.time()
    from sklearn.metrics.pairwise import cosine_similarity
    cpu_result = cosine_similarity(data[:100], data[:100])  # 缩小规模避免太慢
    cpu_time = time.time() - start_time
    print(f"CPU耗时: {cpu_time:.3f}秒")
    
    # GPU测试
    try:
        from gpu_tools import GPUVectorCalculator
        gpu_calc = GPUVectorCalculator()
        
        print(f"🚀 GPU计算 {size}x{size} 相似度矩阵...")
        start_time = time.time()
        gpu_result = gpu_calc.batch_cosine_similarity(data, data)
        gpu_time = time.time() - start_time
        print(f"GPU耗时: {gpu_time:.3f}秒")
        
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time * (100*100) / (size*size)  # 按比例计算
            print(f"🚀 预估加速比: {speedup:.1f}x")
        
    except Exception as e:
        print(f"GPU测试失败: {e}")

if __name__ == "__main__":
    cpu_vs_gpu_benchmark()
"""
    
    with open("gpu_performance_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 性能测试脚本已创建: gpu_performance_test.py")

def main():
    """主安装流程"""
    print("🚀 GPU加速环境自动配置")
    print("=" * 50)
    
    # 检查GPU环境
    nvidia_gpu, cuda_available = check_gpu_availability()
    
    if not nvidia_gpu:
        print("⚠️ 未检测到NVIDIA GPU，将安装CPU版本")
        print("   如果您有GPU但未检测到，请确保安装了NVIDIA驱动")
    
    if not cuda_available:
        print("⚠️ 未检测到CUDA，某些GPU功能可能不可用")
    
    # 安装依赖
    install_other_dependencies()
    
    # 根据环境安装GPU库
    if nvidia_gpu and cuda_available:
        print("\n🎯 检测到完整GPU环境，安装GPU加速库...")
        cupy_success = install_cupy()
        pytorch_success = install_pytorch_gpu()
        
        if not (cupy_success or pytorch_success):
            print("❌ GPU库安装失败，请检查CUDA环境")
    else:
        print("\n🔄 GPU环境不完整，安装CPU版本...")
        run_command("pip install torch torchvision torchaudio", "安装PyTorch CPU版本")
    
    # 测试安装
    test_gpu_installation()
    
    # 创建性能测试脚本
    create_performance_test_script()
    
    print("\n🎉 安装完成！")
    print("\n📋 接下来的步骤:")
    print("1. 运行 python gpu_performance_test.py 测试性能")
    print("2. 运行 python preprocess_gpu.py 开始GPU加速处理")
    print("3. 如果遇到问题，检查NVIDIA驱动和CUDA安装")

if __name__ == "__main__":
    main()
