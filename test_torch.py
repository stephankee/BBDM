import torch
import sys

def test_cuda():
    print("\n" + "="*60)
    print("PyTorch CUDA 环境完整测试")
    print("="*60)
    
    # 1. 基本信息
    print("\n【基本信息】")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version}")
    print(f"CUDA 编译版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA 不可用!")
        print("可能的原因:")
        print("1. 未安装 NVIDIA 驱动")
        print("2. PyTorch 版本与 CUDA 版本不匹配")
        print("3. 环境变量配置错误")
        return False
    
    # 2. GPU 信息
    print("\n【GPU 信息】")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 计算能力: {props.major}.{props.minor}")
        print(f"  - 总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - 多处理器数量: {props.multi_processor_count}")
    
    # 3. 显存测试
    print("\n【显存状态】")
    device = torch.device("cuda:0")
    print(f"当前设备: {device}")
    print(f"已分配显存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # 4. 计算测试
    print("\n【计算测试】")
    try:
        # CPU vs GPU 速度对比
        import time
        
        size = 5000
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        print(f"✓ CPU 矩阵乘法 ({size}x{size}): {cpu_time:.4f} 秒")
        
        x_gpu = torch.randn(size, size).to(device)
        y_gpu = torch.randn(size, size).to(device)
        torch.cuda.synchronize()
        
        start = time.time()
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"✓ GPU 矩阵乘法 ({size}x{size}): {gpu_time:.4f} 秒")
        print(f"✓ 加速比: {cpu_time/gpu_time:.2f}x")
        
        # 测试梯度计算
        x = torch.randn(100, 100, requires_grad=True).to(device)
        y = (x ** 2).sum()
        y.backward()
        print(f"✓ 梯度计算测试成功")
        
        print(f"\n计算后显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"❌ 计算测试失败: {e}")
        return False
    
    # 5. cuDNN 测试
    print("\n【cuDNN 测试】")
    print(f"cuDNN 是否启用: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    
    try:
        # 简单卷积测试
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        y = conv(x)
        print(f"✓ cuDNN 卷积测试成功: {y.shape}")
    except Exception as e:
        print(f"❌ cuDNN 测试失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过! PyTorch CUDA 环境正常")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    test_cuda()