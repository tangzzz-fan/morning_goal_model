#!/usr/bin/env python3
"""
环境配置验证脚本
用于验证Python环境、PyTorch/TensorFlow、CoreML工具链是否正确安装
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """检查Python版本是否满足要求（3.8+）"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}，需要3.8+")
        return False

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """检查包是否安装及版本"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        
        if min_version and version != "unknown":
            from packaging import version as v
            if v.parse(version) >= v.parse(min_version):
                return True, f"✅ {package_name} {version} (>= {min_version})"
            else:
                return False, f"❌ {package_name} {version} (< {min_version})"
        
        return True, f"✅ {package_name} {version}"
    except ImportError:
        return False, f"❌ {package_name} 未安装"

def check_gpu_support() -> Dict[str, bool]:
    """检查GPU支持情况"""
    results = {}
    
    # PyTorch GPU支持
    try:
        import torch
        results["pytorch_gpu"] = torch.cuda.is_available()
        if results["pytorch_gpu"]:
            print(f"✅ PyTorch GPU支持: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch GPU不可用，将使用CPU")
    except ImportError:
        results["pytorch_gpu"] = False
        print("❌ PyTorch未安装")
    
    # TensorFlow GPU支持
    try:
        import tensorflow as tf
        results["tensorflow_gpu"] = len(tf.config.list_physical_devices('GPU')) > 0
        if results["tensorflow_gpu"]:
            print(f"✅ TensorFlow GPU支持: {tf.config.list_physical_devices('GPU')}")
        else:
            print("⚠️ TensorFlow GPU不可用，将使用CPU")
    except ImportError:
        results["tensorflow_gpu"] = False
        print("❌ TensorFlow未安装")
    
    return results

def check_coreml_tools() -> bool:
    """检查CoreML工具链"""
    try:
        import coremltools
        version = coremltools.__version__
        print(f"✅ CoreML Tools {version}")
        
        # 检查ONNX支持
        try:
            import onnx
            print(f"✅ ONNX {onnx.__version__}")
        except ImportError:
            print("❌ ONNX未安装")
            return False
            
        return True
    except ImportError:
        print("❌ CoreML Tools未安装")
        return False

def main():
    """主函数：执行所有环境检查"""
    print("=" * 60)
    print("iOS目标记录应用 - 端侧NLP分析环境配置验证")
    print("=" * 60)
    
    # 1. Python版本检查
    python_ok = check_python_version()
    
    # 2. 核心包检查
    required_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("datasets", "2.0.0"),
        ("numpy", "1.21.0"),
        ("pandas", "1.3.0"),
        ("scikit-learn", "1.0.0"),
        ("matplotlib", "3.5.0"),
        ("seaborn", "0.11.0"),
        ("tqdm", "4.60.0")
    ]
    
    ml_packages = [
        ("coremltools", "5.0"),
        ("onnx", "1.12"),
        ("onnxruntime", "1.12")
    ]
    
    print("\n📦 核心包检查:")
    core_ok = True
    for package, min_version in required_packages:
        ok, msg = check_package(package, min_version)
        print(f"  {msg}")
        if not ok:
            core_ok = False
    
    print("\n🔧 ML工具链检查:")
    ml_ok = True
    for package, min_version in ml_packages:
        ok, msg = check_package(package, min_version)
        print(f"  {msg}")
        if not ok:
            ml_ok = False
    
    # 3. GPU支持检查
    print("\n🎮 GPU支持检查:")
    gpu_results = check_gpu_support()
    
    # 4. CoreML工具链检查
    print("\n🍎 CoreML工具链检查:")
    coreml_ok = check_coreml_tools()
    
    # 5. 总结
    print("\n📋 环境配置总结:")
    print("=" * 60)
    
    all_ok = python_ok and core_ok and ml_ok and coreml_ok
    
    if all_ok:
        print("✅ 环境配置成功！可以开始iOS端侧NLP分析开发")
        print("\n下一步建议:")
        print("1. 准备训练数据集")
        print("2. 下载预训练模型")
        print("3. 开始模型训练与优化")
    else:
        print("❌ 环境配置存在问题，请根据上述提示修复")
        if not python_ok:
            print("  - 升级Python到3.8+")
        if not core_ok:
            print("  - 安装缺失的核心包")
        if not ml_ok:
            print("  - 安装缺失的ML工具链")
        if not coreml_ok:
            print("  - 安装或更新CoreML工具链")
    
    # 6. 生成环境报告
    print("\n📄 生成环境报告...")
    report = {
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": {},
        "gpu_support": gpu_results,
        "coreml_tools": coreml_ok,
        "overall_status": all_ok
    }
    
    # 获取所有包版本
    all_packages = required_packages + ml_packages
    for package, _ in all_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            report["packages"][package] = version
        except:
            report["packages"][package] = "not_installed"
    
    # 保存报告
    import json
    with open("environment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 环境报告已保存至: environment_report.json")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)