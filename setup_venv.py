#!/usr/bin/env python3
"""
虚拟环境配置脚本
支持Windows和macOS的Python虚拟环境创建与管理
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def get_venv_path():
    """获取虚拟环境路径"""
    return Path.cwd() / "venv"

def check_python_availability():
    """检查Python可用性"""
    system = platform.system()
    
    # Windows系统
    if system == "Windows":
        python_commands = ["python", "python3", "py"]
    else:  # macOS/Linux
        python_commands = ["python3", "python"]
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.strip() or result.stderr.strip()
                print(f"✅ 找到Python: {cmd} - {version_line}")
                return cmd
        except FileNotFoundError:
            continue
    
    return None

def create_virtual_env(python_cmd):
    """创建虚拟环境"""
    venv_path = get_venv_path()
    
    print(f"\n🔄 正在创建虚拟环境: {venv_path}")
    
    try:
        # 创建虚拟环境
        result = subprocess.run([python_cmd, "-m", "venv", str(venv_path)], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 虚拟环境创建失败: {result.stderr}")
            return False
        
        print("✅ 虚拟环境创建成功")
        return True
        
    except Exception as e:
        print(f"❌ 创建虚拟环境时出错: {e}")
        return False

def get_venv_python():
    """获取虚拟环境中的Python路径"""
    venv_path = get_venv_path()
    system = platform.system()
    
    if system == "Windows":
        python_path = venv_path / "Scripts" / "python.exe"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # macOS/Linux
        python_path = venv_path / "bin" / "python"
        pip_path = venv_path / "bin" / "pip"
    
    return python_path, pip_path

def install_packages(python_path, pip_path):
    """安装必要的包"""
    print("\n📦 开始安装必要的包...")
    
    # 升级pip
    print("🔄 升级pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                   check=True)
    
    # 基础包
    base_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "jupyter",
        "ipykernel"
    ]
    
    # ML相关包
    ml_packages = [
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "tokenizers"
    ]
    
    # CoreML工具链
    coreml_packages = [
        "coremltools==6.3.0",
        "onnx>=1.12",
        "onnxruntime>=1.12"
    ]
    
    all_packages = base_packages + ml_packages + coreml_packages
    
    # 分批安装，避免内存问题
    batch_size = 5
    for i in range(0, len(all_packages), batch_size):
        batch = all_packages[i:i+batch_size]
        print(f"\n📦 安装第 {i//batch_size + 1} 批包: {', '.join(batch)}")
        
        try:
            subprocess.run([str(pip_path), "install"] + batch, check=True)
            print(f"✅ 第 {i//batch_size + 1} 批安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 第 {i//batch_size + 1} 批安装失败: {e}")
            return False
    
    return True

def create_activation_scripts():
    """创建激活脚本"""
    venv_path = get_venv_path()
    
    # Windows激活脚本
    if platform.system() == "Windows":
        activate_bat = venv_path / "Scripts" / "activate.bat"
        activate_ps1 = venv_path / "Scripts" / "Activate.ps1"
        
        # 创建简单的激活脚本
        desktop_activate = Path.cwd() / "activate_venv.bat"
        with open(desktop_activate, "w") as f:
            f.write(f'@echo off\necho "激活虚拟环境..."\ncall "{activate_bat}"\necho "虚拟环境已激活"\ncmd /k')
        
        print(f"✅ Windows激活脚本已创建: {desktop_activate}")
    
    else:  # macOS/Linux
        activate_script = venv_path / "bin" / "activate"
        
        # 创建激活脚本
        desktop_activate = Path.cwd() / "activate_venv.sh"
        with open(desktop_activate, "w") as f:
            f.write(f'#!/bin/bash\necho "激活虚拟环境..."\nsource "{activate_script}"\necho "虚拟环境已激活"\nexec bash')
        
        # 添加执行权限
        os.chmod(desktop_activate, 0o755)
        print(f"✅ macOS/Linux激活脚本已创建: {desktop_activate}")

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """# iOS目标记录应用 - 端侧NLP分析
# 基础包
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
jupyter
ipykernel

# ML相关包
torch>=2.0.0
torchvision
torchaudio
transformers>=4.30.0
datasets>=2.0.0
tokenizers

# CoreML工具链
coremltools>=5.0
onnx>=1.12
onnxruntime>=1.12

# 开发工具
pytest
black
flake8
mypy
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ requirements.txt 已创建")

def create_project_structure():
    """创建项目目录结构"""
    directories = [
        "data",
        "data/raw",
        "data/processed", 
        "data/annotations",
        "models",
        "models/pretrained",
        "models/trained",
        "models/optimized",
        "models/coreml",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/coreml",
        "notebooks",
        "scripts",
        "tests",
        "docs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # 创建__init__.py文件
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()
    
    print("✅ 项目目录结构已创建")

def main():
    """主函数"""
    print("=" * 60)
    print("iOS目标记录应用 - 虚拟环境配置工具")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {Path.cwd()}")
    
    # 1. 检查Python可用性
    print("\n🔍 检查Python可用性...")
    python_cmd = check_python_availability()
    
    if not python_cmd:
        print("❌ 未找到可用的Python解释器")
        return False
    
    # 2. 创建虚拟环境
    if not create_virtual_env(python_cmd):
        return False
    
    # 3. 获取虚拟环境Python路径
    python_path, pip_path = get_venv_python()
    print(f"\n✅ 虚拟环境Python路径: {python_path}")
    print(f"✅ 虚拟环境pip路径: {pip_path}")
    
    # 4. 安装包
    if not install_packages(python_path, pip_path):
        print("❌ 包安装失败")
        return False
    
    # 5. 创建激活脚本
    create_activation_scripts()
    
    # 6. 创建requirements.txt
    create_requirements_file()
    
    # 7. 创建项目结构
    create_project_structure()
    
    # 8. 验证安装
    print("\n🔍 验证安装...")
    try:
        result = subprocess.run([str(python_path), "-c", 
            "import torch, transformers, coremltools; print('✅ 所有包导入成功')"], 
            capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 验证通过")
        else:
            print(f"❌ 验证失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 验证出错: {e}")
        return False
    
    # 9. 输出使用说明
    print("\n🎉 虚拟环境配置完成！")
    print("\n📖 使用说明:")
    
    if platform.system() == "Windows":
        print("  激活虚拟环境: .\\activate_venv.bat")
        print("  手动激活: venv\\Scripts\\activate.bat")
    else:
        print("  激活虚拟环境: source ./activate_venv.sh")
        print("  手动激活: source venv/bin/activate")
    
    print(f"\n📁 项目结构已创建，包含以下目录:")
    print("  - data/: 数据文件")
    print("  - models/: 模型文件")
    print("  - src/: 源代码")
    print("  - notebooks/: Jupyter笔记本")
    print("  - scripts/: 脚本文件")
    print("  - tests/: 测试文件")
    print("  - docs/: 文档")
    print("  - configs/: 配置文件")
    
    print("\n🚀 下一步:")
    print("1. 激活虚拟环境")
    print("2. 运行 environment_setup.py 验证环境")
    print("3. 开始数据准备和模型训练")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)