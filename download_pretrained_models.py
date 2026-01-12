#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载预训练模型到本地

用于离线环境部署
"""

import os
import torch
import torchvision.models as models
from pathlib import Path


def download_mobilenet_v2(save_dir: str = "pretrained_models"):
    """
    下载MobileNetV2预训练模型
    
    Args:
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_file = save_path / "mobilenet_v2-b0353104.pth"
    
    if model_file.exists():
        print(f"✅ MobileNetV2模型已存在: {model_file}")
        return str(model_file)
    
    print("📥 正在下载MobileNetV2预训练模型...")
    print("   这可能需要几分钟时间...")
    
    try:
        # 下载模型
        model = models.mobilenet_v2(pretrained=True)
        
        # 保存state_dict
        torch.save(model.state_dict(), model_file)
        
        print(f"✅ 模型下载成功!")
        print(f"   保存至: {model_file}")
        print(f"   文件大小: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(model_file)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def verify_model(model_path: str):
    """验证模型文件是否可用"""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"✅ 模型验证成功")
        print(f"   包含 {len(state_dict)} 个参数")
        return True
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        return False


def main():
    print("="*60)
    print("预训练模型下载工具")
    print("="*60)
    print()
    
    # 下载MobileNetV2
    model_path = download_mobilenet_v2()
    
    if model_path:
        print()
        verify_model(model_path)
        
        print()
        print("="*60)
        print("✅ 下载完成!")
        print("="*60)
        print()
        print("下一步:")
        print("1. 将 pretrained_models/ 目录打包")
        print("2. 传输到离线服务器")
        print("3. 在离线服务器上解压到相同位置")
        print()
        print("代码会自动检测并使用本地模型文件")


if __name__ == '__main__':
    main()

