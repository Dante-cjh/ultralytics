import sys
from pathlib import Path
from ultralytics import YOLO

def check_model_classes():
    # 使用当前目录作为搜索根目录
    root_dir = Path(".")
    
    print(f"🔍 正在扫描当前目录 ({root_dir.resolve()}) 下的 .pt 文件...")

    # 查找所有 .pt 文件 (递归)
    # 也可以改为 "*.pt" 只查找当前目录不递归
    model_files = list(root_dir.glob("*.pt")) 
    
    # 如果想同时也找 runs 下的，可以合并列表，或者直接用 rglob("*.pt")
    # 但 rglob 可能会扫到很多无关的，建议先只扫当前目录
    
    if not model_files:
        print("❌ 在当前目录下没有找到任何 '.pt' 文件")
        # 尝试看看 runs 下有没有
        runs_files = list(root_dir.glob("runs/**/*.pt"))
        if runs_files:
             print(f"💡 提示：在 runs 目录下发现了 {len(runs_files)} 个模型，但当前脚本仅配置为扫描根目录。")
        return

    print(f"🔍 找到 {len(model_files)} 个模型文件，开始检查...\n")

    for model_path in model_files:
        try:
            print(f"📂 模型路径: {model_path}")
            model = YOLO(model_path)
            
            # 获取类别名称
            names = model.names
            print(f"✅ 包含类别 ({len(names)} 个):")
            print(f"   {names}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            print("-" * 50)

if __name__ == "__main__":
    check_model_classes()
