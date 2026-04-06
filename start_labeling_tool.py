import os

os.environ["HIP_VISIBLE_DEVICES"] = "0"

import uvicorn
from labeling_tool.backend.main import app

def main():
    # 检测是否在Docker容器中运行
    # 如果在Docker中，使用0.0.0.0以便从外部访问
    # 否则使用127.0.0.1（本地开发）
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "1"
    host = "0.0.0.0" if is_docker else "127.0.0.1"
    
    print("🚀 Starting Labeling Tool...")
    print(f"📂 Backend: http://{host}:8000")
    print("💻 Frontend: Open http://<your-ip>:8000 in your browser")
    if is_docker:
        print("🐳 Running in Docker container mode")
    uvicorn.run(app, host=host, port=8000)

if __name__ == "__main__":
    main()
