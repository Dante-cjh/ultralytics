import os

os.environ["HIP_VISIBLE_DEVICES"] = "0"

import uvicorn
from labeling_tool.backend.main import app

def main():
    print("🚀 Starting Labeling Tool...")
    print("📂 Backend: http://127.0.0.1:8000")
    print("💻 Frontend: Open labeling_tool/frontend/index.html in your browser")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
