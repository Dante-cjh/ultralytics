import tkinter as tk
from tkinter import filedialog
import sys
import ctypes

# 强制 stdout 使用 utf-8 编码，防止 Windows 下中文路径在 subprocess 通信时乱码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def set_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def open_dialog(mode):
    set_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    try:
        if mode == 'folder':
            path = filedialog.askdirectory()
            print(path, end='')
        elif mode == 'file':
            path = filedialog.askopenfilename(
                filetypes=[("Model Weights", "*.pt;*.pth;*.onnx"), ("Images", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")]
            )
            print(path, end='')
        elif mode == 'files':
            paths = filedialog.askopenfilenames(
                filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")]
            )
            # join with |
            print('|'.join(paths), end='')
    except Exception:
        pass
    finally:
        root.destroy()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        open_dialog(sys.argv[1])
