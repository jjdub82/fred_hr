import tkinter as tk
from .ui import MainWindow

def run():
    root = tk.Tk()
    app = MainWindow(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)  # <--- graceful close
    root.mainloop()

if __name__ == "__main__":
    run()
