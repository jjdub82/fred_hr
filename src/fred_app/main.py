# src/fred_app/main.py

def _get_MainWindow():
    """
    Import MainWindow whether we're running as a package (python -m fred_app.main),
    as a script, or inside a frozen EXE.
    """
    try:
        from .ui import MainWindow  # package-relative (dev)
        return MainWindow
    except Exception:
        from fred_app.ui import MainWindow  # absolute (frozen/script)
        return MainWindow

def main():
    MW = _get_MainWindow()

    # Try no-arg constructor first (ttkbootstrap.Window style).
    try:
        app = MW()
        # If this returned a top-level window (common), it will have mainloop():
        if hasattr(app, "mainloop"):
            app.mainloop()
        else:
            # If it returned a Frame-like object, no mainloop—do nothing.
            pass
        return
    except TypeError:
        # Fallback: expects a master/root (classic Tk pattern)
        import tkinter as tk
        root = tk.Tk()
        try:
            app = MW(root)  # your current class signature
        except Exception:
            # ensure we don't leave an orphaned root if construction fails
            try:
                root.destroy()
            finally:
                raise
        root.mainloop()

if __name__ == "__main__":
    main()
