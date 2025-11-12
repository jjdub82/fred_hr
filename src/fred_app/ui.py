# src/fred_app/ui.py
import re
import textwrap
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

import requests
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .api import search_series, get_series_observations
from .charts import observations_to_df  # expects DataFrame with ["date","value"]
from .config import get_fred_api_key

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FRED_BASE = "https://api.stlouisfed.org/fred"


# -------------------- Styling (forced light, always readable tables) --------------------
def _apply_ttk_style(root: tk.Tk):
    """Use a reliable light theme so text is always readable."""
    try:
        import sv_ttk  # optional
        sv_ttk.set_theme("light")  # force LIGHT
    except Exception:
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TButton", padding=6)
        style.configure("TEntry", padding=4)
        style.configure("TCombobox", padding=4)
        style.configure("Treeview.Heading", padding=6)
        style.map(
            "Treeview",
            background=[("selected", "#2a6fdb")],
            foreground=[("selected", "white")],
        )


def _apply_mpl_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    mpl.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


# -------------------- FRED helpers --------------------
def _fred_get(path: str, **params):
    p = {**params, "api_key": get_fred_api_key(), "file_type": "json"}
    r = requests.get(f"{FRED_BASE}/{path}", params=p, timeout=20)
    r.raise_for_status()
    return r.json()


def fred_top_categories():
    return _fred_get("category/children", category_id=0).get("categories", [])


def fred_children(category_id: int):
    return _fred_get("category/children", category_id=category_id).get("categories", [])


def fred_category_series(category_id: int, limit: int = 1000):
    return (
        _fred_get(
            "category/series",
            category_id=category_id,
            order_by="popularity",
            sort_order="desc",
            limit=limit,
        ).get("seriess", [])
    )


# -------------------- Main Window --------------------
class MainWindow(ttk.Frame):
    """FRED desktop UI (light theme). Readable Treeview with wrapped titles, Chart/Table/Insights, simple trend projection."""

    def __init__(self, master: tk.Tk):
        _apply_ttk_style(master)
        _apply_mpl_style()

        super().__init__(master, padding=12)
        self.master.title("FRED App — HR Edition (Light)")
        self.master.geometry("1280x740")

        # state
        self.selected_ids: list[str] = []
        self.series_labels: dict[str, str] = {}  # id -> pretty title
        self.current_df: pd.DataFrame | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None

        # category state
        self.cat_map: dict[str, int] = {}
        self.subcat_map: dict[str, int] = {}

        # my data overlay
        self.my_df: pd.DataFrame | None = None
        self.show_my_data = tk.BooleanVar(master=self, value=False)

        # scenario (simple presets for projection band)
        self.scenario_var = tk.StringVar(master=self, value="Baseline")

        self._build_layout()
        self.pack(fill="both", expand=True)

        # Load categories async
        self._load_categories_bg()

        # Graceful close
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------- Layout --------------------
    def _build_layout(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Search").pack(side="left", padx=(0, 6))
        self.q = tk.StringVar()
        ent_q = ttk.Entry(top, textvariable=self.q, width=36)
        ent_q.pack(side="left")
        ent_q.bind("<Return>", lambda e: self._on_search())
        ttk.Button(top, text="Go", command=self._on_search).pack(side="left", padx=6)

        ttk.Label(top, text="Range").pack(side="left", padx=(16, 6))
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        ent_s = ttk.Entry(top, textvariable=self.start_var, width=12)
        ent_e = ttk.Entry(top, textvariable=self.end_var, width=12)
        ent_s.pack(side="left")
        ttk.Label(top, text="to").pack(side="left", padx=4)
        ent_e.pack(side="left")
        ent_s.bind("<Return>", lambda e: self._run_query())
        ent_e.bind("<Return>", lambda e: self._run_query())

        ttk.Label(top, text="Quick Range").pack(side="left", padx=(16, 6))
        self.range_var = tk.StringVar(value="Max")
        quick = ttk.Combobox(
            top,
            textvariable=self.range_var,
            values=["YTD", "1Y", "5Y", "10Y", "Max"],
            width=6,
            state="readonly",
        )
        quick.pack(side="left")
        quick.bind("<<ComboboxSelected>>", lambda e: self._apply_quick_range())

        self.normalize_var = tk.BooleanVar(master=self, value=False)
        self.yoy_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(
            top, text="Index=100", variable=self.normalize_var, command=self._refresh_chart
        ).pack(side="left", padx=(16, 4))
        ttk.Checkbutton(
            top, text="YoY %", variable=self.yoy_var, command=self._refresh_chart
        ).pack(side="left")

        self.forecast_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(
            top, text="Trend range", variable=self.forecast_var, command=self._refresh_chart
        ).pack(side="left", padx=(16, 0))

        self.forecast_h_var = tk.StringVar(master=self, value="12m")
        ttk.Label(top, text="Horizon").pack(side="left", padx=(12, 4))
        h_cb = ttk.Combobox(
            top,
            textvariable=self.forecast_h_var,
            values=["6m", "12m", "18m", "24m"],
            width=5,
            state="readonly",
        )
        h_cb.pack(side="left")
        h_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_chart())

        ttk.Label(top, text="Scenario").pack(side="left", padx=(12, 4))
        scen_cb = ttk.Combobox(
            top,
            textvariable=self.scenario_var,
            values=["Baseline", "Hot labor", "Cooldown"],
            width=10,
            state="readonly",
        )
        scen_cb.pack(side="left")
        scen_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_chart())

        # My Data
        ttk.Button(top, text="Add My Data", command=self._load_my_data).pack(side="left", padx=(16, 4))
        ttk.Checkbutton(top, text="Show My Data", variable=self.show_my_data, command=self._refresh_chart)\
            .pack(side="left", padx=(4, 0))

        # Right actions
        ttk.Button(top, text="Save View", command=self._save_view).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="Load View", command=self._load_view).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="Export PPT", command=self._export_ppt).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="Export CSV", command=self._export_csv).pack(side="right")
        ttk.Button(top, text="Export PNG", command=self._export_png).pack(side="right", padx=(0, 8))

        # Category/Subcategory bar
        catbar = ttk.Frame(self)
        catbar.pack(fill="x", pady=(0, 8))
        ttk.Label(catbar, text="Category").pack(side="left")
        self.category_var = tk.StringVar(value="")
        self.category_cb = ttk.Combobox(
            catbar, textvariable=self.category_var, width=40, state="readonly"
        )
        self.category_cb.pack(side="left", padx=(6, 12))
        self.category_cb.bind("<<ComboboxSelected>>", lambda e: self._on_category_selected())

        ttk.Label(catbar, text="Subcategory").pack(side="left")
        self.subcategory_var = tk.StringVar(value="")
        self.subcategory_cb = ttk.Combobox(
            catbar, textvariable=self.subcategory_var, width=40, state="readonly"
        )
        self.subcategory_cb.pack(side="left", padx=(6, 12))
        self.subcategory_cb.bind("<<ComboboxSelected>>", lambda e: self._on_subcategory_selected())

        ttk.Button(catbar, text="Load Series", command=self._load_series_from_selection).pack(
            side="left"
        )
        ttk.Button(catbar, text="Clear Filters", command=self._clear_filters).pack(
            side="left", padx=(8, 0)
        )

        # Body split
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)

        # ------ LEFT ------
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left, weight=1)
        left.grid_rowconfigure(0, weight=3)
        left.grid_rowconfigure(3, weight=2)
        left.grid_columnconfigure(0, weight=1)

        res_frame = ttk.Frame(left)
        res_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        res_frame.grid_rowconfigure(0, weight=1)
        res_frame.grid_columnconfigure(0, weight=1)

        # Results Treeview with its own style (so we can bump rowheight for wrapping)
        self.results_tree = ttk.Treeview(
            res_frame,
            columns=("id", "title"),
            show="headings",
            style="Results.Treeview"
        )
        self.results_tree.heading("id", text="Series ID")
        self.results_tree.heading("title", text="Title")
        self.results_tree.column("id", width=180, anchor="w")
        self.results_tree.column("title", width=460, anchor="w")
        res_vsb = ttk.Scrollbar(res_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=res_vsb.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        res_vsb.grid(row=0, column=1, sticky="ns")

        # Data table with a compact style
        # (we don't wrap table cells; it's a numeric snapshot)
        # Right side (tabs)
        # ------ RIGHT ------
        right = ttk.Notebook(body)
        body.add(right, weight=3)

        self.chart_tab = ttk.Frame(right, padding=6)
        right.add(self.chart_tab, text="Chart")

        self.table_tab = ttk.Frame(right, padding=6)
        right.add(self.table_tab, text="Table")

        self.insights_tab = ttk.Frame(right, padding=10)
        right.add(self.insights_tab, text="Insights")
        self.insights_text = tk.Text(self.insights_tab, height=12, wrap="word")
        self.insights_text.configure(state="disabled")
        self.insights_text.pack(fill="both", expand=True)

        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.pack(fill="both", expand=True)

        self.toolbar_frame = ttk.Frame(self.chart_tab)
        self.toolbar_frame.pack(fill="x")

        self.table = ttk.Treeview(self.table_tab, show="headings", style="Data.Treeview")
        self.table.pack(fill="both", expand=True)

        # Apply readable colors + default row heights for both trees
        self._apply_treeview_colors()

        self.results_tree.bind("<Double-1>", lambda e: self._add_from_results())
        self.results_tree.bind("<Return>", lambda e: self._add_from_results())

        # Selected area + controls
        btns = ttk.Frame(left)
        btns.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(btns, text="Add →", command=self._add_from_results).pack(side="left")
        ttk.Button(btns, text="← Remove", command=self._remove_from_selected).pack(
            side="left", padx=6
        )
        ttk.Button(btns, text="Clear Selected", command=self._clear_selected).pack(
            side="left", padx=6
        )

        ttk.Label(left, text="Selected Series").grid(row=2, column=0, sticky="w", pady=(4, 2))

        sel_frame = ttk.Frame(left)
        sel_frame.grid(row=3, column=0, sticky="nsew")
        sel_frame.grid_rowconfigure(0, weight=1)
        sel_frame.grid_columnconfigure(0, weight=1)
        self.sel_list = tk.Listbox(sel_frame, height=8)
        sel_vsb = ttk.Scrollbar(sel_frame, orient="vertical", command=self.sel_list.yview)
        self.sel_list.configure(yscrollcommand=sel_vsb.set)
        self.sel_list.grid(row=0, column=0, sticky="nsew")
        sel_vsb.grid(row=0, column=1, sticky="ns")
        self.sel_list.bind("<Double-1>", lambda e: self._run_query())
        self.sel_list.bind("<Return>", lambda e: self._run_query())

        run_bar = ttk.Frame(left)
        run_bar.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(run_bar, text="← Remove", command=self._remove_from_selected).pack(
            side="left"
        )
        ttk.Button(run_bar, text="Clear", command=self._clear_selected).pack(side="left", padx=(6, 0))
        ttk.Button(run_bar, text="RUN ▶", command=self._run_query).pack(side="right")

        # Status bar
        status = ttk.Frame(self)
        status.pack(fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(fill="x")

    # --- Readable Treeview colors (light palette) ---
    def _apply_treeview_colors(self):
        style = ttk.Style(self)
        base_bg = "#FFFFFF"
        alt_bg = "#F2F4F7"
        base_fg = "#111827"   # dark slate text
        sel_bg = "#2A6FDB"
        sel_fg = "#FFFFFF"

        # Results tree gets larger default rowheight (we will auto-bump after inserts)
        style.configure("Results.Treeview",
                        background=base_bg, fieldbackground=base_bg, foreground=base_fg,
                        rowheight=48)
        style.configure("Results.Treeview.Heading", foreground=base_fg)
        style.map("Results.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", sel_fg)])

        # Data table compact
        style.configure("Data.Treeview",
                        background=base_bg, fieldbackground=base_bg, foreground=base_fg,
                        rowheight=28)
        style.configure("Data.Treeview.Heading", foreground=base_fg)
        style.map("Data.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", sel_fg)])

        # Zebra tags for both
        for tv in (getattr(self, "results_tree", None), getattr(self, "table", None)):
            if tv:
                tv.tag_configure("even", background=base_bg, foreground=base_fg)
                tv.tag_configure("odd",  background=alt_bg,  foreground=base_fg)

    # -------------------- Categories --------------------
    def _load_categories_bg(self):
        self._set_status("Loading categories…")

        def work():
            return fred_top_categories()

        def done(top):
            top_sorted = sorted(top, key=lambda c: c.get("name", "").lower())
            self.cat_map = {c["name"]: c["id"] for c in top_sorted}
            self.category_cb["values"] = list(self.cat_map.keys())
            self._set_status(f"Ready – {len(top_sorted)} categories loaded" if top_sorted else "Ready")

        threading.Thread(target=lambda: self._bg(work, done), daemon=True).start()

    def _on_category_selected(self):
        name = self.category_var.get().strip()
        if not name:
            return
        cat_id = self.cat_map.get(name)
        if not cat_id:
            return

        self._set_status(f"Loading subcategories for “{name}”…")

        def work():
            children = fred_children(cat_id)
            return sorted(children, key=lambda c: c.get("name", "").lower())

        def done(children_sorted):
            self.subcat_map = {c["name"]: c["id"] for c in children_sorted}
            self.subcategory_cb["values"] = list(self.subcat_map.keys())
            self.subcategory_var.set("")
            self._set_status("Ready")

        threading.Thread(target=lambda: self._bg(work, done), daemon=True).start()

    def _on_subcategory_selected(self):
        self._load_series_from_selection()

    def _load_series_from_selection(self):
        subname = self.subcategory_var.get().strip()
        catname = self.category_var.get().strip()
        if subname and subname in self.subcat_map:
            cid, scope = self.subcat_map[subname], subname
        elif catname and catname in self.cat_map:
            cid, scope = self.cat_map[catname], catname
        else:
            messagebox.showinfo("Info", "Choose a Category (and optionally a Subcategory) first.")
            return

        self._set_status(f"Loading series for “{scope}”…")

        def work():
            return fred_category_series(cid, limit=1000)

        def done(series_list):
            self._clear_results()
            for i, s in enumerate(series_list):
                tag   = "even" if i % 2 == 0 else "odd"
                sid   = s.get("id", "")
                title = self._wrap_title(s.get("title", ""), width=50)  # WRAP
                self.results_tree.insert("", "end", values=(sid, title), tags=(tag,))
            # bump row height to fit wrapped titles
            self._refresh_results_rowheight()
            self._set_status(f"Loaded {len(series_list)} series from “{scope}”")

        threading.Thread(target=lambda: self._bg(work, done), daemon=True).start()

    # -------------------- Search --------------------
    def _on_search(self):
        term = self.q.get().strip()
        if not term:
            messagebox.showinfo("Info", "Enter a search term.")
            return

        self._set_status("Searching…")

        def work():
            return search_series(term, limit=200)

        def done(results):
            self._clear_results()
            for i, s in enumerate(results):
                tag   = "even" if i % 2 == 0 else "odd"
                sid   = s.get("id", "")
                title = self._wrap_title(s.get("title", ""), width=50)  # WRAP
                self.results_tree.insert("", "end", values=(sid, title), tags=(tag,))
            # bump row height to fit wrapped titles
            self._refresh_results_rowheight()
            self._set_status(f"Found {len(results)} result(s)")

        threading.Thread(target=lambda: self._bg(work, done), daemon=True).start()

    def _clear_results(self):
        for r in self.results_tree.get_children():
            self.results_tree.delete(r)

    # -------------------- Selection handlers --------------------
    def _add_from_results(self):
        sel = self.results_tree.selection()
        if not sel:
            self._set_status("Pick a series first.")
            return
        item = self.results_tree.item(sel[0])
        vals = item.get("values") or []
        sid = vals[0] if len(vals) > 0 else ""
        title = vals[1] if len(vals) > 1 else ""
        if not sid:
            self._set_status("Couldn’t read a series ID.")
            return
        if sid not in self.selected_ids:
            self.selected_ids.append(sid)
            self.sel_list.insert("end", sid)
        if title:
            self.series_labels[sid] = str(title).replace("\n", " ").strip()
        self._set_status(f"Added {sid}")

    def _remove_from_selected(self):
        sel = list(self.sel_list.curselection())
        sel.reverse()
        for idx in sel:
            sid = self.sel_list.get(idx)
            self.sel_list.delete(idx)
            if sid in self.selected_ids:
                self.selected_ids.remove(sid)

    def _clear_selected(self):
        self.sel_list.delete(0, "end")
        self.selected_ids.clear()

    # -------------------- Run & data --------------------
    def _run_query(self):
        if not self.selected_ids:
            messagebox.showinfo("Info", "Select at least one series first.")
            self._set_status("No series selected.")
            return

        start, end = self._clean_dates(self.start_var.get().strip(), self.end_var.get().strip())
        params = {}
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end

        preview = ", ".join(self.selected_ids[:3]) + ("..." if len(self.selected_ids) > 3 else "")
        self._set_status(f"Fetching series: {preview}")

        def work():
            dfs = []
            for sid in self.selected_ids:
                try:
                    obs = get_series_observations(sid, **params)
                    df = observations_to_df(obs).set_index("date").rename(columns={"value": sid})
                    dfs.append(df)
                except Exception as e:
                    print(f"Error fetching {sid}: {e}")
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, axis=1)

        def done(df: pd.DataFrame):
            self.current_df = df
            if df.empty:
                self._set_status("No data for selection/date range.")
                messagebox.showinfo("No Data", "No observations returned.")
                self._clear_chart()
                self._clear_table()
                self._refresh_insights()
                return
            self._refresh_chart()
            self._refresh_table()
            self._refresh_insights()
            self._set_status(f"Loaded {len(df.columns)} series, {len(df)} rows.")

        threading.Thread(target=lambda: self._bg(work, done), daemon=True).start()

    # -------------------- Dates / ranges --------------------
    def _clean_dates(self, start: str, end: str):
        valid_s = bool(DATE_RE.match(start))
        valid_e = bool(DATE_RE.match(end))
        if not valid_s and start:
            self._set_status("Ignored invalid start date (YYYY-MM-DD).")
            start = ""
        if not valid_e and end:
            self._set_status("Ignored invalid end date (YYYY-MM-DD).")
            end = ""
        if start and end:
            try:
                s, e = pd.to_datetime(start), pd.to_datetime(end)
                if s > e:
                    start, end = end, start
                    self._set_status("Swapped start/end to maintain order.")
            except Exception:
                start, end = "", ""
                self._set_status("Ignored invalid date range.")
        return start, end

    def _apply_quick_range(self):
        choice = self.range_var.get()
        try:
            today = pd.Timestamp.today().normalize()
            if choice == "YTD":
                s = pd.Timestamp(today.year, 1, 1); e = today
            elif choice == "1Y":
                s = today - pd.DateOffset(years=1); e = today
            elif choice == "5Y":
                s = today - pd.DateOffset(years=5); e = today
            elif choice == "10Y":
                s = today - pd.DateOffset(years=10); e = today
            elif choice == "Max":
                self.start_var.set(""); self.end_var.set(""); return
            else:
                return
            self.start_var.set(s.date().isoformat()); self.end_var.set(e.date().isoformat())
        except Exception:
            pass

    # -------------------- Simple forecasting utilities --------------------
    def _infer_freq_and_steps(self, idx: pd.DatetimeIndex):
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
            return None, 0
        f = pd.infer_freq(idx)
        if f in ("M", "MS", "ME", "BM", "BMS"):
            return "M", 24
        if f and str(f).upper().startswith("Q"):
            return "Q", 8
        return "M", 24  # fallback

    def _forecast_steps_from_ui(self, freq: str) -> int:
        m = {"6m": 6, "12m": 12, "18m": 18, "24m": 24}
        months = m.get(self.forecast_h_var.get(), 12)
        if freq == "Q":
            import math
            return max(2, int(math.ceil(months / 3)))
        return months

    def _scenario_params(self):
        scen = (self.scenario_var.get() or "Baseline").lower()
        if "hot" in scen:
            return 0.45, 1.35, -0.10  # kappa_annual, band_mult, mu_shift
        if "cool" in scen:
            return 0.30, 1.10, +0.10
        return 0.35, 1.28, 0.0

    def _trend_projection_band(self, y: pd.Series, steps: int, freq: str, lookback: int = 60):
        y = y.dropna()
        if len(y) < 12:
            return None
        if len(y) > lookback:
            y = y.iloc[-lookback:]

        last = y.index[-1]
        idx_future = (
            pd.date_range(last + pd.offsets.QuarterEnd(1), periods=steps, freq="Q")
            if freq == "Q"
            else pd.date_range(last + pd.offsets.MonthEnd(1), periods=steps, freq="M")
        )

        level_med = float(np.nanmedian(y.values))
        level_max = float(np.nanmax(y.values))
        is_rate_like = (level_med < 20.0 and level_max < 50.0)

        diffs = y.diff().dropna()
        sigma = float(diffs.std()) if len(diffs) >= 6 else 0.0
        kappa_annual, band_mult_k, mu_shift = self._scenario_params()

        if is_rate_like:
            try:
                mu = float(y.rolling(60, min_periods=12).mean().iloc[-1])
                if np.isnan(mu):
                    mu = float(y.mean())
            except Exception:
                mu = float(y.mean())
            mu = max(0.0, mu * (1.0 + mu_shift))
            kappa = kappa_annual / (12 if freq == "M" else 4)
            mid = np.empty(steps, dtype=float)
            x = float(y.iloc[-1])
            for i in range(steps):
                x = x + kappa * (mu - x)
                mid[i] = x
        else:
            t = np.arange(len(y), dtype=float)
            a, b = np.polyfit(t, y.values.astype(float), deg=1)
            mean_level = float(np.nanmean(y.values))
            cap = 0.02 * max(mean_level, 1.0)
            b = float(np.clip(b, -cap, cap))
            tf = np.arange(len(y), len(y) + steps, dtype=float)
            mid = a + b * tf

        h = np.arange(1, steps + 1, dtype=float)
        band = band_mult_k * sigma * np.sqrt(h)
        upper = mid + band
        lower = mid - band

        ymax_recent = max(level_max, np.nanmax(y.values))
        cap_hi = max(ymax_recent * 1.5, np.nanmax(mid))
        cap_lo = 0.0 if is_rate_like else -np.inf

        mid = np.clip(mid, cap_lo, cap_hi)
        upper = np.clip(upper, cap_lo, cap_hi)
        lower = np.clip(lower, cap_lo, cap_hi)

        return pd.DataFrame({"mid": mid, "low": lower, "high": upper}, index=idx_future)

    # -------------------- Chart & Table --------------------
    def _wrap_label(self, s: str, width: int = 32) -> str:
        s = " ".join(str(s or "").split())
        return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s

    def _refresh_chart(self):
        df = self.current_df
        if df is None or df.empty:
            self._clear_chart()
            return

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        plot = df.copy()
        if self.yoy_var.get():
            plot = plot.pct_change(12) * 100.0
        if self.normalize_var.get() and not self.yoy_var.get():
            first = plot.dropna(how="all").iloc[0]
            plot = plot.divide(first) * 100.0

        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            plot = plot.join(self.my_df, how="outer")

        plot = plot.astype("float64").dropna(how="all", axis=1)
        if plot.empty:
            self._clear_chart()
            self._set_status("No plottable data.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)

        labels = []
        eligible_cnt = 0

        for col in plot.columns:
            label = self.series_labels.get(col, col)
            labels.append(label)
            series = plot[col].astype("float64")
            ax.plot(plot.index, series.values, label=self._wrap_label(label, 32))

            if self.forecast_var.get() and (not col.startswith("My_")):
                freq, steps_cap = self._infer_freq_and_steps(series.index)
                if freq:
                    steps = min(steps_cap, self._forecast_steps_from_ui(freq))
                    band = self._trend_projection_band(series, steps=steps, freq=freq)
                    if band is not None and not band.empty:
                        x = band.index
                        mid = band["mid"].to_numpy(dtype=float)
                        low = band["low"].to_numpy(dtype=float)
                        high = band["high"].to_numpy(dtype=float)
                        ax.fill_between(x, low, high, alpha=0.12, linewidth=0)
                        ax.plot(x, mid, linestyle="--", linewidth=2.2,
                                label=self._wrap_label(f"{label} (trend range)", 32))
                        eligible_cnt += 1

        ax.set_title(self._wrap_label(labels[0], 48) if len(labels) == 1 else "FRED Series", pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("YoY % " if self.yoy_var.get() else "Value")

        if len(labels) <= 2:
            ax.legend(loc="best", frameon=False, fontsize=9)
            fig.tight_layout()
        else:
            fig.subplots_adjust(right=0.78)
            lg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
            for t in lg.get_texts():
                t.set_ha("left")

        fig.autofmt_xdate(rotation=0, ha="center")
        self._set_chart(fig)

        if self.forecast_var.get():
            self._set_status(f"Projected {eligible_cnt} series" if eligible_cnt else "No series eligible for trend range.")

    def _refresh_table(self):
        df = self.current_df
        if df is None or df.empty:
            self._clear_table()
            return

        table_df = df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            table_df = table_df.join(self.my_df, how="outer")

        last = table_df.index.max()
        start = last - pd.DateOffset(months=12)
        snap = table_df.loc[table_df.index >= start].copy()

        self.table.delete(*self.table.get_children())
        self.table["columns"] = ["date"] + list(table_df.columns)
        for c in self.table["columns"]:
            self.table.heading(c, text=c)
            self.table.column(c, width=120 if c == "date" else 110, anchor="e")

        for i, row in enumerate(snap.reset_index().itertuples(index=False)):
            vals = [row.date.date().isoformat()] + [
                "" if pd.isna(getattr(row, col)) else f"{getattr(row, col):,.3f}"
                for col in table_df.columns
            ]
            tag = "even" if i % 2 == 0 else "odd"
            self.table.insert("", "end", values=vals, tags=(tag,))

    def _set_chart(self, fig):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="right", padx=4, pady=2)

    def _clear_chart(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None

    def _clear_table(self):
        self.table.delete(*self.table.get_children())
        self.table["columns"] = []

    # -------------------- Insights --------------------
    def _refresh_insights(self):
        self.insights_text.configure(state="normal")
        self.insights_text.delete("1.0", "end")

        df = self.current_df
        if df is None or df.empty:
            self.insights_text.insert("end", "Run a query to see insights.")
            self.insights_text.configure(state="disabled"); return

        work_df = df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            work_df = work_df.join(self.my_df, how="outer")

        last_idx = work_df.dropna(how="all").index.max()
        if pd.isna(last_idx):
            self.insights_text.insert("end", "No recent data to summarize.")
            self.insights_text.configure(state="disabled"); return

        lines = [f"Snapshot: {last_idx.date().isoformat()}\n"]
        for col in work_df.columns:
            s = work_df[col].dropna()
            if s.empty: continue
            latest = s.iloc[-1]
            def delta(n):
                try: return latest - s.iloc[-n]
                except Exception: return np.nan
            d3, d6, d12 = delta(3), delta(6), delta(12)
            latest_str = f"{latest:,.2f}" if pd.notna(latest) else "—"
            d3_str = "—" if pd.isna(d3) else f"{d3:+.2f}"
            d6_str = "—" if pd.isna(d6) else f"{d6:+.2f}"
            d12_str = "—" if pd.isna(d12) else f"{d12:+.2f}"
            title = self.series_labels.get(col, col)
            lines.append(f"{title}")
            lines.append(f"  Latest: {latest_str} | Δ3m {d3_str} | Δ6m {d6_str} | Δ12m {d12_str}")
        txt = "\n".join(lines).strip() or "No series to summarize."
        self.insights_text.insert("end", txt)
        self.insights_text.configure(state="disabled")

    # -------------------- Save/Load + My Data + Export + Filters --------------------
    def _save_view(self):
        import json
        state = {
            "selected_ids": self.selected_ids,
            "series_labels": self.series_labels,
            "start": self.start_var.get(),
            "end": self.end_var.get(),
            "quick_range": self.range_var.get(),
            "normalize": bool(self.normalize_var.get()),
            "yoy": bool(self.yoy_var.get()),
            "trend": bool(self.forecast_var.get()),
            "horizon": self.forecast_h_var.get(),
            "scenario": self.scenario_var.get(),
            "category": self.category_var.get(),
            "subcategory": self.subcategory_var.get(),
            "show_my_data": bool(self.show_my_data.get()),
        }
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("View files", "*.json")],
                                            title="Save view")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self._set_status(f"Saved view → {path}")
        except Exception as e:
            self._error(e)

    def _load_view(self):
        import json
        path = filedialog.askopenfilename(filetypes=[("View files", "*.json")], title="Load view")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.selected_ids = list(state.get("selected_ids", []))
            self.series_labels = dict(state.get("series_labels", {}))
            self.sel_list.delete(0, "end")
            for sid in self.selected_ids:
                self.sel_list.insert("end", sid)
            self.start_var.set(state.get("start", ""))
            self.end_var.set(state.get("end", ""))
            self.range_var.set(state.get("quick_range", "Max"))
            self.normalize_var.set(state.get("normalize", False))
            self.yoy_var.set(state.get("yoy", False))
            self.forecast_var.set(state.get("trend", False))
            self.forecast_h_var.set(state.get("horizon", "12m"))
            self.scenario_var.set(state.get("scenario", "Baseline"))
            self.category_var.set(state.get("category", ""))
            self.subcategory_var.set(state.get("subcategory", ""))
            self.show_my_data.set(state.get("show_my_data", False))
            self._set_status(f"Loaded view ← {path}")
            self._run_query()
        except Exception as e:
            self._error(e)

    def _load_my_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        try:
            raw = pd.read_csv(path)
            if raw.empty:
                messagebox.showinfo("Info", "Selected CSV is empty."); return
            raw.columns = [str(c) for c in raw.columns]
            date_col = raw.columns[0]
            df = raw.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.resample("M").mean()
            df = df.rename(columns={c: f"My_{c}" for c in df.columns})
            self.my_df = df
            self.show_my_data.set(True)
            self._set_status(f"Loaded My Data: {path}")
            self._refresh_chart(); self._refresh_table(); self._refresh_insights()
        except Exception as e:
            self._error(e)

    def _export_csv(self):
        if self.current_df is None or self.current_df.empty:
            messagebox.showinfo("Info", "Nothing to export yet."); return
        df = self.current_df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            df = df.join(self.my_df, how="outer")
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")], title="Save data as CSV")
        if not path: return
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df.to_csv(path, index_label="date")
            self._set_status(f"Saved CSV → {path}")
        except Exception as e:
            self._error(e)

    def _export_png(self):
        if not self.canvas:
            messagebox.showinfo("Info", "No chart to export yet."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")], title="Save chart as PNG")
        if not path: return
        try:
            self.canvas.figure.savefig(path, dpi=144, bbox_inches="tight")
            self._set_status(f"Saved chart → {path}")
        except Exception as e:
            self._error(e)

    def _export_ppt(self):
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.dml.color import RGBColor
            from pptx.enum.text import PP_ALIGN
        except Exception:
            messagebox.showinfo("Missing dependency",
                                "PowerPoint export needs 'python-pptx'.\nInstall:\n  pip install python-pptx")
            return
        if not self.canvas:
            messagebox.showinfo("Info", "No chart to export yet."); return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        self.canvas.figure.savefig(tmp.name, dpi=144, bbox_inches="tight")

        self.insights_text.configure(state="normal")
        insight_str = self.insights_text.get("1.0", "end").strip() or "No insights available."
        self.insights_text.configure(state="disabled")

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        TF_RED = RGBColor(226, 0, 26)
        SLATE = RGBColor(45, 55, 72)

        title_box = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(1.0))
        fill = title_box.fill; fill.solid(); fill.fore_color.rgb = TF_RED
        title_box.line.fill.background()
        p = title_box.text_frame.paragraphs[0]
        p.text = "Labor Market Dashboard"; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor(255,255,255)

        slide.shapes.add_picture(tmp.name, Inches(0.5), Inches(1.2), height=Inches(4.5))

        tx_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.9), Inches(12.3), Inches(2.3))
        tf = tx_box.text_frame; tf.word_wrap = True; tf.clear()
        p = tf.paragraphs[0]; p.text = "Insights"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = SLATE; p.alignment = PP_ALIGN.LEFT
        for line in insight_str.splitlines():
            if not line.strip(): continue
            q = tf.add_paragraph(); q.text = line; q.level = 1; q.font.size = Pt(12); q.font.color.rgb = SLATE

        path = filedialog.asksaveasfilename(defaultextension=".pptx",
                                            filetypes=[("PowerPoint", "*.pptx")],
                                            title="Save dashboard slide")
        if not path: return
        try:
            prs.save(path); self._set_status(f"Saved PowerPoint → {path}")
        except Exception as e:
            self._error(e)

    def _clear_filters(self):
        self.q.set(""); self.start_var.set(""); self.end_var.set(""); self.range_var.set("Max")
        self.normalize_var.set(False); self.yoy_var.set(False)
        self.forecast_var.set(False); self.forecast_h_var.set("12m"); self.scenario_var.set("Baseline")
        self.category_var.set(""); self.category_cb.set(""); self.subcategory_var.set(""); self.subcategory_cb.set("")
        self._clear_results(); self._set_status("Filters cleared.")

    # -------------------- Wrapping helpers for Treeview --------------------
    def _wrap_title(self, text: str, width: int = 50) -> str:
        """Insert newlines so text wraps in a Treeview cell."""
        text = (text or "").strip()
        return "\n".join(textwrap.wrap(text, width=width)) if text else ""

    def _refresh_results_rowheight(self, style_name: str = "Results.Treeview",
                                   base_pad_px: int = 6, max_lines: int = 5):
        """
        Measure how many wrapped lines are present in the results Treeview and
        bump the row height so all lines are visible (capped by max_lines).
        """
        longest = 1
        for iid in self.results_tree.get_children(""):
            vals = self.results_tree.item(iid, "values")
            if not vals:
                continue
            title = str(vals[1]) if len(vals) > 1 else ""
            lines = title.count("\n") + 1 if title else 1
            if lines > longest:
                longest = lines

        longest = min(longest, max_lines)

        try:
            f = tkfont.nametofont("TkDefaultFont")
            line_px = f.metrics("linespace")
            row_h = max(28, int(longest * (line_px + base_pad_px)))
        except Exception:
            row_h = 24 + (longest - 1) * 16

        ttk.Style(self).configure(style_name, rowheight=row_h)

    # -------------------- Helpers & Close --------------------
    def _bg(self, func, on_done):
        try:
            res = func(); self.after(0, lambda: on_done(res))
        except Exception as e:
            self.after(0, lambda: self._error(e))

    def _set_status(self, text: str):
        self.status_var.set(text); self.update_idletasks()

    def _error(self, e: Exception):
        self._set_status(f"Error: {e}"); messagebox.showerror("Error", str(e))

    def _on_close(self):
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            self.master.quit(); self.master.destroy()
        except Exception:
            pass
        import sys; sys.exit(0)
