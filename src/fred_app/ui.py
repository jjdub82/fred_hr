# src/fred_app/ui.py
import re
import io
import textwrap
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

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


# -------------------- Styling --------------------
def _apply_ttk_style(root: tk.Tk):
    """Modern ttk look; works with or without sv-ttk."""
    try:
        import sv_ttk  # optional
        sv_ttk.set_theme("dark")
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
        style.configure("Treeview", rowheight=48, borderwidth=0)
        style.configure("Treeview.Heading", padding=6)
        style.map(
            "Treeview",
            background=[("selected", "#2a6fdb")],
            foreground=[("selected", "white")],
        )


def _apply_mpl_style(dark: bool = False):
    try:
        plt.style.use("seaborn-v0_8-darkgrid" if dark else "seaborn-v0_8-whitegrid")
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
    """FRED desktop UI for HR: Chart/Table/Insights, Trend Range, Horizon, Scenarios, Save/Load views, My Data, Exports."""

    def __init__(self, master: tk.Tk):
        _apply_ttk_style(master)
        dark = False
        try:
            import sv_ttk
            dark = sv_ttk.get_theme() == "dark"
        except Exception:
            pass
        _apply_mpl_style(dark=dark)

        super().__init__(master, padding=12)
        self.master.title("FRED App — HR Edition (R7)")
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

        # scenario tuning (affects projection)
        self.scenario_var = tk.StringVar(master=self, value="Baseline")  # Baseline / Hot labor / Cooldown

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

        # Trend range toggle + horizon + scenario
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

        # Right-side actions
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

        # ------ LEFT PANE ------
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left, weight=1)
        left.grid_rowconfigure(0, weight=3)
        left.grid_rowconfigure(3, weight=2)
        left.grid_columnconfigure(0, weight=1)

        res_frame = ttk.Frame(left)
        res_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        res_frame.grid_rowconfigure(0, weight=1)
        res_frame.grid_columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(res_frame, columns=("id", "title"), show="headings")
        self.results_tree.heading("id", text="Series ID")
        self.results_tree.heading("title", text="Title")
        self.results_tree.column("id", width=180, anchor="w")
        self.results_tree.column("title", width=460, anchor="w")
        res_vsb = ttk.Scrollbar(res_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=res_vsb.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        res_vsb.grid(row=0, column=1, sticky="ns")

        self.results_tree.tag_configure("odd", background="#FFFFFF")
        self.results_tree.tag_configure("even", background="#F2F4F7")
        self.results_tree.bind("<Double-1>", lambda e: self._add_from_results())
        self.results_tree.bind("<Return>", lambda e: self._add_from_results())

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

        # ------ RIGHT PANE ------
        right = ttk.Notebook(body)
        body.add(right, weight=3)

        self.chart_tab = ttk.Frame(right, padding=6)
        right.add(self.chart_tab, text="Chart")

        self.table_tab = ttk.Frame(right, padding=6)
        right.add(self.table_tab, text="Table")

        # Insights tab
        self.insights_tab = ttk.Frame(right, padding=10)
        right.add(self.insights_tab, text="Insights")
        self.insights_text = tk.Text(self.insights_tab, height=12, wrap="word")
        self.insights_text.configure(state="disabled")
        self.insights_text.pack(fill="both", expand=True)

        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.pack(fill="both", expand=True)

        self.toolbar_frame = ttk.Frame(self.chart_tab)
        self.toolbar_frame.pack(fill="x")

        self.table = ttk.Treeview(self.table_tab, show="headings")
        self.table.pack(fill="both", expand=True)

        # Status bar
        status = ttk.Frame(self)
        status.pack(fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(fill="x")

    # -------------------- Categories --------------------
    def _load_categories_bg(self):
        self._set_status("Loading categories…")

        def work():
            return fred_top_categories()

        def done(top):
            top_sorted = sorted(top, key=lambda c: c.get("name", "").lower())
            self.cat_map = {c["name"]: c["id"] for c in top_sorted}
            self.category_cb["values"] = list(self.cat_map.keys())
            self._set_status(
                f"Ready – {len(top_sorted)} categories loaded" if top_sorted else "Ready"
            )

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
                tag = "even" if i % 2 == 0 else "odd"
                sid = s.get("id", "")
                title = s.get("title", "")
                if title:
                    title = "\n".join(textwrap.wrap(title, width=50))
                self.results_tree.insert("", "end", values=(sid, title), tags=(tag,))
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
                tag = "even" if i % 2 == 0 else "odd"
                sid = s.get("id", "")
                title = s.get("title", "")
                if title:
                    title = "\n".join(textwrap.wrap(title, width=50))
                self.results_tree.insert("", "end", values=(sid, title), tags=(tag,))
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
                s = pd.Timestamp(today.year, 1, 1)
                e = today
            elif choice == "1Y":
                s = today - pd.DateOffset(years=1)
                e = today
            elif choice == "5Y":
                s = today - pd.DateOffset(years=5)
                e = today
            elif choice == "10Y":
                s = today - pd.DateOffset(years=10)
                e = today
            elif choice == "Max":
                self.start_var.set("")
                self.end_var.set("")
                return
            else:
                return
            self.start_var.set(s.date().isoformat())
            self.end_var.set(e.date().isoformat())
        except Exception:
            pass

    # -------------------- Forecasting utilities --------------------
    def _infer_freq_and_steps(self, idx: pd.DatetimeIndex) -> tuple[str | None, int]:
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
            return None, 0
        f = pd.infer_freq(idx)
        if f is None:
            deltas = np.diff(idx.values).astype("timedelta64[D]").astype(int)
            if len(deltas):
                med = int(np.median(deltas))
                if 25 <= med <= 35:
                    f = "M"
                elif 80 <= med <= 100:
                    f = "Q"
        if f in ("M", "MS", "ME", "BM", "BMS"):
            return "M", 24  # allow up to 24 months if user selects it
        if f and str(f).upper().startswith("Q"):
            return "Q", 8
        return None, 0

    def _series_quality(self, s: pd.Series) -> tuple[bool, str]:
        y = s.dropna()
        if y.empty:
            return False, "all missing"
        freq, _ = self._infer_freq_and_steps(y.index)
        n = len(y)
        if freq == "M" and n < 36:
            return False, "needs >=36 monthly obs"
        if n < 20:
            return False, "needs >=20 points"
        miss_ratio = 1.0 - (len(y) / len(s))
        if miss_ratio > 0.2:
            return False, "too many missing"
        if float(np.nanstd(y.values)) < 1e-9:
            return False, "no variance"
        try:
            lag1 = y.autocorr(lag=1)
            if lag1 is None or np.isnan(lag1) or abs(lag1) < 0.2:
                return False, "weak autocorrelation"
        except Exception:
            pass
        return True, ""

    def _is_forecastable(self, s: pd.Series) -> tuple[bool, str, int, str | None]:
        ok, why = self._series_quality(s)
        if not ok:
            return False, why, 0, None
        freq, steps_cap = self._infer_freq_and_steps(s.index)
        if not freq:
            return False, "unknown frequency", 0, None
        return True, "", steps_cap, freq

    def _forecast_steps_from_ui(self, freq: str) -> int:
        m = {"6m": 6, "12m": 12, "18m": 18, "24m": 24}
        months = m.get(self.forecast_h_var.get(), 12)
        if freq == "Q":
            import math
            return max(2, int(math.ceil(months / 3)))
        return months

    def _scenario_params(self) -> tuple[float, float, float]:
        """
        Returns (kappa_annual, band_mult_k, mu_shift) where:
        - kappa_annual: speed of mean reversion for rate-like series
        - band_mult_k: width multiplier for the cone (1.28 ~ 80% band)
        - mu_shift: shift to long-run mean (e.g., -10% for hot labor = tighter market)
        """
        scen = (self.scenario_var.get() or "Baseline").lower()
        if "hot" in scen:
            return 0.45, 1.35, -0.10
        if "cool" in scen:
            return 0.30, 1.10, +0.10
        return 0.35, 1.28, 0.0

    def _trend_projection_band(self, y: pd.Series, steps: int, freq: str, lookback: int = 60):
        """
        Projection band:
          - Rate-like series (values <~20) use mean reversion toward a long-run mean (scenario-adjusted).
          - Others use linear trend with capped slope.
          - Uncertainty is ADDITIVE using recent absolute-change volatility and grows with sqrt(h).
          - Bounds clipped to [0, 1.5 * recent max] (>=0 for rate-like).
        Returns DataFrame ['mid','low','high'] indexed forward.
        """
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
            # Mean reversion to 5y average (or overall mean), shifted by scenario (±10%)
            try:
                mu = float(y.rolling(60, min_periods=12).mean().iloc[-1])
                if np.isnan(mu):
                    mu = float(y.mean())
            except Exception:
                mu = float(y.mean())
            mu = max(0.0, mu * (1.0 + mu_shift))  # scenario shift
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
            cap = 0.02 * max(mean_level, 1.0)  # ~2% of mean per step
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

        # Ensure clean, numeric data and sorted datetime index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Transforms
        plot = df.copy()
        if self.yoy_var.get():
            plot = plot.pct_change(12) * 100.0
        if self.normalize_var.get() and not self.yoy_var.get():
            first = plot.dropna(how="all").iloc[0]
            plot = plot.divide(first) * 100.0

        plot = plot.astype("float64").dropna(how="all", axis=1)

        # add My Data overlay if available
        overlay = None
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            overlay = self._prep_my_data_for_merge()
            if overlay is not None and not overlay.empty:
                plot = plot.join(overlay, how="outer")
                # don't apply transforms to my data beyond what we already did

        if plot.empty:
            self._clear_chart()
            self._set_status("No plottable data (all NaN after transforms/overlay).")
            return

        fig = plt.figure(constrained_layout=False)
        ax = fig.add_subplot(111)

        labels = []
        eligible_cnt = 0
        skipped = []
        my_cols = set(overlay.columns) if overlay is not None else set()

        for col in plot.columns:
            label = self.series_labels.get(col, col)
            if col in my_cols:
                label = f"[My] {label}"
            labels.append(label)

            series = plot[col].astype("float64")
            ax.plot(plot.index, series.values, label=self._wrap_label(label, 32))

            # optional trend range (only for FRED series, not My Data)
            if self.forecast_var.get() and (col not in my_cols):
                ok, why, steps_cap, freq = self._is_forecastable(series)
                if ok and steps_cap > 0 and freq:
                    steps = min(steps_cap, self._forecast_steps_from_ui(freq))
                    band = self._trend_projection_band(series, steps=steps, freq=freq)
                    if band is not None and not band.empty:
                        x = band.index
                        low = band["low"].astype("float64").to_numpy()
                        high = band["high"].astype("float64").to_numpy()
                        mid = band["mid"].astype("float64").to_numpy()
                        mask = ~(np.isnan(low) | np.isnan(high) | np.isnan(mid))
                        if mask.any():
                            x = x[mask]; low = low[mask]; high = high[mask]; mid = mid[mask]
                            ax.fill_between(x, low, high, alpha=0.12, linewidth=0, zorder=1)
                            ax.plot(
                                x, mid, linestyle="--", linewidth=2.5, zorder=3,
                                label=self._wrap_label(f"{label} (trend range)", 32),
                            )
                            # endpoint marker + value
                            x_last = x[-1]; y_last = float(mid[-1])
                            ax.plot([x_last], [y_last], marker="o", markersize=5, zorder=4)
                            ax.annotate(
                                f"{y_last:,.2f}", xy=(x_last, y_last), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=9,
                            )
                            eligible_cnt += 1
                        else:
                            skipped.append(f"{col}: trend band all-NaN")
                    else:
                        skipped.append(f"{col}: trend band failed")
                else:
                    skipped.append(f"{col}: {why or 'not eligible'}")

        # Titles/labels
        if len(labels) == 1:
            ax.set_title(self._wrap_label(labels[0], 48), pad=10)
        else:
            ax.set_title("FRED Series", pad=10)

        ax.set_xlabel("Date")
        ax.set_ylabel("YoY % " if self.yoy_var.get() else "Value")

        # Legend
        if len(labels) <= 2:
            ax.legend(loc="best", frameon=False, fontsize=9)
            fig.tight_layout()
        else:
            fig.subplots_adjust(right=0.78)
            lg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
            for t in lg.get_texts():
                t.set_ha("left")

        # Shade future region if projections plotted
        if self.forecast_var.get() and eligible_cnt > 0:
            try:
                last_hist = plot.index.max()
                ax.axvspan(last_hist, ax.get_xlim()[1], alpha=0.06)
            except Exception:
                pass

        fig.autofmt_xdate(rotation=0, ha="center")
        self._set_chart(fig)

        # Status + refresh insights/alerts
        alerts = self._compute_alerts()
        if alerts:
            self._set_status(" | ".join(alerts))
        elif self.forecast_var.get():
            if eligible_cnt == 0:
                self._set_status("No series eligible for trend range. Select longer histories or turn off YoY.")
            else:
                msg = f"Projected {eligible_cnt} series"
                if skipped:
                    msg += f" | Skipped {len(skipped)}"
                self._set_status(msg)

    def _refresh_table(self):
        df = self.current_df
        if df is None or df.empty:
            self._clear_table()
            return

        # Include My Data in table view if shown
        table_df = df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            overlay = self._prep_my_data_for_merge()
            if overlay is not None and not overlay.empty:
                table_df = table_df.join(overlay, how="outer")

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
        self.table.tag_configure("odd", background="#FFFFFF")
        self.table.tag_configure("even", background="#F2F4F7")

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

    # -------------------- Insights & Alerts --------------------
    def _refresh_insights(self):
        self.insights_text.configure(state="normal")
        self.insights_text.delete("1.0", "end")

        df = self.current_df
        if df is None or df.empty:
            self.insights_text.insert("end", "Run a query to see insights.")
            self.insights_text.configure(state="disabled")
            return

        # Include My Data in insights calculations if displayed
        work_df = df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            overlay = self._prep_my_data_for_merge()
            if overlay is not None and not overlay.empty:
                work_df = work_df.join(overlay, how="outer")

        last_idx = work_df.dropna(how="all").index.max()
        if pd.isna(last_idx):
            self.insights_text.insert("end", "No recent data to summarize.")
            self.insights_text.configure(state="disabled")
            return

        lines = []
        lines.append(f"Snapshot: {last_idx.date().isoformat()}\n")

        # Alerts banner
        alerts = self._compute_alerts()
        if alerts:
            lines.append("⚠ Alerts: " + " | ".join(alerts) + "\n")

        for col in work_df.columns:
            s = work_df[col].dropna()
            if s.empty:
                continue
            latest = s.iloc[-1]

            def delta(n):
                try:
                    prev = s.iloc[-n]
                    return latest - prev
                except Exception:
                    return np.nan

            d3, d6, d12 = delta(3), delta(6), delta(12)
            latest_str = f"{latest:,.2f}" if pd.notna(latest) else "—"
            d3_str = "—" if pd.isna(d3) else f"{d3:+.2f}"
            d6_str = "—" if pd.isna(d6) else f"{d6:+.2f}"
            d12_str = "—" if pd.isna(d12) else f"{d12:+.2f}"

            title = self.series_labels.get(col, col)
            lines.append(f"{title}")
            lines.append(f"  Latest: {latest_str} | Δ3m {d3_str} | Δ6m {d6_str} | Δ12m {d12_str}")
            lines.append(f"  → {self._series_insight(col, latest, d3, d12)}\n")

        txt = "\n".join(lines).strip() or "No series to summarize."
        self.insights_text.insert("end", txt)
        self.insights_text.configure(state="disabled")

    def _series_insight(self, sid: str, latest: float, d3: float, d12: float) -> str:
        sid_up = (sid or "").upper()

        def dir_str(x, up_word="rising", down_word="falling"):
            if pd.isna(x):
                return "stable"
            if x > 0:
                return up_word
            if x < 0:
                return down_word
            return "stable"

        if sid_up.startswith("[MY] "):
            sid_up = sid_up[5:]

        # Unemployment rate-like
        if "UNRATE" in sid_up or ("RATE" in sid_up and "UNEMP" in sid_up):
            trend = dir_str(d3, "rising", "falling")
            if pd.notna(latest):
                if latest <= 4.0:
                    risk = "tight labor market; expect higher turnover and wage pressure."
                elif latest <= 6.0:
                    risk = "balanced market; normal hiring difficulty."
                else:
                    risk = "slackening market; easier hiring, lower wage pressure."
            else:
                risk = "labor market conditions uncertain."
            return f"Unemployment is {trend}. Result: {risk}"

        # CPI headline inflation
        if "CPIAUCSL" in sid_up or ("CPI" in sid_up and "U" in sid_up):
            trend = dir_str(d3, "accelerating", "cooling")
            return f"Inflation {trend}. Plan COLA/merit budgets accordingly; watch real wage gap."

        # Payrolls / Manufacturing
        if "PAYEMS" in sid_up or ("CES" in sid_up and "300" in sid_up):
            trend = dir_str(d3, "expanding", "contracting")
            return f"Payrolls {trend}. Anticipate {'headcount growth' if d3>0 else 'hiring caution'}."

        # Avg hourly earnings
        if ("AHE" in sid_up) or (sid_up.startswith("CES") and sid_up.endswith("0008")):
            trend = dir_str(d3, "rising", "easing")
            return f"Wage growth {trend}. Align merit budgets; benchmark internal wages vs. market."

        trend = dir_str(d3, "up", "down")
        return f"Trend is {trend} over the last quarter; monitor vs. internal KPIs."

    def _compute_alerts(self) -> list[str]:
        """Simple business rules -> alerts."""
        alerts = []
        df = self.current_df
        if df is None or df.empty:
            return alerts
        # UNRATE very low
        if "UNRATE" in df.columns:
            try:
                val = float(df["UNRATE"].dropna().iloc[-1])
                if val < 4.0:
                    alerts.append(f"UNRATE {val:.2f}% < 4% → tight labor market")
            except Exception:
                pass
        # AHE YoY high (needs YoY transform)
        if "CES3000000008" in df.columns:
            try:
                y = df["CES3000000008"].dropna().pct_change(12).dropna() * 100
                if not y.empty and y.iloc[-1] > 4.0:
                    alerts.append(f"Manufacturing AHE YoY {y.iloc[-1]:.1f}% > 4%")
            except Exception:
                pass
        # CPI YoY high
        if "CPIAUCSL" in df.columns:
            try:
                y = df["CPIAUCSL"].dropna().pct_change(12).dropna() * 100
                if not y.empty and y.iloc[-1] > 3.0:
                    alerts.append(f"CPI YoY {y.iloc[-1]:.1f}% > 3%")
            except Exception:
                pass
        return alerts

    # -------------------- My Data overlay --------------------
    def _load_my_data(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            raw = pd.read_csv(path)
            if raw.empty:
                messagebox.showinfo("Info", "Selected CSV is empty.")
                return
            # Expect first column to be date; others numeric
            raw.columns = [str(c) for c in raw.columns]
            date_col = raw.columns[0]
            df = raw.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            # Attempt to coerce numerics
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # Resample to month-end for alignment with most FRED series
            df = df.resample("M").mean()
            # Prefix user columns to distinguish in legend
            df = df.rename(columns={c: f"My_{c}" for c in df.columns})
            self.my_df = df
            self.show_my_data.set(True)
            self._set_status(f"Loaded My Data: {path}")
            self._refresh_chart()
            self._refresh_table()
            self._refresh_insights()
        except Exception as e:
            self._error(e)

    def _prep_my_data_for_merge(self) -> pd.DataFrame | None:
        if not isinstance(self.my_df, pd.DataFrame) or self.my_df.empty:
            return None
        return self.my_df.copy()

    # -------------------- Save / Load Views --------------------
    def _save_view(self):
        """Save current UI state (series, dates, toggles, horizon) to a JSON file."""
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
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("View files", "*.json")],
            title="Save view",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self._set_status(f"Saved view → {path}")
        except Exception as e:
            self._error(e)

    def _load_view(self):
        """Load UI state from a JSON view file, update controls, and re-run."""
        import json
        path = filedialog.askopenfilename(
            filetypes=[("View files", "*.json")], title="Load view"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # restore selections
            self.selected_ids = list(state.get("selected_ids", []))
            self.series_labels = dict(state.get("series_labels", {}))

            self.sel_list.delete(0, "end")
            for sid in self.selected_ids:
                self.sel_list.insert("end", sid)

            # restore filters / toggles
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
    def _clear_filters(self):
        """Reset search/date/range/toggles/category filters and clear the results grid."""
        # search + dates
        self.q.set("")
        self.start_var.set("")
        self.end_var.set("")
        self.range_var.set("Max")

        # transforms / projection
        self.normalize_var.set(False)
        self.yoy_var.set(False)
        self.forecast_var.set(False)
        self.forecast_h_var.set("12m")
        self.scenario_var.set("Baseline")

        # category/subcategory pickers
        self.category_var.set("")
        self.category_cb.set("")
        self.subcategory_var.set("")
        self.subcategory_cb.set("")

        # clear the search results list (does NOT touch your Selected Series on purpose)
        self._clear_results()

        # optional: leave My Data as-is; uncomment next two lines if you want to reset it too
        # self.show_my_data.set(False)
        # self.my_df = None

        self._set_status("Filters cleared.")

    # -------------------- Exporters --------------------
    def _export_csv(self):
        if self.current_df is None or self.current_df.empty:
            messagebox.showinfo("Info", "Nothing to export yet.")
            return
        # include My Data if visible
        df = self.current_df.copy()
        if self.show_my_data.get() and isinstance(self.my_df, pd.DataFrame) and not self.my_df.empty:
            overlay = self._prep_my_data_for_merge()
            if overlay is not None and not overlay.empty:
                df = df.join(overlay, how="outer")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save data as CSV"
        )
        if not path:
            return
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df.to_csv(path, index_label="date")
            self._set_status(f"Saved CSV → {path}")
        except Exception as e:
            self._error(e)

    def _export_png(self):
        if not self.canvas:
            messagebox.showinfo("Info", "No chart to export yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")], title="Save chart as PNG"
        )
        if not path:
            return
        try:
            self.canvas.figure.savefig(path, dpi=144, bbox_inches="tight")
            self._set_status(f"Saved chart → {path}")
        except Exception as e:
            self._error(e)

    def _export_ppt(self):
        """Export a single Thermo Fisher–style slide with chart + insights."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.enum.text import PP_ALIGN
            from pptx.dml.color import RGBColor
        except Exception:
            messagebox.showinfo(
                "Missing dependency",
                "PowerPoint export requires the 'python-pptx' package.\n\nInstall with:\n  pip install python-pptx",
            )
            return

        if not self.canvas:
            messagebox.showinfo("Info", "No chart to export yet.")
            return

        # Save current chart to a temp PNG
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        self.canvas.figure.savefig(tmp.name, dpi=144, bbox_inches="tight")

        # Gather insights text
        self.insights_text.configure(state="normal")
        insight_str = self.insights_text.get("1.0", "end").strip()
        self.insights_text.configure(state="disabled")
        if not insight_str:
            insight_str = "No insights available."

        # Build PPT
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

        # Brand colors (Thermo Fisher red + slate)
        TF_RED = RGBColor(226, 0, 26)      # approximate Thermo Fisher red
        SLATE = RGBColor(45, 55, 72)

        # Title bar shape
        left, top, width, height = Inches(0), Inches(0), Inches(13.33), Inches(1.0)
        title_box = slide.shapes.add_shape(1, left, top, width, height)  # rectangle
        fill = title_box.fill
        fill.solid()
        fill.fore_color.rgb = TF_RED
        title_box.line.fill.background()

        # Title text
        title_tf = title_box.text_frame
        title_tf.clear()
        p = title_tf.paragraphs[0]
        p.text = "Labor Market Dashboard"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

        # Chart image
        slide.shapes.add_picture(tmp.name, Inches(0.5), Inches(1.2), height=Inches(4.5))

        # Insights text box
        tx_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.9), Inches(12.3), Inches(2.3))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.clear()
        p = tf.paragraphs[0]
        p.text = "Insights"
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = SLATE
        p.alignment = PP_ALIGN.LEFT

        for line in insight_str.splitlines():
            if not line.strip():
                continue
            p = tf.add_paragraph()
            p.text = line
            p.level = 1
            p.font.size = Pt(12)
            p.font.color.rgb = SLATE

        path = filedialog.asksaveasfilename(
            defaultextension=".pptx",
            filetypes=[("PowerPoint", "*.pptx")],
            title="Save dashboard slide",
        )
        if not path:
            return
        try:
            prs.save(path)
            self._set_status(f"Saved PowerPoint → {path}")
        except Exception as e:
            self._error(e)

    # -------------------- Helpers & Close --------------------
    def _bg(self, func, on_done):
        try:
            res = func()
            self.after(0, lambda: on_done(res))
        except Exception as e:
            self.after(0, lambda: self._error(e))

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def _error(self, e: Exception):
        self._set_status(f"Error: {e}")
        messagebox.showerror("Error", str(e))

    def _on_close(self):
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            self.master.quit()
            self.master.destroy()
        except Exception:
            pass
        import sys

        sys.exit(0)
