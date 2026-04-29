# agent_gui.py  ·  Dallas County Tax Defence Agent
# Run: python agent_gui.py

import os, sys, threading, pickle, warnings, platform
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

warnings.filterwarnings('ignore')

# ── THEME ─────────────────────────────────────────────────────────────────────
BG        = '#1a1612'   # deep warm brown-black
SURFACE   = '#211e19'   # card surface
RAISED    = '#2a261f'   # input fields
BORDER    = '#3d3830'   # subtle borders
GOLD      = '#c9a84c'   # primary accent
GOLD_DIM  = '#7a6530'   # muted gold
RED       = '#c0392b'   # reject
GREEN     = '#27ae60'   # accept
CREAM     = '#f0e6d0'   # primary text
CREAM_DIM = '#8a7d68'   # secondary text
HEADER_BG = '#12100d'   # darkest strip for header

# Map to existing variable names used in the GUI
ACCENT  = GOLD
SUCCESS = GREEN
DANGER  = RED
WARN    = GOLD_DIM
FG      = CREAM
FG_DIM  = CREAM_DIM

# Fonts – use serif for UI, monospace for code
_mac   = platform.system() == 'Darwin'
F_UI   = 'Times New Roman'          
F_CODE = 'Courier New'              
Fh  = (F_UI,  13, 'bold')
Fb  = (F_UI,  10)
Fs  = (F_UI,   9)
Fm  = (F_CODE, 10)

HIGH_VALUE_THRESHOLD = 10_000_000

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
_NEEDED = ['final_model.pkl', 'hv_model.pkl', 'feature_cols.pkl',
           'categorical_features.pkl', 'impute_values.pkl', 'cat_impute_values.pkl']
_miss = [f for f in _NEEDED if not os.path.exists(f'model/{f}')]
if _miss:
    sys.exit(f"ERROR: Missing model files: {_miss}")

print("Loading models…")
_ld = lambda f: pickle.load(open(f'model/{f}', 'rb'))

final_model       = _ld('final_model.pkl')
hv_model          = _ld('hv_model.pkl')
feature_cols      = _ld('feature_cols.pkl')
cat_feats         = _ld('categorical_features.pkl')
impute_num        = _ld('impute_values.pkl')
impute_cat        = _ld('cat_impute_values.pkl')

for _c in ['NBHD_CD', 'PROPERTY_ZIPCODE', 'SPTD_DESC', 'ZONING']:
    if _c not in cat_feats:
        cat_feats.append(_c)

explainer_main = shap.TreeExplainer(final_model)
explainer_hv   = shap.TreeExplainer(hv_model)
print("Ready.")

# ── PROPERTY TYPES (from DCAD SPTD_DESC values in notebook dataset) ───────────
PROPERTY_TYPES = [
    "COMMERCIAL IMPROVEMENTS",
    "OFFICE",
    "RETAIL",
    "WAREHOUSE",
    "INDUSTRIAL",
    "HOTEL/MOTEL",
    "APARTMENT",
    "MIXED USE",
    "RESTAURANT",
    "MEDICAL/DENTAL OFFICE",
    "BANK/FINANCIAL",
    "SERVICE STATION",
    "PARKING LOT/GARAGE",
    "OTHER",
]

# ── FEATURE ENGINEERING (mirrors notebook exactly) ────────────────────────────
_EXEMPT_KEYS = ['TOTAL_CNTY_EXEMPT', 'TOTAL_CITY_EXEMPT', 'TOTAL_ISD_EXEMPT',
                'TOTAL_HOSPITAL_EXEMPT', 'TOTAL_COLLEGE_EXEMPT', 'TOTAL_SPCL_EXEMPT']

def engineer(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Notebook cleaning mirrors
    if df['REMODEL_YR'].iloc[0] == 0:
        df['REMODEL_YR'] = df['YEAR_BUILT']
    df['NUM_STORIES']  = df['NUM_STORIES'].replace(0, 1)
    df['NUM_UNITS']    = df['NUM_UNITS'].replace(0, 1)
    df['TOT_DEPR_PCT'] = df['TOT_DEPR_PCT'].clip(upper=100)
    df['PROPERTY_ZIPCODE'] = (df['PROPERTY_ZIPCODE'].astype(str).str.strip()
                               .replace(['', 'nan', 'None'], 'MISSING'))

    # Engineered features — mirror notebook cell In[62]
    df['IMPR_VAL_PER_SQFT']  = df['VAL_AMT']         / (df['GROSS_BLDG_AREA'] + 1)
    df['BLDG_AREA_PER_UNIT'] = df['GROSS_BLDG_AREA']  / (df['NUM_UNITS']       + 1)
    df['EFFECTIVE_AGE']      = 2026 - df['REMODEL_YR']
    df['STRUCTURAL_AGE']     = 2026 - df['YEAR_BUILT']
    df['IS_RENOVATED']       = (df['REMODEL_YR'] > df['YEAR_BUILT']).astype(int)
    df['TOTAL_EXEMPTIONS']   = df[_EXEMPT_KEYS].sum(axis=1)
    df['PREV_VAL_PER_SQFT']  = df['PREV_MKT_VAL']    / (df['GROSS_BLDG_AREA'] + 1)
    df['FLOOR_AREA_RATIO']   = df['GROSS_BLDG_AREA']  / (df['AREA_SIZE']       + 1)
    df['DEPRECIATION_SPREAD']= df['TOT_DEPR_PCT']     -  df['PHYS_DEPR_PCT']
    df['EXEMPTION_RATIO']    = df['TOTAL_EXEMPTIONS'] / (df['PREV_MKT_VAL']    + 1)

    # Fill missing columns with training medians
    for col in feature_cols:
        if col not in df.columns:
            df[col] = impute_num.get(col, 0)
    for col, val in impute_num.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Categorical columns — use saved training modes
    for col in cat_feats:
        if col in df.columns:
            v = df[col].iloc[0]
            if v in ['', None] or (isinstance(v, float) and np.isnan(v)):
                df[col] = impute_cat.get(col, 'MISSING')
            df[col] = df[col].astype('category')

    # Coerce remaining objects
    for col in df.columns:
        if col not in cat_feats and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df[feature_cols]

# ── PREDICTION ────────────────────────────────────────────────────────────────
def predict(raw: dict, offer: float) -> dict:
    X    = engineer(raw)
    is_hv = raw.get('PREV_MKT_VAL', 0) > HIGH_VALUE_THRESHOLD

    # Floor always from main model (well-calibrated, 9.7% hit rate)
    floor = float(np.expm1(final_model.predict(X)[0]))

    # SHAP: use HV explainer for >$10M (captures tier-specific dynamics)
    exp = explainer_hv if is_hv else explainer_main
    sv  = exp.shap_values(X)[0]

    top5 = (pd.DataFrame({'feature': X.columns, 'sv': sv})
            .assign(abs_sv=lambda d: d.sv.abs())
            .sort_values('abs_sv', ascending=False).head(5))

    drivers = [{
        'feature':   r.feature,
        'direction': 'increased' if r.sv > 0 else 'decreased',
        'pct':       round(r.abs_sv / np.log1p(floor) * 100, 1) if floor else 0,
        'sv':        float(r.sv),
    } for r in top5.itertuples()]

    gap = floor - offer
    return dict(
        floor    = round(floor, 2),
        offer    = round(offer, 2),
        gap      = round(gap, 2),
        gap_pct  = round(gap / floor * 100 if floor else 0, 1),
        decision = 'REJECT' if offer < floor else 'ACCEPT',
        is_hv    = is_hv,
        drivers  = drivers,
        X        = X,
        sv       = sv,
        ev       = exp.expected_value,
    )

# ── HUMAN-READABLE FEATURE LABELS ─────────────────────────────────────────────
_LABELS = {
    'PREV_MKT_VAL':        lambda v: f"Prior certified value ${v:,.0f} — strongest historical signal.",
    'GROSS_BLDG_AREA':     lambda v: f"{v:,.0f} sqft — {'large commercial' if v > 20000 else 'mid-size'} tier.",
    'EFFECTIVE_AGE':       lambda v: f"Effective age {int(v)} yrs ({'recently renovated' if v < 15 else 'ageing stock'}).",
    'STRUCTURAL_AGE':      lambda v: f"Structural age {int(v)} yrs since original construction.",
    'FLOOR_AREA_RATIO':    lambda v: f"FAR {v:.2f} — {'dense' if v > 1 else 'low-density'} development.",
    'PREV_VAL_PER_SQFT':   lambda v: f"${v:.2f}/sqft prior-year rate.",
    'DEPRECIATION_SPREAD': lambda v: f"Spread {v:.1f}% — {'significant obsolescence' if v > 10 else 'minimal obsolescence'}.",
    'TOTAL_EXEMPTIONS':    lambda v: f"Total exemptions ${v:,.0f}.",
    'EXEMPTION_RATIO':     lambda v: f"Exemption ratio {v:.4f} — share of value shielded from tax.",
    'IMPR_VAL_PER_SQFT':   lambda v: f"${v:.2f}/sqft improvement value.",
    'BLDG_AREA_PER_UNIT':  lambda v: f"{v:,.0f} sqft per unit.",
    'VAL_AMT':             lambda v: f"Land value ${v:,.0f}.",
    'NBHD_CD':             lambda v: f"Neighborhood {v} — primary location driver.",
    'SPTD_DESC':           lambda v: f"Property type: {v}.",
    'PROPERTY_ZIPCODE':    lambda v: f"ZIP {v} — geographic market segment.",
    'ZONING':              lambda v: f"Zoning {v} — land-use class.",
    'NUM_UNITS':           lambda v: f"{int(v)} income-producing units.",
    'YEAR_BUILT':          lambda v: f"Built {int(v)}.",
    'REMODEL_YR':          lambda v: f"Last remodelled {int(v)}.",
}

def format_result(res: dict, raw: dict) -> str:
    d, f, o, g, gp = res['decision'], res['floor'], res['offer'], res['gap'], res['gap_pct']
    model_tag = "Main + HV SHAP" if res['is_hv'] else "Main"

    lines = [
        "=" * 58,
        f"  DECISION: {d}",
        "=" * 58,
        f"  Predicted Floor  :  ${f:>13,.0f}",
        f"  Lawyer's Offer   :  ${o:>13,.0f}",
        f"  Gap              :  ${g:>13,.0f}  ({gp:.1f}%)",
        f"  Model            :  {model_tag}",
        "",
        (f"  ${o:,.0f} falls {gp:.1f}% below the defensible floor.\n"
         f"  Recommend rejection at ARB."
         if d == 'REJECT'
         else f"  ${o:,.0f} meets or exceeds the floor. Acceptable."),
        "",
        "─" * 58,
        "  TOP VALUATION DRIVERS",
        "─" * 58,
        "",
    ]

    for i, drv in enumerate(res['drivers'], 1):
        feat    = drv['feature']
        val_raw = res['X'][feat].iloc[0]
        arrow   = "▲" if drv['direction'] == 'increased' else "▼"
        lines.append(f"  {i}. {arrow} {feat}  ({drv['direction']}, {drv['pct']:.1f}%)")
        fn = _LABELS.get(feat)
        try:
            v = val_raw if feat in cat_feats else float(val_raw)
            lines.append(f"       {fn(v)}" if fn else f"       Value: {v}")
        except Exception:
            lines.append(f"       Value: {val_raw}")
        lines.append("")

    lines += [
        "─" * 58,
        "  PROPERTY SUMMARY",
        "─" * 58,
        f"  ZIP              :  {raw.get('PROPERTY_ZIPCODE', 'N/A')}",
        f"  Neighborhood     :  {raw.get('NBHD_CD', 'N/A')}",
        f"  Type             :  {raw.get('SPTD_DESC', 'N/A')}",
        f"  Building Area    :  {raw.get('GROSS_BLDG_AREA', 0):,.0f} sqft",
        f"  Year Built       :  {raw.get('YEAR_BUILT', 0)}",
        f"  Remodel Year     :  {raw.get('REMODEL_YR', 0)}",
        f"  Stories          :  {raw.get('NUM_STORIES', 1)}",
        f"  Prior Value      :  ${raw.get('PREV_MKT_VAL', 0):,.0f}",
        f"  Total Deprec.    :  {raw.get('TOT_DEPR_PCT', 0):.1f}%",
        "=" * 58,
    ]
    return "\n".join(lines)

# ── WIDGET HELPERS ────────────────────────────────────────────────────────────
def btn(parent, text, cmd, bg=ACCENT, fg='white', **kw):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=Fb,
                     relief='flat', bd=0, activebackground=bg,
                     activeforeground=fg, cursor='hand2', **kw)

def entry(parent, var):
    return tk.Entry(parent, textvariable=var, bg=RAISED, fg=FG,
                    insertbackground=FG, font=Fb, relief='flat', bd=0,
                    highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=ACCENT)

def field(parent, label, var, padx=0):
    tk.Label(parent, text=label, bg=SURFACE, fg=FG_DIM, font=Fs).pack(
        anchor='w', pady=(5, 1), padx=padx)
    entry(parent, var).pack(fill=tk.X, ipady=6, pady=(0, 2), padx=padx)

def pair(parent, la, va, lb, vb):
    row = tk.Frame(parent, bg=SURFACE)
    row.pack(fill=tk.X)
    for lbl, var, pad in [(la, va, (0, 3)), (lb, vb, (0, 0))]:
        col = tk.Frame(row, bg=SURFACE)
        col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=pad)
        tk.Label(col, text=lbl, bg=SURFACE, fg=FG_DIM, font=Fs).pack(
            anchor='w', pady=(5, 1))
        entry(col, var).pack(fill=tk.X, ipady=6, pady=(0, 2))

def section(parent, text):
    f = tk.Frame(parent, bg=SURFACE)
    f.pack(fill=tk.X, pady=(14, 3))
    tk.Frame(f, bg=ACCENT, width=3).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {text}", bg=SURFACE, fg=FG,
             font=(F_UI, 10, 'bold')).pack(side=tk.LEFT, anchor='w')

def bind_scroll(canvas: tk.Canvas):
    def _scroll(e):
        if _mac:
            canvas.yview_scroll(int(-e.delta / 8), 'units')
        else:
            canvas.yview_scroll(int(-e.delta / 120), 'units')

    def _recurse(w):
        w.bind('<MouseWheel>', _scroll)
        w.bind('<Button-4>', lambda e: canvas.yview_scroll(-3, 'units'))
        w.bind('<Button-5>', lambda e: canvas.yview_scroll(3, 'units'))
        for c in w.winfo_children():
            _recurse(c)

    canvas.bind('<MouseWheel>', _scroll)
    return _recurse

# ── CUSTOM DROPDOWN (theme-native, replaces ttk.Combobox) ────────────────────
class Dropdown(tk.Frame):
    """
    A dark-themed dropdown that matches the app palette exactly.
    Uses a Toplevel popup list instead of the platform-native ttk widget,
    so colours and fonts are fully controllable.
    """
    def __init__(self, parent, variable: tk.StringVar, values: list, **kw):
        super().__init__(parent, bg=RAISED,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=ACCENT, **kw)
        self._var    = variable
        self._values = values
        self._popup  = None

        self._lbl = tk.Label(self, textvariable=variable, bg=RAISED, fg=FG,
                             font=Fb, anchor='w', padx=8, pady=6, cursor='hand2')
        self._lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Chevron indicator
        self._arr = tk.Label(self, text='⌄', bg=RAISED, fg=FG_DIM,
                             font=(F_UI, 11), padx=6, cursor='hand2')
        self._arr.pack(side=tk.RIGHT)

        for w in (self, self._lbl, self._arr):
            w.bind('<Button-1>', self._toggle)
            w.bind('<FocusOut>', self._close_if_lost)

    def _toggle(self, _=None):
        if self._popup and self._popup.winfo_exists():
            self._close()
        else:
            self._open()

    def _open(self):
        self._close()
        self.config(highlightbackground=ACCENT)
        pop = tk.Toplevel(self)
        pop.wm_overrideredirect(True)
        pop.configure(bg=BORDER)

        # Position below the widget
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        w = self.winfo_width()
        pop.geometry(f"{w}x{min(len(self._values), 10) * 28}+{x}+{y}")

        cvs = tk.Canvas(pop, bg=RAISED, highlightthickness=0, bd=0)
        sb  = tk.Scrollbar(pop, orient='vertical', command=cvs.yview,
                           bg=RAISED, troughcolor=BG, bd=0, relief='flat')
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = tk.Frame(cvs, bg=RAISED)
        cvs.create_window((0, 0), window=frm, anchor='nw')
        frm.bind('<Configure>', lambda e: cvs.configure(scrollregion=cvs.bbox('all')))

        current = self._var.get()
        for val in self._values:
            is_sel = val == current
            item_bg = ACCENT if is_sel else RAISED
            item_fg = 'white' if is_sel else FG
            row = tk.Frame(frm, bg=item_bg, cursor='hand2')
            row.pack(fill=tk.X)
            lbl = tk.Label(row, text=val, bg=item_bg, fg=item_fg,
                           font=Fb, anchor='w', padx=12, pady=5)
            lbl.pack(fill=tk.X)
            for w in (row, lbl):
                w.bind('<Button-1>', lambda e, v=val: self._select(v))
                w.bind('<Enter>',    lambda e, r=row, l=lbl, s=is_sel:
                       (r.config(bg=ACCENT if not s else ACCENT),
                        l.config(bg=ACCENT if not s else ACCENT, fg='white')))
                w.bind('<Leave>',    lambda e, r=row, l=lbl, s=is_sel, ib=item_bg, _f=item_fg:
                       (r.config(bg=ib), l.config(bg=ib, fg=_f)))

        pop.bind('<FocusOut>', self._close_if_lost)
        pop.bind('<Escape>',   lambda e: self._close())
        cvs.bind('<MouseWheel>', lambda e: cvs.yview_scroll(
            int(-e.delta / (8 if _mac else 120)), 'units'))
        pop.focus_set()
        self._popup = pop

    def _select(self, val):
        self._var.set(val)
        self._close()

    def _close(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None
        self.config(highlightbackground=BORDER)

    def _close_if_lost(self, e):
        self.after(100, lambda: self._close()
                   if not (self._popup and self._popup.winfo_exists()
                           and self._popup.focus_get()) else None)

# ── POPUPS ────────────────────────────────────────────────────────────────────
def show_shap(res: dict):
    pop = tk.Toplevel()
    pop.title("SHAP Waterfall — Valuation Drivers")
    pop.geometry("820x520")
    pop.configure(bg=BG)
    plt.clf()
    shap.waterfall_plot(shap.Explanation(
        values        = res['sv'],
        base_values   = res['ev'],
        data          = res['X'].iloc[0],
        feature_names = res['X'].columns.tolist(),
    ), show=False, max_display=10)
    plt.tight_layout(pad=0.8)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=pop)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    btn(pop, "Close", pop.destroy, DANGER).pack(pady=8)

def show_scenario(floor: float):
    pop = tk.Toplevel()
    pop.title("Scenario Analysis")
    pop.geometry("420x290")
    pop.configure(bg=BG)
    pop.resizable(False, False)

    tk.Label(pop, text="Scenario Analysis", bg=BG, fg=FG,    font=Fh).pack(pady=(20, 4))
    tk.Label(pop, text=f"Floor: ${floor:,.0f}", bg=BG, fg=FG_DIM, font=Fb).pack()
    tk.Label(pop, text="Test an alternative offer ($):", bg=BG, fg=FG_DIM,
             font=Fs).pack(pady=(14, 3))

    var = tk.StringVar(value=str(int(floor * .90)))
    tk.Entry(pop, textvariable=var, bg=RAISED, fg=FG, insertbackground=FG,
             font=Fb, relief='flat', highlightthickness=1,
             highlightbackground=BORDER, highlightcolor=ACCENT).pack(
             ipady=7, padx=50, fill=tk.X)

    result_lbl = tk.Label(pop, text="", bg=BG, font=(F_UI, 11, 'bold'), wraplength=380)
    result_lbl.pack(pady=12)

    def evaluate():
        try:
            alt = float(var.get().replace(',', ''))
        except ValueError:
            result_lbl.config(text="Enter a valid number.", fg=DANGER)
            return
        if alt >= floor:
            result_lbl.config(
                text=f"✅  ACCEPT — ${alt:,.0f} meets the floor.", fg=SUCCESS)
        else:
            deficit = floor - alt
            result_lbl.config(
                text=f"❌  REJECT — ${alt:,.0f} is ${deficit:,.0f} "
                     f"({deficit / floor * 100:.1f}%) below floor.",
                fg=DANGER)

    btn(pop, "Evaluate", evaluate, ACCENT).pack(pady=(0, 4))
    btn(pop, "Close",    pop.destroy, RAISED, FG_DIM).pack()

# ── MAIN APPLICATION ──────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.result  = None
        self.raw     = None
        root.title("Dallas County · Tax Defence Agent")
        root.geometry("1180x840")
        root.minsize(900, 640)
        root.configure(bg=BG)
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=HEADER_BG, pady=16)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Dallas County  ·  Tax Defence Agent",
                 bg=HEADER_BG, fg=GOLD, font=Fh).pack()
        tk.Label(hdr,
                 text="Minimum defensible settlement floor  ·  LightGBM quantile regression  ·  10th-percentile calibrated",
                 bg=HEADER_BG, fg=GOLD, font=Fs).pack(pady=(3, 0))
        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill=tk.X)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        left = tk.Frame(body, bg=SURFACE, width=390)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)
        self._build_form(left)

        right = tk.Frame(body, bg=SURFACE)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_results(right)

    def _build_form(self, parent):
        tk.Label(parent, text="Property Details", bg=SURFACE, fg=FG,
                 font=(F_UI, 12, 'bold')).pack(anchor='w', padx=14, pady=(14, 6))

        cvs = tk.Canvas(parent, bg=SURFACE, highlightthickness=0, bd=0)
        sb  = ttk.Scrollbar(parent, orient='vertical', command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = tk.Frame(cvs, bg=SURFACE, padx=14)
        win = cvs.create_window((0, 0), window=frm, anchor='nw')
        frm.bind('<Configure>', lambda e: cvs.configure(scrollregion=cvs.bbox('all')))
        cvs.bind('<Configure>', lambda e: cvs.itemconfig(win, width=e.width))

        recurse = bind_scroll(cvs)
        parent.after(200, lambda: recurse(frm))

        # ── Building ──
        section(frm, "Building")
        self.v_area    = tk.StringVar(value='0')
        self.v_land    = tk.StringVar(value='0')
        field(frm, 'Building Area (sqft) *', self.v_area)
        field(frm, 'Land Area (sqft)',        self.v_land)

        self.v_yr      = tk.StringVar(value='2000')
        self.v_rem     = tk.StringVar(value='0')
        self.v_stories = tk.StringVar(value='1')
        self.v_units   = tk.StringVar(value='1')
        pair(frm, 'Year Built *',            self.v_yr,      'Remodel Year (0=none)', self.v_rem)
        pair(frm, 'Stories',                 self.v_stories, 'Units',                 self.v_units)

        self.v_taxobj = tk.StringVar(value='1')
        field(frm, 'Number of Tax Objects', self.v_taxobj)

        # ── Valuation ──
        section(frm, "Valuation")
        self.v_land_val = tk.StringVar(value='0')
        self.v_prev     = tk.StringVar(value='0')
        field(frm, 'Land Value ($)',                   self.v_land_val)
        field(frm, 'Prior Year Certified Value ($) *', self.v_prev)

        # ── Location & Classification ──
        section(frm, "Location & Classification")
        self.v_zip  = tk.StringVar(value='75201')
        self.v_nbhd = tk.StringVar(value='')
        self.v_zone = tk.StringVar(value='O')
        field(frm, 'ZIP Code',          self.v_zip)
        field(frm, 'Neighborhood Code', self.v_nbhd)

        tk.Label(frm, text='Property Type', bg=SURFACE, fg=FG_DIM,
                 font=Fs).pack(anchor='w', pady=(5, 1))
        self.v_sptd = tk.StringVar(value='COMMERCIAL IMPROVEMENTS')
        Dropdown(frm, self.v_sptd, PROPERTY_TYPES).pack(
            fill=tk.X, pady=(0, 4))

        field(frm, 'Zoning Code', self.v_zone)

        # ── Depreciation ──
        section(frm, "Depreciation (%)")
        self.v_tot_d  = tk.StringVar(value='0.0')
        self.v_phys_d = tk.StringVar(value='0.0')
        self.v_func_d = tk.StringVar(value='0.0')
        self.v_ext_d  = tk.StringVar(value='0.0')
        pair(frm, 'Total %',      self.v_tot_d,  'Physical %',  self.v_phys_d)
        pair(frm, 'Functional %', self.v_func_d, 'External %',  self.v_ext_d)

        # ── Exemptions ──
        section(frm, "Exemptions ($)")
        _labels = ['County', 'City', 'ISD', 'Hospital', 'College', 'Special']
        self.v_exempts = [tk.StringVar(value='0') for _ in _labels]
        for i in range(0, 6, 2):
            pair(frm, _labels[i], self.v_exempts[i],
                      _labels[i+1], self.v_exempts[i+1])

        # ── Protest Details ──
        section(frm, "Protest Details")
        self.v_offer = tk.StringVar(value='0')
        field(frm, "Lawyer's Proposed Value ($) *", self.v_offer)

        tk.Frame(frm, bg=SURFACE, height=8).pack()
        btn(frm, "  🔍  Analyse Protest", self._on_analyse, pady=12).pack(
            fill=tk.X, pady=(8, 3))
        btn(frm, "↺  Clear", self._on_clear, RAISED, FG_DIM, pady=7).pack(
            fill=tk.X, pady=(0, 20))

    def _build_results(self, parent):
        tk.Label(parent, text="Analysis Results", bg=SURFACE, fg=FG,
                 font=(F_UI, 12, 'bold')).pack(anchor='w', padx=14, pady=(14, 6))

        self.status_var = tk.StringVar(
            value="Enter property details and click Analyse Protest.")
        self.status_lbl = tk.Label(
            parent, textvariable=self.status_var,
            bg=RAISED, fg=FG_DIM, font=(F_UI, 11, 'bold'),
            pady=12, padx=14, wraplength=640, justify='center')
        self.status_lbl.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.txt = scrolledtext.ScrolledText(
            parent, font=Fm, bg='#050505', fg=FG,
            insertbackground=FG, relief='flat', bd=0,
            padx=14, pady=12, state='disabled', wrap=tk.WORD)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=12)

        for tag, fg_c, size, bold in [
            ('reject', RED,   11, True),
            ('accept', GREEN,  11, True),
            ('head',   GOLD,   10, True),
            ('normal', CREAM,       10, False),
        ]:
            self.txt.tag_config(tag, foreground=fg_c,
                                font=(F_CODE, size, 'bold' if bold else 'normal'))

        bf = tk.Frame(parent, bg=SURFACE)
        bf.pack(fill=tk.X, padx=12, pady=10)

        self.shap_btn = btn(bf, "📊  SHAP Chart", self._on_shap,
                            RAISED, '#818cf8', padx=18, pady=8, state='disabled')
        self.shap_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.scen_btn = btn(bf, "🔄  Scenario", self._on_scenario,
                            RAISED, WARN, padx=18, pady=8, state='disabled')
        self.scen_btn.pack(side=tk.LEFT)

        self.hv_lbl = tk.Label(bf, text="", bg=SURFACE, fg=CREAM_DIM,
                                font=(F_UI, 9, 'bold'))
        self.hv_lbl.pack(side=tk.LEFT, padx=(16, 0))

    # ── INPUT HELPERS ─────────────────────────────────────────────────────────
    def _flt(self, v, d=0.0):
        try:    return float(str(v.get()).replace(',', ''))
        except: return d

    def _int(self, v, d=0):
        try:    return int(float(str(v.get()).replace(',', '')))
        except: return d

    def _collect(self) -> dict:
        area = self._flt(self.v_area)
        _EKEYS = ['TOTAL_CNTY_EXEMPT', 'TOTAL_CITY_EXEMPT', 'TOTAL_ISD_EXEMPT',
                  'TOTAL_HOSPITAL_EXEMPT', 'TOTAL_COLLEGE_EXEMPT', 'TOTAL_SPCL_EXEMPT']
        return {
            'GROSS_BLDG_AREA':  area,
            'AREA_SIZE':        self._flt(self.v_land) or area,
            'YEAR_BUILT':       self._int(self.v_yr, 2000),
            'REMODEL_YR':       self._int(self.v_rem, 0),
            'NUM_STORIES':      self._int(self.v_stories, 1),
            'NUM_UNITS':        self._int(self.v_units, 1),
            'NUM_TAX_OBJECTS':  self._int(self.v_taxobj, 1),
            'VAL_AMT':          self._flt(self.v_land_val),
            'PREV_MKT_VAL':     self._flt(self.v_prev),
            'TOT_DEPR_PCT':     self._flt(self.v_tot_d),
            'PHYS_DEPR_PCT':    self._flt(self.v_phys_d),
            'FUNCT_DEPR_PCT':   self._flt(self.v_func_d),
            'EXTRNL_DEPR_PCT':  self._flt(self.v_ext_d),
            'PROPERTY_ZIPCODE': self.v_zip.get().strip() or 'MISSING',
            'NBHD_CD':          self.v_nbhd.get().strip() or impute_cat.get('NBHD_CD', 'UNASSIGNED'),
            'SPTD_DESC':        self.v_sptd.get(),
            'ZONING':           self.v_zone.get().strip() or impute_cat.get('ZONING', 'O'),
            **{k: self._flt(v) for k, v in zip(_EKEYS, self.v_exempts)},
        }

    # ── STATUS / TEXT WRITERS ─────────────────────────────────────────────────
    def _set_status(self, text, colour):
        self.status_var.set(text)
        self.status_lbl.config(fg=colour)

    def _write(self, text):
        self.txt.config(state='normal')
        self.txt.delete('1.0', tk.END)
        for line in text.split('\n'):
            tag = ('reject' if 'DECISION: REJECT' in line else
                   'accept' if 'DECISION: ACCEPT' in line else
                   'head'   if any(k in line for k in ['═', '─', 'DRIVERS', 'SUMMARY']) else
                   'normal')
            self.txt.insert(tk.END, line + '\n', tag)
        self.txt.config(state='disabled')
        self.txt.see('1.0')

    # ── EVENTS ────────────────────────────────────────────────────────────────
    def _on_analyse(self):
        if self._flt(self.v_offer) <= 0:
            messagebox.showerror("Missing Input", "Please enter the lawyer's proposed value.")
            return
        if self._flt(self.v_area) <= 0:
            messagebox.showerror("Missing Input", "Please enter a valid building area.")
            return

        self._set_status("⏳  Analysing…", WARN)
        self.shap_btn.config(state='disabled')
        self.scen_btn.config(state='disabled')
        self.hv_lbl.config(text="")
        self.root.update()

        def _run():
            try:
                raw    = self._collect()
                res    = predict(raw, self._flt(self.v_offer))
                text   = format_result(res, raw)
                self.result, self.raw = res, raw

                f, o, g, gp = res['floor'], res['offer'], res['gap'], res['gap_pct']
                dec = res['decision']
                status = (f"❌  REJECT — ${o:,.0f} is ${g:,.0f} ({gp:.1f}%) below floor ${f:,.0f}"
                          if dec == 'REJECT'
                          else f"✅  ACCEPT — ${o:,.0f} meets or exceeds floor ${f:,.0f}")
                colour = DANGER if dec == 'REJECT' else SUCCESS
                hv_tag = "⬡ HV SHAP active" if res['is_hv'] else ""

                self.root.after(0, lambda: self._set_status(status, colour))
                self.root.after(0, lambda: self._write(text))
                self.root.after(0, lambda: self.shap_btn.config(state='normal'))
                self.root.after(0, lambda: self.scen_btn.config(state='normal'))
                self.root.after(0, lambda: self.hv_lbl.config(text=hv_tag))

            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}", DANGER))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_clear(self):
        for var, val in [
            (self.v_area, '0'), (self.v_land, '0'), (self.v_yr, '2000'),
            (self.v_rem, '0'),  (self.v_stories, '1'), (self.v_units, '1'),
            (self.v_taxobj, '1'), (self.v_land_val, '0'), (self.v_prev, '0'),
            (self.v_zip, '75201'), (self.v_nbhd, ''), (self.v_zone, 'O'),
            (self.v_offer, '0'),   (self.v_tot_d, '0.0'), (self.v_phys_d, '0.0'),
            (self.v_func_d, '0.0'), (self.v_ext_d, '0.0'),
        ]:
            var.set(val)
        for v in self.v_exempts:
            v.set('0')
        self.v_sptd.set('COMMERCIAL IMPROVEMENTS')
        self.result = self.raw = None
        self._set_status("Enter property details and click Analyse Protest.", FG_DIM)
        self.txt.config(state='normal')
        self.txt.delete('1.0', tk.END)
        self.txt.config(state='disabled')
        self.shap_btn.config(state='disabled')
        self.scen_btn.config(state='disabled')
        self.hv_lbl.config(text="")

    def _on_shap(self):
        if self.result:
            show_shap(self.result)

    def _on_scenario(self):
        if self.result:
            show_scenario(self.result['floor'])

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.quit)
    App(root)
    root.mainloop()