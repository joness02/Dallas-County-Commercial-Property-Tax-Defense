# agent_chat.py — Dallas County Tax Defence Agent (Conversational)
# Run: python agent_chat.py

import json, pickle, threading, warnings
import numpy as np
import pandas as pd
import shap
import tkinter as tk
from tkinter import font as tkfont
from groq import Groq
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = "[YOUR_API_KEY_HERE]"
GROQ_MODEL   = "llama-3.3-70b-versatile"
HV_THRESHOLD = 10_000_000

# ── PALETTE — warm parchment / legal document aesthetic ───────────────────────
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
HEADER_BG = '#12100d'   # darkest strip

F_SERIF = 'Times New Roman'
F_MONO  = 'Courier New'

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models…")
_ld = lambda f: pickle.load(open(f'model/{f}', 'rb'))
final_model    = _ld('final_model.pkl')
hv_model       = _ld('hv_model.pkl')
feature_cols   = _ld('feature_cols.pkl')
cat_feats      = _ld('categorical_features.pkl')
impute_num     = _ld('impute_values.pkl')
impute_cat     = _ld('cat_impute_values.pkl')
explainer_main = shap.TreeExplainer(final_model)
explainer_hv   = shap.TreeExplainer(hv_model)
print("Ready.")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
_EXEMPT = ['TOTAL_CNTY_EXEMPT','TOTAL_CITY_EXEMPT','TOTAL_ISD_EXEMPT',
           'TOTAL_HOSPITAL_EXEMPT','TOTAL_COLLEGE_EXEMPT','TOTAL_SPCL_EXEMPT']

def engineer(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    if df['REMODEL_YR'].iloc[0] == 0:
        df['REMODEL_YR'] = df['YEAR_BUILT']
    df['NUM_STORIES']  = df['NUM_STORIES'].replace(0, 1)
    df['NUM_UNITS']    = df['NUM_UNITS'].replace(0, 1)
    df['TOT_DEPR_PCT'] = df['TOT_DEPR_PCT'].clip(upper=100)
    df['PROPERTY_ZIPCODE'] = df['PROPERTY_ZIPCODE'].astype(str).str.strip().replace(['','nan','None'], 'MISSING')
    df['IMPR_VAL_PER_SQFT']  = df['VAL_AMT']        / (df['GROSS_BLDG_AREA'] + 1)
    df['BLDG_AREA_PER_UNIT'] = df['GROSS_BLDG_AREA'] / (df['NUM_UNITS']       + 1)
    df['EFFECTIVE_AGE']      = 2026 - df['REMODEL_YR']
    df['STRUCTURAL_AGE']     = 2026 - df['YEAR_BUILT']
    df['IS_RENOVATED']       = (df['REMODEL_YR'] > df['YEAR_BUILT']).astype(int)
    df['TOTAL_EXEMPTIONS']   = df[_EXEMPT].sum(axis=1)
    df['PREV_VAL_PER_SQFT']  = df['PREV_MKT_VAL']   / (df['GROSS_BLDG_AREA'] + 1)
    df['FLOOR_AREA_RATIO']   = df['GROSS_BLDG_AREA'] / (df['AREA_SIZE']       + 1)
    df['DEPRECIATION_SPREAD']= df['TOT_DEPR_PCT']    -  df['PHYS_DEPR_PCT']
    df['EXEMPTION_RATIO']    = df['TOTAL_EXEMPTIONS']/ (df['PREV_MKT_VAL']    + 1)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = impute_num.get(col, 0)
    for col, val in impute_num.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in cat_feats:
        if col in df.columns:
            v = df[col].iloc[0]
            if v in ['', None] or (isinstance(v, float) and np.isnan(v)):
                df[col] = impute_cat.get(col, 'MISSING')
            df[col] = df[col].astype('category')
    for col in df.columns:
        if col not in cat_feats and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df[feature_cols]

# ── TOOLS ─────────────────────────────────────────────────────────────────────
def tool_analyse(props: dict, offer: float) -> dict:
    X     = engineer(props)
    is_hv = props.get('PREV_MKT_VAL', 0) > HV_THRESHOLD
    floor = float(np.expm1(final_model.predict(X)[0]))
    exp   = explainer_hv if is_hv else explainer_main
    sv    = exp.shap_values(X)[0]
    top5  = (pd.DataFrame({'feature': X.columns, 'sv': sv})
             .assign(abs=lambda d: d.sv.abs())
             .sort_values('abs', ascending=False).head(5))
    drivers = [{'feature': r.feature,
                'direction': 'increased' if r.sv > 0 else 'decreased',
                'pct': round(r.abs / np.log1p(floor) * 100, 1)}
               for r in top5.itertuples()]
    gap = floor - offer
    return {'floor': round(floor, 2), 'offer': round(offer, 2),
            'gap': round(gap, 2), 'gap_pct': round(gap/floor*100 if floor else 0, 1),
            'decision': 'REJECT' if offer < floor else 'ACCEPT',
            'model': 'HV specialist SHAP' if is_hv else 'Main',
            'drivers': drivers}

def tool_scenario(floor: float, alt_offer: float) -> dict:
    gap = floor - alt_offer
    return {'floor': round(floor, 2), 'alt_offer': round(alt_offer, 2),
            'gap': round(gap, 2), 'gap_pct': round(gap/floor*100 if floor else 0, 1),
            'revenue_loss': round(max(0, gap), 2),
            'decision': 'ACCEPT' if alt_offer >= floor else 'REJECT'}

TOOLS = [
    {"type": "function", "function": {
        "name": "analyse_protest",
        "description": "Analyse a commercial property tax protest. Call when the user provides property details and a lawyer's offer.",
        "parameters": {"type": "object", "properties": {
            "property_details": {"type": "object", "description":
                "All known property fields: GROSS_BLDG_AREA, AREA_SIZE, YEAR_BUILT, REMODEL_YR, "
                "NUM_STORIES, NUM_UNITS, VAL_AMT, PREV_MKT_VAL, TOT_DEPR_PCT, PHYS_DEPR_PCT, "
                "FUNCT_DEPR_PCT, EXTRNL_DEPR_PCT, PROPERTY_ZIPCODE, NBHD_CD, SPTD_DESC, ZONING, "
                "NUM_TAX_OBJECTS, TOTAL_CNTY_EXEMPT, TOTAL_CITY_EXEMPT, TOTAL_ISD_EXEMPT, "
                "TOTAL_HOSPITAL_EXEMPT, TOTAL_COLLEGE_EXEMPT, TOTAL_SPCL_EXEMPT. Use 0 for unknowns."},
            "lawyers_offer": {"type": "number", "description": "Lawyer's proposed settlement in dollars."}},
            "required": ["property_details", "lawyers_offer"]}}},
    {"type": "function", "function": {
        "name": "scenario_analysis",
        "description": "Evaluate a what-if alternative offer against a known floor.",
        "parameters": {"type": "object", "properties": {
            "floor":     {"type": "number", "description": "Previously predicted floor."},
            "alt_offer": {"type": "number", "description": "Alternative offer to evaluate."}},
            "required": ["floor", "alt_offer"]}}}
]

SYSTEM = """You are the Dallas County Commercial Tax Defence Agent.

Your job: help county appraisers evaluate commercial property tax protests.
When given property details and a lawyer's offer, call analyse_protest, then explain the result clearly.

Rules:
- Lead every analysis with REJECT or ACCEPT
- State floor and gap in exact dollar amounts
- Explain each SHAP driver in plain English (no jargon)
- For what-if questions, call scenario_analysis with the floor from the last analysis
- If details are partial, use what's given and state what you assumed
- Be direct — this is an operational tool, not a chatbot
- Never suggest accepting below floor without stating the exact revenue loss"""

# ── AGENT LOOP ────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

def dispatch(name: str, args: dict) -> str:
    if name == 'analyse_protest':
        return json.dumps(tool_analyse(args['property_details'], args['lawyers_offer']))
    if name == 'scenario_analysis':
        return json.dumps(tool_scenario(args['floor'], args['alt_offer']))
    return json.dumps({'error': f'Unknown tool: {name}'})

def get_response(history: list) -> str:
    while True:
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=history,
            tools=TOOLS, tool_choice='auto',
            max_tokens=1024, temperature=0.3)
        msg = resp.choices[0].message
        if msg.tool_calls:
            history.append({'role': 'assistant', 'content': msg.content or '',
                            'tool_calls': [{'id': tc.id, 'type': 'function',
                                'function': {'name': tc.function.name,
                                             'arguments': tc.function.arguments}}
                                           for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                history.append({'role': 'tool', 'tool_call_id': tc.id,
                                'content': dispatch(tc.function.name,
                                                    json.loads(tc.function.arguments))})
        else:
            text = msg.content or ''
            history.append({'role': 'assistant', 'content': text})
            return text

# ── THINKING DOTS ANIMATOR ────────────────────────────────────────────────────
class ThinkingDots:
    """Pulses a label with animated dots while agent is processing."""
    def __init__(self, label: tk.Label):
        self.label   = label
        self.running = False
        self.step    = 0

    def start(self):
        self.running = True
        self.step    = 0
        self._tick()

    def _tick(self):
        if not self.running:
            return
        self.label.config(text='  Analysing' + ('·' * (self.step % 4)))
        self.step += 1
        self.label.after(380, self._tick)

    def stop(self):
        self.running = False
        self.label.config(text='')

# ── MAIN APPLICATION ──────────────────────────────────────────────────────────
class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.history = [{'role': 'system', 'content': SYSTEM}]
        self._setup_window()
        self._build()
        self._agent_msg(
            "Ready. Describe the protest — property size, year built, location, "
            "depreciation, prior year value, and the lawyer's offer.\n"
            "I will predict the defensible floor and advise whether to reject or accept.")

    def _setup_window(self):
        self.root.title("Dallas County  ·  Tax Defence Agent")
        self.root.geometry("920x740")
        self.root.minsize(700, 500)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    # ── HEADER ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HEADER_BG)
        hdr.pack(fill=tk.X)
        tk.Frame(hdr, bg=GOLD, height=2).pack(fill=tk.X)

        inner = tk.Frame(hdr, bg=HEADER_BG, pady=12)
        inner.pack(fill=tk.X, padx=20)

        # Seal icon left
        tk.Label(inner, text='⚖', bg=HEADER_BG, fg=GOLD,
                 font=(F_SERIF, 28)).pack(side=tk.LEFT, padx=(0, 14))

        # Title block centre
        centre = tk.Frame(inner, bg=HEADER_BG)
        centre.pack(side=tk.LEFT, expand=True)
        tk.Label(centre,
                 text='DALLAS COUNTY APPRAISAL DISTRICT',
                 bg=HEADER_BG, fg=GOLD,
                 font=(F_SERIF, 9, 'bold')).pack()
        tk.Label(centre,
                 text='Commercial Tax Defence Agent',
                 bg=HEADER_BG, fg=CREAM,
                 font=(F_SERIF, 18, 'bold')).pack(pady=(1, 0))
        tk.Label(centre,
                 text='Protest Analysis  ·  ARB Settlement Floor  ·  LightGBM + Groq',
                 bg=HEADER_BG, fg=CREAM_DIM,
                 font=(F_SERIF, 9, 'italic')).pack(pady=(2, 0))

        tk.Frame(hdr, bg=GOLD_DIM, height=1).pack(fill=tk.X, pady=(2, 0))
        tk.Frame(hdr, bg=GOLD,     height=1).pack(fill=tk.X)

    # ── CHAT AREA ─────────────────────────────────────────────────────────────
    def _build_chat(self):
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 0))

        sb = tk.Scrollbar(wrap, bg=RAISED, troughcolor=BG,
                          activebackground=GOLD_DIM, bd=0, relief='flat')
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat = tk.Text(
            wrap,
            font=(F_SERIF, 11),
            bg=SURFACE, fg=CREAM,
            insertbackground=CREAM,
            relief='flat', bd=0,
            padx=22, pady=16,
            state='disabled',
            wrap=tk.WORD,
            spacing1=2, spacing3=5,
            yscrollcommand=sb.set,
            cursor='arrow',
            selectbackground=GOLD_DIM,
            selectforeground=CREAM,
        )
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.chat.yview)

        # Tags
        self.chat.tag_config('agent_name',
            foreground=GOLD,      font=(F_SERIF, 10, 'bold'))
        self.chat.tag_config('user_name',
            foreground='#7aab8a', font=(F_SERIF, 10, 'bold'))
        self.chat.tag_config('body',
            foreground=CREAM,     font=(F_SERIF, 11),
            lmargin1=10, lmargin2=10)
        self.chat.tag_config('body_user',
            foreground='#c8d8c8', font=(F_SERIF, 11),
            lmargin1=10, lmargin2=10)
        self.chat.tag_config('reject',
            foreground='#e05c4a', font=(F_SERIF, 14, 'bold'),
            lmargin1=10, lmargin2=10)
        self.chat.tag_config('accept',
            foreground='#4aba74', font=(F_SERIF, 14, 'bold'),
            lmargin1=10, lmargin2=10)
        self.chat.tag_config('rule',
            foreground=BORDER,    font=(F_MONO, 5))

    # ── INPUT BAR ─────────────────────────────────────────────────────────────
    def _build_input(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X, padx=14)

        bar = tk.Frame(self.root, bg=BG, pady=10)
        bar.pack(fill=tk.X, padx=14)

        # Input field with focus-reactive border
        self._inp_frame = tk.Frame(bar, bg=RAISED,
                                   highlightthickness=1,
                                   highlightbackground=BORDER,
                                   highlightcolor=GOLD)
        self._inp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.inp = tk.Text(self._inp_frame,
                           font=(F_SERIF, 11),
                           bg=RAISED, fg=CREAM,
                           insertbackground=GOLD,
                           relief='flat', bd=8,
                           height=3, wrap=tk.WORD)
        self.inp.pack(fill=tk.BOTH, expand=True)
        self.inp.bind('<Return>',       lambda e: (self._send(), 'break')[1])
        self.inp.bind('<Shift-Return>', lambda e: None)
        self.inp.bind('<FocusIn>',
            lambda e: self._inp_frame.config(highlightbackground=GOLD))
        self.inp.bind('<FocusOut>',
            lambda e: self._inp_frame.config(highlightbackground=BORDER))

        # Buttons
        btns = tk.Frame(bar, bg=BG)
        btns.pack(side=tk.RIGHT)

        self.send_btn = tk.Button(
            btns, text='Send  ↵',
            command=self._send,
            bg=GOLD, fg=HEADER_BG,
            font=(F_SERIF, 10, 'bold'),
            relief='flat', bd=0,
            padx=22, pady=11,
            cursor='hand2',
            activebackground='#dbb85a',
            activeforeground=HEADER_BG)
        self.send_btn.pack(pady=(0, 6))

        tk.Button(
            btns, text='Clear',
            command=self._clear,
            bg=RAISED, fg=CREAM_DIM,
            font=(F_SERIF, 9),
            relief='flat', bd=0,
            padx=12, pady=5,
            cursor='hand2',
            activebackground=BORDER,
            activeforeground=CREAM).pack()

    # ── FOOTER ────────────────────────────────────────────────────────────────
    def _build_footer(self):
        tk.Frame(self.root, bg=GOLD, height=1).pack(fill=tk.X)
        foot = tk.Frame(self.root, bg=HEADER_BG, pady=5)
        foot.pack(fill=tk.X)

        row = tk.Frame(foot, bg=HEADER_BG)
        row.pack(fill=tk.X, padx=16, pady=(3, 2))

        tk.Label(row,
                 text='Enter → send   ·   Shift+Enter → new line',
                 bg=HEADER_BG, fg=CREAM_DIM,
                 font=(F_SERIF, 8, 'italic')).pack(side=tk.LEFT)

        self.thinking_lbl = tk.Label(row, text='',
                                     bg=HEADER_BG, fg=GOLD_DIM,
                                     font=(F_SERIF, 9, 'italic'))
        self.thinking_lbl.pack(side=tk.RIGHT)
        self.dots = ThinkingDots(self.thinking_lbl)

    # ── BUILD ALL ─────────────────────────────────────────────────────────────
    def _build(self):
        self._build_header()
        self._build_chat()
        self._build_input()
        self._build_footer()

    # ── WRITE HELPERS ─────────────────────────────────────────────────────────
    def _write(self, text: str, *tags):
        self.chat.config(state='normal')
        self.chat.insert(tk.END, text, tags)
        self.chat.config(state='disabled')
        self.chat.see(tk.END)

    def _rule(self):
        self._write('\n' + '─' * 68 + '\n', 'rule')

    def _agent_msg(self, text: str):
        self._rule()
        self._write('  Agent\n', 'agent_name')
        for line in text.split('\n'):
            ul = line.upper()
            if 'REJECT' in ul:
                self._write(line + '\n', 'reject')
            elif 'ACCEPT' in ul:
                self._write(line + '\n', 'accept')
            else:
                self._write(line + '\n', 'body')

    def _user_msg(self, text: str):
        self._rule()
        self._write('  You\n', 'user_name')
        self._write(text.replace('\n', '\n  ') + '\n', 'body_user')

    # ── EVENTS ────────────────────────────────────────────────────────────────
    def _send(self):
        text = self.inp.get('1.0', tk.END).strip()
        if not text:
            return
        self.inp.delete('1.0', tk.END)
        self._user_msg(text)
        self.send_btn.config(state='disabled')
        self.dots.start()
        self.history.append({'role': 'user', 'content': text})

        def _run():
            try:
                reply = get_response(self.history)
                self.root.after(0, self.dots.stop)
                self.root.after(0, lambda: self._agent_msg(reply))
            except Exception as e:
                self.root.after(0, self.dots.stop)
                self.root.after(0, lambda: self._agent_msg(
                    f"Error: {e}\nCheck your API key or connection."))
            finally:
                self.root.after(0, lambda: self.send_btn.config(state='normal'))

        threading.Thread(target=_run, daemon=True).start()

    def _clear(self):
        self.history = [{'role': 'system', 'content': SYSTEM}]
        self.chat.config(state='normal')
        self.chat.delete('1.0', tk.END)
        self.chat.config(state='disabled')
        self._agent_msg("Session cleared. Ready for the next protest.")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()