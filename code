# -*- coding: utf-8 -*-

"""
Contact : flora.parrotin@univ-rennes.fr
Code name : NORMA -  NORmative Mineral Analysis

Nettoyage & refactoring : Claude (Anthropic)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import math
import csv

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Chemical constants (dynamically extensible)
# =============================================================================

ATOMIC_WEIGHTS = {
    'H': 1.00794,   'O': 15.999,    'Si': 28.0855,   'Al': 26.9815385,
    'Fe': 55.845,   'Mg': 24.305,   'Na': 22.98976928, 'K': 39.0983,
    'Ca': 40.078,   'Ti': 47.867,   'Ba': 137.327,   'Mn': 54.938044,
    'P':  30.973761998,
}

# Mutable list to allow adding custom oxides
OXIDES = ['SiO2', 'Al2O3', 'Na2O', 'Fe2O3', 'MgO', 'TiO2', 'K2O', 'BaO', 'CaO', 'MnO', 'P2O5']

# Mutable: element -> (normative oxide, nb metal atoms in oxide)
ELEM_TO_OXIDE = {
    'Si': ('SiO2',  1), 'Al': ('Al2O3', 2), 'Na': ('Na2O',  2),
    'Fe': ('Fe2O3', 2), 'Mg': ('MgO',   1), 'Ti': ('TiO2',  1),
    'K':  ('K2O',   2), 'Ba': ('BaO',   1), 'Ca': ('CaO',   1),
    'Mn': ('MnO',   1), 'P':  ('P2O5',  2),
}

OXIDE_TO_ELEMENT = {v[0]: k for k, v in ELEM_TO_OXIDE.items()}

FORMULA_TOKEN_RE = re.compile(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)')


# =============================================================================
# Default minerals library text
# =============================================================================

DEFAULT_MINERALS_TEXT = (
    "Chlorite_Fe_red       = Si2.88 Al2.63 Fe3.08 Mg0.95 O18 H6\n"
    "Chlorite_Fe_ox        = Si3.0  Al2.87 Fe2.08 Mg1.2  O18 H6\n"
    "Chlorite_Fe_red_HYTEC = Si3    Al2    Fe3.685 Mg1.315 O18 H6\n"
    "Chlorite_Fe_ox_HYTEC  = Si3.0  Al3    Fe1.7   Mg1.8   O18 H6\n"
    "Analcime              = Na1 Al1 Si2 O6 H2\n"
    "Hematite              = Fe2 O3\n"
    "Anatase               = Ti1 O2\n"
    "Felspath_K_Ba         = K0.71 Ba0.29 Al1.29 Si2.71 O8\n"
    "Orthose_Ba            = K0.74 Ba0.26 Al1.26 Si2.74 O8\n"
    "Orthose               = K1 Al1 Si3 O8\n"
    "Feldspath_Ba          = Ba1 Al2 Si2 O8\n"
    "Apatite               = Ca4 Mn1 P3 O13 H1\n"
    "Harmotome             = Ba1 Al2 Si6 O16 H12\n"
    "Kaolinite             = Al2 Si2 O9 H4\n"
)

# Predefined minerals available in one click
PREDEFINED_MINERALS = {
    "Kaolinite":       "Al2 Si2 O9 H4",
    "Illite":          "K0.6 Al2.3 Si3.5 O12 H2",
    "Muscovite":       "K1 Al3 Si3 O12 H2",
    "Smectite-Na":     "Na0.4 Al2 Si3.6 O12 H6",
    "Smectite-Ca":     "Ca0.2 Al2 Si3.6 O12 H6",
    "Phlogopite":      "K1 Mg3 Al1 Si3 O12 H2",
    "Biotite":         "K1 Mg2 Fe1 Al1 Si3 O12 H2",
    "Albite":          "Na1 Al1 Si3 O8",
    "Anorthite":       "Ca1 Al2 Si2 O8",
    "K-Feldspar":      "K1 Al1 Si3 O8",
    "Oligoclase":      "Na0.8 Ca0.2 Al1.2 Si2.8 O8",
    "Enstatite":       "Mg2 Si2 O6",
    "Diopside":        "Ca1 Mg1 Si2 O6",
    "Augite":          "Ca1 Mg0.7 Fe0.3 Si2 O6",
    "Quartz":          "Si1 O2",
    "Goethite":        "Fe1 O2 H1",
    "Magnetite":       "Fe3 O4",
    "Gibbsite":        "Al1 O3 H3",
    "Boehmite":        "Al1 O2 H1",
    "Rutile":          "Ti1 O2",
    "Ilmenite":        "Fe1 Ti1 O3",
    "Analcime":        "Na1 Al1 Si2 O6 H2",
    "Apatite":         "Ca5 P3 O12",
    "Hematite":        "Fe2 O3",
    "Anatase":         "Ti1 O2",
    "Plagioclase-Ab70":"Na0.7 Ca0.3 Al1.3 Si2.7 O8",
}


# =============================================================================
# Chemical parsing and conversions
# =============================================================================

def parse_formula(formula_text: str) -> dict:
    """Parse a structural formula into a dict atom -> coefficient."""
    counts: dict = {}
    for m in FORMULA_TOKEN_RE.finditer(formula_text.replace(" ", "")):
        elem = m.group(1)
        coeff = float(m.group(2)) if m.group(2) else 1.0
        counts[elem] = counts.get(elem, 0.0) + coeff
    return counts


def format_formula(counts: dict) -> str:
    """Reconstruct a readable string from a dict atom -> coefficient."""
    if not counts:
        return ""
    parts = []
    for el, n in counts.items():
        if abs(n - 1.0) < 1e-12:
            parts.append(el)
        else:
            txt = f"{n:.4f}".rstrip("0").rstrip(".")
            parts.append(f"{el}{txt}")
    return " ".join(parts)


def oxide_molar_mass(oxide: str) -> float:
    """Compute the molar mass of an oxide from its chemical formula."""
    mass = 0.0
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', oxide):
        if el in ATOMIC_WEIGHTS:
            mass += ATOMIC_WEIGHTS[el] * (int(n) if n else 1)
    return mass


_OXIDE_MOLAR_MASS = {ox: oxide_molar_mass(ox) for ox in OXIDES}


def add_custom_oxide(oxide_name: str, element: str, atoms_in_oxide: int,
                     atomic_weight=None) -> bool:
    """Dynamically add a custom oxide to the global system.

    Returns True if successful, False if the oxide already exists.
    """
    if oxide_name in OXIDES:
        return False
    if atomic_weight is not None:
        ATOMIC_WEIGHTS[element] = float(atomic_weight)
    OXIDES.append(oxide_name)
    ELEM_TO_OXIDE[element] = (oxide_name, atoms_in_oxide)
    OXIDE_TO_ELEMENT[oxide_name] = element
    _OXIDE_MOLAR_MASS[oxide_name] = oxide_molar_mass(oxide_name)
    return True


def mineral_to_oxide_wtpercent(counts: dict) -> dict:
    """Convert a structural formula to normative oxide wt%."""
    oxide_moles = {ox: 0.0 for ox in OXIDES}
    for el, n_el in counts.items():
        if el in ELEM_TO_OXIDE:
            ox, atoms_per_ox = ELEM_TO_OXIDE[el]
            if ox in oxide_moles:
                oxide_moles[ox] += n_el / atoms_per_ox
    oxide_g = {ox: oxide_moles[ox] * _OXIDE_MOLAR_MASS[ox] for ox in OXIDES}
    total = sum(oxide_g.values())
    if total <= 0:
        return {ox: 0.0 for ox in OXIDES}
    return {ox: 100.0 * oxide_g[ox] / total for ox in OXIDES}


def build_A_from_minerals(minerals: list) -> np.ndarray:
    """Build the mass balance matrix A (oxides x phases)."""
    cols = [
        [mineral_to_oxide_wtpercent(counts)[ox] for ox in OXIDES]
        for _, counts in minerals
    ]
    return np.array(cols, dtype=float).T


def molar_mass_from_counts(counts: dict) -> float:
    """Molar mass of a phase from its atom -> coefficient dict."""
    return sum(ATOMIC_WEIGHTS[el] * n for el, n in counts.items() if el in ATOMIC_WEIGHTS)


# =============================================================================
# NNLS solvers
# =============================================================================

def nnls_coordinate_descent(A: np.ndarray, b: np.ndarray,
                             max_iter: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """NNLS by projected coordinate descent.

    Solves:  min ||Ax - b||^2   subject to   x >= 0
    """
    m, n = A.shape
    x = np.zeros(n)
    col_norm2 = np.einsum('ij,ij->j', A, A)

    for _ in range(max_iter):
        x_old = x.copy()
        r = b - A @ x
        for j in range(n):
            if col_norm2[j] <= 0:
                continue
            delta = (A[:, j] @ r) / col_norm2[j]
            xj_new = max(0.0, x[j] + delta)
            delta = xj_new - x[j]
            x[j] = xj_new
            r -= delta * A[:, j]
        if np.linalg.norm(x - x_old, 1) < tol:
            break
    return x


def greedy_phase_selection(A: np.ndarray, b: np.ndarray,
                            rmse_threshold: float = 0.05,
                            weights: np.ndarray = None) -> list:
    """Greedy phase selection: adds phases one by one to minimise RMSE.

    weights : weighting vector (same length as b).
              If None, uniform weighting (standard absolute RMSE).
              For relative errors: weights = 1/b.
    """
    if weights is None:
        weights = np.ones_like(b)

    def _rmse(fit, obs):
        return float(np.sqrt(np.mean((weights * (obs - fit)) ** 2)))

    n = A.shape[1]
    selected = []
    remaining = list(range(n))
    current_rmse = _rmse(np.zeros_like(b), b)

    while remaining:
        best_rmse = current_rmse
        best_j = None

        for j in remaining:
            trial_idx = selected + [j]
            A_trial = A[:, trial_idx]
            x_trial = nnls_coordinate_descent(A_trial, b)
            if x_trial.sum() > 0:
                fit = A_trial @ (x_trial / x_trial.sum())
            else:
                fit = np.zeros_like(b)
            rmse = _rmse(fit, b)
            if rmse < best_rmse:
                best_rmse = rmse
                best_j = j

        if best_j is None or (current_rmse - best_rmse) < rmse_threshold:
            break

        selected.append(best_j)
        remaining.remove(best_j)
        current_rmse = best_rmse

    return selected


def solve_mass_balance(oxide_wt: dict, minerals: list,
                       fixed_wt: dict = None,
                       mode: str = "nnls",
                       rmse_threshold: float = 0.05,
                       weights: np.ndarray = None,
                       forced_order: dict = None) -> tuple:
    """Orchestrate the full mass balance (Step 1).

    weights : weighting vector (len = nb oxides).
              None or array of 1s -> no weighting.
    """
    if fixed_wt     is None: fixed_wt      = {}
    if forced_order is None: forced_order  = {}

    b = np.array([oxide_wt.get(ox, 0.0) for ox in OXIDES], dtype=float)
    if b.sum() <= 0:
        raise ValueError("No non-zero oxide content.")
    b = 100.0 * b / b.sum()

    # Weighting vector
    if weights is None or len(weights) != len(b):
        w = np.ones(len(b))
    else:
        w = np.asarray(weights, dtype=float)

    A = build_A_from_minerals(minerals)
    names = [m[0] for m in minerals]
    n_phases = len(minerals)

    def _nnls(A_, b_):
        return nnls_coordinate_descent(A_ * w[:, np.newaxis], b_ * w)

    # Fast path: Best Fit with no fixed phases
    if mode == "nnls" and not fixed_wt:
        x = _nnls(A, b)
        if x.sum() > 0:
            x = 100.0 * x / x.sum()
        fit = A @ (x / 100.0)
        resid = b - fit
        rms = float(np.sqrt(np.mean(resid ** 2)))
        return ({names[j]: float(x[j]) for j in range(n_phases)},
                {OXIDES[i]: float(resid[i]) for i in range(len(OXIDES))},
                rms, fit.tolist(), b.tolist(), [])

    # Separate fixed / free phases
    fixed_idx = [i for i, name in enumerate(names) if name in fixed_wt]
    free_idx  = [i for i in range(n_phases) if i not in fixed_idx]

    x = np.zeros(n_phases)
    b_eff = b.copy()
    sum_fixed = 0.0
    for i in fixed_idx:
        f_pct = max(0.0, float(fixed_wt[names[i]]))
        x[i] = f_pct
        sum_fixed += f_pct
        b_eff = b_eff - (f_pct / 100.0) * A[:, i]

    remaining_budget = max(0.0, 100.0 - sum_fixed)
    normative_selected_names = []

    if free_idx:
        A_free = A[:, free_idx]

        if mode == "greedy":
            # Forced-order phases: injected first according to their number
            ordered_free = []
            for name, pos in sorted(forced_order.items(), key=lambda x: x[1]):
                if name in names:
                    gi = names.index(name)
                    if gi in free_idx:
                        local_j = free_idx.index(gi)
                        if local_j not in ordered_free:
                            ordered_free.append(local_j)

            # Remaining phases: free greedy selection
            unordered_free_idx = [j for j in range(len(free_idx))
                                  if j not in ordered_free]
            A_unordered = A_free[:, unordered_free_idx]
            greedy_local = greedy_phase_selection(A_unordered, b_eff,
                                                  rmse_threshold, weights=w)
            greedy_global = [unordered_free_idx[j] for j in greedy_local]

            # Final sequence: forced phases first, then selected phases
            sel_local = ordered_free + [j for j in greedy_global
                                        if j not in ordered_free]
            normative_selected_names = [names[free_idx[j]] for j in sel_local]

            x_free_raw = np.zeros(len(free_idx))
            if sel_local:
                x_sel = _nnls(A_free[:, sel_local], b_eff)
                for k, j in enumerate(sel_local):
                    x_free_raw[j] = x_sel[k]
        else:
            # Best Fit with fixed phases
            x_free_raw = _nnls(A_free, b_eff)

        if x_free_raw.sum() > 0:
            x_free_norm = x_free_raw / x_free_raw.sum() * remaining_budget
        else:
            x_free_norm = x_free_raw
        for k, i in enumerate(free_idx):
            x[i] = x_free_norm[k]

    fit = A @ (x / 100.0)
    resid = b - fit
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return ({names[j]: float(x[j]) for j in range(n_phases)},
            {OXIDES[i]: float(resid[i]) for i in range(len(OXIDES))},
            rms, fit.tolist(), b.tolist(), normative_selected_names)


# =============================================================================
# Equivalent dissolution depth
# =============================================================================

def compute_depths_um(sample: dict, minerals_wt: dict,
                      mineral_formulas_used: dict, cations_mol: dict) -> list:
    """Compute equivalent dissolution depth for each phase / tracer element."""
    t_mm = float(sample.get("thickness_mm",      0.0))
    d_cm = float(sample.get("diameter_cm",        0.0))
    rho  = float(sample.get("density_bulk_g_cm3", 0.0))

    if t_mm <= 0 or d_cm <= 0 or rho <= 0:
        return []

    m_bulk_g = rho * math.pi * (d_cm / 2.0) ** 2 * (t_mm / 10.0)
    wt_l  = {k.lower(): float(v) for k, v in minerals_wt.items()}
    fmt_l = {k.lower(): v        for k, v in mineral_formulas_used.items()}
    results = []

    def _depth(wt_frac, counts, el):
        M = molar_mass_from_counts(counts)
        if M <= 0 or counts.get(el, 0.0) <= 0:
            return None
        n_phase = (wt_frac * m_bulk_g) / M
        n_total = n_phase * counts[el]
        n_sol   = cations_mol.get(el, 0.0)
        if n_total <= 0 or n_sol <= 0:
            return None
        return (n_sol / n_total) * t_mm * 1000.0

    for key, wt in wt_l.items():
        counts = fmt_l.get(key, {})
        if not counts or wt <= 0:
            continue
        wt_frac = wt / 100.0
        if "analcime" in key:
            d = _depth(wt_frac, counts, "Na")
            if d is not None:
                results.append((f"{key} (Na)", d))
        elif "chlorite" in key:
            for el in ("Mg", "Fe"):
                d = _depth(wt_frac, counts, el)
                if d is not None:
                    results.append((f"{key} ({el})", d))

    return results


# =============================================================================
# Utility widget: scrollable frame
# =============================================================================

class ScrollableFrame(ttk.Frame):
    """Frame with integrated vertical scrollbar."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",  self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll( 1, "units"))

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._win_id, width=event.width)

    def _on_mousewheel(self, event):
        step = int(-1 * (event.delta / 120)) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(step, "units")


# =============================================================================
# Main application
# =============================================================================

class App(tk.Tk):
    """Main window with two tabs."""

    def __init__(self):
        super().__init__()
        self.title("Oxides -> Minerals + Dissolution depth")
        self.geometry("1400x940")
        self.minsize(1200, 800)
        self._init_state()
        self._build_ui()

    # -------------------------------------------------------------------------
    # Internal state
    # -------------------------------------------------------------------------

    def _init_state(self):
        self._last_solution:        dict = None
        self._last_resid:           dict = None
        self._last_rms:            float = None
        self._last_fit:             list = None
        self._last_b:               list = None
        self._last_formulas_used:   dict = None
        self._last_greedy_selected: list = []
        self._last_oxide_labels:    list = list(OXIDES)

    # -------------------------------------------------------------------------
    # Interface construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Big.TNotebook.Tab",      font=("TkDefaultFont", 11, "bold"), padding=(14, 8))
        style.configure("Bold.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("RMSE.TLabel",            font=("TkDefaultFont", 12, "bold"))

        self.notebook = ttk.Notebook(self, style="Big.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_step1 = ttk.Frame(self.notebook)
        self.tab_step2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_step1, text="Step 1: Mineral quantification")
        self.notebook.add(self.tab_step2, text="Step 2: Dissolution depth")

        self.var_status = tk.StringVar(value="Ready.")

        self._build_step1_tab()
        self._build_step2_tab()
        status_bar = ttk.Frame(self)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(status_bar, textvariable=self.var_status,
                  relief=tk.SUNKEN, anchor="w").pack(fill=tk.X)

    # =========================================================================
    # Tab Step 1
    # =========================================================================

    def _build_step1_tab(self):
        sc   = ScrollableFrame(self.tab_step1)
        sc.pack(fill=tk.BOTH, expand=True)
        root = sc.inner

        # Variables shared with step 2
        self.var_thick = tk.StringVar(value="")
        self.var_rho   = tk.StringVar(value="")
        self.var_diam  = tk.StringVar(value="")

        # ------------------------------------------------------------------ #
        # Section 1: Experimental oxides                                      #
        # ------------------------------------------------------------------ #
        fox = ttk.LabelFrame(
            root,
            text="1) Experimental data — oxide wt% of bulk rock (geochemical analysis)",
            style="Bold.TLabelframe"
        )
        fox.pack(fill=tk.X, padx=10, pady=6)

        # Iron valence note
        note_fe = ttk.Label(
            fox,
            text="  Fe expressed as Fe2O3 (total Fe3+ convention). "
                 "Actual iron valence in each mineral is encoded in its structural formula.",
            foreground="#7B3F00",
            font=("TkDefaultFont", 9, "italic")
        )
        note_fe.grid(row=0, column=0, columnspan=14, sticky="w", padx=6, pady=(4, 2))

        self.vars_ox   = {}
        self._ox_frame = fox
        self._ox_row   = 1
        self._ox_col   = 0

        for ox in OXIDES:
            self._add_oxide_field(ox, 0.0)

        # Control row
        ctrl = ttk.Frame(fox)
        ctrl.grid(row=99, column=0, columnspan=14, sticky="w", padx=6, pady=(6, 4))
        self.var_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto-renormalise to 100 %",
                        variable=self.var_norm,
                        command=self._update_oxide_sum).pack(side=tk.LEFT)
        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill="y", padx=10)
        self.lbl_oxide_sum = ttk.Label(ctrl, text="Sum: ---", foreground="#555555")
        self.lbl_oxide_sum.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(ctrl, text="Add custom oxide...",
                   command=self.on_add_custom_oxide).pack(side=tk.LEFT)
        self._update_oxide_sum()

        # ------------------------------------------------------------------ #
        # Sections 2 + 3: Library & Selection                                 #
        # ------------------------------------------------------------------ #
        line_mid = ttk.Frame(root)
        line_mid.pack(fill=tk.X, padx=10, pady=6)
        self._build_library_panel(line_mid)
        self._build_selection_panel(line_mid)

        # ------------------------------------------------------------------ #
        # Action buttons                                                       #
        # ------------------------------------------------------------------ #
        fbtn = ttk.Frame(root)
        fbtn.pack(fill=tk.X, padx=10, pady=4)

        ttk.Button(fbtn, text="Compute mineral wt%",
                   command=self.on_solve_minerals).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Separator(fbtn, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)
        ttk.Button(fbtn, text="Import CSV",
                   command=self.on_import_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(fbtn, text="Export CSV",
                   command=self.on_export).pack(side=tk.LEFT, padx=4)
        ttk.Separator(fbtn, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)
        ttk.Button(fbtn, text="Go to Step 2",
                   command=self.goto_step2_from_step1).pack(side=tk.LEFT, padx=4)
        ttk.Button(fbtn, text="Clear results",
                   command=self.on_clear).pack(side=tk.LEFT, padx=4)

        # ------------------------------------------------------------------ #
        # Results area                                                         #
        # ------------------------------------------------------------------ #
        results_row = ttk.Frame(root)
        results_row.pack(fill=tk.X, padx=10, pady=6)

        # Left column: mineral wt% + RMSE
        left_col = ttk.Frame(results_row)
        left_col.pack(side=tk.LEFT, fill=tk.Y)

        left = ttk.LabelFrame(left_col, text="Results — mineral wt%")
        left.pack(fill=tk.BOTH)
        self.tree = self._make_treeview(left, [
            ("mineral", "Mineral",   140, "w"),
            ("perc",    "wt%",        80, "e"),
            ("flag",    "",           30, "center"),
        ], height=12)
        self.tree.tag_configure("fixed",       foreground="#1565C0")
        self.tree.tag_configure("greedy_skip", foreground="#AAAAAA")

        self.lbl_rmse = ttk.Label(left_col, text="", style="RMSE.TLabel",
                                  anchor="center", foreground="#1565C0")
        self.lbl_rmse.pack(fill=tk.X, pady=(4, 0))

        # Residuals
        right = ttk.LabelFrame(results_row, text="Residuals per oxide (Obs - Calc)")
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        right.configure(width=520, height=330)
        right.pack_propagate(False)
        self.tree_r = self._make_treeview(right, [
            ("elem",  "Element",   60, "center"),
            ("resid", "Residual",  80, "e"),
            ("obs",   "Obs.",      70, "e"),
            ("calc",  "Calc.",     70, "e"),
            ("err",   "Err (%)",   70, "e"),
        ], height=12)

        # Plot
        mid = ttk.LabelFrame(results_row, text="Oxide wt%: experimental vs model")
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        mid.configure(width=570, height=400)
        mid.pack_propagate(False)

        graph_ctrl = ttk.Frame(mid)
        graph_ctrl.pack(fill=tk.X, padx=8, pady=(6, 0))
        self.var_plot_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(graph_ctrl, text="Log scale",
                        variable=self.var_plot_log,
                        command=self.update_oxide_plot).pack(side=tk.LEFT)
        ttk.Button(graph_ctrl, text="Save PNG",
                   command=self.on_save_png).pack(side=tk.RIGHT)

        self.fig_ox = Figure(figsize=(5.3, 3.3), dpi=100)
        self.ax_ox  = self.fig_ox.add_subplot(111)
        self.canvas_ox = FigureCanvasTkAgg(self.fig_ox, master=mid)
        self.canvas_ox.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_oxide_plot()

    # -------------------------------------------------------------------------
    # Step 1 sub-panels
    # -------------------------------------------------------------------------

    def _add_oxide_field(self, ox: str, default_val: float = 0.0):
        """Add an oxide field to the grid (init and custom oxides)."""
        r, c = self._ox_row, self._ox_col
        ttk.Label(self._ox_frame, text=f"{ox} :").grid(
            row=r, column=c, sticky="w", padx=(6 if c == 0 else 2, 4), pady=2)
        v = tk.StringVar(value="")
        ent = ttk.Entry(self._ox_frame, textvariable=v, width=9)
        ent.grid(row=r, column=c + 1, sticky="w", pady=2)
        ent.bind("<KeyRelease>", lambda e: self._update_oxide_sum())
        self.vars_ox[ox] = v
        self._ox_col += 2
        if self._ox_col >= 12:
            self._ox_row += 1
            self._ox_col = 0

    def _build_library_panel(self, parent):
        """Panel 2) Mineral library."""
        fmin = ttk.LabelFrame(parent, text="2) Mineral library",
                              style="Bold.TLabelframe")
        fmin.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        fmin.configure(width=730, height=330)
        fmin.pack_propagate(False)

        # Predefined minerals row
        predef_row = ttk.Frame(fmin)
        predef_row.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(predef_row, text="Predefined:").pack(side=tk.LEFT)
        self.var_predef = tk.StringVar()
        combo = ttk.Combobox(predef_row, textvariable=self.var_predef,
                             values=list(PREDEFINED_MINERALS.keys()),
                             width=22, state="readonly")
        combo.pack(side=tk.LEFT, padx=(4, 6))
        ttk.Button(predef_row, text="Add to library",
                   command=self.on_add_predefined_mineral).pack(side=tk.LEFT)

        ttk.Label(fmin,
                  text="One formula per line  |  Format: Name = Structural formula",
                  foreground="#555555").pack(anchor="w", padx=6)

        # Text area with state indicator (background colour)
        txt_frame = ttk.Frame(fmin)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 2))
        self.txt = tk.Text(txt_frame, height=13, width=90,
                           bg="white", relief=tk.SOLID, borderwidth=1)
        sb = ttk.Scrollbar(txt_frame, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill="y")
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt.insert("1.0", DEFAULT_MINERALS_TEXT)
        self.txt.bind("<KeyPress>", lambda e: self._on_lib_modified())

        # Library state label
        self.lbl_lib_state = ttk.Label(
            fmin,
            text="Library not verified — click Refresh to validate.",
            foreground="#795548"
        )
        self.lbl_lib_state.pack(anchor="w", padx=6, pady=(0, 4))

    def _build_selection_panel(self, parent):
        """Panel 3) Phase selection + fit mode."""
        fselect = ttk.LabelFrame(parent, text="3) Minerals used in fit",
                                 style="Bold.TLabelframe")
        fselect.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        fselect.configure(width=430, height=330)
        fselect.pack_propagate(False)

        # --- Fit mode ---
        fmode = ttk.Frame(fselect)
        fmode.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(fmode, text="Mode:",
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.var_fit_mode = tk.StringVar(value="nnls")
        for i, (text, val) in enumerate([
            ("Best Fit",  "nnls"),
            ("Normative", "greedy"),
        ]):
            ttk.Radiobutton(fmode, text=text, variable=self.var_fit_mode,
                            value=val, command=self._on_fit_mode_change).grid(
                row=0, column=i + 1, padx=4, sticky="w")

        # Normative threshold (visible only in Normative mode)
        self._greedy_frame = ttk.Frame(fselect)
        ttk.Label(self._greedy_frame,
                  text="Normative RMSE threshold:").pack(side=tk.LEFT, padx=(6, 4))
        self.var_greedy_thresh = tk.StringVar(value="0.05")
        ttk.Entry(self._greedy_frame, textvariable=self.var_greedy_thresh,
                  width=6).pack(side=tk.LEFT)
        ttk.Label(self._greedy_frame, text="wt%").pack(side=tk.LEFT, padx=4)

        # --- Weighting option ---
        fopt = ttk.Frame(fselect)
        fopt.pack(fill=tk.X, padx=6, pady=(0, 2))

        pondrow = ttk.Frame(fopt)
        pondrow.pack(fill=tk.X)
        ttk.Label(pondrow, text="Weighting:",
                  font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self.var_weighting = tk.StringVar(value="None")
        self._weighting_combo = ttk.Combobox(
            pondrow, textvariable=self.var_weighting, state="readonly", width=22,
            values=["None",
                    "Relative  (1/b)",
                    "Square root  (1/sqrt(b))",
                    "Analytical  (1/sigma)"])
        self._weighting_combo.pack(side=tk.LEFT, padx=(6, 0))
        self._weighting_combo.bind("<<ComboboxSelected>>",
                                   lambda e: self._on_weighting_change())

        # Dynamic description
        self._lbl_weighting_desc = ttk.Label(
            fopt, text="", foreground="#555555",
            font=("TkDefaultFont", 8, "italic"), wraplength=390, justify="left")
        self._lbl_weighting_desc.pack(anchor="w", padx=(4, 0), pady=(2, 0))

        # Sigma panel (visible only in Analytical mode)
        self._sigma_frame = ttk.LabelFrame(fopt, text="Analytical uncertainties (1 sigma, wt%)")
        self.vars_sigma = {}
        for ox in OXIDES:
            self.vars_sigma[ox] = tk.StringVar(value="")
        self._build_sigma_fields()
        self._on_weighting_change()

        ttk.Separator(fselect, orient="horizontal").pack(fill=tk.X, padx=6, pady=2)

        # --- Selection buttons ---
        btn_row = ttk.Frame(fselect)
        btn_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(btn_row, text="Refresh",
                   command=self.refresh_fit_mineral_list).pack(side=tk.LEFT)
        ttk.Separator(btn_row, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
        ttk.Button(btn_row, text="Check all",
                   command=self.select_all_fit_minerals).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Uncheck all",
                   command=self.unselect_all_fit_minerals).pack(side=tk.LEFT, padx=4)
        ttk.Separator(btn_row, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
        ttk.Button(btn_row, text="View phases",
                   command=self.show_parsed_phases).pack(side=tk.LEFT)

        ttk.Separator(fselect, orient="horizontal").pack(fill=tk.X, padx=6, pady=2)

        # Column headers
        hdr = ttk.Frame(fselect)
        hdr.pack(fill=tk.X, padx=8)
        ttk.Label(hdr, text="Phase", width=20,
                  font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self._lbl_fixed_col = ttk.Label(
            hdr, text="Fixed (wt%)",
            font=("TkDefaultFont", 9, "bold"), foreground="#1565C0", width=9)
        self._lbl_order_col = ttk.Label(
            hdr, text="Order",
            font=("TkDefaultFont", 9, "bold"), foreground="#5D4037", width=6)

        # Scrollable checkbox area
        scroll_sel = ScrollableFrame(fselect)
        scroll_sel.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        self.fit_select_container = scroll_sel.inner
        self.fit_mineral_vars: dict = {}
        self.fit_fixed_vars:   dict = {}
        self.fit_order_vars:   dict = {}
        self.refresh_fit_mineral_list()

    # =========================================================================
    # Tab Step 2
    # =========================================================================

    def _build_step2_tab(self):
        sc   = ScrollableFrame(self.tab_step2)
        sc.pack(fill=tk.BOTH, expand=True)
        root = sc.inner

        fmode = ttk.LabelFrame(root, text="Source of mineral wt%")
        fmode.pack(fill=tk.X, padx=10, pady=6)
        self.var_depth_mode = tk.StringVar(value="auto")
        for text, val in [("Use Step 1 results", "auto"),
                          ("Enter mineral wt% manually", "manual")]:
            ttk.Radiobutton(fmode, text=text, variable=self.var_depth_mode,
                            value=val, command=self._update_depth_mode_ui).pack(
                anchor="w", padx=8, pady=2)
        self.var_auto_info = tk.StringVar(value="No Step 1 results available.")
        ttk.Label(fmode, textvariable=self.var_auto_info,
                  foreground="#444444", wraplength=1100, justify="left").pack(
            anchor="w", padx=8, pady=(4, 6))

        fs = ttk.LabelFrame(root, text="Sample parameters for depth calculation")
        fs.pack(fill=tk.X, padx=10, pady=6)
        for label, var, col in [
            ("Thickness (mm):",        self.var_thick, 0),
            ("Dry bulk density (g/cm3):", self.var_rho, 2),
            ("Diameter (cm):",          self.var_diam, 4),
        ]:
            ttk.Label(fs, text=label).grid(row=0, column=col, sticky="w",
                                           padx=(12 if col else 0, 0))
            ttk.Entry(fs, textvariable=var, width=10).grid(row=0, column=col + 1, sticky="w")

        fcat = ttk.LabelFrame(root, text="Final dissolved cations (mol)")
        fcat.pack(fill=tk.X, padx=10, pady=6)
        self.vars_cat = {}
        for c, el in enumerate(['Al', 'Si', 'Na', 'Mg', 'Fe']):
            ttk.Label(fcat, text=f"{el} :").grid(row=0, column=c * 2, sticky="w")
            v = tk.StringVar(value="")
            ttk.Entry(fcat, textvariable=v, width=12).grid(
                row=0, column=c * 2 + 1, sticky="w", padx=(0, 10))
            self.vars_cat[el] = v

        fmanual = ttk.LabelFrame(root, text="Manual mineral wt% input")
        fmanual.pack(fill=tk.X, padx=10, pady=6)
        self.frame_manual = fmanual
        self.manual_wt_vars:  dict = {}
        self._manual_widgets: list = []
        self._build_manual_mineral_entries(self._read_minerals_safe())

        ttk.Label(fmanual,
                  text='Formulas used are those defined in Step 1. '
                       'If you modify the library, click "Update manual list".',
                  wraplength=1100, justify="left").grid(
            row=99, column=0, columnspan=6, sticky="w", padx=6, pady=(8, 2))
        ttk.Button(fmanual, text="Update manual list",
                   command=self.refresh_manual_entries_from_text).grid(
            row=100, column=0, sticky="w", padx=6, pady=(4, 6))

        fbtn = ttk.Frame(root)
        fbtn.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(fbtn, text="Compute depths",
                   command=self.on_compute_depths).pack(side=tk.LEFT, padx=4)
        ttk.Button(fbtn, text="Back to Step 1",
                   command=lambda: self.notebook.select(self.tab_step1)).pack(
            side=tk.LEFT, padx=4)

        fres = ttk.LabelFrame(root, text="Results — equivalent dissolution depths")
        fres.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.tree_d = self._make_treeview(fres, [
            ("mineral", "Phase (tracer element)", 350, "w"),
            ("depth",   "Depth (um)",             180, "e"),
        ], height=10)
        self._update_depth_mode_ui()

    # -------------------------------------------------------------------------
    # UI helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_treeview(parent, columns: list, height: int = 10) -> ttk.Treeview:
        ids = [c[0] for c in columns]
        tv  = ttk.Treeview(parent, columns=ids, show="headings", height=height)
        for col_id, title, width, anchor in columns:
            tv.heading(col_id, text=title)
            tv.column( col_id, width=width, anchor=anchor)
        tv.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return tv

    @staticmethod
    def _clear_treeview(*trees):
        for tree in trees:
            tree.delete(*tree.get_children())

    # -------------------------------------------------------------------------
    # Field readers
    # -------------------------------------------------------------------------

    def _read_oxides(self) -> dict:
        """Read oxide fields with non-numeric entry handling."""
        vals = {}
        bad  = []
        for ox, v in self.vars_ox.items():
            s = v.get().strip()
            try:
                vals[ox] = float(s) if s else 0.0
            except ValueError:
                vals[ox] = 0.0
                bad.append(ox)
        if bad:
            self.var_status.set(
                f"Non-numeric value(s) ignored -> treated as 0: {', '.join(bad)}")
        if self.var_norm.get():
            total = sum(vals.values())
            if total > 0:
                vals = {k: 100.0 * v / total for k, v in vals.items()}
        return vals

    def _read_minerals(self) -> list:
        """Parse the library text area -> list of (name, atom_dict) tuples."""
        minerals = []
        for ln in self.txt.get("1.0", "end").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            name, form = ln.split("=", 1) if "=" in ln else (f"Mineral_{len(minerals) + 1}", ln)
            minerals.append((name.strip(), parse_formula(form)))
        return minerals

    def _read_minerals_safe(self) -> list:
        try:
            return self._read_minerals()
        except Exception:
            return []

    def _read_selected_minerals(self) -> tuple:
        """Return (selected_minerals, fixed_wt, forced_order) according to mode."""
        all_minerals = self._read_minerals()
        mode = self.var_fit_mode.get()
        minerals  = []
        fixed_wt  = {}
        forced_order = {}

        for name, counts in all_minerals:
            var = self.fit_mineral_vars.get(name)
            if var is None or not var.get():
                continue
            minerals.append((name, counts))

            # Fixed (wt%) column — both modes
            fvar = self.fit_fixed_vars.get(name)
            if fvar:
                s = fvar.get().strip()
                if s:
                    try:
                        fixed_wt[name] = float(s)
                    except ValueError:
                        pass

            # Order column — Normative mode only
            if mode == "greedy":
                ovar = self.fit_order_vars.get(name)
                if ovar:
                    s = ovar.get().strip()
                    if s:
                        try:
                            val = int(s)
                            if val >= 1:
                                forced_order[name] = val
                            else:
                                ovar.set("")
                                self.var_status.set(
                                    f"Order ignored for '{name}': must be integer >= 1.")
                        except ValueError:
                            ovar.set("")

        return minerals, fixed_wt, forced_order

    def _read_sample(self) -> dict:
        return dict(
            thickness_mm=       float(self.var_thick.get() or 0.0),
            density_bulk_g_cm3= float(self.var_rho.get()   or 0.0),
            diameter_cm=        float(self.var_diam.get()   or 0.0),
        )

    def _read_cations(self) -> dict:
        out = {}
        for el, var in self.vars_cat.items():
            try:
                out[el] = float(var.get() or 0.0)
            except ValueError:
                out[el] = 0.0
        return out

    def _get_manual_mineral_wt(self) -> dict:
        return {
            name: float(var.get().strip()) if var.get().strip() else 0.0
            for name, var in self.manual_wt_vars.items()
        }

    # -------------------------------------------------------------------------
    # Dynamic phase management
    # -------------------------------------------------------------------------

    def refresh_fit_mineral_list(self):
        """Re-read the library and rebuild checkboxes + fixed fields + order."""
        previous_check = {n: v.get() for n, v in self.fit_mineral_vars.items()}
        previous_fixed = {n: v.get() for n, v in self.fit_fixed_vars.items()}
        previous_order = {n: v.get() for n, v in self.fit_order_vars.items()}

        for child in self.fit_select_container.winfo_children():
            child.destroy()

        minerals = self._read_minerals_safe()
        self.fit_mineral_vars = {}
        self.fit_fixed_vars   = {}
        self.fit_order_vars   = {}

        if not minerals:
            ttk.Label(self.fit_select_container,
                      text="No valid mineral.").pack(anchor="w")
            self.lbl_lib_state.config(
                text="No valid phase — check the library.",
                foreground="#C62828")
            self.txt.configure(bg="#FFEBEE")
            return

        mode = self.var_fit_mode.get()
        is_normative = (mode == "greedy")

        # Headers
        self._lbl_fixed_col.pack(side=tk.LEFT)
        if is_normative:
            self._lbl_order_col.pack(side=tk.LEFT)
        else:
            self._lbl_order_col.pack_forget()

        for name, _ in minerals:
            row = ttk.Frame(self.fit_select_container)
            row.pack(fill=tk.X, pady=1)

            var = tk.BooleanVar(value=previous_check.get(name, True))
            self.fit_mineral_vars[name] = var
            ttk.Checkbutton(row, text=name, variable=var, width=22).pack(
                side=tk.LEFT, anchor="w")

            fvar = tk.StringVar(value=previous_fixed.get(name, ""))
            self.fit_fixed_vars[name] = fvar
            ttk.Entry(row, textvariable=fvar, width=7,
                      foreground="#1565C0").pack(side=tk.LEFT, padx=(2, 0))

            ovar = tk.StringVar(value=previous_order.get(name, ""))
            self.fit_order_vars[name] = ovar
            if is_normative:
                ttk.Entry(row, textvariable=ovar, width=5,
                          foreground="#5D4037").pack(side=tk.LEFT, padx=(4, 0))

        self.txt.configure(bg="#E8F5E9")
        self.lbl_lib_state.config(
            text=f"{len(minerals)} phase(s) loaded successfully.",
            foreground="#2E7D32")
        self.var_status.set(f"Library refreshed: {len(minerals)} phase(s).")

    def select_all_fit_minerals(self):
        for var in self.fit_mineral_vars.values():
            var.set(True)

    def unselect_all_fit_minerals(self):
        for var in self.fit_mineral_vars.values():
            var.set(False)

    def _build_manual_mineral_entries(self, minerals: list):
        for w in self._manual_widgets:
            try:
                w.destroy()
            except tk.TclError:
                pass
        self._manual_widgets = []
        self.manual_wt_vars  = {}
        row = col = 0
        for name, _ in minerals:
            lbl = ttk.Label(self.frame_manual, text=f"{name} :")
            lbl.grid(row=row, column=col, sticky="w", padx=(6, 4), pady=3)
            var = tk.StringVar(value="")
            ent = ttk.Entry(self.frame_manual, textvariable=var, width=10)
            ent.grid(row=row, column=col + 1, sticky="w", padx=(0, 14), pady=3)
            self._manual_widgets.extend([lbl, ent])
            self.manual_wt_vars[name] = var
            col += 2
            if col >= 6:
                row += 1; col = 0

    def refresh_manual_entries_from_text(self):
        try:
            minerals = self._read_minerals()
            if not minerals:
                self.var_status.set("No valid mineral in the library.")
                return
            previous = {k: v.get() for k, v in self.manual_wt_vars.items()}
            self._build_manual_mineral_entries(minerals)
            for k, v in previous.items():
                if k in self.manual_wt_vars:
                    self.manual_wt_vars[k].set(v)
            self._update_depth_mode_ui()
            self.frame_manual.update_idletasks()
            self.var_status.set(f"List updated: {len(minerals)} phase(s).")
        except Exception:
            import traceback
            messagebox.showerror("Refresh error", traceback.format_exc())

    def _update_depth_mode_ui(self):
        manual = (self.var_depth_mode.get() == "manual")
        for w in self._manual_widgets:
            try:
                w.configure(state="normal" if manual else "disabled")
            except tk.TclError:
                pass
        if self._last_solution:
            summary = ", ".join(
                f"{k} = {v:.2f} wt%"
                for k, v in sorted(self._last_solution.items(), key=lambda x: -x[1])
            )
            self.var_auto_info.set(f"Step 1 results: {summary}")
        else:
            self.var_auto_info.set("No Step 1 results available.")

    # -------------------------------------------------------------------------
    # Visual indicators
    # -------------------------------------------------------------------------

    def _on_lib_modified(self):
        """Called on each keystroke in the library text area."""
        self.txt.configure(bg="#FFFDE7")
        self.lbl_lib_state.config(
            text="Library modified — click Refresh.",
            foreground="#E65100")

    def _update_oxide_sum(self, *_):
        total = 0.0
        for v in self.vars_ox.values():
            try:
                total += float(v.get() or 0.0)
            except ValueError:
                pass
        color = "#2E7D32" if abs(total - 100) < 0.5 else "#B71C1C"
        self.lbl_oxide_sum.config(text=f"Sum: {total:.2f} %", foreground=color)

    def _on_weighting_change(self):
        """Update description and show/hide sigma fields."""
        w = self.var_weighting.get()
        descs = {
            "None":
                "Minimises absolute error. SiO2 dominates the optimisation.",
            "Relative  (1/b)":
                "Divides by observed value: TiO2 at 1% counts as much as SiO2 at 50%.",
            "Square root  (1/sqrt(b))":
                "Compromise: attenuates majors without over-amplifying traces.",
            "Analytical  (1/sigma)":
                "Classic chi2: weighted by the measurement uncertainty of each oxide.",
        }
        self._lbl_weighting_desc.config(text=descs.get(w, ""))
        if w == "Analytical  (1/sigma)":
            self._sigma_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            self._sigma_frame.pack_forget()

    def _build_sigma_fields(self):
        """Build the sigma field grid inside _sigma_frame."""
        for child in self._sigma_frame.winfo_children():
            child.destroy()
        r = c = 0
        for ox in OXIDES:
            ttk.Label(self._sigma_frame, text=f"{ox}:").grid(
                row=r, column=c, sticky="w", padx=(4, 2), pady=1)
            ttk.Entry(self._sigma_frame, textvariable=self.vars_sigma[ox],
                      width=6).grid(row=r, column=c + 1, sticky="w", pady=1)
            c += 2
            if c >= 8:
                r += 1; c = 0

    def _on_fit_mode_change(self):
        """Show/hide normative threshold and refresh phase list."""
        mode = self.var_fit_mode.get()
        if mode == "greedy":
            self._greedy_frame.pack(fill=tk.X, padx=6, pady=(0, 2))
        else:
            self._greedy_frame.pack_forget()
        self.refresh_fit_mineral_list()

    def _read_weighting(self) -> np.ndarray:
        """Return the weighting vector w according to the user's choice.

        The NNLS solves:  min ||diag(w) A x - diag(w) b||^2

        "None"                      -> w = 1  (original behaviour)
        "Relative  (1/b)"           -> w = 1/b
        "Square root  (1/sqrt(b))"  -> w = 1/sqrt(b)
        "Analytical  (1/sigma)"     -> w = 1/sigma (user-entered sigma)
        """
        b_vals = np.array([float(self.vars_ox[ox].get() or 0.0) for ox in OXIDES],
                          dtype=float)
        if self.var_norm.get() and b_vals.sum() > 0:
            b_vals = 100.0 * b_vals / b_vals.sum()

        choice = self.var_weighting.get()

        if choice == "Relative  (1/b)":
            return np.where(b_vals > 0, 1.0 / b_vals, 0.0)

        elif choice == "Square root  (1/sqrt(b))":
            return np.where(b_vals > 0, 1.0 / np.sqrt(b_vals), 0.0)

        elif choice == "Analytical  (1/sigma)":
            sigmas = []
            for ox in OXIDES:
                s = self.vars_sigma[ox].get().strip()
                try:
                    val = float(s)
                    sigmas.append(val if val > 0 else None)
                except ValueError:
                    sigmas.append(None)
            w = []
            for i, ox in enumerate(OXIDES):
                if sigmas[i] is not None:
                    w.append(1.0 / sigmas[i])
                elif b_vals[i] > 0:
                    w.append(1.0 / b_vals[i])
                else:
                    w.append(0.0)
            return np.array(w)

        else:  # "None"
            return np.ones(len(OXIDES))

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    def update_oxide_plot(self):
        ax = self.ax_ox
        ax.clear()
        ax.set_title("Oxides: experimental vs model")
        ax.set_xlabel("Oxides")
        ax.set_ylabel("wt%")
        ax.grid(True, alpha=0.3)

        if self._last_b is None or self._last_fit is None:
            self.fig_ox.tight_layout()
            self.canvas_ox.draw()
            return

        labels_all = self._last_oxide_labels
        obs  = np.array(self._last_b,  dtype=float)
        calc = np.array(self._last_fit, dtype=float)

        n = min(len(labels_all), len(obs), len(calc))
        labels_all = labels_all[:n]
        obs  = obs[:n]
        calc = calc[:n]

        idx = list(range(n))

        if not idx:
            self.fig_ox.tight_layout()
            self.canvas_ox.draw()
            return

        labels = [labels_all[i] for i in idx]
        obs_p  = obs[idx]
        calc_p = calc[idx]

        order  = np.argsort(obs_p)[::-1]
        labels = [labels[i] for i in order]
        obs_p  = obs_p[order]
        calc_p = calc_p[order]

        if self.var_plot_log.get():
            ax.set_yscale("log")
            eps = max(obs[obs > 0].min() * 0.05, 1e-4) if np.any(obs > 0) else 1e-4
            obs_p  = np.where(obs_p  > 0, obs_p,  eps)
            calc_p = np.where(calc_p > 0, calc_p, eps)
        else:
            ax.set_yscale("linear")

        x = np.arange(len(labels))
        ax.plot(x, obs_p,  marker="o", label="Experimental")
        ax.plot(x, calc_p, marker="s", label="Model")
        ax.set_xticks(x)
        fontsize = max(6, 9 - max(0, len(labels) - 11))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=fontsize)
        ax.legend()
        self.fig_ox.tight_layout()
        self.canvas_ox.draw()

    # -------------------------------------------------------------------------
    # Step 1 actions
    # -------------------------------------------------------------------------

    def on_solve_minerals(self):
        """Compute mineral wt% according to the selected fit mode."""
        try:
            ox = self._read_oxides()
            minerals, fixed_wt, forced_order = self._read_selected_minerals()
            if not minerals:
                messagebox.showinfo("Fit", "No mineral selected for fit.")
                return

            mode = self.var_fit_mode.get()
            try:
                rmse_thresh = float(self.var_greedy_thresh.get())
            except ValueError:
                rmse_thresh = 0.05

            rel_w = self._read_weighting()
            all_minerals  = self._read_minerals()
            formulas_used = {name: counts.copy() for name, counts in all_minerals}

            sol, resid, rms, fit, b, norm_sel = solve_mass_balance(
                ox, minerals, fixed_wt, mode, rmse_thresh, rel_w, forced_order)

            self._last_solution        = sol
            self._last_resid           = resid
            self._last_rms             = rms
            self._last_fit             = fit
            self._last_b               = b
            self._last_formulas_used   = formulas_used
            self._last_greedy_selected = norm_sel
            self._last_oxide_labels    = list(OXIDES)

            # Mineral wt% table
            self._clear_treeview(self.tree)
            for name, perc in sorted(sol.items(), key=lambda x: -x[1]):
                flag = "F" if name in fixed_wt else (
                       str(forced_order[name]) if name in forced_order else "")
                if norm_sel and name not in norm_sel and name not in fixed_wt:
                    tag = "greedy_skip"
                elif name in fixed_wt:
                    tag = "fixed"
                else:
                    tag = ""
                self.tree.insert("", "end",
                                 values=(name, f"{perc:.3f}", flag),
                                 tags=(tag,))

            # Residuals table
            self._clear_treeview(self.tree_r)
            for i, oxname in enumerate(OXIDES):
                elem = OXIDE_TO_ELEMENT.get(oxname, "?")
                err  = abs(resid[oxname]) / abs(b[i]) * 100 if b[i] != 0 else 0.0
                self.tree_r.insert("", "end", values=(
                    elem, f"{resid[oxname]:+.3f}", f"{b[i]:.3f}",
                    f"{fit[i]:.3f}", f"{err:.1f}"
                ))

            self._sync_solution_to_manual_entries()
            self._update_depth_mode_ui()
            self.update_oxide_plot()

            self.lbl_rmse.config(text=f"RMSE = {rms:.3f} wt%")

            mode_lbl  = {"nnls": "Best Fit", "greedy": "Normative"}[mode]
            pond_lbl  = self.var_weighting.get().split("  ")[0]
            pond_sfx  = f" | {pond_lbl}" if pond_lbl != "None" else ""
            suffix    = (f"  |  {len(norm_sel)} phase(s) retained" if norm_sel else "")
            self.var_status.set(
                f"Step 1 [{mode_lbl}{pond_sfx}]{suffix}  |  RMSE = {rms:.3f} wt%")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _sync_solution_to_manual_entries(self):
        if not self._last_solution:
            return
        for name, var in self.manual_wt_vars.items():
            if name in self._last_solution:
                var.set(f"{self._last_solution[name]:.3f}")

    def goto_step2_from_step1(self):
        if self._last_solution:
            self.var_depth_mode.set("auto")
        self._update_depth_mode_ui()
        self.notebook.select(self.tab_step2)

    def on_compute_depths(self):
        try:
            sample  = self._read_sample()
            cations = self._read_cations()
            current_formulas = {name: counts.copy() for name, counts in self._read_minerals()}

            if self.var_depth_mode.get() == "auto":
                if not self._last_solution:
                    messagebox.showinfo("Depths",
                        "No Step 1 results available. "
                        "Use manual mode or compute wt% first.")
                    return
                minerals_wt   = self._last_solution
                formulas_used = self._last_formulas_used or current_formulas
            else:
                minerals_wt   = self._get_manual_mineral_wt()
                formulas_used = current_formulas

            depths = compute_depths_um(sample, minerals_wt, formulas_used, cations)
            self._clear_treeview(self.tree_d)

            if depths:
                for name, d in depths:
                    self.tree_d.insert("", "end", values=(name, f"{d:.3f}"))
                self.var_status.set(
                    f"Step 2 done. {len(depths)} depth(s) computed.")
            else:
                self.tree_d.insert("", "end", values=("No result", "---"))
                self.var_status.set("Step 2: no depth computed.")

        except Exception as e:
            messagebox.showerror("Depth error", str(e))

    # -------------------------------------------------------------------------
    # Export / Import CSV
    # -------------------------------------------------------------------------

    def on_export(self):
        """Export oxides, library and results to a reloadable CSV."""
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")

            w.writerow(["[Experimental oxides]"])
            for ox, v in self.vars_ox.items():
                w.writerow([ox, v.get()])
            w.writerow([])

            # Custom oxide definitions (for reimport)
            standard_oxides = ['SiO2','Al2O3','Na2O','Fe2O3','MgO',
                                'TiO2','K2O','BaO','CaO','MnO','P2O5']
            custom_oxides = [ox for ox in OXIDES if ox not in standard_oxides]
            if custom_oxides:
                w.writerow(["[Custom oxides]"])
                w.writerow(["Oxide", "Element", "Nb_atoms", "Atomic_weight"])
                for ox in custom_oxides:
                    el = OXIDE_TO_ELEMENT.get(ox, "?")
                    atoms = ELEM_TO_OXIDE.get(el, (None, 1))[1]
                    aw    = ATOMIC_WEIGHTS.get(el, "")
                    w.writerow([ox, el, atoms, aw])
                w.writerow([])

            w.writerow(["[Mineral library]"])
            w.writerow(["Name", "Formula"])
            for name, counts in self._read_minerals_safe():
                w.writerow([name, format_formula(counts)])
            w.writerow([])

            if self._last_solution:
                w.writerow(["[Step 1: Mineral quantification]"])
                w.writerow(["Mineral", "wt%", "Formula", "Mode"])
                mode = self.var_fit_mode.get()
                for name, wt in sorted(self._last_solution.items(), key=lambda x: -x[1]):
                    formula = (format_formula(self._last_formulas_used.get(name, {}))
                               if self._last_formulas_used else "")
                    w.writerow([name, f"{wt:.4f}", formula, mode])
                w.writerow(["RMSE (wt%)", f"{self._last_rms:.4f}"])
                w.writerow([])

                if self._last_b and self._last_fit:
                    w.writerow(["[Residuals]"])
                    w.writerow(["Oxide", "Obs.", "Calc.", "Residual", "Error (%)"])
                    for i, oxname in enumerate(OXIDES):
                        b_i   = self._last_b[i]
                        fit_i = self._last_fit[i]
                        res   = b_i - fit_i
                        err   = abs(res) / abs(b_i) * 100 if b_i != 0 else 0.0
                        w.writerow([oxname, f"{b_i:.4f}", f"{fit_i:.4f}",
                                    f"{res:+.4f}", f"{err:.2f}"])
                    w.writerow([])

            if self.tree_d.get_children():
                w.writerow(["[Step 2: Dissolution depth]"])
                w.writerow(["Phase (tracer element)", "Depth (um)"])
                for item in self.tree_d.get_children():
                    w.writerow(self.tree_d.item(item)["values"])

        messagebox.showinfo("Export", f"File saved:\n{path}")

    def on_import_csv(self):
        """Import oxides and library from a CSV exported by this application."""
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            sections: dict = {}
            current_section = None
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    if not row or all(c.strip() == "" for c in row):
                        current_section = None
                        continue
                    first = row[0].strip()
                    if first.startswith("[") and first.endswith("]"):
                        current_section = first[1:-1]
                        sections[current_section] = []
                        continue
                    if current_section is not None:
                        sections[current_section].append(row)

            loaded = []

            # 1. Re-create custom oxides BEFORE reading their values
            if "Custom oxides" in sections:
                for row in sections["Custom oxides"]:
                    if len(row) < 3 or row[0].strip() == "Oxide":
                        continue
                    ox_name = row[0].strip()
                    element = row[1].strip()
                    try:
                        atoms = int(row[2].strip())
                    except ValueError:
                        continue
                    aw = float(row[3].strip()) if len(row) >= 4 and row[3].strip() else None
                    if ox_name not in OXIDES:
                        add_custom_oxide(ox_name, element, atoms, aw)
                        self._add_oxide_field(ox_name, 0.0)
                        self._ox_frame.update_idletasks()

            # 2. Load oxide values
            if "Experimental oxides" in sections:
                for row in sections["Experimental oxides"]:
                    if len(row) >= 2 and row[0].strip() in self.vars_ox:
                        self.vars_ox[row[0].strip()].set(row[1].strip())
                loaded.append("oxides")
                self._update_oxide_sum()

            if "Mineral library" in sections:
                lines = []
                for row in sections["Mineral library"]:
                    if len(row) >= 2 and row[0].strip() != "Name":
                        lines.append(f"{row[0].strip()} = {row[1].strip()}")
                if lines:
                    self.txt.delete("1.0", "end")
                    self.txt.insert("1.0", "\n".join(lines) + "\n")
                    self.txt.configure(bg="white")
                    self.lbl_lib_state.config(
                        text="Library loaded from CSV — click Refresh.",
                        foreground="#E65100")
                    loaded.append("library")

            if loaded:
                messagebox.showinfo("Import CSV",
                                    f"Successfully imported: {', '.join(loaded)}.\n"
                                    "Click Refresh to validate the library.")
                self.var_status.set(f"CSV import: {', '.join(loaded)} loaded.")
            else:
                messagebox.showwarning("Import CSV",
                                       "No recognised section in this file.\n"
                                       "Make sure it was exported by this application.")
        except Exception as e:
            messagebox.showerror("Import error", str(e))

    def on_clear(self):
        self._clear_treeview(self.tree, self.tree_r, self.tree_d)
        self._init_state()
        self._update_depth_mode_ui()
        self.update_oxide_plot()
        self.lbl_rmse.config(text="")
        self.var_status.set("Results cleared.")

    # -------------------------------------------------------------------------
    # Additional actions
    # -------------------------------------------------------------------------

    def on_save_png(self):
        """Save the plot as PNG, PDF or SVG."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        self.fig_ox.savefig(path, dpi=150, bbox_inches="tight")
        self.var_status.set(f"Plot saved: {path}")

    def on_add_custom_oxide(self):
        """Dialog to add a custom oxide to the system."""
        dlg = tk.Toplevel(self)
        dlg.title("Add custom oxide")
        dlg.resizable(False, False)
        dlg.grab_set()

        fields = {}
        rows = [
            ("Oxide formula (e.g. ZrO2):",               ""),
            ("Element symbol (e.g. Zr):",                 ""),
            ("Nb of element atoms in the oxide:",         "1"),
            ("Atomic weight (leave blank if known):",     ""),
            ("Initial value (wt%):",                      ""),
        ]
        for i, (label, default) in enumerate(rows):
            ttk.Label(dlg, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=4)
            v = tk.StringVar(value=default)
            ttk.Entry(dlg, textvariable=v, width=18).grid(row=i, column=1, padx=10, pady=4)
            fields[i] = v

        def do_add():
            oxide_name = fields[0].get().strip()
            element    = fields[1].get().strip()
            try:
                atoms = int(fields[2].get().strip())
            except ValueError:
                messagebox.showerror("Error", "Invalid atom count.", parent=dlg)
                return
            aw_str = fields[3].get().strip()
            aw = float(aw_str) if aw_str else None
            try:
                val = float(fields[4].get().strip() or 0.0)
            except ValueError:
                val = 0.0

            if not oxide_name or not element:
                messagebox.showerror("Error", "Formula and symbol are required.", parent=dlg)
                return
            if element not in ATOMIC_WEIGHTS and aw is None:
                messagebox.showerror("Error",
                    f"Unknown element '{element}' — please enter its atomic weight.",
                    parent=dlg)
                return

            if not add_custom_oxide(oxide_name, element, atoms, aw):
                messagebox.showwarning("Already present",
                    f"Oxide '{oxide_name}' already exists.", parent=dlg)
                dlg.destroy()
                return

            self._add_oxide_field(oxide_name, val)
            self._ox_frame.update_idletasks()
            self._update_oxide_sum()
            dlg.destroy()
            self.var_status.set(
                f"Oxide '{oxide_name}' added. Refresh the library if needed.")

        ttk.Button(dlg, text="Add", command=do_add).grid(
            row=len(rows), column=0, columnspan=2, pady=10)

    def on_add_predefined_mineral(self):
        """Add the selected predefined mineral to the library."""
        name = self.var_predef.get()
        if not name or name not in PREDEFINED_MINERALS:
            messagebox.showinfo("Predefined", "Select a mineral from the list.")
            return
        formula = PREDEFINED_MINERALS[name]
        current = self.txt.get("1.0", "end").rstrip("\n")
        line = f"{name} = {formula}"
        if current:
            self.txt.insert("end", f"\n{line}")
        else:
            self.txt.insert("end", line)
        self._on_lib_modified()

    def show_parsed_phases(self):
        """Display parsed phases with oxide composition in a popup."""
        minerals = self._read_minerals_safe()
        if not minerals:
            messagebox.showinfo("Parsed phases",
                                "No valid phase in the library.")
            return

        top = tk.Toplevel(self)
        top.title(f"Phases ({len(minerals)})")

        n_ox_show = len(OXIDES)
        win_width = min(1200, 400 + n_ox_show * 70)
        top.geometry(f"{win_width}x420")

        cols = ([("name",    "Phase",    160, "w"),
                 ("formula", "Formula",  230, "w")]
                + [(OXIDES[i], OXIDES[i], 65, "e") for i in range(n_ox_show)])

        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tv = ttk.Treeview(frame, columns=[c[0] for c in cols],
                          show="headings", height=min(len(minerals), 18))
        for col_id, title, width, anchor in cols:
            tv.heading(col_id, text=title)
            tv.column(col_id, width=width, anchor=anchor)

        vsb = ttk.Scrollbar(frame, command=tv.yview)
        tv.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill="y")
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for name, counts in minerals:
            ox_wt     = mineral_to_oxide_wtpercent(counts)
            formula_s = format_formula(counts)
            row = ([name, formula_s]
                   + [f"{ox_wt[OXIDES[i]]:.1f}" for i in range(n_ox_show)])
            tv.insert("", "end", values=row)


# =============================================================================
if __name__ == "__main__":
    app = App()
    app.state("zoomed")
    app.mainloop()
