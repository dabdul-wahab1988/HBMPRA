#!/usr/bin/env python3
"""
plot_results.py (slim)

Generates exactly four figures:
  (1) 4×3 posterior density panel for up to nine organ-target HIs
  (2) 2×2 posterior density panel for HI_overall, CR_total, BLL (fourth cell blank)
  (3) 4×3 probability-of-exceedance panel for the same nine organ-target HIs
  (4) 2×2 probability-of-exceedance panel for HI_overall, CR_total, BLL

Also writes a single compact CSV summary with group-wise mean, 94% HDI,
and probability of exceeding policy thresholds (HI=1, CR=1e-6, BLL from --bll-thresholds).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter

# Global variable to hold unit information loaded from ASSUMPTIONS.json
_UNIT_INFO = {
    'input_units_original': 'µg/L',
    'concentration_units': 'µg/L',
    'bll_units': 'µg/dL'
}

def load_unit_info(results_dir):
    """Load unit information from ASSUMPTIONS.json if available."""
    global _UNIT_INFO
    assumptions_path = os.path.join(results_dir, "ASSUMPTIONS.json")
    if os.path.exists(assumptions_path):
        try:
            with open(assumptions_path) as f:
                assumptions = json.load(f)
            _UNIT_INFO['input_units_original'] = assumptions.get('input_units_original', 'µg/L')
            _UNIT_INFO['concentration_units'] = assumptions.get('concentration_units', 'µg/L')
            _UNIT_INFO['bll_units'] = assumptions.get('bll_units', 'µg/dL')
            print(f"  Loaded unit info: input={_UNIT_INFO['input_units_original']}, BLL={_UNIT_INFO['bll_units']}")
        except Exception as e:
            print(f"  Warning: Could not load unit info from ASSUMPTIONS.json: {e}")
    return _UNIT_INFO

# ------------------------ Helpers (only what's needed) ------------------------

def _discover_organ_hi_vars(idata, max_vars=9):
    """Return up to `max_vars` HI_* variables that look like organ targets (dims: site,group).
    
    Only includes variables that have actual non-zero data.
    """
    post = idata.posterior
    organ_vars = []
    for v in post.data_vars:
        if not str(v).startswith("HI_"):
            continue
        if v in ("HI_overall",):
            continue
        dims = tuple(getattr(post[v], "dims", ()))
        # Check if the variable has 'site' and 'group' dimensions (may also have 'chain' and 'draw')
        if "site" in dims and "group" in dims:
            # Check if variable has actual non-zero data
            vals = post[v].values
            if np.isfinite(vals).any() and (vals > 0).any():
                organ_vars.append(v)

    # Preferred order for drinking water relevant organs
    # NOTE: Respiratory removed - not applicable for oral/dermal exposure routes
    preferred = [
        "HI_neuro","HI_nephro","HI_hepato","HI_derm","HI_gi",
        "HI_cardiovascular","HI_endocrine","HI_hematologic",
        "HI_skeletal_dental","HI_hematological","HI_systemic"
    ]
    ordered = [v for v in preferred if v in organ_vars]
    remaining = sorted([v for v in organ_vars if v not in ordered])
    return (ordered + remaining)[:max_vars]

def _latex_label(var):
    if var == "HI_overall":
        return r"$HI_{\mathrm{overall}}$"
    if var == "CR_total":
        return r"$CR_{\mathrm{total}}$"
    if var == "BLL":
        # Use unit from loaded ASSUMPTIONS.json
        bll_unit = _UNIT_INFO.get('bll_units', 'µg/dL')
        # Convert to LaTeX-friendly format
        if bll_unit == 'µg/dL':
            return r"$\mathrm{Pb\ BLL}\ (\mu\mathrm{g}/\mathrm{dL})$"
        elif bll_unit == 'mg/dL':
            return r"$\mathrm{Pb\ BLL}\ (\mathrm{mg}/\mathrm{dL})$"
        else:
            return rf"$\mathrm{{Pb\ BLL}}\ ({bll_unit})$"
    if var.startswith("HI_"):
        organ = var[3:]
        return rf"$HI_{{\mathrm{{{organ}}}}}$"
    return var

def _threshold_for(var, bll_thr):
    if var == "CR_total":
        return 1e-6
    if var == "BLL":
        return float(bll_thr)
    if var.startswith("HI_"):
        return 1.0
    return 0.0

def _stack_vals(da):
    """Flatten chains/draws, drop NaNs/inf."""
    vals = da.stack(s=("chain","draw")).values.astype(float).ravel()
    return vals[np.isfinite(vals)]

def _hdi(vals, prob=0.94):
    from arviz.stats import hdi
    lo, hi = [float(x) for x in hdi(vals, hdi_prob=prob)]
    return lo, hi

def _sci_tex(x, sig=3):
    """Return MathTeX scientific notation: e.g., 1.23e-4 -> r'$1.23\\times10^{-4}$'."""
    if not np.isfinite(x) or x == 0:
        return r"$0$"
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    e = int(np.floor(np.log10(x)))
    m = x / (10 ** e)
    s = rf"${sign}{m:.{sig}f}\\times 10^{{{e}}}$"
    # tidy trailing zeros in mantissa (optional)
    s = s.replace(".000", "").replace(".00", "").replace(".0", "")
    return s

def _sci_tex_inner(x, sig=3):
    """Return TeX (no $) scientific notation: e.g., 1.23e-4 -> '1.23\\times 10^{-4}'."""
    if not np.isfinite(x) or x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    e = int(np.floor(np.log10(x)))
    m = x / (10 ** e)
    s = rf"{sign}{m:.{sig}f}\\times 10^{{{e}}}"
    s = s.replace(".000", "").replace(".00", "").replace(".0", "")
    return s

def _format_val_tex(x, sig=2):
    """Format value for legend: use 10^{e} notation for values far from 1, otherwise simple number."""
    if not np.isfinite(x) or x == 0:
        return "0"
    x = abs(float(x))
    e = int(np.floor(np.log10(x)))
    # For values between 0.01 and 1000, use simple format
    if -2 <= e <= 3:
        if e >= 0:
            return f"{x:.{sig}f}".rstrip('0').rstrip('.')
        else:
            return f"{x:.{-e+sig}f}".rstrip('0').rstrip('.')
    # Otherwise use MathTeX scientific notation
    m = x / (10 ** e)
    # Round mantissa
    m_str = f"{m:.{sig}f}".rstrip('0').rstrip('.')
    return rf"${m_str}\times10^{{{e}}}$"

def _tex_group(name: str) -> str:
    """Return group name for legend; plain text to avoid mathtext parsing issues."""
    return str(name)

def _exceedance_curve_sorted(samples, grid):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.zeros_like(grid)
    s.sort()
    p = (s.size - np.searchsorted(s, grid, side="right")) / s.size
    return np.clip(p, 1.0 / max(1, s.size), 1.0)

# ------------------------ Log-space KDE helper -------------------------------

def _kde_on_log10(ax, values, color, label=None, bw_adjust=1.2, fill=False):
    """Plot KDE on log10-transformed positives. Returns (min_log10, max_log10, positives).

    - Drops non-positive values (cannot be shown in log10 space).
    - Uses no tails (cut=0) and a slightly wider bandwidth for stability.
    - If fewer than 5 positive values, returns (None, None, None) and does not plot.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return None, None, None
    xlog = np.log10(v)
    sns.kdeplot(x=xlog, ax=ax, label=label, fill=fill, linewidth=1.2,
                color=color, bw_adjust=bw_adjust, cut=0, common_norm=False)
    return float(np.min(xlog)), float(np.max(xlog)), v

# ------------------------ Panels: posterior densities -------------------------

def plot_hi_organs_posterior_4x3(idata, output_dir):
    post = idata.posterior
    groups = list(post.coords["group"].values)
    vars9 = _discover_organ_hi_vars(idata, max_vars=12)  # Allow up to 12 for 4x3
    if not vars9:
        print("No organ HI variables found; skipping organ HI posterior panel.")
        return

    # Dynamically size the grid based on number of variables
    n_vars = len(vars9)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    plt.subplots_adjust(hspace=0.40, wspace=0.35)
    pal = sns.color_palette("tab10", n_colors=max(10, len(groups)))
    
    # Handle single row case where axes is 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(vars9):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.text(0.02, 0.95, f"({chr(97+idx)})", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        # Track global log10 limits to set ticks later
        lo_k_all = np.inf
        hi_k_all = -np.inf
        # Also compute mass at zero
        p_zero_overall = 0.0
        total_count = 0
        for gi, g in enumerate(groups):
            da = post[var].sel(group=g)
            vals = _stack_vals(da)
            if vals.size == 0:
                continue
            p_zero_overall += float(np.sum(vals <= 0))
            total_count += float(vals.size)
            color = pal[gi % len(pal)]
            lo_k, hi_k, vpos = _kde_on_log10(ax, vals, color, label=None, bw_adjust=1.2, fill=False)
            if vpos is not None:
                med = float(np.median(vpos))
                lo_log, hi_log = _hdi(np.log(vpos), prob=0.94)
                lo_b, hi_b = float(np.exp(lo_log)), float(np.exp(hi_log))
                gtex = _tex_group(g)
                # MathTeX format for legend values
                med_str = _format_val_tex(med)
                lo_str = _format_val_tex(lo_b)
                hi_str = _format_val_tex(hi_b)
                ax.plot([], [], label=f"{gtex}: {med_str} [{lo_str}, {hi_str}]", color=color)
                if lo_k is not None:
                    lo_k_all = min(lo_k_all, lo_k)
                if hi_k is not None:
                    hi_k_all = max(hi_k_all, hi_k)
        # Include HI=1 threshold in log10 units
        try:
            thr_k = 0.0  # log10(1) = 0
            cur_lo, cur_hi = ax.get_xlim()
            ax.set_xlim(min(cur_lo, thr_k) - 0.1, max(cur_hi, thr_k) + 0.1)
            ax.axvline(thr_k, color="red", linestyle="--", linewidth=1.2)
        except Exception:
            pass
        # Nice decade ticks
        try:
            cur_lo, cur_hi = ax.get_xlim()
            kmin = int(np.floor(cur_lo)); kmax = int(np.ceil(cur_hi))
            xticks = list(range(kmin, kmax + 1))
            ax.set_xticks(xticks)
            ax.set_xticklabels([rf"$10^{{{t}}}$" for t in xticks])
        except Exception:
            pass
        lbl = _latex_label(var)
        ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel("Density (log10 space)", fontsize=12)
        title_extra = ""
        if total_count > 0 and p_zero_overall > 0:
            p0 = 100.0 * p_zero_overall / total_count
            title_extra = f"  •  P(=0): {p0:.1f}%"
        ax.set_title(f"Posterior {lbl}{title_extra}", fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        # Position legend at upper right to avoid overlap with (a), (b) annotations
        leg = ax.legend(fontsize=11, loc='upper right', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')
            leg.get_frame().set_linewidth(0.0)

    # Hide unused axes
    total_cells = n_rows * n_cols
    for k in range(len(vars9), total_cells):
        r, c = divmod(k, n_cols); axes[r][c].axis("off")

    out = os.path.join(output_dir, "posterior_organs_panel.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)

def plot_core_posterior_2x2(idata, output_dir, bll_thresholds="3.5,5,10", cr_axis="log"):
    post = idata.posterior
    groups = list(post.coords["group"].values)
    bll_thr = float(str(bll_thresholds).split(",")[0])
    panels = ["HI_overall", "CR_total", "BLL", None]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.40, wspace=0.35)
    pal = sns.color_palette("tab10", n_colors=max(10, len(groups)))

    letter_idx = 0
    for idx, var in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        if var is None:
            ax.axis("off"); continue
        if var not in post:
            # More helpful message for common skip scenarios
            if var == "CR_total":
                msg = "No carcinogens in data\n(Anions F/NO₃ are not carcinogenic)"
            elif var == "BLL":
                msg = "No lead (Pb) in data\n(BLL requires lead exposure)"
            else:
                msg = "Not in trace"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(_latex_label(var), fontsize=14)
            continue

        # Place label only for actual plotted panels
        ax.text(0.02, 0.95, f"({chr(97+letter_idx)})", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        letter_idx += 1

        thr = _threshold_for(var, bll_thr)
        all_vals = []
        for g in groups:
            v = _stack_vals(post[var].sel(group=g))
            if v.size:
                all_vals.append(v)
        # Determine safe left/right limits
        left_limit = 1e-12
        right_limit = None
        if all_vals:
            # Use smallest positive across groups for left limit
            positives = [vv[vv > 0] for vv in all_vals]
            positives = [pp for pp in positives if pp.size]
            if positives:
                left_limit = max(1e-12, float(min(map(np.min, positives))))
            # Right limit as max finite across groups
            max_vals = []
            for vv in all_vals:
                finite = vv[np.isfinite(vv)]
                if finite.size:
                    max_vals.append(float(np.max(finite)))
            if max_vals:
                right_limit = max(max_vals)

        # Plot in log10 space for HI_overall, CR_total, and BLL
        use_log = (var in ("HI_overall", "CR_total", "BLL"))
        lo_k_all = np.inf
        hi_k_all = -np.inf
        p_zero_overall = 0.0
        total_count = 0
        for gi, g in enumerate(groups):
            vals = _stack_vals(post[var].sel(group=g))
            if vals.size == 0:
                continue
            p_zero_overall += float(np.sum(vals <= 0))
            total_count += float(vals.size)
            color = pal[gi % len(pal)]
            if use_log:
                lo_k, hi_k, vpos = _kde_on_log10(ax, vals, color, label=None, bw_adjust=1.2, fill=False)
                if vpos is not None:
                    med = float(np.median(vpos))
                    lo_log, hi_log = _hdi(np.log(vpos), prob=0.94)
                    lo_b, hi_b = float(np.exp(lo_log)), float(np.exp(hi_log))
                    gtex = _tex_group(g)
                    # MathTeX format for legend values
                    med_str = _format_val_tex(med)
                    lo_str = _format_val_tex(lo_b)
                    hi_str = _format_val_tex(hi_b)
                    ax.plot([], [], label=f"{gtex}: {med_str} [{lo_str}, {hi_str}]", color=color)
                    if lo_k is not None:
                        lo_k_all = min(lo_k_all, lo_k)
                    if hi_k is not None:
                        hi_k_all = max(hi_k_all, hi_k)
            else:
                sns.kdeplot(vals, ax=ax, label=None, fill=False, linewidth=1.2, color=color,
                            bw_adjust=1.2, cut=0, common_norm=False)
                med = float(np.median(vals))
                lo_b, hi_b = _hdi(vals, prob=0.94)
                # MathTeX format for legend values
                med_str = _format_val_tex(med)
                lo_str = _format_val_tex(lo_b)
                hi_str = _format_val_tex(hi_b)
                ax.plot([], [], label=f"{g}: {med_str} [{lo_str}, {hi_str}]", color=color)

        # Threshold line; convert to log10 if needed
        lbl = _latex_label(var)
        if use_log:
            try:
                thr_k = np.log10(thr) if thr > 0 else lo_k_all - 1
                cur_lo, cur_hi = ax.get_xlim()
                ax.set_xlim(min(cur_lo, thr_k) - 0.1, max(cur_hi, thr_k) + 0.1)
                ax.axvline(thr_k, color="red", linestyle="--", linewidth=1.2)
                # Nice decade ticks
                cur_lo, cur_hi = ax.get_xlim()
                kmin = int(np.floor(cur_lo)); kmax = int(np.ceil(cur_hi))
                ax.set_xticks(list(range(kmin, kmax + 1)))
                ax.set_xticklabels([rf"$10^{{{t}}}$" for t in range(kmin, kmax + 1)])
            except Exception:
                pass
            ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel("Density (log10 space)", fontsize=12)
        else:
            ax.axvline(thr, color="red", linestyle="--", linewidth=1.2)
            ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel("Density", fontsize=12)
        
        ax.set_title(f"Posterior {lbl}", fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        # Position legend at upper right to avoid overlap with (a), (b) annotations
        leg = ax.legend(fontsize=13, loc='upper right', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')
            leg.get_frame().set_linewidth(0.0)

    out = os.path.join(output_dir, "posterior_core_2x2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)

# ------------------------ Panels: exceedance curves ---------------------------

def plot_hi_organs_exceedance_4x3(idata, output_dir):
    post = idata.posterior
    groups = list(post.coords["group"].values)
    vars9 = _discover_organ_hi_vars(idata, max_vars=12)  # Allow up to 12 for 4x3
    if not vars9:
        print("No organ HI variables found; skipping organ HI exceedance panel.")
        return

    # Dynamically size the grid based on number of variables
    n_vars = len(vars9)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    plt.subplots_adjust(hspace=0.40, wspace=0.35)
    pal = sns.color_palette("tab10", n_colors=max(10, len(groups)))
    
    # Handle single row case where axes is 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(vars9):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.text(0.02, 0.95, f"({chr(97+idx)})", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        thr = 1.0

        # Build global grid
        all_pos = []
        for g in groups:
            vals = _stack_vals(post[var].sel(group=g))
            pos = vals[vals > 0]
            if pos.size:
                all_pos.append(pos)
        if not all_pos:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes); continue
        lo = max(1e-12, float(min(map(np.min, all_pos))))
        hi = float(max(map(np.max, all_pos)))
        if hi <= lo: hi = lo * 10
        grid = np.logspace(np.log10(lo), np.log10(hi), 200)

        for gi, g in enumerate(groups):
            vals = _stack_vals(post[var].sel(group=g))
            curve = _exceedance_curve_sorted(vals, grid)
            p_thr = float((vals > thr).mean()) * 100.0
            color = pal[gi % len(pal)]
            # Format threshold properly for legend
            if var == "CR_total":
                thr_str = f"{thr:.1e}"  # Simple scientific notation
            else:
                thr_str = f"{thr:.1f}"
            ax.plot(grid, curve, label=f"{g} (P≥{thr_str}: {p_thr:.1f}%)", linewidth=1.4, color=color)
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        # Set dynamic y-axis limits based on actual curve data
        try:
            min_prob = min([float(c[c > 0].min()) if (c > 0).any() else 1e-3 for c in [curve]])
            ax.set_ylim(bottom=max(1e-7, min_prob / 10), top=10)
        except Exception:
            ax.set_ylim(bottom=1e-5, top=10)
        lbl = _latex_label(var)
        ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel("Probability of Exceedance", fontsize=12)
        var_escaped = var.replace('_', r'\_')
        ax.set_title(rf"$P({var_escaped}\geq {thr})$ by Group", fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        # Position legend at upper right to avoid overlap with (a), (b) annotations
        leg = ax.legend(fontsize=11, loc='lower left', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')
            leg.get_frame().set_linewidth(0.0)

    # Hide unused cells
    total_cells = n_rows * n_cols
    for k in range(len(vars9), total_cells):
        r, c = divmod(k, n_cols); axes[r][c].axis("off")

    out = os.path.join(output_dir, "exceedance_organs_panel.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)

def plot_core_exceedance_2x2(idata, output_dir, bll_thresholds="3.5,5,10", cr_axis="log"):
    post = idata.posterior
    groups = list(post.coords["group"].values)
    bll_thr = float(str(bll_thresholds).split(",")[0])
    panels = ["HI_overall", "CR_total", "BLL", None]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.40, wspace=0.35)
    pal = sns.color_palette("tab10", n_colors=max(10, len(groups)))

    letter_idx = 0
    for idx, var in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        if var is None:
            ax.axis("off"); continue
        if var not in post:
            # More helpful message for common skip scenarios
            if var == "CR_total":
                msg = "No carcinogens in data\n(Anions F/NO₃ are not carcinogenic)"
            elif var == "BLL":
                msg = "No lead (Pb) in data\n(BLL requires lead exposure)"
            else:
                msg = "Not in trace"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(_latex_label(var), fontsize=14)
            continue

        # Place label only for actual plotted panels
        ax.text(0.02, 0.95, f"({chr(97+letter_idx)})", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        letter_idx += 1

        thr = _threshold_for(var, bll_thr)
        all_pos = []
        for g in groups:
            vals = _stack_vals(post[var].sel(group=g))
            pos = vals[vals > 0]
            if pos.size:
                all_pos.append(pos)
        if not all_pos:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes); continue
        lo = max(1e-12, float(min(map(np.min, all_pos))))
        hi = float(max(map(np.max, all_pos)))
        if hi <= lo: hi = lo * 10
        grid = np.logspace(np.log10(lo), np.log10(hi), 200)

        all_curves = []
        for gi, g in enumerate(groups):
            vals = _stack_vals(post[var].sel(group=g))
            curve = _exceedance_curve_sorted(vals, grid)
            all_curves.append(curve)
            p_thr = float((vals > thr).mean()) * 100.0
            color = pal[gi % len(pal)]
            # Format threshold properly for legend
            if var == "CR_total":
                thr_str = f"{thr:.1e}"  # Simple scientific notation
            else:
                thr_str = f"{thr:.1f}"
            ax.plot(grid, curve, label=f"{g} (P≥{thr_str}: {p_thr:.1f}%)", linewidth=1.4, color=color)

        ax.axvline(thr, color="red", linestyle="--", linewidth=1.2)
        if str(cr_axis).lower() == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
        else:
            ax.set_xscale("linear")
        ax.set_yscale("log")
        # Set dynamic y-axis limits based on actual curve data
        try:
            min_probs = []
            for c in all_curves:
                pos = c[c > 0]
                if pos.size > 0:
                    min_probs.append(float(pos.min()))
            if min_probs:
                min_prob = min(min_probs)
                ax.set_ylim(bottom=max(1e-7, min_prob / 10), top=10)
            else:
                ax.set_ylim(bottom=1e-5, top=10)
        except Exception:
            ax.set_ylim(bottom=1e-5, top=10)
        # For CR_total panel, also cap x-axis at global maximum
        if var == "CR_total":
            try:
                max_vals = []
                for g in groups:
                    v = _stack_vals(post[var].sel(group=g))
                    if v.size:
                        mv = np.nanmax(v[np.isfinite(v)])
                        if np.isfinite(mv):
                            max_vals.append(float(mv))
                if max_vals:
                    ax.set_xlim(right=max(max_vals))
            except Exception:
                pass
        lbl = _latex_label(var)
        ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel("Probability of Exceedance", fontsize=12)
        var_escaped = var.replace('_', r'\_')
        ax.set_title(rf"$P({var_escaped}\geq {thr})$ by Group", fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        # Position legend at lower left to avoid overlap with (a), (b) annotations
        leg = ax.legend(fontsize=13, loc='lower left', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')
            leg.get_frame().set_linewidth(0.0)

    out = os.path.join(output_dir, "exceedance_core_2x2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)

# ------------------------ Single summary CSV (compact) ------------------------

def write_single_summary_csv(idata, output_dir, bll_thresholds="3.5,5,10"):
    """One CSV: rows=metrics; columns for each group: 3%, Mean, 94%, P(>thr)."""
    post = idata.posterior
    groups = list(post.coords["group"].values)
    bll_thr = float(str(bll_thresholds).split(",")[0])

    # Collect variables to summarize
    vars9 = _discover_organ_hi_vars(idata, max_vars=9)
    core = [v for v in ["HI_overall","CR_total","BLL"] if v in post]
    vars_all = vars9 + core

    cols = []
    for g in groups:
        cols += [f"{g} 3%", f"{g} Mean", f"{g} 94%", f"{g} P(>thr)"]
    df = pd.DataFrame(index=vars_all, columns=cols, dtype=float)

    for var in vars_all:
        thr = _threshold_for(var, bll_thr)
        for g in groups:
            vals = _stack_vals(post[var].sel(group=g))
            if vals.size == 0:
                q3 = q94 = mean = pthr = np.nan
            else:
                q3, q94 = np.percentile(vals, [3, 97])  # approx 94% HDI proxy
                mean = float(np.mean(vals))
                pthr = float((vals > thr).mean())
            df.loc[var, f"{g} 3%"]   = q3
            df.loc[var, f"{g} 94%"]  = q94
            df.loc[var, f"{g} Mean"] = mean
            df.loc[var, f"{g} P(>thr)"] = pthr

    out = os.path.join(output_dir, "summary_compact.csv")
    df.to_csv(out, index_label="Metric")
    print("Saved", out)

# ------------------------ CLI / main -----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, help="Directory containing trace.nc")
    p.add_argument("--output-dir", default=None, help="Output directory for figures/CSV")
    p.add_argument("--bll-thresholds", default="3.5,5,10", help="Comma list; first is used")
    p.add_argument("--cr-axis", choices=["log","linear","auto"], default="log", help="Axis scale for CR_total panels")
    args = p.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load unit information from ASSUMPTIONS.json
    load_unit_info(args.results_dir)

    # Load idata
    idata_path = os.path.join(args.results_dir, "trace.nc")
    if not os.path.exists(idata_path):
        raise FileNotFoundError(f"trace.nc not found in {args.results_dir}")
    idata = az.from_netcdf(idata_path)

    # Try to read thresholds from RUNLOG (if present) to keep in sync
    runlog_path = os.path.join(args.results_dir, "RUNLOG.json")
    bll_thr = args.bll_thresholds
    if os.path.exists(runlog_path):
        try:
            runlog = json.load(open(runlog_path))
            if "bll_thresholds" in runlog and isinstance(runlog["bll_thresholds"], list) and runlog["bll_thresholds"]:
                bll_thr = str(runlog["bll_thresholds"][0])
        except Exception:
            pass

    # Resolve CR axis choice (auto detection if requested)
    cr_axis_choice = args.cr_axis
    if cr_axis_choice == "auto" and "CR_total" in idata.posterior:
        try:
            post = idata.posterior
            # Stack all groups/sites and chains into one vector
            vals = post["CR_total"].stack(s=("chain","draw")).values.astype(float).ravel()
            vals = vals[np.isfinite(vals)]
            pos = vals[vals > 0]
            if pos.size >= 10:
                lo = float(np.percentile(pos, 1))
                hi = float(np.percentile(pos, 99))
            elif pos.size > 0:
                lo = float(np.min(pos)); hi = float(np.max(pos))
            else:
                lo = 0.0; hi = 0.0
            ratio = (hi / lo) if (lo > 0 and hi > 0) else np.inf
            cr_axis_choice = "log" if (ratio >= 100.0) else "linear"
        except Exception:
            cr_axis_choice = "log"

    # Generate outputs
    write_single_summary_csv(idata, output_dir, bll_thr)
    plot_hi_organs_posterior_4x3(idata, output_dir)
    plot_core_posterior_2x2(idata, output_dir, bll_thr, cr_axis=cr_axis_choice)
    plot_hi_organs_exceedance_4x3(idata, output_dir)
    plot_core_exceedance_2x2(idata, output_dir, bll_thr, cr_axis=cr_axis_choice)

if __name__ == "__main__":
    main()
