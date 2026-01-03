#!/usr/bin/env python3
"""
plot_panel.py - Multi-Panel Plot System for HBMPRA

Provides a user-friendly text-based syntax for non-coders to create
custom multi-panel figures using simple layout specifications.

Syntax:
    [a|b]         → 1 row, 2 columns (a and b side-by-side)
    [a][b]        → 2 rows, 1 column (a on top, b below)
    [a|b][c]      → 2 rows; row1: a,b; row2: c (aligned under a)
    [a|b|c][d|e|f] → 2×3 grid

Plot Codes:
    Core Posterior:     a=HI_overall, b=CR_total, c=BLL
    Core Exceedance:    d=HI_overall, e=CR_total, f=BLL
    Organ Posterior:    a1=neuro, a2=nephro, a3=hepato, a4=derm, a5=gi,
                        a6=cardiovascular, a7=endocrine, a8=hematologic,
                        a9=skeletal_dental
    Organ Exceedance:   d1-d10 (same organs as above)

Presets:
    all         → All available plots
    core        → HI, CR, BLL (posterior + exceedance)
    organs      → All organ HI posteriors
    organs_exc  → All organ HI exceedance curves
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# Import helpers from plot_result.py
try:
    from plot_result import (
        _stack_vals, _hdi, _latex_label, _threshold_for,
        _kde_on_log10, _exceedance_curve_sorted, _format_val_tex,
        _tex_group, _discover_organ_hi_vars, load_unit_info, _UNIT_INFO
    )
except ImportError:
    from .plot_result import (
        _stack_vals, _hdi, _latex_label, _threshold_for,
        _kde_on_log10, _exceedance_curve_sorted, _format_val_tex,
        _tex_group, _discover_organ_hi_vars, load_unit_info, _UNIT_INFO
    )

# =============================================================================
# PLOT REGISTRY
# =============================================================================

# Organ mapping for numbered codes (drinking water relevant organs only)
# NOTE: Respiratory removed - not applicable for drinking water exposure (oral/dermal routes)
ORGAN_MAP = {
    1: 'neuro',
    2: 'nephro', 
    3: 'hepato',
    4: 'derm',
    5: 'gi',
    6: 'cardiovascular',
    7: 'endocrine',
    8: 'hemato',  # Fixed: was 'hematologic', toxref uses 'hemato'
    9: 'skeletal_dental',
}

# Reverse mapping for display
ORGAN_NAMES = {
    'neuro': 'Neurotoxicity',
    'nephro': 'Nephrotoxicity',
    'hepato': 'Hepatotoxicity',
    'derm': 'Dermal Effects',
    'gi': 'Gastrointestinal',
    'cardiovascular': 'Cardiovascular',
    'endocrine': 'Endocrine',
    'hemato': 'Hematological',  # Fixed: was 'hematologic'
    'skeletal_dental': 'Skeletal/Dental',
}

# Plot type definitions
PLOT_REGISTRY = {
    # Core posterior distributions
    'a': {'var': 'HI_overall', 'type': 'posterior', 'name': 'HI Overall Posterior'},
    'b': {'var': 'CR_total', 'type': 'posterior', 'name': 'Cancer Risk Posterior'},
    'c': {'var': 'BLL', 'type': 'posterior', 'name': 'Blood Lead Level Posterior'},
    
    # Core exceedance curves
    'd': {'var': 'HI_overall', 'type': 'exceedance', 'name': 'HI Overall Exceedance'},
    'e': {'var': 'CR_total', 'type': 'exceedance', 'name': 'Cancer Risk Exceedance'},
    'f': {'var': 'BLL', 'type': 'exceedance', 'name': 'BLL Exceedance'},
}

# Add organ-specific plots (a1-a12 for posterior, d1-d12 for exceedance)
for num, organ in ORGAN_MAP.items():
    PLOT_REGISTRY[f'a{num}'] = {
        'var': f'HI_{organ}',
        'type': 'posterior',
        'name': f'HI {ORGAN_NAMES.get(organ, organ)} Posterior'
    }
    PLOT_REGISTRY[f'd{num}'] = {
        'var': f'HI_{organ}',
        'type': 'exceedance',
        'name': f'HI {ORGAN_NAMES.get(organ, organ)} Exceedance'
    }

# Preset layouts
PRESETS = {
    'core': '[a|b|c]',                              # Core posteriors only
    'core_exc': '[d|e|f]',                          # Core exceedance only
    'core_all': '[a|b|c][d|e|f]',                   # Both posterior + exceedance
    'organs': '[a1|a2|a3][a4|a5|a6][a7|a8|a9]',     # Organ posteriors
    'organs_exc': '[d1|d2|d3][d4|d5|d6][d7|d8|d9]', # Organ exceedance
    'organs_all': '[a1|a2|a3][a4|a5|a6][a7|a8|a9][d1|d2|d3][d4|d5|d6][d7|d8|d9]',  # All organs
    'all': '[a|b|c][d|e|f][a1|a2|a3][a4|a5|a6][d1|d2|d3][d4|d5|d6]'
}

# =============================================================================
# LAYOUT PARSER
# =============================================================================

def parse_layout(layout_str: str) -> list:
    """
    Parse layout string into 2D grid of plot identifiers.
    
    Parameters
    ----------
    layout_str : str
        Layout specification like "[a|b][c]" or "[a|b|c][d|e|f]"
    
    Returns
    -------
    list of list of str
        2D grid of plot codes, e.g., [['a', 'b'], ['c']]
    
    Examples
    --------
    >>> parse_layout("[a|b]")
    [['a', 'b']]
    >>> parse_layout("[a][b]")
    [['a'], ['b']]
    >>> parse_layout("[a|b][c]")
    [['a', 'b'], ['c']]
    """
    # Check for preset
    layout_str = layout_str.strip().lower()
    if layout_str in PRESETS:
        layout_str = PRESETS[layout_str]
    
    # Parse row brackets
    rows = re.findall(r'\[([^\]]+)\]', layout_str)
    
    if not rows:
        raise ValueError(f"Invalid layout syntax: '{layout_str}'. Use format like [a|b][c]")
    
    grid = []
    for row in rows:
        # Split by pipe for columns
        cols = [c.strip() for c in row.split('|')]
        # Validate plot codes
        for code in cols:
            if code and code not in PLOT_REGISTRY:
                raise ValueError(f"Unknown plot code: '{code}'. Type '?' for available codes.")
        grid.append([c for c in cols if c])  # Filter empty strings
    
    return grid


def expand_preset(name: str) -> str:
    """
    Expand a preset name to its full layout string.
    
    Parameters
    ----------
    name : str
        Preset name like 'core', 'organs', 'all'
    
    Returns
    -------
    str
        Full layout string
    """
    return PRESETS.get(name.lower(), name)


def get_grid_dimensions(grid: list) -> tuple:
    """
    Get the dimensions of the grid.
    
    Returns
    -------
    tuple
        (n_rows, max_cols)
    """
    n_rows = len(grid)
    max_cols = max(len(row) for row in grid) if grid else 0
    return n_rows, max_cols


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_posterior_panel(ax, idata, var_name, groups, pal, bll_thr=3.5, panel_label=None):
    """
    Plot posterior density for a single variable on the given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes
    idata : arviz.InferenceData
        Inference data with posterior
    var_name : str
        Variable name (e.g., 'HI_overall', 'HI_neuro')
    groups : list
        List of group names
    pal : list
        Color palette
    bll_thr : float
        BLL threshold for reference line
    panel_label : str, optional
        Panel label like '(a)'
    """
    post = idata.posterior
    
    if var_name not in post:
        ax.text(0.5, 0.5, f"{var_name}\nnot available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(_latex_label(var_name), fontsize=12)
        return
    
    # Add panel label
    if panel_label:
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    # Track log10 limits
    lo_k_all = np.inf
    hi_k_all = -np.inf
    
    for gi, g in enumerate(groups):
        da = post[var_name].sel(group=g)
        vals = _stack_vals(da)
        if vals.size == 0:
            continue
        
        color = pal[gi % len(pal)]
        lo_k, hi_k, vpos = _kde_on_log10(ax, vals, color, label=None, bw_adjust=1.2, fill=False)
        
        if vpos is not None:
            med = float(np.median(vpos))
            lo_log, hi_log = _hdi(np.log(vpos), prob=0.94)
            lo_b, hi_b = float(np.exp(lo_log)), float(np.exp(hi_log))
            gtex = _tex_group(g)
            med_str = _format_val_tex(med)
            lo_str = _format_val_tex(lo_b)
            hi_str = _format_val_tex(hi_b)
            ax.plot([], [], label=f"{gtex}: {med_str} [{lo_str}, {hi_str}]", color=color)
            
            if lo_k is not None:
                lo_k_all = min(lo_k_all, lo_k)
            if hi_k is not None:
                hi_k_all = max(hi_k_all, hi_k)
    
    # Threshold line
    thr = _threshold_for(var_name, bll_thr)
    try:
        thr_k = np.log10(thr) if thr > 0 else lo_k_all - 1
        cur_lo, cur_hi = ax.get_xlim()
        ax.set_xlim(min(cur_lo, thr_k) - 0.1, max(cur_hi, thr_k) + 0.1)
        ax.axvline(thr_k, color='red', linestyle='--', linewidth=1.2)
        
        # Nice decade ticks
        cur_lo, cur_hi = ax.get_xlim()
        kmin = int(np.floor(cur_lo))
        kmax = int(np.ceil(cur_hi))
        ax.set_xticks(list(range(kmin, kmax + 1)))
        ax.set_xticklabels([rf"$10^{{{t}}}$" for t in range(kmin, kmax + 1)])
    except Exception:
        pass
    
    lbl = _latex_label(var_name)
    ax.set_xlabel(lbl, fontsize=10)
    ax.set_ylabel("Density (log10)", fontsize=10)
    ax.set_title(f"Posterior {lbl}", fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    # Only create legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')


def plot_exceedance_panel(ax, idata, var_name, groups, pal, bll_thr=3.5, panel_label=None):
    """
    Plot exceedance curve for a single variable on the given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes
    idata : arviz.InferenceData
        Inference data with posterior
    var_name : str
        Variable name (e.g., 'HI_overall', 'CR_total')
    groups : list
        List of group names
    pal : list
        Color palette
    bll_thr : float
        BLL threshold for reference line
    panel_label : str, optional
        Panel label like '(a)'
    """
    post = idata.posterior
    
    if var_name not in post:
        ax.text(0.5, 0.5, f"{var_name}\nnot available",
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(_latex_label(var_name), fontsize=12)
        return
    
    # Add panel label
    if panel_label:
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
    
    thr = _threshold_for(var_name, bll_thr)
    
    # Build global grid
    all_pos = []
    for g in groups:
        vals = _stack_vals(post[var_name].sel(group=g))
        pos = vals[vals > 0]
        if pos.size:
            all_pos.append(pos)
    
    if not all_pos:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    lo = max(1e-12, float(min(map(np.min, all_pos))))
    hi = float(max(map(np.max, all_pos)))
    if hi <= lo:
        hi = lo * 10
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    
    for gi, g in enumerate(groups):
        vals = _stack_vals(post[var_name].sel(group=g))
        curve = _exceedance_curve_sorted(vals, grid)
        p_thr = float((vals > thr).mean()) * 100.0
        color = pal[gi % len(pal)]
        
        if var_name == 'CR_total':
            thr_str = f"{thr:.1e}"
        else:
            thr_str = f"{thr:.1f}"
        
        ax.plot(grid, curve, label=f"{g} (P≥{thr_str}: {p_thr:.1f}%)", 
                linewidth=1.2, color=color)
    
    ax.axvline(thr, color='red', linestyle='--', linewidth=1.2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    
    try:
        ax.set_ylim(bottom=1e-5, top=10)
    except Exception:
        pass
    
    lbl = _latex_label(var_name)
    ax.set_xlabel(lbl, fontsize=10)
    ax.set_ylabel("P(Exceedance)", fontsize=10)
    var_escaped = var_name.replace('_', r'\_')
    ax.set_title(rf"$P({var_escaped}\geq {thr})$", fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    # Only create legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(fontsize=8, loc='lower left', framealpha=0.0)
        if leg is not None:
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('none')


# =============================================================================
# MAIN MULTI-PANEL FUNCTION
# =============================================================================

def create_multi_panel_figure(
    idata,
    layout_str: str,
    output_path: str,
    bll_thresholds: str = "3.5,5,10",
    figsize_per_cell: tuple = (6, 5),
    dpi: int = 300,
    results_dir: str = None
) -> str:
    """
    Create multi-panel figure from layout specification.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data with posterior samples
    layout_str : str
        Layout specification like "[a|b][c]" or preset name
    output_path : str
        Output file path for the figure
    bll_thresholds : str
        Comma-separated BLL thresholds (first is used)
    figsize_per_cell : tuple
        (width, height) per subplot cell in inches
    dpi : int
        Figure resolution
    results_dir : str, optional
        Results directory to load unit info from
    
    Returns
    -------
    str
        Path to saved figure
    """
    # Load unit info if available
    if results_dir:
        load_unit_info(results_dir)
    
    # Parse layout
    grid = parse_layout(layout_str)
    n_rows, n_cols = get_grid_dimensions(grid)
    
    # Calculate figure size (auto-size based on layout)
    fig_width = n_cols * figsize_per_cell[0]
    fig_height = n_rows * figsize_per_cell[1]
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.35, wspace=0.30)
    
    # Handle single row/col cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get groups and palette
    post = idata.posterior
    groups = list(post.coords['group'].values)
    pal = sns.color_palette('tab10', n_colors=max(10, len(groups)))
    
    # Parse BLL threshold
    bll_thr = float(str(bll_thresholds).split(',')[0])
    
    # Plot each cell
    panel_idx = 0
    for r_idx, row in enumerate(grid):
        for c_idx, code in enumerate(row):
            ax = axes[r_idx, c_idx]
            plot_info = PLOT_REGISTRY.get(code)
            
            if plot_info is None:
                ax.axis('off')
                continue
            
            # Generate panel label
            panel_label = f"({chr(97 + panel_idx)})"  # (a), (b), (c), ...
            panel_idx += 1
            
            var_name = plot_info['var']
            plot_type = plot_info['type']
            
            if plot_type == 'posterior':
                plot_posterior_panel(ax, idata, var_name, groups, pal, bll_thr, panel_label)
            elif plot_type == 'exceedance':
                plot_exceedance_panel(ax, idata, var_name, groups, pal, bll_thr, panel_label)
    
    # Hide unused cells (when rows have different column counts)
    for r_idx in range(n_rows):
        row_len = len(grid[r_idx]) if r_idx < len(grid) else 0
        for c_idx in range(row_len, n_cols):
            axes[r_idx, c_idx].axis('off')
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def get_available_plots(idata) -> dict:
    """
    Get list of plots available for the given inference data.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data with posterior
    
    Returns
    -------
    dict
        Dictionary mapping codes to availability info
    """
    post = idata.posterior
    available = {}
    
    for code, info in PLOT_REGISTRY.items():
        var_name = info['var']
        is_available = var_name in post
        available[code] = {
            'name': info['name'],
            'var': var_name,
            'available': is_available
        }
    
    return available


# =============================================================================
# HELP SYSTEM
# =============================================================================

def show_help():
    """Print help information for the plot builder."""
    help_text = """
══════════════════════════════════════════════════════════════════════════════
  MULTI-PANEL PLOT BUILDER - HELP
══════════════════════════════════════════════════════════════════════════════

SYNTAX:
  [a|b]           → 1 row, 2 columns (a and b side-by-side)
  [a][b]          → 2 rows, 1 column (a on top, b below)
  [a|b][c]        → 2 rows; row1 has a,b; row2 has c (aligned under a)
  [a|b|c][d|e|f]  → 2×3 grid (2 rows, 3 columns)

──────────────────────────────────────────────────────────────────────────────
PLOT CODES:
──────────────────────────────────────────────────────────────────────────────

  CORE POSTERIOR DISTRIBUTIONS:
    a  = HI_overall   (Overall Hazard Index)
    b  = CR_total     (Total Cancer Risk)
    c  = BLL          (Blood Lead Level)

  CORE EXCEEDANCE CURVES:
    d  = HI_overall   (HI Exceedance Probability)
    e  = CR_total     (CR Exceedance Probability)
    f  = BLL          (BLL Exceedance Probability)

  ORGAN-SPECIFIC HI POSTERIOR (drinking water relevant organs):
    a1 = Neurotoxicity        a2 = Nephrotoxicity      a3 = Hepatotoxicity
    a4 = Dermal               a5 = Gastrointestinal    a6 = Cardiovascular
    a7 = Endocrine            a8 = Hematological       a9 = Skeletal/Dental

  ORGAN-SPECIFIC HI EXCEEDANCE:
    d1 = Neurotoxicity        d2 = Nephrotoxicity      d3 = Hepatotoxicity
    d4 = Dermal               d5 = Gastrointestinal    d6 = Cardiovascular
    d7 = Endocrine            d8 = Hematological       d9 = Skeletal/Dental

──────────────────────────────────────────────────────────────────────────────
PRESETS:
──────────────────────────────────────────────────────────────────────────────
  core        → [a|b|c]                     (HI, CR, BLL posteriors)
  core_exc    → [d|e|f]                     (HI, CR, BLL exceedance)
  core_all    → [a|b|c][d|e|f]              (Both posterior + exceedance)
  organs      → [a1|a2|a3][a4|a5|a6]...     (All organ posteriors)
  organs_exc  → [d1|d2|d3][d4|d5|d6]...     (All organ exceedance)
  organs_all  → organs + organs_exc         (All organ plots)
  all         → All available plots in grid layout

──────────────────────────────────────────────────────────────────────────────
EXAMPLES:
──────────────────────────────────────────────────────────────────────────────
  [a|b|c]              → Single row: HI, CR, BLL posteriors
  [a|d]                → Compare HI posterior vs exceedance side-by-side
  [a][d]               → HI posterior on top, exceedance below
  [a1|a2|a3][a4|a5|a6] → 2×3 grid of organ HI posteriors
  [a|b][c|d][e|f]      → 3×2 grid mixing posterior and exceedance

══════════════════════════════════════════════════════════════════════════════
  Type 'done' to exit the plot builder
══════════════════════════════════════════════════════════════════════════════
"""
    print(help_text)


def show_available_plots(idata):
    """Show which plots are available for the current data."""
    available = get_available_plots(idata)
    
    print("\n" + "="*60)
    print("  AVAILABLE PLOTS FOR YOUR DATA")
    print("="*60)
    
    # Core plots
    print("\n  CORE PLOTS:")
    for code in ['a', 'b', 'c', 'd', 'e', 'f']:
        info = available.get(code, {})
        status = "✓" if info.get('available', False) else "✗"
        name = info.get('name', 'Unknown')
        print(f"    {status} {code} = {name}")
    
    # Organ plots
    print("\n  ORGAN POSTERIOR (a1-a9):")
    for i in range(1, 10):
        code = f'a{i}'
        info = available.get(code, {})
        status = "✓" if info.get('available', False) else "✗"
        name = info.get('name', 'Unknown').replace(' Posterior', '')
        print(f"    {status} {code:3} = {name}")
    
    print("\n  ORGAN EXCEEDANCE (d1-d9):")
    for i in range(1, 10):
        code = f'd{i}'
        info = available.get(code, {})
        status = "✓" if info.get('available', False) else "✗"
        name = info.get('name', 'Unknown').replace(' Exceedance', '')
        print(f"    {status} {code:3} = {name}")
    
    print("="*60)


# =============================================================================
# INTERACTIVE PLOT BUILDER
# =============================================================================

def interactive_plot_builder(idata, output_dir: str, bll_thresholds: str = "3.5,5,10"):
    """
    Run interactive plot builder loop.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data with posterior
    output_dir : str
        Directory to save generated figures
    bll_thresholds : str
        BLL thresholds for reference lines
    
    Returns
    -------
    list
        List of generated figure paths
    """
    generated_figures = []
    figure_count = 1
    
    print("\n" + "="*70)
    print("  CUSTOM MULTI-PANEL PLOT BUILDER")
    print("="*70)
    print("  Create custom figure layouts with simple text syntax!")
    print("  Type '?' or 'help' for syntax guide and plot codes")
    print("  Type 'list' to see available plots for your data")
    print("  Type 'done' to exit")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n  Enter layout (e.g. [a|b][c]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting plot builder...")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['?', 'help']:
            show_help()
            continue
        
        if user_input.lower() == 'list':
            show_available_plots(idata)
            continue
        
        if user_input.lower() == 'done':
            print("\n  Exiting plot builder...")
            break
        
        # Try to create the figure
        try:
            output_filename = f"custom_panel_{figure_count}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  Generating figure: {output_filename}...")
            
            saved_path = create_multi_panel_figure(
                idata=idata,
                layout_str=user_input,
                output_path=output_path,
                bll_thresholds=bll_thresholds,
                results_dir=output_dir
            )
            
            print(f"  [OK] Saved: {saved_path}")
            generated_figures.append(saved_path)
            figure_count += 1
            
        except ValueError as e:
            print(f"  [ERROR] {e}")
        except Exception as e:
            print(f"  [ERROR] Failed to create figure: {e}")
    
    if generated_figures:
        print(f"\n  Generated {len(generated_figures)} custom figure(s):")
        for fig_path in generated_figures:
            print(f"    • {os.path.basename(fig_path)}")
    
    return generated_figures


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for plot_panel.py"""
    import argparse
    import arviz as az
    
    parser = argparse.ArgumentParser(
        description="Create multi-panel figures with custom layouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_panel.py --results-dir ./results --layout "[a|b][c]"
  python plot_panel.py --results-dir ./results --layout "core"
  python plot_panel.py --results-dir ./results --interactive
        """
    )
    
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing trace.nc")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results-dir/figures)")
    parser.add_argument("--layout", default=None,
                        help="Layout string like '[a|b][c]' or preset name")
    parser.add_argument("--output-name", default="custom_panel.png",
                        help="Output filename (default: custom_panel.png)")
    parser.add_argument("--bll-thresholds", default="3.5,5,10",
                        help="BLL thresholds (first is used)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive plot builder")
    parser.add_argument("--help-plots", action="store_true",
                        help="Show help for plot codes and syntax")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_plots:
        show_help()
        return 0
    
    # Load inference data
    trace_path = os.path.join(args.results_dir, "trace.nc")
    if not os.path.exists(trace_path):
        print(f"Error: trace.nc not found in {args.results_dir}")
        return 1
    
    print(f"Loading inference data from {trace_path}...")
    idata = az.from_netcdf(trace_path)
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Interactive mode
    if args.interactive:
        interactive_plot_builder(idata, output_dir, args.bll_thresholds)
        return 0
    
    # Single layout mode
    if args.layout:
        output_path = os.path.join(output_dir, args.output_name)
        try:
            saved = create_multi_panel_figure(
                idata=idata,
                layout_str=args.layout,
                output_path=output_path,
                bll_thresholds=args.bll_thresholds,
                results_dir=args.results_dir
            )
            print(f"Saved: {saved}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        print("Error: Specify --layout or --interactive")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
