#!/usr/bin/env python3
"""
plot_diagnostics_panel.py - Custom Diagnostic Plot Builder

Provides interactive diagnostic plotting with:
- Trace plots (standalone)
- R-hat/ESS summary (standalone)  
- Prior vs Posterior multi-panel (customizable layout)

Hyperparameters are detected dynamically from the trace file.

Usage:
    from plot_diagnostics_panel import interactive_diagnostic_builder
    interactive_diagnostic_builder(trace_file, model_file, output_dir)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import arviz as az
except ImportError:
    az = None

try:
    import pymc as pm
except ImportError:
    pm = None

try:
    import dill
except ImportError:
    dill = None


# =============================================================================
# HYPERPARAMETER DETECTION
# =============================================================================

# Known hyperparameter patterns (non-deterministic sampling parameters)
HYPERPARAM_PATTERNS = [
    r'^z_log_',      # Non-centered z-scores
    r'^mu_log_',     # Hierarchical means
    r'^sigma_log_',  # Hierarchical standard deviations
    r'^log_k$',      # Log of k parameter
    r'^k$',          # BLL slope parameter
    r'^b0$',         # BLL intercept
    r'^z_b0$',       # Non-centered b0
    r'_g$',          # Group-level parameters (BW_g, IR_perkg_g)
]

# Exclude patterns (deterministic outputs, not hyperparameters)
EXCLUDE_PATTERNS = [
    r'^HI_',         # Hazard indices
    r'^CR_',         # Cancer risk
    r'^BLL$',        # Blood lead level
    r'^HQ_',         # Hazard quotients
    r'^EDI_',        # Exposure doses
]


def detect_hyperparameters(trace):
    """
    Detect hyperparameters in the trace file.
    
    Returns dict mapping code (c1, c2, ...) to variable name.
    """
    if trace is None or not hasattr(trace, 'posterior'):
        return {}
    
    all_vars = list(trace.posterior.data_vars)
    hyperparams = []
    
    for var in all_vars:
        # Check if matches any hyperparam pattern
        is_hyperparam = any(re.match(pat, var) for pat in HYPERPARAM_PATTERNS)
        # Check if should be excluded
        is_excluded = any(re.match(pat, var) for pat in EXCLUDE_PATTERNS)
        
        if is_hyperparam and not is_excluded:
            hyperparams.append(var)
    
    # Also include common hyperparameters even if not matching patterns
    common = ['k', 'b0', 'z_b0', 'log_k', 'mu_log_k', 'sigma_log_k', 
              'z_log_bw', 'z_log_ir', 'z_log_k', 'log_BW_g', 'log_IR_perkg_g',
              'BW_g', 'IR_perkg_g']
    for var in common:
        if var in all_vars and var not in hyperparams:
            hyperparams.append(var)
    
    # Create code mapping
    hyperparam_map = {}
    for i, var in enumerate(sorted(hyperparams), 1):
        hyperparam_map[f'c{i}'] = var
    
    return hyperparam_map


# =============================================================================
# STANDALONE PLOTS
# =============================================================================

def plot_trace(trace, hyperparam_map, output_dir):
    """Generate standalone trace plots for hyperparameters."""
    if az is None:
        print("  [SKIP] ArviZ not available")
        return None
    
    vars_to_plot = list(hyperparam_map.values())
    if not vars_to_plot:
        print("  [SKIP] No hyperparameters found")
        return None
    
    # Limit to first 9 for reasonable plot size
    vars_to_plot = vars_to_plot[:9]
    
    try:
        az.plot_trace(trace, var_names=vars_to_plot)
        fig = plt.gcf()
        fig.set_size_inches(12, max(4, len(vars_to_plot) * 1.5))
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, 'trace_hyperparams.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"  [WARN] Failed to create trace plot: {e}")
        return None


def plot_rhat_ess(trace, output_dir):
    """Generate standalone R-hat and ESS summary plot."""
    if az is None:
        print("  [SKIP] ArviZ not available")
        return None
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get variable names from posterior
        posterior = getattr(trace, 'posterior', None)
        if posterior is None:
            raise ValueError("No posterior in trace")
        
        var_names = list(posterior.data_vars)
        
        # Filter to non-deterministic variables (smaller dimensions typically)
        # and limit to reasonable number for display
        filtered_vars = []
        for v in var_names:
            data = posterior[v]
            # Skip very large arrays (likely deterministic outputs)
            if data.size < 100000:
                filtered_vars.append(v)
        
        # Limit to first 30 variables
        plot_vars = filtered_vars[:30] if len(filtered_vars) > 30 else filtered_vars
        
        if not plot_vars:
            axes[0].text(0.5, 0.5, 'No variables to plot', ha='center', va='center',
                        transform=axes[0].transAxes, fontsize=12)
            axes[1].text(0.5, 0.5, 'No variables to plot', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
        else:
            # Compute R-hat and ESS
            rhat_vals = []
            ess_vals = []
            valid_vars = []
            
            try:
                rhat = az.rhat(trace, var_names=plot_vars)
                ess = az.ess(trace, var_names=plot_vars, method='bulk')
            except Exception:
                rhat = None
                ess = None
            
            for v in plot_vars:
                try:
                    if rhat is not None and v in rhat:
                        rv = rhat[v]
                        r_val = float(np.nanmean(rv.values))
                    else:
                        r_val = np.nan
                    
                    if ess is not None and v in ess:
                        ev = ess[v]
                        e_val = float(np.nanmean(ev.values))
                    else:
                        e_val = np.nan
                    
                    if not np.isnan(r_val) or not np.isnan(e_val):
                        rhat_vals.append(r_val if not np.isnan(r_val) else 1.0)
                        ess_vals.append(e_val if not np.isnan(e_val) else 0)
                        valid_vars.append(v[:15] if len(v) > 15 else v)  # Truncate long names
                except Exception:
                    continue
            
            if valid_vars:
                x = np.arange(len(valid_vars))
                
                # R-hat barplot
                colors_rhat = ['green' if r < 1.05 else 'orange' if r < 1.1 else 'red' 
                              for r in rhat_vals]
                axes[0].bar(x, rhat_vals, color=colors_rhat, alpha=0.7)
                axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                axes[0].axhline(1.05, color='red', linestyle='--', linewidth=1, label='R-hat=1.05')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(valid_vars, rotation=45, ha='right', fontsize=8)
                axes[0].set_ylabel('R-hat')
                axes[0].set_title('R-hat by Variable (< 1.05 is good)', fontsize=14)
                axes[0].set_ylim(0.95, max(1.2, max(rhat_vals) * 1.05))
                
                # ESS barplot
                axes[1].bar(x, ess_vals, color='steelblue', alpha=0.7)
                axes[1].axhline(400, color='red', linestyle='--', linewidth=1, label='ESS=400')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(valid_vars, rotation=45, ha='right', fontsize=8)
                axes[1].set_ylabel('Bulk ESS')
                axes[1].set_title('Effective Sample Size (> 400 is good)', fontsize=14)
            else:
                axes[0].text(0.5, 0.5, 'Could not compute R-hat', ha='center', va='center',
                            transform=axes[0].transAxes, fontsize=12)
                axes[1].text(0.5, 0.5, 'Could not compute ESS', ha='center', va='center',
                            transform=axes[1].transAxes, fontsize=12)
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, 'rhat_ess_summary.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"  [WARN] Failed to create R-hat/ESS plot: {e}")
        return None


# =============================================================================
# PRIOR VS POSTERIOR MULTI-PANEL
# =============================================================================

def parse_layout(layout_str):
    """Parse layout string into 2D grid of plot codes."""
    layout_str = layout_str.strip()
    
    # Find all [...] groups
    row_pattern = re.compile(r'\[([^\]]+)\]')
    rows = row_pattern.findall(layout_str)
    
    if not rows:
        # Single code or comma-separated
        codes = [c.strip() for c in layout_str.replace(',', '|').split('|') if c.strip()]
        return [codes] if codes else []
    
    grid = []
    for row in rows:
        cols = [c.strip() for c in row.split('|') if c.strip()]
        if cols:
            grid.append(cols)
    
    return grid


def plot_prior_posterior_panel(trace, model, hyperparam_map, layout_grid, output_dir, 
                                fig_num=1, prior_samples=1000):
    """
    Create multi-panel prior vs posterior plot.
    
    Parameters
    ----------
    trace : InferenceData
        ArviZ trace with posterior samples
    model : PyMC model
        Model for sampling priors
    hyperparam_map : dict
        Mapping of codes (c1, c2, ...) to variable names
    layout_grid : list of lists
        2D grid of codes to plot
    """
    if not layout_grid:
        print("  [WARN] Empty layout")
        return None
    
    # Calculate figure dimensions
    n_rows = len(layout_grid)
    n_cols = max(len(row) for row in layout_grid)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Ensure axes is 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get variables to plot and sample priors
    all_vars = []
    for row in layout_grid:
        for code in row:
            if code in hyperparam_map:
                all_vars.append(hyperparam_map[code])
    
    # Sample priors if model available
    idata_prior = None
    if model is not None and pm is not None:
        try:
            with model:
                idata_prior = pm.sample_prior_predictive(
                    prior_samples, 
                    var_names=all_vars, 
                    return_inferencedata=True
                )
        except Exception as e:
            print(f"  [WARN] Could not sample priors: {e}")
    
    # Plot each cell
    for r, row in enumerate(layout_grid):
        for c, code in enumerate(row):
            ax = axes[r, c]
            
            if code not in hyperparam_map:
                ax.text(0.5, 0.5, f'Unknown: {code}', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(code)
                continue
            
            var = hyperparam_map[code]
            
            # Plot posterior
            if var in trace.posterior.data_vars:
                post_vals = trace.posterior[var].values.flatten()
                sns.kdeplot(post_vals, ax=ax, color='C1', fill=True, alpha=0.6, label='Posterior')
            else:
                ax.text(0.5, 0.5, f'{var}\nnot in posterior', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(var)
                continue
            
            # Plot prior if available
            prior_group = getattr(idata_prior, 'prior', None) if idata_prior else None
            if prior_group is not None and var in prior_group.data_vars:
                prior_vals = prior_group[var].values.flatten()
                sns.kdeplot(prior_vals, ax=ax, color='C0', fill=False, alpha=0.8, label='Prior')
            
            # Format title with LaTeX if possible
            tex_labels = {
                'z_log_bw': r'$z_{\log BW}$',
                'z_log_ir': r'$z_{\log IR}$',
                'mu_log_k': r'$\mu_{\log k}$',
                'sigma_log_k': r'$\sigma_{\log k}$',
                'z_log_k': r'$z_{\log k}$',
                'log_k': r'$\log k$',
                'k': r'$k$',
                'b0': r'$b_0$',
                'z_b0': r'$z_{b_0}$',
                'BW_g': r'$BW_g$',
                'IR_perkg_g': r'$IR_{perkg,g}$',
                'log_BW_g': r'$\log BW_g$',
                'log_IR_perkg_g': r'$\log IR_{perkg,g}$',
            }
            ax.set_title(tex_labels.get(var, var), fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Density')
            
            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=9, loc='upper right', framealpha=0.5)
    
    # Hide unused axes
    for r in range(n_rows):
        for c in range(len(layout_grid[r]), n_cols):
            axes[r, c].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'prior_posterior_panel_{fig_num}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


# =============================================================================
# INTERACTIVE BUILDER
# =============================================================================

def show_hyperparam_help(hyperparam_map):
    """Display available hyperparameters."""
    print("\n  Detected hyperparameters in trace:")
    
    # Format in columns
    items = list(hyperparam_map.items())
    n_cols = 3
    n_rows = (len(items) + n_cols - 1) // n_cols
    
    for r in range(n_rows):
        row_str = "    "
        for c in range(n_cols):
            idx = r + c * n_rows
            if idx < len(items):
                code, var = items[idx]
                row_str += f"{code} = {var:20}"
        print(row_str)
    
    print("\n  Layout syntax: [c1|c2][c3|c4] for 2 rows × 2 cols")
    print("  Or 'all' for all hyperparameters in a grid")


def interactive_diagnostic_builder(trace_file, model_file, output_dir, 
                                   figures_dir=None,
                                   auto_trace=None, auto_rhat=None, 
                                   auto_prior_posterior=None):
    """
    Interactive diagnostic plot builder.
    
    Parameters
    ----------
    trace_file : str
        Path to trace.nc file
    model_file : str
        Path to model.pkl file
    output_dir : str
        Base output directory
    figures_dir : str, optional
        Override figures directory (default: output_dir/figures)
    auto_trace : bool, optional
        If True, auto-generate trace plots without prompting
    auto_rhat : bool, optional
        If True, auto-generate R-hat/ESS plots without prompting
    auto_prior_posterior : str, optional
        If 'all', auto-generate all prior vs posterior plots
    """
    if figures_dir is None:
        figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load trace
    if not os.path.exists(trace_file):
        print(f"  [ERROR] Trace file not found: {trace_file}")
        return 0
    
    try:
        trace = az.from_netcdf(trace_file)
    except Exception as e:
        print(f"  [ERROR] Failed to load trace: {e}")
        return 0
    
    # Load model (optional, for prior sampling)
    model = None
    if os.path.exists(model_file) and dill is not None:
        try:
            with open(model_file, 'rb') as f:
                model = dill.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load model for prior sampling: {e}")
    
    # Detect hyperparameters
    hyperparam_map = detect_hyperparameters(trace)
    
    # Check if running in auto mode
    is_auto = auto_trace is not None or auto_rhat is not None or auto_prior_posterior is not None
    
    if not is_auto:
        print("\n" + "="*60)
        print("  DIAGNOSTIC PLOTS")
        print("="*60)
    
    figures_generated = 0
    
    # --- Trace plots ---
    if auto_trace is True:
        # Auto mode - generate without prompting
        print("  Generating trace plots...")
        result = plot_trace(trace, hyperparam_map, figures_dir)
        if result:
            print(f"    [OK] Saved: {os.path.basename(result)}")
            figures_generated += 1
    elif auto_trace is None and not is_auto:
        # Interactive mode
        trace_choice = input("\n  Generate trace plots? [y/n, default=y]: ").strip().lower()
        if trace_choice != 'n':
            print("  Generating trace plots...")
            result = plot_trace(trace, hyperparam_map, figures_dir)
            if result:
                print(f"  [OK] Saved: {os.path.basename(result)}")
                figures_generated += 1
    
    # --- R-hat/ESS ---
    if auto_rhat is True:
        # Auto mode
        print("  Generating R-hat/ESS summary...")
        result = plot_rhat_ess(trace, figures_dir)
        if result:
            print(f"    [OK] Saved: {os.path.basename(result)}")
            figures_generated += 1
    elif auto_rhat is None and not is_auto:
        # Interactive mode
        rhat_choice = input("\n  Generate R-hat/ESS summary? [y/n, default=y]: ").strip().lower()
        if rhat_choice != 'n':
            print("  Generating R-hat/ESS summary...")
            result = plot_rhat_ess(trace, figures_dir)
            if result:
                print(f"  [OK] Saved: {os.path.basename(result)}")
                figures_generated += 1
    
    # --- Prior vs Posterior ---
    if auto_prior_posterior == 'all':
        # Auto mode - generate all
        if hyperparam_map:
            print("  Generating prior vs posterior panel...")
            codes = list(hyperparam_map.keys())
            n_cols = 3
            n_rows_grid = (len(codes) + n_cols - 1) // n_cols
            layout_grid = []
            for r in range(n_rows_grid):
                row = codes[r * n_cols:(r + 1) * n_cols]
                layout_grid.append(row)
            
            result = plot_prior_posterior_panel(
                trace, model, hyperparam_map, layout_grid, 
                figures_dir, fig_num=1
            )
            if result:
                print(f"    [OK] Saved: {os.path.basename(result)}")
                figures_generated += 1
    elif auto_prior_posterior is None and not is_auto:
        # Interactive mode
        print("\n" + "-"*60)
        print("  PRIOR VS POSTERIOR PLOTS")
        print("-"*60)
        
        if not hyperparam_map:
            print("  [SKIP] No hyperparameters detected in trace")
        else:
            show_hyperparam_help(hyperparam_map)
            
            pp_fig_num = 1
            while True:
                layout_input = input(f"\n  Enter layout (or 'done' to finish): ").strip()
                
                if layout_input.lower() in ('done', 'q', 'quit', ''):
                    break
                
                # Handle 'all' preset
                if layout_input.lower() == 'all':
                    codes = list(hyperparam_map.keys())
                    n_cols = 3
                    n_rows_grid = (len(codes) + n_cols - 1) // n_cols
                    layout_grid = []
                    for r in range(n_rows_grid):
                        row = codes[r * n_cols:(r + 1) * n_cols]
                        layout_grid.append(row)
                else:
                    layout_grid = parse_layout(layout_input)
                
                if not layout_grid:
                    print("  [WARN] Invalid layout. Try again.")
                    continue
                
                print(f"  Generating prior vs posterior panel...")
                result = plot_prior_posterior_panel(
                    trace, model, hyperparam_map, layout_grid, 
                    figures_dir, pp_fig_num
                )
                if result:
                    print(f"  [OK] Saved: {os.path.basename(result)}")
                    figures_generated += 1
                    pp_fig_num += 1
    
    if not is_auto:
        print("="*60)
        print(f"  Diagnostic plots complete: {figures_generated} figure(s) generated")
        print("="*60)
    
    return figures_generated


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnostic plot builder')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    args = parser.parse_args()
    
    trace_file = os.path.join(args.results_dir, 'trace.nc')
    model_file = os.path.join(args.results_dir, 'model.pkl')
    
    interactive_diagnostic_builder(trace_file, model_file, args.results_dir)
