import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def check_prior_plausibility(results_dir):
    f = os.path.join(results_dir, "prior_pred_BLL.npy")
    if not os.path.exists(f):
        print("No prior predictive BLL saved; skip.")
        return
    arr = np.load(f, allow_pickle=True)
    if arr.size == 0:
        print("No BLL in model; skip.")
        return
    q2_5, q97_5 = np.quantile(arr, [0.025, 0.975])
    assert q2_5 >= 0.0 and q97_5 <= 20.0, (
        f"Prior BLL too wide: 2.5%={q2_5:.1f}, 97.5%={q97_5:.1f} µg/dL (tighten priors)")

def plot_convergence_summaries(trace_file, out_png=None):
    """Create a compact convergence summary (R-hat and ESS) from a trace NetCDF file.

    This function is robust to environments where ArviZ plotting helpers may fail.
    It computes R-hat and ESS and attempts to use `az.plot_rhat`/`az.plot_ess`; if
    those fail, it falls back to barplots of mean R-hat and bulk ESS per variable.
    """
    try:
        idata = az.from_netcdf(trace_file)
    except Exception as e:
        print(f"Failed to read trace file for convergence summary ({e})")
        return

    # Try compute diagnostics; wrap in try/except separately
    try:
        rhat = az.rhat(idata)
    except Exception:
        rhat = None
    try:
        ess = az.ess(idata, method="bulk")
    except Exception:
        ess = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Attempt to use ArviZ plot functions first
    try:
        if rhat is not None:
            az.plot_rhat(idata, ax=axes[0])
        else:
            raise RuntimeError("rhat not available")
        if ess is not None:
            az.plot_ess(idata, kind="evolution", ax=axes[1])
        else:
            raise RuntimeError("ess not available")
    except Exception:
        # Fallback: compute mean rhat per variable and bulk ESS per variable and barplot
        axes[0].clear()
        axes[1].clear()
        axes[0].set_title("R-hat (mean per variable)")
        axes[1].set_title("Bulk ESS per variable")

        # Collect variables and numeric values
        var_names = []
        rhat_means = []
        ess_vals = []
        # idata.posterior may be an xarray Dataset
        try:
            for var in idata.posterior.data_vars:
                var_names.append(var)
                if rhat is not None and var in rhat:
                    # rhat[var] can be array; take mean
                    try:
                        rhat_val = float(np.nanmean(rhat[var].values))
                    except Exception:
                        rhat_val = np.nan
                else:
                    rhat_val = np.nan
                rhat_means.append(rhat_val)

                if ess is not None and var in ess:
                    try:
                        ess_val = float(np.nanmean(ess[var].values))
                    except Exception:
                        ess_val = np.nan
                else:
                    ess_val = np.nan
                ess_vals.append(ess_val)
        except Exception:
            var_names = []

        if var_names:
            x = np.arange(len(var_names))
            axes[0].bar(x, rhat_means, color='C2')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(var_names, rotation=90, fontsize=8)
            axes[0].axhline(1.05, color='red', linestyle='--', linewidth=1)

            axes[1].bar(x, ess_vals, color='C3')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(var_names, rotation=90, fontsize=8)
        else:
            axes[0].text(0.5, 0.5, 'R-hat/ESS not available', ha='center', va='center')

    plt.tight_layout()
    if out_png:
        try:
            fig.savefig(out_png, dpi=200)
        except Exception as e:
            print(f"Failed to save convergence plot ({e})")
    plt.close(fig)
import argparse
import os
import dill as pickle  # use dill to load model serialized with dill
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
from demographics import GROUP_INFO as GROUPS

# Import constants from hbmpra.py
# Focused diagnostics for hbmpra_optimized hyperparameters
def _detect_hbmpra_hyperparams(trace):
    """Return a list of hyperparameter variable names present in the trace.

    We target the hierarchical and non-centered parameters defined in
    `hbmpra_optimized.py` such as: z_log_bw, z_log_ir, mu_log_k, sigma_log_k,
    z_log_k, log_k, k, b0, z_b0, and any group-level non-centered z-parameters.
    """
    candidates = [
        "z_log_bw", "z_log_ir", "mu_log_k", "sigma_log_k", "z_log_k",
        "log_k", "k", "b0", "z_b0", "mu_log_bw", "sigma_log_bw",
        "z_log_ir", "z_log_bw"
    ]
    present = [v for v in candidates if v in trace.posterior.data_vars]
    # also include any var that startswith 'z_' or endswith '_k' or is 'b0'
    for v in trace.posterior.data_vars:
        if v.startswith("z_") and v not in present:
            present.append(v)
        if (v.endswith("_k") or v == "b0" or v == "log_k") and v not in present:
            present.append(v)
    return present


def plot_prior_vs_posterior_vars(model, trace, vars_list, output_dir, prior_samples=2000):
    """Plot prior vs posterior densities for selected variables.

    model: loaded PyMC model (used for sampling priors)
    trace: ArviZ InferenceData from the fit
    vars_list: list of variable names to plot
    """
    if not vars_list:
        print("No HBMPRA hyperparameters detected for prior/posterior plotting.")
        return
    # draw prior predictive samples for these variables
    with model:
        try:
            idata_prior = pm.sample_prior_predictive(prior_samples, var_names=vars_list, return_inferencedata=True)
        except Exception as e:
            print(f"Could not draw prior samples ({e}); skipping prior plots.")
            idata_prior = None

    for var in vars_list:
        if var not in trace.posterior.data_vars:
            continue
        plt.figure(figsize=(6, 4))
        # posterior
        post_vals = trace.posterior[var].values.flatten()
        sns.kdeplot(post_vals, label="Posterior", color="C1", fill=True, alpha=0.6)
        # prior if available
        if idata_prior is not None and var in idata_prior.prior:
            prior_vals = idata_prior.prior[var].values.flatten()
            sns.kdeplot(prior_vals, label="Prior", color="C0", fill=True, alpha=0.4)
        plt.title(f"Prior vs Posterior: {var}")
        plt.xlabel(var)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prior_posterior_{var}.png'), dpi=200)
        plt.close()


def plot_prior_posterior_panel(model, trace, vars_list, output_dir, prior_samples=2000):
    """Create a 4x4 panel plot (up to 16 vars) showing prior vs posterior densities with MathTeX labels."""
    if not vars_list:
        print("No variables provided for panel plot.")
        return
    # limit to 16 variables
    vars_panel = vars_list[:16]
    # sample priors
    with model:
        try:
            idata_prior = pm.sample_prior_predictive(prior_samples, var_names=vars_panel, return_inferencedata=True)
        except Exception as e:
            print(f"Could not draw prior samples for panel ({e}); skipping panel plot.")
            idata_prior = None

    # MathTeX label mapping (extend as needed)
    tex_labels = {
        "z_log_bw": r"$z_{\log BW}$",
        "z_log_ir": r"$z_{\log IR}$",
        "mu_log_k": r"$\mu_{\log k}$",
        "sigma_log_k": r"$\sigma_{\log k}$",
        "z_log_k": r"$z_{\log k}$",
        "log_k": r"$\log k$",
        "k": r"$k$",
        "b0": r"$b_0$",
        "z_b0": r"$z_{b_0}$",
        "mu_log_bw": r"$\mu_{\log BW}$",
        "sigma_log_bw": r"$\sigma_{\log BW}$",
    }

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()

    for i, var in enumerate(vars_panel):
        ax = axes[i]
        # posterior
        if var in trace.posterior.data_vars:
            post_vals = trace.posterior[var].values.flatten()
            sns.kdeplot(post_vals, fill=True, color='C1', ax=ax, label='Posterior', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'No posterior', ha='center', va='center')

        # prior
        if idata_prior is not None and var in idata_prior.prior:
            prior_vals = idata_prior.prior[var].values.flatten()
            sns.kdeplot(prior_vals, fill=False, color='C0', ax=ax, label='Prior', alpha=0.6)

        ax.set_title(tex_labels.get(var, var), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=9)

    # blank remaining axes
    for j in range(len(vars_panel), 16):
        axes[j].axis('off')

    plt.tight_layout()
    outp = os.path.join(output_dir, 'prior_posterior_panel.png')
    fig.savefig(outp, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved panel prior/posterior plot to {outp}")


def plot_traces_for_vars(trace, vars_list, output_dir):
    """Create trace (time-series) plots for the selected vars using ArviZ."""
    if not vars_list:
        print("No HBMPRA hyperparameters detected for trace plotting.")
        return
    try:
        az.plot_trace(trace, var_names=vars_list)
        fig = plt.gcf()
        fig.set_size_inches(12, max(4, len(vars_list)*1.5))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trace_hbmpra_hyperparams.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Failed to create trace plots ({e})")

def plot_convergence(trace, output_dir):
    # Trace plots for hyperparameters only (deterministic nodes don't have meaningful convergence diagnostics)
    params = ["IRkg", "AF", "BW", "EF_ing_frac", "EF_der_frac", "ET_frac"]
    # Filter to only existing variables
    params = [p for p in params if p in trace.posterior.data_vars]
    if not params:
        print("No hyperparameters found in trace for convergence plots.")
        return
    # Use TeX formatted labels
    tex_labels = {
        "IRkg": r"$IR_{kg}$",
        "AF": r"$AF$",
        "BW": r"$BW$",
        "EF_ing_frac": r"$EF_{ing}$",
        "EF_der_frac": r"$EF_{der}$",
        "ET_frac": r"$ET$",
        "HI_total_Adults": r"$HI_{total}^{Adults}$",
        "HI_total_Children": r"$HI_{total}^{Children}$",
        "HI_total_Teens": r"$HI_{total}^{Teens}$",
        "HI_total_Pregnant": r"$HI_{total}^{Pregnant}$",
        "CR_total_Adults": r"$CR_{total}^{Adults}$",
        "CR_total_Children": r"$CR_{total}^{Children}$",
        "CR_total_Teens": r"$CR_{total}^{Teens}$",
        "CR_total_Pregnant": r"$CR_{total}^{Pregnant}$"
    }
    az.plot_trace(trace, var_names=params)
    # Replace default subplot titles with TeX labels
    fig = plt.gcf()
    for ax in fig.axes:
        title = ax.get_title()
        if title in tex_labels:
            ax.set_title(tex_labels[title], fontsize=16)
        # Set log scale for CR variables
        if 'CR_total' in title:
            ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trace_hyperparameters.png'))
    plt.close()
    
    # R-hat statistics
    rhat = az.rhat(trace, var_names=params)
    rhats = {var: float(np.mean(rhat[var].values)) for var in params}
    # Save rhat values
    with open(os.path.join(output_dir, 'rhat_hyperparameters.txt'), 'w') as f:
        for var, val in rhats.items():
            # Use ASCII 'R-hat' instead of Unicode circumflex to avoid encoding issues
            f.write(f"{var}: R-hat = {val:.3f}\n")

def plot_mechanism_diagnostics(trace, output_dir):
    """Create comprehensive diagnostics for mechanism-specific risks"""
    
    # Create summary statistics for mechanism risks
    mechanism_stats = []
    
    for grp in GROUPS:
        for mechanism, metals in ORGAN_GROUPS.items():
            # HI statistics
            hi_sum = calculate_additive_hi(trace, grp, metals)
            if hi_sum.max() > 0:
                mechanism_stats.append({
                    'group': grp,
                    'mechanism': mechanism,
                    'risk_type': 'HI',
                    'metals': list(metals),
                    'mean': np.mean(hi_sum),
                    'median': np.median(hi_sum),
                    'p95': np.percentile(hi_sum, 95),
                    'prob_exceed_1': (hi_sum > 1.0).mean(),
                    'max': np.max(hi_sum)
                })
            
            # CR statistics (only for carcinogenic metals)
            carcinogenic_metals = {metal for metal in METALS if SF[metal] > 0}
            carcinogenic_in_group = metals.intersection(carcinogenic_metals)
            
            if carcinogenic_in_group:
                cr_vals = []
                for metal in carcinogenic_in_group:
                    var_name = f"CR_{grp}_{metal}"
                    if var_name in trace.posterior.data_vars:
                        cr_vals.append(trace.posterior[var_name].values.flatten())
                
                if cr_vals:
                    cr_sum = np.sum(cr_vals, axis=0)
                    if cr_sum.max() > 0:
                        mechanism_stats.append({
                            'group': grp,
                            'mechanism': mechanism,
                            'risk_type': 'CR',
                            'metals': list(carcinogenic_in_group),
                            'mean': np.mean(cr_sum),
                            'median': np.median(cr_sum),
                            'p95': np.percentile(cr_sum, 95),
                            'prob_exceed_1e6': (cr_sum > 1e-6).mean(),
                            'max': np.max(cr_sum)
                        })
    
    if mechanism_stats:
        mech_stats_df = pd.DataFrame(mechanism_stats)
        mech_stats_df.to_csv(os.path.join(output_dir, "mechanism_diagnostics_summary.csv"), index=False)
        print("Saved mechanism_diagnostics_summary.csv")
        
        # Create heatmap of mechanism risks by group
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # HI heatmap - Probability of exceeding HI > 1
        hi_data = mech_stats_df[mech_stats_df['risk_type'] == 'HI'].pivot(
            index='group', columns='mechanism', values='prob_exceed_1'
        )
        if not hi_data.empty:
            sns.heatmap(hi_data, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax1, vmin=0, vmax=1)
            ax1.set_title('Probability HI > 1 by Mechanism and Group', fontsize=16)
            ax1.set_ylabel('Demographic Group', fontsize=14)
            ax1.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "mechanism_risk_heatmap.png"), dpi=300, bbox_inches='tight')
        print("Saved mechanism_risk_heatmap.png")
        plt.close(fig)

def calculate_additive_hi(trace, group, metals_in_group):
    """Calculate HI for metals with similar toxic mechanisms"""
    hi_values = []
    for metal in metals_in_group:
        var_name = f"HI_{group}_{metal}"
        if var_name in trace.posterior.data_vars:
            hi_values.append(trace.posterior[var_name].values.flatten())
    
    if hi_values:
        return np.sum(hi_values, axis=0)
    # return zero array matching number of posterior draws
    n_draws = trace.posterior.sizes.get("draw", 1)
    return np.zeros(n_draws)

def plot_mechanism_risks(trace, output_dir):
    """Plot mechanism-specific HI for each demographic group"""
    
    # Mechanism-specific HI plot
    fig_hi, axes_hi = plt.subplots(2, 2, figsize=(14, 10))
    axes_hi = axes_hi.flatten()
    
    for i, grp in enumerate(GROUPS):
        ax = axes_hi[i]
        mechanism_names = []
        hi_values = []
        
        for mechanism, metals in ORGAN_GROUPS.items():
            hi_sum = calculate_additive_hi(trace, grp, metals)
            if hi_sum.max() > 0:
                mechanism_names.append(mechanism)
                hi_values.append(hi_sum)
        
        if hi_values:
            ax.boxplot(hi_values, tick_labels=mechanism_names, showfliers=False)
            ax.set_title(f'Mechanism-Specific HI - {grp}', fontsize=16)
            ax.set_ylabel('Hazard Index', fontsize=14)
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='HI=1')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend(fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Mechanism-Specific HI - {grp}', fontsize=16)
    
    plt.tight_layout()
    fig_hi.savefig(os.path.join(output_dir, "mechanism_hi_by_group.png"), dpi=300, bbox_inches='tight')
    print("Saved mechanism_hi_by_group.png")
    plt.close(fig_hi)
    

def plot_posterior_distributions(trace, output_dir):
    """Plot posterior distributions for all variables including deterministic ones"""
    # All variables to plot
    all_vars = ["IRkg", "AF", "BW", "EF_ing_frac", "EF_der_frac", "ET_frac", "HI_total_Adults", "HI_total_Children", "HI_total_Teens", "HI_total_Pregnant", "CR_total_Adults", "CR_total_Children", "CR_total_Teens", "CR_total_Pregnant"]
    
    # Mapping for MathTeX labels
    tex_labels = {
        "IRkg": r"$IR_{kg}$",
        "AF": r"$AF$",
        "BW": r"$BW$",
        "EF_ing_frac": r"$EF_{ing}$",
        "EF_der_frac": r"$EF_{der}$",
        "ET_frac": r"$ET$",
        "HI_total_Adults": r"$HI_{total}^{Adults}$",
        "HI_total_Children": r"$HI_{total}^{Children}$",
        "HI_total_Teens": r"$HI_{total}^{Teens}$",
        "HI_total_Pregnant": r"$HI_{total}^{Pregnant}$",
        "CR_total_Adults": r"$CR_{total}^{Adults}$",
        "CR_total_Children": r"$CR_{total}^{Children}$",
        "CR_total_Teens": r"$CR_{total}^{Teens}$",
        "CR_total_Pregnant": r"$CR_{total}^{Pregnant}$"
    }
    
    colors = plt.cm.tab10.colors  # Use a colormap for consistent colors
    
    # Plot all variables in separate figures
    for i, var in enumerate(all_vars):
        if var in trace.posterior.data_vars:
            plt.figure(figsize=(8, 5))
            vals = trace.posterior[var].values.flatten()
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(vals, fill=True, color=color, alpha=0.7)
            
            # Calculate statistics
            mean_val = np.mean(vals)
            hdi_val = az.hdi(vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            plt.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            plt.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            plt.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = plt.ylim()[1] * 0.95
            if var.startswith('CR'):
                plt.text(mean_val * 1.1, y_pos, f'{mean_val:.2e}', 
                        color=color, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                plt.xscale('log')
            else:
                plt.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                        color=color, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.xlabel(tex_labels.get(var, var), fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.title(f"Posterior Distribution of {tex_labels.get(var, var)}\n(with Mean and 94% HDI)", fontsize=16)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'posterior_{var}.png'), dpi=300)
            plt.close()
            print(f"Saved posterior_{var}.png")

def plot_combined_cr_posteriors(trace, output_dir):
    """Create a 2x2 plot combining CR_total posterior distributions for all groups"""
    cr_vars = ["CR_total_Adults", "CR_total_Children", "CR_total_Teens", "CR_total_Pregnant"]
    
    # Mapping for MathTeX labels
    tex_labels = {
        "CR_total_Adults": r"$CR_{total}^{Adults}$",
        "CR_total_Children": r"$CR_{total}^{Children}$",
        "CR_total_Teens": r"$CR_{total}^{Teens}$",
        "CR_total_Pregnant": r"$CR_{total}^{Pregnant}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, var in enumerate(cr_vars):
        ax = axes[i]
        if var in trace.posterior.data_vars:
            vals = trace.posterior[var].values.flatten()
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(vals)
            hdi_val = az.hdi(vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.2e}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.2e}, {hdi_val[1]:.2e}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val * 1.1, y_pos, f'{mean_val:.2e}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xscale('log')
            ax.set_xlabel(tex_labels.get(var, var), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Posterior Distribution\n{tex_labels.get(var, var)}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_cr_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined_cr_posteriors.png")

def plot_combined_hi_posteriors(trace, output_dir):
    """Create a 2x2 plot combining HI_total posterior distributions for all groups"""
    hi_vars = ["HI_total_Adults", "HI_total_Children", "HI_total_Teens", "HI_total_Pregnant"]
    
    # Mapping for MathTeX labels
    tex_labels = {
        "HI_total_Adults": r"$HI_{total}^{Adults}$",
        "HI_total_Children": r"$HI_{total}^{Children}$",
        "HI_total_Teens": r"$HI_{total}^{Teens}$",
        "HI_total_Pregnant": r"$HI_{total}^{Pregnant}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, var in enumerate(hi_vars):
        ax = axes[i]
        if var in trace.posterior.data_vars:
            vals = trace.posterior[var].values.flatten()
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(vals)
            hdi_val = az.hdi(vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(tex_labels.get(var, var), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Posterior Distribution\n{tex_labels.get(var, var)}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_hi_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined_hi_posteriors.png")

def plot_combined_hyperparameter_posteriors(trace, output_dir):
    """Create a 2x2 plot combining hyperparameter posterior distributions"""
    params = ["IRkg", "AF", "BW", "EF_ing_frac"]
    
    # Mapping for MathTeX labels
    tex_labels = {
        "IRkg": r"$IR_{kg}$",
        "AF": r"$AF$",
        "BW": r"$BW$",
        "EF_ing_frac": r"$EF_{ing}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(params):
        ax = axes[i]
        if param in trace.posterior.data_vars:
            vals = trace.posterior[param].values.flatten()
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(vals)
            hdi_val = az.hdi(vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(tex_labels.get(param, param), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Posterior Distribution\n{tex_labels.get(param, param)}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_hyperparameter_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined_hyperparameter_posteriors.png")

def plot_combined_prior_posterior(model, trace, output_dir, prior_samples=2000):
    """Create a 3x3 plot combining prior vs posterior distributions for hyperparameters"""
    params = ["IRkg", "AF", "BW", "EF_ing_frac", "EF_der_frac", "ET_frac"]
    
    # Sample from priors
    with model:
        idata_prior = pm.sample_prior_predictive(prior_samples, var_names=params, return_inferencedata=True)
    
    # Mapping for MathTeX labels
    tex_labels = {
        "IRkg": r"$IR_{kg}$",
        "AF": r"$AF$",
        "BW": r"$BW$",
        "EF_ing_frac": r"$EF_{ing}$",
        "EF_der_frac": r"$EF_{der}$",
        "ET_frac": r"$ET$"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Prior density
        prior_vals = idata_prior.prior[param].values.flatten()
        sns.kdeplot(prior_vals, label='Prior', ax=ax, color='blue', alpha=0.7)
        
        # Posterior density
        post_vals = trace.posterior[param].values.flatten()
        sns.kdeplot(post_vals, label='Posterior', ax=ax, color='orange', alpha=0.7)
        
        ax.set_xlabel(tex_labels.get(param, param), fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"Prior vs Posterior\n{tex_labels.get(param, param)}", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_prior_posterior.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined_prior_posterior.png")

def plot_neurotoxic_hi_posteriors(trace, output_dir):
    """Create a 2x2 plot for neurotoxic HI posterior distributions across all groups"""
    groups = ["Adults", "Children", "Teens", "Pregnant"]
    neurotoxic_metals = ORGAN_GROUPS['neurotoxic']  # {'Mn'} - base case
    
    # Mapping for MathTeX labels
    tex_labels = {
        "Adults": r"$HI_{neurotoxic}^{Adults}$",
        "Children": r"$HI_{neurotoxic}^{Children}$",
        "Teens": r"$HI_{neurotoxic}^{Teens}$",
        "Pregnant": r"$HI_{neurotoxic}^{Pregnant}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, group in enumerate(groups):
        ax = axes[i]
        
        # Calculate neurotoxic HI for this group
        hi_vals = calculate_additive_hi(trace, group, neurotoxic_metals)
        
        if hi_vals.max() > 0:
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(hi_vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(hi_vals)
            hdi_val = az.hdi(hi_vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(tex_labels.get(group, group), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Neurotoxic HI\n{group}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            ax.text(0.5, 0.5, 'No neurotoxic HI data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Neurotoxic HI\n{group}", fontsize=16)
    
    plt.suptitle("Neurotoxic Hazard Index by Demographic Group", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neurotoxic_hi_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved neurotoxic_hi_posteriors.png")

def plot_nephrotoxic_hi_posteriors(trace, output_dir):
    """Create a 2x2 plot for nephrotoxic HI posterior distributions across all groups"""
    groups = ["Adults", "Children", "Teens", "Pregnant"]
    nephrotoxic_metals = ORGAN_GROUPS['nephrotoxic']  # {'Cd', 'Hg', 'CrVI'}
    
    # Mapping for MathTeX labels
    tex_labels = {
        "Adults": r"$HI_{nephrotoxic}^{Adults}$",
        "Children": r"$HI_{nephrotoxic}^{Children}$",
        "Teens": r"$HI_{nephrotoxic}^{Teens}$",
        "Pregnant": r"$HI_{nephrotoxic}^{Pregnant}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, group in enumerate(groups):
        ax = axes[i]
        
        # Calculate nephrotoxic HI for this group
        hi_vals = calculate_additive_hi(trace, group, nephrotoxic_metals)
        
        if hi_vals.max() > 0:
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(hi_vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(hi_vals)
            hdi_val = az.hdi(hi_vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(tex_labels.get(group, group), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Nephrotoxic HI\n{group}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            ax.text(0.5, 0.5, 'No nephrotoxic HI data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Nephrotoxic HI\n{group}", fontsize=16)
    
    plt.suptitle("Nephrotoxic Hazard Index by Demographic Group", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nephrotoxic_hi_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved nephrotoxic_hi_posteriors.png")

def plot_hepatotoxic_hi_posteriors(trace, output_dir):
    """Create a 2x2 plot for hepatotoxic HI posterior distributions across all groups"""
    groups = ["Adults", "Children", "Teens", "Pregnant"]
    hepatotoxic_metals = ORGAN_GROUPS['hepatotoxic']  # {'As', 'Cd', 'Cu', 'CrVI'}
    
    # Mapping for MathTeX labels
    tex_labels = {
        "Adults": r"$HI_{hepatotoxic}^{Adults}$",
        "Children": r"$HI_{hepatotoxic}^{Children}$",
        "Teens": r"$HI_{hepatotoxic}^{Teens}$",
        "Pregnant": r"$HI_{hepatotoxic}^{Pregnant}$"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, group in enumerate(groups):
        ax = axes[i]
        
        # Calculate hepatotoxic HI for this group
        hi_vals = calculate_additive_hi(trace, group, hepatotoxic_metals)
        
        if hi_vals.max() > 0:
            color = colors[i % len(colors)]
            
            # Plot KDE
            sns.kdeplot(hi_vals, fill=True, color=color, alpha=0.7, ax=ax)
            
            # Calculate statistics
            mean_val = np.mean(hi_vals)
            hdi_val = az.hdi(hi_vals, hdi_prob=0.94)
            
            # Add vertical lines for mean and HDI
            ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(hdi_val[0], color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            ax.axvline(hdi_val[1], color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'94% HDI: [{hdi_val[0]:.3f}, {hdi_val[1]:.3f}]')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(mean_val + 0.01, y_pos, f'{mean_val:.3f}', 
                    color=color, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(tex_labels.get(group, group), fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(f"Hepatotoxic HI\n{group}", fontsize=16)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            ax.text(0.5, 0.5, 'No hepatotoxic HI data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Hepatotoxic HI\n{group}", fontsize=16)
    
    plt.suptitle("Hepatotoxic Hazard Index by Demographic Group", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hepatotoxic_hi_posteriors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved hepatotoxic_hi_posteriors.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot prior vs posterior and convergence diagnostics')
    parser.add_argument('--model-file', required=True, help='Pickle file of PyMC model (model.pkl)')
    parser.add_argument('--trace-file', required=True, help='NetCDF trace file (trace.nc)')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Load trace
    trace = az.from_netcdf(args.trace_file)

    # Attempt to load model for prior sampling (may fail if pickle is not compatible)
    model = None
    try:
        with open(args.model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Warning: could not load model for prior sampling ({e}), proceeding with trace-only diagnostics.")

    # Focused HBMPRA hyperparameter diagnostics
    vars_list = _detect_hbmpra_hyperparams(trace)
    print('Detected HBMPRA hyperparameters:', vars_list)

    # Prior vs posterior (requires model)
    if model is not None:
        plot_prior_vs_posterior_vars(model, trace, vars_list, args.output_dir)
        # Also create a compact 4x4 panel view
        plot_prior_posterior_panel(model, trace, vars_list, args.output_dir)
    else:
        print('Model not available; skipping prior vs posterior plots that require prior sampling.')

    # Trace plots for the hyperparameters
    plot_traces_for_vars(trace, vars_list, args.output_dir)

    # Convergence summaries (R-hat, ESS) - pass the exact trace file path
    plot_convergence_summaries(args.trace_file, out_png=os.path.join(args.output_dir, 'convergence_summary.png'))

    print('Focused HBMPRA diagnostics saved in', args.output_dir)
