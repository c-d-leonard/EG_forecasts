import json
import numpy as np
import matplotlib.pyplot as plt
import math

def analyze_results(filename, make_plot=True):
    condition_1_count = 0
    condition_2_count = 0
    condition_either_count = 0
    total = 0

    OmM_true_vals = []
    OmM_fit_means = []

    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            if data['const_bad_fit']:
                condition_1_count += 1
                condition_either_count += 1
            if data['outside_95']:
                condition_2_count += 1
                condition_either_count += 1
            if data['const_bad_fit'] and data['outside_95']:
                print('getting both outside95 and const_bad_fit, should not occur')
            
        # Store OmM_true and OmM_fit_mean if present
            # Only collect Ωm values if fit was done (OmM_fit_mean is not NaN)
            if not math.isnan(data['OmegaM_fit_mean']):
                OmM_true_vals.append(data['OmegaM_true'])
                OmM_fit_means.append(data['OmegaM_fit_mean'])
        

    print(f"Total runs: {total}")
    print(f"Constant bad fit: {condition_1_count} ({condition_1_count/total:.2%})")
    print(f"Outside 95: {condition_2_count} ({condition_2_count/total:.2%})")
    print(f"Reject GR either way: {condition_either_count} ({condition_either_count/total:.2%})")

    if OmM_true_vals:
        OmM_true_vals = np.array(OmM_true_vals)
        OmM_fit_means = np.array(OmM_fit_means)
        diff = OmM_true_vals - OmM_fit_means

        print(f"Number of runs with GR fit: {len(OmM_true_vals)}")
        print(f"Mean ΔΩm (fit - true): {np.mean(diff):.4f}")
        print(f"Std dev ΔΩm: {np.std(diff):.4f}")

        if make_plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(OmM_true_vals, OmM_fit_means, alpha=0.5, s=15, edgecolor='none')
            min_val = min(OmM_true_vals.min(), OmM_fit_means.min())
            max_val = max(OmM_true_vals.max(), OmM_fit_means.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
            plt.xlabel(r"$\Omega_{\mathrm{M,true}}$")
            plt.ylabel(r"$\Omega_{\mathrm{M,fit}}$")
            plt.legend()                    
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('../plots/OmegaMdiag_fR-4_DESY3Prior_LSSTY1.pdf')
            plt.close()
        

analyze_results("../txtfiles/post_pred_test_fR-4_CMBPrior_LSSTY10_gc_seed_100runs.jsonl", make_plot=False)
