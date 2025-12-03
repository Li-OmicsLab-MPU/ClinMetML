"""Standalone DCA test script for ClinMetML.

This script generates a simple synthetic binary outcome and model
probabilities, then calls the DCA utilities to verify that the
net benefit curves and plotting work as expected.

Usage
-----
    cd /home/ljr/tabular_biomarker/ClinMetML
    python -m tests.test_dca_standalone

It will create outputs under `clinmetml_tests/dca_demo/`.
"""

from pathlib import Path

import numpy as np

from clinmetml.core.dca import compute_dca_curves, plot_dca_curves


def make_synthetic_data(n: int = 1000, prevalence: float = 0.2, seed: int = 42):
    """Generate a simple synthetic y_true and predicted probabilities.

    Parameters
    ----------
    n : int
        Number of samples.
    prevalence : float
        Desired proportion of positive cases.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Binary labels with given prevalence
    y_true = rng.binomial(1, prevalence, size=n)

    # Construct probabilities with some discrimination:
    # - positives have probabilities centered higher
    # - negatives have probabilities centered lower
    pos_probs = rng.normal(loc=0.7, scale=0.15, size=n)
    neg_probs = rng.normal(loc=0.2, scale=0.15, size=n)

    y_pred_proba = np.where(y_true == 1, pos_probs, neg_probs)
    y_pred_proba = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)

    return y_true, y_pred_proba


def main():
    output_dir = Path("clinmetml_tests") / "dca_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data with moderate imbalance
    y_true, y_proba = make_synthetic_data(n=2000, prevalence=0.2, seed=123)

    # Compute DCA curves using the core function
    thresholds, model_curves, treat_all = compute_dca_curves(
        y_true,
        {"synthetic_model": y_proba},
    )

    # Save raw DCA values to CSV for inspection
    import pandas as pd

    df = pd.DataFrame(
        {
            "threshold": thresholds,
            "synthetic_model_net_benefit": model_curves["synthetic_model"],
            "treat_all": treat_all,
        }
    )
    csv_path = output_dir / "synthetic_dca.csv"
    df.to_csv(csv_path, index=False)

    # Plot the DCA curve to a PNG file
    png_path = output_dir / "synthetic_dca.png"
    plot_dca_curves(
        y_true,
        {"synthetic_model": y_proba},
        title="DCA - Synthetic Model",
        output_path=str(png_path),
    )

    print("Synthetic DCA test completed.")
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    main()
