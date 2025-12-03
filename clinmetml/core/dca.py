import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional
import matplotlib.pyplot as plt


def calculate_net_benefit_model(thresholds: np.ndarray, y_pred_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba)
    thresholds = np.asarray(thresholds)
    n = float(len(y_true))
    net_benefit = np.zeros_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        if t <= 0.0:
            y_pred = np.ones_like(y_true)
        elif t >= 1.0:
            y_pred = np.zeros_like(y_true)
        else:
            y_pred = (y_pred_proba >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        if t >= 1.0:
            net_benefit[i] = -np.inf if fp > 0 else tp / n
        else:
            net_benefit[i] = (tp / n) - (fp / n) * (t / (1.0 - t))

    return net_benefit


def calculate_net_benefit_all(thresholds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    thresholds = np.asarray(thresholds)
    n = float(len(y_true))
    prevalence = np.mean(y_true)
    net_benefit = np.zeros_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        if t >= 1.0:
            net_benefit[i] = -np.inf
        else:
            net_benefit[i] = prevalence - (1.0 - prevalence) * (t / (1.0 - t))

    return net_benefit


def compute_dca_curves(
    y_true: Iterable[int],
    scores: Dict[str, Iterable[float]],
    thresholds: Optional[np.ndarray] = None,
    threshold_limits: Tuple[float, float] = (0.05, 1.0),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Compute DCA curves for given scores.

    Parameters
    ----------
    y_true : Iterable[int]
        True binary labels (0/1).
    scores : Dict[str, Iterable[float]]
        Mapping from model name to predicted probabilities for the positive class.
    thresholds : Optional[np.ndarray]
        Explicit array of threshold probabilities. If None, a uniform grid will
        be generated within ``threshold_limits``.
    threshold_limits : Tuple[float, float], default=(0.05, 1.0)
        Lower and upper bounds for the automatically generated threshold grid
        when ``thresholds`` is None.
    """

    if thresholds is None:
        low, high = threshold_limits
        thresholds = np.linspace(low, high, 101)
    thresholds = np.asarray(thresholds)

    y_true_arr = np.asarray(list(y_true)).astype(int)

    model_curves: Dict[str, np.ndarray] = {}
    for name, s in scores.items():
        model_curves[name] = calculate_net_benefit_model(thresholds, np.asarray(list(s)), y_true_arr)

    treat_all = calculate_net_benefit_all(thresholds, y_true_arr)
    return thresholds, model_curves, treat_all


def plot_dca_curves(
    y_true: Iterable[int],
    scores: Dict[str, Iterable[float]],
    title: str,
    output_path: Optional[str] = None,
    threshold_limits: Tuple[float, float] = (0.05, 0.9),
):
    """Plot DCA curves for one or more models.

    By default, thresholds are sampled uniformly between 0.05 and 1.0, which is
    typically a more clinically relevant range for decision thresholds. This
    can be overridden either by passing explicit ``thresholds`` to
    ``compute_dca_curves`` or by changing ``threshold_limits`` here.
    """

    thresholds, model_curves, treat_all = compute_dca_curves(
        y_true,
        scores,
        threshold_limits=threshold_limits,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for i, (name, nb) in enumerate(model_curves.items()):
        ax.plot(thresholds, nb, label=name, color=colors[i % len(colors)], linewidth=2.0)

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.5, label="Treat None")
    ax.plot(thresholds, treat_all, color="navy", linestyle="--", linewidth=1.5, label="Treat All")

    ax.set_xlim(0.0, 1.0)
    finite_vals = np.isfinite(treat_all)
    for nb in model_curves.values():
        finite_vals |= np.isfinite(nb)
    if np.any(finite_vals):
        ymin = min([np.min(v[np.isfinite(v)]) for v in [treat_all, *model_curves.values()] if np.any(np.isfinite(v))])
        ymax = max([np.max(v[np.isfinite(v)]) for v in [treat_all, *model_curves.values()] if np.any(np.isfinite(v))])
        ax.set_ylim(ymin - 0.05, ymax + 0.05)

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)
