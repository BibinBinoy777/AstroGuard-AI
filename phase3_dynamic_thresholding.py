#!/usr/bin/env python3
"""
Phase 3: EWMA Error Smoothing, Non-Parametric Dynamic Thresholding & Pruning Engine
===================================================================================
Project 3 — Spacecraft Anomaly Detection (NASA JPL Telemanom Architecture)
Dataset: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Reference: https://github.com/khundman/telemanom

What Phase 3 Builds & Verifies:
  1. Loads test predictions from Phase 2 (`phase2_predictions.npz` or runs Phase 1 -> 2
     automatically) for spacecraft channel `P-1`.
  2. Computes raw multi-step horizon residuals `e_raw` and applies Exponentially Weighted
     Moving Average (EWMA) smoothing `e_s` to suppress transient single-step sensor noise.
  3. Implements NASA JPL's Batch Non-Parametric Dynamic Thresholding (Hundman et al., KDD 2018,
     Section 3.3 `process_batches` & `find_epsilon`):
       - Evaluates historical error batches of length `batch_window = 2100` steps.
       - In each batch, searches `epsilon(z) = mu(e_s) + z * sigma(e_s)` for z in [2.5, 12.0]:
             Score(z) = [ (Delta_mu / mu) + (Delta_sigma / sigma) ] / ( |e_a| + |E_seq|^2 )
       - Applies bidirectional error buffer padding (`error_buffer = 75`) and minimum percent
         decrease pruning (`p_min = 0.13`) across candidate sequences to reject false-positive
         candidates in normal batches.
  4. Runs a 3-way ablation benchmark comparing:
       (a) Static 2.5-Sigma Threshold on Raw Residuals (`e_raw`)
       (b) Unpruned Dynamic Threshold (`EWMA + Batch NDT without p_min pruning`)
       (c) Full Telemanom Pipeline (`EWMA + Batch NDT + p_min = 0.13 Pruning`)
  5. Exports a 3-panel Phase 3 Dynamic Thresholding & Optimization Diagnostic figure.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase1_data_windowing import Phase1TelemetryDataset
from phase2_lstm_forecaster import Phase2ForecasterTrainer, Phase2ModelConfig


# ==============================================================================
# 1. PHASE 3 CONFIGURATION & DATA STRUCTURES
# ==============================================================================

@dataclass
class Phase3ThresholdConfig:
    """Hyperparameters matching NASA JPL's Telemanom (`telemanom/errors.py`, KDD 2018)."""
    l_s: int = 250
    n_predictions: int = 10
    ewma_span: int = 40                     # Smoothing window span for EWMA residuals
    batch_window: int = 2100                # Batch window size (70 steps * 30 windows in Telemanom)
    error_buffer: int = 75                  # Bidirectional padding steps around anomalous points
    z_min: float = 2.5                      # Minimum z multiplier for epsilon = mu + z * sigma
    z_max: float = 12.0                     # Maximum z multiplier
    z_step: float = 0.25                    # Search grid resolution for z
    p_min: float = 0.13                     # Minimum relative peak drop (13%) to retain sequence


@dataclass
class CandidateSequenceInfo:
    """Diagnostic metadata for each candidate anomalous sequence before/after pruning."""
    batch_id: int
    start_t: int
    end_t: int
    peak_error: float
    pct_drop_to_next: float
    kept_after_pruning: bool


@dataclass
class Phase3DetectionOutput:
    """Full diagnostic output from the Phase 3 thresholding engine."""
    e_raw: np.ndarray
    e_s: np.ndarray
    epsilon_trace: np.ndarray               # Time-varying dynamic threshold across batches
    z_grid: np.ndarray                      # z search grid for primary anomaly batch
    score_grid: np.ndarray                  # Objective score curve for primary anomaly batch
    optimal_z_primary: float
    epsilon_primary: float
    epsilon_static_raw: float
    candidate_sequences: List[CandidateSequenceInfo] = field(default_factory=list)
    pruned_dynamic_intervals: List[Tuple[int, int]] = field(default_factory=list)
    unpruned_dynamic_intervals: List[Tuple[int, int]] = field(default_factory=list)
    static_raw_intervals: List[Tuple[int, int]] = field(default_factory=list)


# ==============================================================================
# 2. NON-PARAMETRIC DYNAMIC THRESHOLDING & PRUNING ENGINE
# ==============================================================================

class Phase3DynamicThresholder:
    """
    Implements the complete unsupervised error thresholding and pruning pipeline
    from Hundman et al. (KDD 2018), including batch windowing (`process_batches`).
    """

    def __init__(self, cfg: Phase3ThresholdConfig):
        self.cfg = cfg

    def compute_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute multi-step prediction residual `e_raw` and EWMA-smoothed
        residual `e_s`.
        """
        e_raw = np.mean(np.abs(y_true - y_pred), axis=1).astype(np.float64)
        e_s = (
            pd.Series(e_raw)
            .ewm(span=self.cfg.ewma_span, adjust=False)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        return e_raw, e_s

    @staticmethod
    def group_padded_intervals(
        anomalous_indices: np.ndarray, buffer_steps: int, max_len: int
    ) -> List[Tuple[int, int]]:
        """
        Pad each above-threshold index by `[-buffer_steps, +buffer_steps]` and merge
        overlapping or adjacent windows into contiguous `[start_idx, end_idx]` sequences.
        """
        if len(anomalous_indices) == 0:
            return []

        merged: List[Tuple[int, int]] = []
        for idx in anomalous_indices:
            s = max(0, int(idx) - buffer_steps)
            e = min(max_len - 1, int(idx) + buffer_steps)
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def optimize_epsilon_for_batch(
        self, batch_e_s: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate candidate thresholds `epsilon(z) = mu(e_s) + z * sigma(e_s)` across
        `z in [z_min, z_max]` on a batch window and select the `z*` that maximizes
        the KDD 2018 `find_epsilon` objective:
            Score(z) = ( (Delta_mu / mu) + (Delta_sigma / sigma) ) / ( |e_a| + |E_seq|^2 )
        where `e_a = {e_s > epsilon(z)}` and `E_seq` are the contiguous sequences of `e_a`.
        """
        mu_all = float(np.mean(batch_e_s))
        sigma_all = float(np.std(batch_e_s)) + 1e-12

        z_grid = np.arange(self.cfg.z_min, self.cfg.z_max + 1e-6, self.cfg.z_step, dtype=np.float64)
        score_grid = np.zeros_like(z_grid)

        best_score = -1.0
        best_z = 2.5
        best_eps = mu_all + best_z * sigma_all

        for idx, z in enumerate(z_grid):
            eps = mu_all + float(z) * sigma_all
            raw_above = np.where(batch_e_s > eps)[0]
            if len(raw_above) == 0 or len(raw_above) >= int(0.5 * len(batch_e_s)):
                score_grid[idx] = 0.0
                continue

            normal_points = batch_e_s[batch_e_s <= eps]
            if len(normal_points) < 20:
                score_grid[idx] = 0.0
                continue

            seqs = self.group_padded_intervals(raw_above, self.cfg.error_buffer, len(batch_e_s))
            delta_mu = mu_all - float(np.mean(normal_points))
            delta_sigma = sigma_all - float(np.std(normal_points))
            denom = float(len(raw_above) + (len(seqs) ** 2))
            score = ((delta_mu / mu_all) + (delta_sigma / sigma_all)) / denom
            score_grid[idx] = max(0.0, score)

            if score > best_score:
                best_score = score
                best_z = float(z)
                best_eps = eps

        return best_eps, best_z, z_grid, score_grid

    def prune_batch_sequences(
        self,
        batch_e_s: np.ndarray,
        raw_sequences: List[Tuple[int, int]],
        normal_max: float,
        batch_start_offset: int,
        batch_id: int,
    ) -> Tuple[List[Tuple[int, int]], List[CandidateSequenceInfo]]:
        """
        Sort candidate sequences in a batch descending by peak error `e_max` and retain
        sequences whose peak drops by at least `p_min` (13%) relative to the next lower
        peak (or `normal_max` for the lowest candidate sequence in the batch).
        """
        if not raw_sequences:
            return [], []

        seq_peaks = []
        for s, e in raw_sequences:
            seq_peaks.append((s, e, float(np.max(batch_e_s[s : e + 1]))))

        # Sort descending by peak error
        seq_peaks.sort(key=lambda item: item[2], reverse=True)
        comparison_peaks = [item[2] for item in seq_peaks] + [normal_max]

        last_kept_idx = -1
        pct_drops: List[float] = []
        for i in range(len(seq_peaks)):
            drop = (comparison_peaks[i] - comparison_peaks[i + 1]) / (comparison_peaks[i] + 1e-12)
            pct_drops.append(float(drop))
            if drop >= self.cfg.p_min:
                last_kept_idx = i

        candidates_info: List[CandidateSequenceInfo] = []
        kept_intervals: List[Tuple[int, int]] = []

        for i, (s, e, pk) in enumerate(seq_peaks):
            is_kept = i <= last_kept_idx
            abs_s = int(s + batch_start_offset)
            abs_e = int(e + batch_start_offset)
            candidates_info.append(
                CandidateSequenceInfo(
                    batch_id=batch_id,
                    start_t=abs_s,
                    end_t=abs_e,
                    peak_error=round(pk, 4),
                    pct_drop_to_next=round(pct_drops[i], 4),
                    kept_after_pruning=is_kept,
                )
            )
            if is_kept:
                kept_intervals.append((abs_s, abs_e))

        kept_intervals.sort(key=lambda x: x[0])
        return kept_intervals, candidates_info

    @staticmethod
    def _merge_global_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        sorted_ints = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_ints[0]]
        for s, e in sorted_ints[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def run(
        self, y_true: np.ndarray, y_pred: np.ndarray, offset: int = 250
    ) -> Phase3DetectionOutput:
        """Execute batch-windowed Non-Parametric Dynamic Thresholding + Pruning."""
        e_raw, e_s = self.compute_residuals(y_true, y_pred)
        N = len(e_s)
        W = self.cfg.batch_window

        # Channel-wide normal baseline reference (85th percentile of smoothed error)
        # prevents normal-max reference from collapsing in anomaly-free batches
        channel_normal_baseline = float(np.percentile(e_s, 85) * 1.18)

        epsilon_trace = np.zeros(N, dtype=np.float64)
        all_candidates: List[CandidateSequenceInfo] = []
        pruned_global: List[Tuple[int, int]] = []
        unpruned_global: List[Tuple[int, int]] = []

        primary_z_grid = np.arange(self.cfg.z_min, self.cfg.z_max + 1e-6, self.cfg.z_step)
        primary_score_grid = np.zeros_like(primary_z_grid)
        primary_opt_z = 2.5
        primary_eps = 0.3

        batch_id = 0
        for b_start in range(0, N, W):
            b_end = min(N, b_start + W)
            if b_end - b_start < 500 and b_start > 0:
                b_start_eff = max(0, b_end - W)
            else:
                b_start_eff = b_start

            batch_id += 1
            batch_slice = e_s[b_start_eff:b_end]
            eps_b, z_b, z_grid, score_grid = self.optimize_epsilon_for_batch(batch_slice)
            epsilon_trace[b_start:b_end] = eps_b

            if batch_id == 1:
                primary_z_grid = z_grid
                primary_score_grid = score_grid
                primary_opt_z = z_b
                primary_eps = eps_b

            above_b = np.where(batch_slice > eps_b)[0]
            raw_b_seqs = self.group_padded_intervals(above_b, self.cfg.error_buffer, len(batch_slice))

            normal_mask = np.ones(len(batch_slice), dtype=bool)
            for s, e in raw_b_seqs:
                normal_mask[s : e + 1] = False
            local_normal_max = (
                float(np.max(batch_slice[normal_mask]))
                if np.any(normal_mask)
                else float(np.min(batch_slice))
            )
            effective_normal_max = max(local_normal_max, channel_normal_baseline)

            batch_offset = offset + b_start_eff
            kept_b, cand_b = self.prune_batch_sequences(
                batch_slice, raw_b_seqs, effective_normal_max, batch_offset, batch_id
            )
            all_candidates.extend(cand_b)
            pruned_global.extend(kept_b)
            unpruned_global.extend([(int(s + batch_offset), int(e + batch_offset)) for s, e in raw_b_seqs])

        pruned_merged = self._merge_global_intervals(pruned_global)
        unpruned_merged = self._merge_global_intervals(unpruned_global)

        # Baseline 1: Static 2.5-sigma threshold directly on unsmoothed residuals `e_raw`
        eps_static_raw = float(np.mean(e_raw) + 2.5 * np.std(e_raw))
        above_static = np.where(e_raw > eps_static_raw)[0]
        static_seqs = [
            (int(s + offset), int(e + offset))
            for s, e in self.group_padded_intervals(above_static, 25, len(e_raw))
        ]

        return Phase3DetectionOutput(
            e_raw=e_raw,
            e_s=e_s,
            epsilon_trace=epsilon_trace,
            z_grid=primary_z_grid,
            score_grid=primary_score_grid,
            optimal_z_primary=primary_opt_z,
            epsilon_primary=primary_eps,
            epsilon_static_raw=eps_static_raw,
            candidate_sequences=all_candidates,
            pruned_dynamic_intervals=pruned_merged,
            unpruned_dynamic_intervals=unpruned_merged,
            static_raw_intervals=static_seqs,
        )


# ==============================================================================
# 3. SEGMENT-LEVEL EVALUATOR (KDD 2018 BENCHMARK METRICS)
# ==============================================================================

def evaluate_intervals(
    predicted: List[Tuple[int, int]], ground_truth: List[Tuple[int, int]]
) -> Dict[str, object]:
    """Compute segment-level TP, FP, FN, Precision, Recall, and F1 score."""
    def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return max(a[0], b[0]) <= min(a[1], b[1])

    tp = sum(1 for gt in ground_truth if any(_overlaps(gt, p) for p in predicted))
    fn = len(ground_truth) - tp
    fp = sum(1 for p in predicted if not any(_overlaps(p, gt) for gt in ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "detected_count": len(predicted),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "intervals": [list(x) for x in predicted],
    }


# ==============================================================================
# 4. PHASE 3 VISUALIZATION (3-PANEL DYNAMIC THRESHOLDING & ABLATION REPORT)
# ==============================================================================

def plot_phase3_diagnostics(
    out: Phase3DetectionOutput,
    t_indices: np.ndarray,
    ground_truth_seqs: List[Tuple[int, int]],
    ablation_metrics: Dict[str, Dict[str, object]],
    chan_id: str,
    output_path: str,
) -> None:
    """Render a 3-panel Phase 3 diagnostic chart."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 9.6), dpi=160)
    fig.patch.set_facecolor("#0F172A")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.05], hspace=0.40, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, :])        # Full width top panel: Residuals + Dynamic Threshold Trace + Windows
    ax2 = fig.add_subplot(gs[1, :])        # Full width middle panel: Objective Score(z) optimization curve
    ax3_left = fig.add_subplot(gs[2, 0])   # Bottom left: Candidate sequence peak pruning across batches
    ax3_right = fig.add_subplot(gs[2, 1])  # Bottom right: 3-Way Ablation Precision/Recall/F1 comparison

    # --- Panel 1: Raw vs EWMA Error + Batch Dynamic Threshold Trace + Detected vs Ground Truth ---
    ax1.set_facecolor("#1E293B")
    ax1.plot(t_indices, out.e_raw, color="#64748B", lw=0.75, alpha=0.45, label="Raw Horizon Residual ($e_t$)")
    ax1.plot(t_indices, out.e_s, color="#A855F7", lw=1.7, label="EWMA Smoothed Error ($e_s$, span=40)")
    ax1.plot(
        t_indices, out.epsilon_trace, color="#10B981", linestyle="--", lw=1.8,
        label="Adaptive Batch Dynamic Threshold ($\epsilon_b^*$)"
    )
    ax1.axhline(
        out.epsilon_static_raw, color="#F43F5E", linestyle=":", lw=1.3,
        label=f"Static 2.5$\sigma$ Raw Threshold (val={out.epsilon_static_raw:.3f})"
    )

    for i, (s, e) in enumerate(ground_truth_seqs):
        ax1.axvspan(s, e, color="#EF4444", alpha=0.28, label="Ground-Truth Anomaly" if i == 0 else None)

    for i, (s, e) in enumerate(out.pruned_dynamic_intervals):
        ax1.axvspan(
            s, e, color="#F59E0B", alpha=0.22, hatch="//",
            label="Telemanom Detected Window (After Pruning)" if i == 0 else None
        )

    ax1.set_title(
        f"Phase 3A: EWMA Smoothed Error ($e_s$), Batch Dynamic Threshold ($\epsilon_b^*$) & Pruned Detections — Channel {chan_id}",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax1.set_xlabel("Mission Timestep (t)", color="#E2E8F0", fontsize=9.5)
    ax1.set_ylabel("Prediction Error", color="#E2E8F0", fontsize=9.5)
    ax1.set_xlim(int(t_indices[0]), int(t_indices[-1]))
    ax1.tick_params(colors="#CBD5E1")
    ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", ncol=2, facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    # --- Panel 2: Non-Parametric Dynamic Threshold Objective Curve Score(z) ---
    ax2.set_facecolor("#1E293B")
    ax2.plot(
        out.z_grid, out.score_grid * 1e4, color="#38BDF8", lw=2.2, marker="o", markersize=4,
        label=r"KDD 2018 Objective Score($z$) $\times 10^4$ (Batch 1)"
    )
    opt_idx = int(np.argmin(np.abs(out.z_grid - out.optimal_z_primary)))
    opt_score = out.score_grid[opt_idx] * 1e4
    ax2.axvline(
        out.optimal_z_primary, color="#10B981", linestyle="--", lw=1.6,
        label=f"Selected Argmax $z^* = {out.optimal_z_primary:.2f}$ ($\epsilon_1^* = {out.epsilon_primary:.3f}$)"
    )
    ax2.scatter([out.optimal_z_primary], [opt_score], color="#10B981", s=70, zorder=5)

    ax2.set_title(
        r"Phase 3B: Unsupervised Threshold Optimization Curve — $\arg\max_z [(\Delta\mu/\mu + \Delta\sigma/\sigma) / (|e_a| + |E_{seq}|^2)]$",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax2.set_xlabel("Threshold Multiplier (z) in $\epsilon = \mu(e_s) + z \cdot \sigma(e_s)$", color="#E2E8F0", fontsize=9.5)
    ax2.set_ylabel(r"Objective Score ($\times 10^{-4}$)", color="#E2E8F0", fontsize=9.5)
    ax2.tick_params(colors="#CBD5E1")
    ax2.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    # --- Panel 3 Left: Candidate Sequence Peak Error & Minimum Drop Pruning ---
    ax3_left.set_facecolor("#1E293B")
    labels = [f"B{c.batch_id}\n[{c.start_t},{c.end_t}]" for c in out.candidate_sequences]
    peaks = [c.peak_error for c in out.candidate_sequences]
    colors = ["#10B981" if c.kept_after_pruning else "#EF4444" for c in out.candidate_sequences]
    x_pos = np.arange(len(labels))
    bars = ax3_left.bar(x_pos, peaks, color=colors, width=0.55, edgecolor="#E2E8F0", linewidth=0.8)

    for bar, c in zip(bars, out.candidate_sequences):
        status_tag = "KEPT" if c.kept_after_pruning else "PRUNED"
        ax3_left.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{status_tag}\n\u0394={int(c.pct_drop_to_next*100)}%",
            ha="center", va="bottom", color="#F8FAFC", fontsize=7.5, fontweight="bold",
        )

    ax3_left.set_title(
        f"Phase 3C-1: Candidate Peak Pruning Across Batches ($p_{{min}} = 13\%$)",
        color="#F8FAFC", fontsize=10.8, fontweight="bold", pad=8,
    )
    ax3_left.set_xticks(x_pos)
    ax3_left.set_xticklabels(labels, color="#CBD5E1", fontsize=7.8)
    ax3_left.set_ylabel("Peak Smoothed Error ($e_{max}$)", color="#E2E8F0", fontsize=9)
    ax3_left.set_ylim(0, max(peaks + [0.5]) * 1.30)
    ax3_left.tick_params(colors="#CBD5E1")
    ax3_left.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6, axis="y")

    # --- Panel 3 Right: 3-Way Ablation Benchmark (Precision, Recall, F1) ---
    ax3_right.set_facecolor("#1E293B")
    methods = ["Static 2.5\u03c3\n(Raw Residual)", "Dynamic NDT\n(Unpruned)", "Full Telemanom\n(EWMA+NDT+Prune)"]
    keys = ["static_2.5sigma_raw", "unpruned_dynamic_ndt", "full_telemanom_ndt_pruned"]
    prec_vals = [float(ablation_metrics[k]["Precision"]) for k in keys]
    rec_vals = [float(ablation_metrics[k]["Recall"]) for k in keys]
    f1_vals = [float(ablation_metrics[k]["F1"]) for k in keys]

    xm = np.arange(len(methods))
    w = 0.24
    ax3_right.bar(xm - w, prec_vals, width=w, color="#38BDF8", label="Precision")
    ax3_right.bar(xm, rec_vals, width=w, color="#A855F7", label="Recall")
    bars_f1 = ax3_right.bar(xm + w, f1_vals, width=w, color="#10B981", label="F1-Score")

    for bar, k in zip(bars_f1, keys):
        fp_cnt = ablation_metrics[k]["FP"]
        ax3_right.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f"F1={bar.get_height():.2f}\n(FP={fp_cnt})",
            ha="center", va="bottom", color="#F8FAFC", fontsize=7.8, fontweight="bold",
        )

    ax3_right.set_title(
        "Phase 3C-2: Thresholding Ablation Benchmark",
        color="#F8FAFC", fontsize=10.8, fontweight="bold", pad=8,
    )
    ax3_right.set_xticks(xm)
    ax3_right.set_xticklabels(methods, color="#CBD5E1", fontsize=8.5)
    ax3_right.set_ylabel("Score [0.0 - 1.0]", color="#E2E8F0", fontsize=9)
    ax3_right.set_ylim(0, 1.25)
    ax3_right.tick_params(colors="#CBD5E1")
    ax3_right.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6, axis="y")
    ax3_right.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    for ax in (ax1, ax2, ax3_left, ax3_right):
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: NASA Telemanom Dynamic Thresholding & Ablation")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SMAP/MSL data folder")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to labeled_anomalies.csv")
    parser.add_argument("--chan_id", type=str, default="P-1", help="Channel ID (default: P-1)")
    parser.add_argument("--preds_npz", type=str, default=None, help="Optional path to Phase 2 saved predictions .npz")
    parser.add_argument("--plot_path", type=str, default="phase3_dynamic_thresholding_diagnostics.png", help="Output PNG path")
    args = parser.parse_args()

    cfg3 = Phase3ThresholdConfig()
    dataset = Phase1TelemetryDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        l_s=cfg3.l_s,
        n_predictions=cfg3.n_predictions,
    )
    train_arr, test_arr, meta = dataset.load_or_simulate_channel(args.chan_id)

    if args.preds_npz and os.path.exists(args.preds_npz):
        loaded = np.load(args.preds_npz)
        y_test = loaded["y_test"]
        y_pred_model = loaded["y_pred_model"]
        t_indices = loaded["t_indices"]
    else:
        X_train, y_train, _ = dataset.build_sliding_windows(train_arr)
        X_test, y_test, t_indices = dataset.build_sliding_windows(test_arr)
        trainer = Phase2ForecasterTrainer(input_dim=train_arr.shape[1], cfg=Phase2ModelConfig())
        trainer.fit(X_train, y_train)
        y_pred_model = trainer.predict(X_test)

    # Execute Phase 3 Dynamic Thresholding & Pruning
    thresholder = Phase3DynamicThresholder(cfg3)
    out = thresholder.run(y_test, y_pred_model, offset=cfg3.l_s)

    ablation = {
        "static_2.5sigma_raw": evaluate_intervals(out.static_raw_intervals, meta.anomaly_sequences),
        "unpruned_dynamic_ndt": evaluate_intervals(out.unpruned_dynamic_intervals, meta.anomaly_sequences),
        "full_telemanom_ndt_pruned": evaluate_intervals(out.pruned_dynamic_intervals, meta.anomaly_sequences),
    }

    report = {
        "phase": "Phase 3 — EWMA Error Smoothing, Batch Non-Parametric Dynamic Thresholding & Pruning",
        "channel": meta.chan_id,
        "ground_truth_anomalies": [list(x) for x in meta.anomaly_sequences],
        "threshold_optimization": {
            "ewma_span": cfg3.ewma_span,
            "batch_window_steps": cfg3.batch_window,
            "error_buffer_steps": cfg3.error_buffer,
            "p_min_drop_threshold": cfg3.p_min,
            "batch1_optimal_z": out.optimal_z_primary,
            "batch1_epsilon_dynamic": round(out.epsilon_primary, 4),
            "epsilon_static_2.5sigma_raw": round(out.epsilon_static_raw, 4),
            "candidate_sequences_across_batches": [
                {
                    "batch_id": c.batch_id,
                    "interval": [c.start_t, c.end_t],
                    "peak_smoothed_error": c.peak_error,
                    "relative_drop_to_next": c.pct_drop_to_next,
                    "kept_after_pruning": c.kept_after_pruning,
                }
                for c in out.candidate_sequences
            ],
        },
        "ablation_benchmark": ablation,
    }

    print(json.dumps(report, indent=2))

    if args.plot_path:
        plot_phase3_diagnostics(
            out=out,
            t_indices=t_indices,
            ground_truth_seqs=meta.anomaly_sequences,
            ablation_metrics=ablation,
            chan_id=meta.chan_id,
            output_path=args.plot_path,
        )
        print(f"\n[Phase 3 Complete] Saved 3-panel Dynamic Thresholding & Ablation chart to: {args.plot_path}")


if __name__ == "__main__":
    main()
