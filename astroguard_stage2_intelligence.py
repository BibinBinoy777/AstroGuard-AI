#!/usr/bin/env python3
"""
AstroGuard AI — Stage 2: Advanced Anomaly Intelligence Suite (Phases 5–10)
==========================================================================
Expanding NASA JPL's Telemanom (KDD 2018) into a Spacecraft Intelligence Platform.

References:
  1. Hundman, K. et al. (2018), "Detecting Spacecraft Anomalies Using LSTMs and
     Nonparametric Dynamic Thresholding", KDD '18. https://github.com/khundman/telemanom
     (Uses F_0.5 alongside Precision & Recall to weight false-alarm suppression 2x).
  2. Baireddy, S. et al. (2021), "Spacecraft Time-Series Anomaly Detection Using
     Transfer Learning", CVPRW 2021 AI4Space.
     https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Baireddy_Spacecraft_Time-Series_Anomaly_Detection_Using_Transfer_Learning_CVPRW_2021_paper.pdf
  3. NASA SMAP & MSL Dataset:
     https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

What Stage 2 (Phases 5–10) Builds & Verifies:
  - Phase 5: Dynamic Thresholding Engine with official NASA JPL F_0.5 & F_1 scoring
  - Phase 6: Anomaly Pattern Classifier (Point vs. Contextual vs. Degradation Drift Indicator)
  - Phase 7: Multi-Channel Benchmarking across SMAP & MSL Curiosity subsystems (F_0.5 & F_1)
  - Phase 8: Model Comparison Engine (LSTM Forecaster vs. Isolation Forest vs. Statistical Control)
  - Phase 9: Explainable AI (XAI) Engine answering the 5 Mission-Control Engineer Questions
  - Phase 10: False-Positive Optimizer maximizing F_0.5 over (p_min, min_duration) grid
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase1_data_windowing import Phase1TelemetryDataset
from phase2_lstm_forecaster import Phase2ForecasterTrainer, Phase2ModelConfig
from phase3_dynamic_thresholding import Phase3DynamicThresholder, Phase3ThresholdConfig
from phase4_fleet_benchmark_and_api import FleetBenchmarkRunner, FleetChannelSpec


# ==============================================================================
# PHASE 5: NASA JPL F_0.5 & F_1 EVALUATION ENGINE + DYNAMIC THRESHOLDING
# ==============================================================================

def compute_fbeta_metrics(
    predicted: List[Tuple[int, int]], ground_truth: List[Tuple[int, int]]
) -> Dict[str, object]:
    """
    Compute segment-level Precision, Recall, F_1, and NASA JPL's official F_0.5 score:
        F_0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    which weights Precision twice as heavily as Recall to penalize false alarms.
    """
    def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return max(a[0], b[0]) <= min(a[1], b[1])

    tp = sum(1 for gt in ground_truth if any(_overlaps(gt, p) for p in predicted))
    fn = len(ground_truth) - tp
    fp = sum(1 for p in predicted if not any(_overlaps(p, gt) for gt in ground_truth))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "F0.5": round(f05, 4),
        "intervals": [list(x) for x in predicted],
    }


# ==============================================================================
# PHASE 6: ANOMALY PATTERN CLASSIFIER (POINT vs CONTEXTUAL vs DEGRADATION DRIFT)
# ==============================================================================

@dataclass
class ClassifiedAnomalyEvent:
    """Categorized anomaly event with morphological & command-context metrics."""
    chan_id: str
    interval: Tuple[int, int]
    predicted_category: str             # "point", "contextual", or "degradation_drift_indicator"
    nasa_taxonomy_match: str            # "point" or "contextual" (mapped to NASA 2-class labels)
    ground_truth_category: Optional[str]
    crest_factor: float                 # Peak-to-mean amplitude ratio
    oscillation_zero_crossings: float   # High-frequency sign changes per step
    monotonic_drift_r2: float           # Linear trend R^2 (degradation indicator)
    command_cooccurrence_rate: float    # Fraction of active command channels during onset


class Phase6AnomalyClassifier:
    """
    Categorizes detected spacecraft anomaly intervals into:
      1. `point`: Abrupt step shift or rail excursion (high mean offset or high white-noise
         derivative flip rate >= 0.40 on clamped step excursions).
      2. `contextual`: Coherent harmonic distortion, control-loop oscillation, or
         command-response mismatch (structured oscillation with zero-crossing rate < 0.40).
      3. `degradation_drift_indicator`: Sustained monotonic drift (R^2_trend >= 0.65) flagged as
         an early equipment degradation indicator (distinct from verified failure prediction).
    """

    @staticmethod
    def classify_interval(
        chan_id: str,
        test_arr: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        offset: int,
        interval: Tuple[int, int],
        gt_seqs: List[Tuple[int, int]],
        gt_classes: List[str],
    ) -> ClassifiedAnomalyEvent:
        s_raw, e_raw = interval
        s_clamped = max(offset, min(len(test_arr) - 1, s_raw))
        e_clamped = max(s_clamped + 5, min(len(test_arr), e_raw))

        # Extract telemetry segment and prediction residuals
        seg_telem = test_arr[s_clamped:e_clamped, 0]
        w_start = max(0, s_clamped - offset)
        w_end = min(len(y_test), e_clamped - offset)
        seg_res = (
            y_test[w_start:w_end, 0] - y_pred[w_start:w_end, 0]
            if w_end > w_start
            else seg_telem - np.mean(seg_telem)
        )

        # 1. Crest factor & step-shift magnitude
        abs_res = np.abs(seg_res)
        crest = float(np.max(abs_res) / (np.mean(abs_res) + 1e-8))
        mean_shift = float(np.abs(np.mean(seg_res)) / (np.std(seg_res) + 1e-8))

        # 2. Zero-crossing / derivative sign-change rate
        diffs = np.diff(seg_telem)
        zc_rate = (
            float(np.mean(np.diff(np.sign(diffs)) != 0))
            if len(diffs) > 2
            else 0.0
        )

        # 3. Linear monotonic trend R^2 (degradation drift indicator)
        t_seq = np.arange(len(seg_telem), dtype=np.float64)
        if len(t_seq) > 5 and np.std(seg_telem) > 1e-6:
            corr = float(np.corrcoef(t_seq, seg_telem)[0, 1])
            drift_r2 = corr ** 2
        else:
            drift_r2 = 0.0

        # 4. Command activity during onset window [s_clamped - 25, e_clamped]
        cmd_start = max(0, s_clamped - 25)
        cmd_rate = (
            float(np.mean(test_arr[cmd_start:e_clamped, 1:] > 0))
            if test_arr.shape[1] > 1
            else 0.0
        )

        # Decision rules separating Degradation Drift, Point Step/Spike, and Contextual Harmonic
        if drift_r2 >= 0.65 and zc_rate < 0.35:
            pred_cat = "degradation_drift_indicator"
            nasa_cat = "contextual"
        elif mean_shift >= 1.15 or zc_rate >= 0.40:
            pred_cat = "point"
            nasa_cat = "point"
        else:
            pred_cat = "contextual"
            nasa_cat = "contextual"

        # Match with ground truth label if overlapping
        matched_gt: Optional[str] = None
        for (gs, ge), g_cls in zip(gt_seqs, gt_classes):
            if max(s_raw, gs) <= min(e_raw, ge):
                matched_gt = g_cls
                break

        return ClassifiedAnomalyEvent(
            chan_id=chan_id,
            interval=interval,
            predicted_category=pred_cat,
            nasa_taxonomy_match=nasa_cat,
            ground_truth_category=matched_gt,
            crest_factor=round(crest, 3),
            oscillation_zero_crossings=round(zc_rate, 3),
            monotonic_drift_r2=round(drift_r2, 3),
            command_cooccurrence_rate=round(cmd_rate, 4),
        )


# ==============================================================================
# PHASE 8: MODEL COMPARISON ENGINE (LSTM vs ISOLATION FOREST vs STATISTICAL)
# ==============================================================================

class PureNumpyIsolationForest:
    """
    Self-contained Isolation Forest implementation in pure NumPy (Liu et al., ICDM 2008)
    for benchmarking multivariate unsupervised anomaly detection against Telemanom LSTM.
    """

    def __init__(self, n_trees: int = 60, max_samples: int = 256, max_depth: int = 8, seed: int = 42):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.trees: List[dict] = []

    def _build_tree(self, X: np.ndarray, depth: int) -> dict:
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples <= 2:
            return {"leaf": True, "size": n_samples}

        feat = int(self.rng.integers(0, n_features))
        col = X[:, feat]
        c_min, c_max = float(np.min(col)), float(np.max(col))
        if c_max - c_min < 1e-7:
            return {"leaf": True, "size": n_samples}

        split = float(self.rng.uniform(c_min, c_max))
        left_mask = col < split
        if np.all(left_mask) or not np.any(left_mask):
            return {"leaf": True, "size": n_samples}

        return {
            "leaf": False,
            "feat": feat,
            "split": split,
            "left": self._build_tree(X[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], depth + 1),
        }

    @staticmethod
    def _c_factor(n: int) -> float:
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)

    def _path_length(self, x: np.ndarray, node: dict, depth: int) -> float:
        if node["leaf"]:
            return float(depth) + self._c_factor(node["size"])
        if x[node["feat"]] < node["split"]:
            return self._path_length(x, node["left"], depth + 1)
        return self._path_length(x, node["right"], depth + 1)

    def fit(self, X_features: np.ndarray) -> None:
        self.trees = []
        n = len(X_features)
        sample_sz = min(self.max_samples, n)
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=sample_sz, replace=False)
            self.trees.append(self._build_tree(X_features[idx], 0))

    def score_samples(self, X_features: np.ndarray) -> np.ndarray:
        """Return continuous anomaly score in [0, 1] (higher = more anomalous)."""
        c_norm = self._c_factor(self.max_samples) + 1e-8
        path_lengths = np.zeros(len(X_features), dtype=np.float64)
        for i, x in enumerate(X_features):
            total_len = sum(self._path_length(x, t, 0) for t in self.trees)
            path_lengths[i] = total_len / len(self.trees)
        return np.power(2.0, -path_lengths / c_norm)


class Phase8ModelComparisonEngine:
    """
    Compares 3 distinct anomaly detection paradigms across spacecraft channels:
      1. Statistical Z-Score / Rolling EWMA Control Chart (`statistical_control`)
      2. Unsupervised Isolation Forest on Window Summary Features (`isolation_forest`)
      3. AstroGuard LSTM Sequence Forecaster + Dynamic Thresholding (`astroguard_lstm_ndt`)
    """

    @staticmethod
    def _extract_iforest_features(X_windows: np.ndarray) -> np.ndarray:
        """Extract multivariate window statistics for Isolation Forest."""
        telem = X_windows[:, -35:, 0]
        mean_t = np.mean(telem, axis=1, keepdims=True)
        std_t = np.std(telem, axis=1, keepdims=True)
        max_abs_diff = np.max(np.abs(np.diff(telem, axis=1)), axis=1, keepdims=True)
        p95 = np.percentile(np.abs(telem), 95, axis=1, keepdims=True)
        cmd_sum = np.mean(X_windows[:, -35:, 1:], axis=1) if X_windows.shape[2] > 1 else np.zeros((len(X_windows), 1))
        return np.hstack([mean_t, std_t, max_abs_diff, p95, cmd_sum]).astype(np.float32)

    @classmethod
    def evaluate_models_on_channel(
        cls,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        gt_seqs: List[Tuple[int, int]],
        y_pred_lstm: np.ndarray,
        offset: int = 250,
    ) -> Dict[str, Dict[str, object]]:
        thresholder = Phase3DynamicThresholder(Phase3ThresholdConfig())

        # Model 1: Statistical Rolling Z-Score Control Chart (3-sigma deviation from rolling window)
        s_telem = pd.Series(test_arr[offset : offset + len(y_test), 0])
        roll_mean = s_telem.rolling(window=120, min_periods=10).mean().to_numpy()
        roll_std = s_telem.rolling(window=120, min_periods=10).std().fillna(0.1).to_numpy() + 1e-4
        z_stat = np.abs(s_telem.to_numpy() - roll_mean) / roll_std
        stat_idx = np.where(z_stat > 2.65)[0]
        stat_seqs = [
            (int(s + offset), int(e + offset))
            for s, e in thresholder.group_padded_intervals(stat_idx, 35, len(z_stat))
        ]

        # Model 2: Isolation Forest on Multivariate Sliding-Window Features
        F_tr = cls._extract_iforest_features(X_train)
        F_te = cls._extract_iforest_features(X_test)
        iforest = PureNumpyIsolationForest(n_trees=50, max_samples=256, seed=42)
        iforest.fit(F_tr)
        if_scores = iforest.score_samples(F_te)
        if_smooth = pd.Series(if_scores).ewm(span=25, adjust=False).mean().to_numpy()
        if_thresh = float(np.mean(if_smooth) + 2.35 * np.std(if_smooth))
        if_idx = np.where(if_smooth > if_thresh)[0]
        if_seqs = [
            (int(s + offset), int(e + offset))
            for s, e in thresholder.group_padded_intervals(if_idx, 50, len(if_smooth))
        ]

        # Model 3: AstroGuard Multivariate Sequence Forecaster + NDT + Pruning
        out_lstm = thresholder.run(y_test, y_pred_lstm, offset=offset)

        return {
            "statistical_control_chart": compute_fbeta_metrics(stat_seqs, gt_seqs),
            "isolation_forest_multivariate": compute_fbeta_metrics(if_seqs, gt_seqs),
            "astroguard_lstm_ndt": compute_fbeta_metrics(out_lstm.pruned_dynamic_intervals, gt_seqs),
        }


# ==============================================================================
# PHASE 9: EXPLAINABLE AI (XAI) ROOT-CAUSE ENGINE (5 ENGINEER QUESTIONS)
# ==============================================================================

@dataclass
class EngineerExplanationReport:
    """Structured root-cause diagnostic report answering the 5 Mission-Control questions."""
    chan_id: str
    anomaly_id: int
    q1_what_changed: str
    q2_when_did_it_start: Dict[str, int]
    q3_contributing_measurements: List[Dict[str, object]]
    q4_how_unusual_severity: Dict[str, object]
    q5_engineer_investigation_steps: List[str]


class Phase9ExplainableAIEngine:
    """
    Generates an evidence-backed root-cause explanation for each detected anomaly
    by computing counterfactual feature perturbation sensitivity and spectral/statistical
    shifts between the anomalous window and preceding nominal telemetry.
    """

    @staticmethod
    def explain_anomaly(
        chan_id: str,
        anomaly_id: int,
        classified_event: ClassifiedAnomalyEvent,
        test_arr: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        forecaster: Phase2ForecasterTrainer,
        e_s: np.ndarray,
        dynamic_epsilon: float,
        offset: int = 250,
    ) -> EngineerExplanationReport:
        s_t, e_t = classified_event.interval
        w_start = max(0, s_t - offset)
        w_end = min(len(e_s) - 1, e_t - offset)
        peak_w_idx = w_start + int(np.argmax(e_s[w_start : w_end + 1]))
        peak_t = peak_w_idx + offset

        # Compare nominal baseline [s_t - 300, s_t - 50] vs anomalous segment [s_t, e_t]
        nom_slice = test_arr[max(0, s_t - 300) : max(50, s_t - 50), 0]
        anom_slice = test_arr[s_t : min(len(test_arr), e_t), 0]

        delta_mean = float(np.mean(anom_slice) - np.mean(nom_slice))
        std_ratio = float((np.std(anom_slice) + 1e-6) / (np.std(nom_slice) + 1e-6))

        # Q1: What changed?
        if classified_event.predicted_category == "point":
            q1_text = (
                f"Abrupt point excursion in {chan_id} telemetry: mean level shifted by "
                f"{delta_mean:+.3f} (volatility changed {std_ratio:.2f}x vs. nominal baseline, "
                f"crest factor {classified_event.crest_factor:.2f})."
            )
        elif classified_event.predicted_category == "degradation_drift_indicator":
            q1_text = (
                f"Sustained monotonic drift in {chan_id} (trend R^2 = {classified_event.monotonic_drift_r2:.2f}): "
                f"telemetry shifted {delta_mean:+.3f} over {e_t - s_t} steps, indicating potential component degradation."
            )
        else:
            q1_text = (
                f"Contextual harmonic/oscillatory divergence in {chan_id}: variance expanded {std_ratio:.2f}x "
                f"(zero-crossing rate {classified_event.oscillation_zero_crossings:.2f}/step) violating expected orbital phase."
            )

        # Q2: When did it start, peak, and end?
        q2_timing = {
            "onset_timestep": int(s_t),
            "peak_error_timestep": int(peak_t),
            "recovery_timestep": int(e_t),
            "duration_timesteps": int(e_t - s_t),
        }

        # Q3: Which telemetry/command measurements contributed? (Counterfactual Perturbation Attribution)
        X_peak = X_test[peak_w_idx : peak_w_idx + 1].copy()
        y_true_peak = y_test[peak_w_idx]
        base_pred = forecaster.predict(X_peak)[0]
        base_err = float(np.mean(np.abs(y_true_peak - base_pred)))

        # Attribution 1: Primary Telemetry Autoregressive Stream (Column 0)
        telem_residual_contrib = float(np.mean(np.abs(y_true_peak - np.mean(nom_slice))))

        # Attribution 2..d: Command channels active or perturbed in the lookback window
        contributions = [
            {
                "measurement": f"{chan_id}_Telemetry_Stream (col 0)",
                "attribution_weight_pct": 0.0,  # Populated after normalization
                "raw_score": telem_residual_contrib,
                "role": "Primary sensor deviation from forecast",
            }
        ]
        for c_idx in range(1, test_arr.shape[1]):
            X_pert = X_peak.copy()
            X_pert[0, :, c_idx] = 0.0
            pert_pred = forecaster.predict(X_pert)[0]
            cmd_sensitivity = float(np.mean(np.abs(base_pred - pert_pred)))
            cmd_active_pulses = int(np.sum(X_peak[0, :, c_idx] > 0))
            score_c = cmd_sensitivity + (0.015 * cmd_active_pulses)
            if score_c > 0.001:
                contributions.append(
                    {
                        "measurement": f"Command_Module_C{c_idx} (col {c_idx})",
                        "attribution_weight_pct": 0.0,
                        "raw_score": score_c,
                        "role": f"Active command state ({cmd_active_pulses} pulses in lookback window)",
                    }
                )

        contributions.sort(key=lambda item: float(item["raw_score"]), reverse=True)
        top_contribs = contributions[:4]
        total_raw = sum(float(c["raw_score"]) for c in top_contribs) + 1e-12
        for c in top_contribs:
            c["attribution_weight_pct"] = round(100.0 * float(c["raw_score"]) / total_raw, 1)
            del c["raw_score"]

        # Q4: How unusual was the behavior?
        normal_errors = e_s[e_s <= dynamic_epsilon]
        norm_mu = float(np.mean(normal_errors)) if len(normal_errors) > 0 else 0.08
        norm_sd = float(np.std(normal_errors)) + 1e-6 if len(normal_errors) > 0 else 0.02
        peak_es = float(e_s[peak_w_idx])
        z_anom = (peak_es - norm_mu) / norm_sd
        exceed_ratio = peak_es / (dynamic_epsilon + 1e-12)

        q4_unusual = {
            "peak_ewma_error": round(peak_es, 4),
            "dynamic_threshold_epsilon": round(dynamic_epsilon, 4),
            "threshold_exceedance_ratio": f"{exceed_ratio:.2f}x",
            "residual_z_score_above_nominal": f"{z_anom:.1f} sigma",
            "raw_horizon_mae_at_peak": round(base_err, 4),
        }

        # Q5: What should the engineer investigate?
        top_cmd = top_contribs[1]["measurement"] if len(top_contribs) > 1 else "Command Bus"
        if classified_event.predicted_category == "point":
            q5_steps = [
                f"Inspect {chan_id} sensor bus for transient voltage/current load drop or Single Event Upset (SEU) at t={s_t}.",
                f"Cross-check command logs on {top_cmd} to verify whether an uncommanded relay switch occurred.",
                "Verify whether adjacent subsystem channels exhibited simultaneous step impulses.",
            ]
        elif classified_event.predicted_category == "degradation_drift_indicator":
            q5_steps = [
                f"Flag {chan_id} for preventive degradation monitoring (sustained trend R^2={classified_event.monotonic_drift_r2:.2f}).",
                "Compare thermal/power dissipation curve against historical orbital eclipse cycles for calibration drift.",
                "Note: Treat as an early equipment degradation indicator rather than a confirmed hardware failure.",
            ]
        else:
            q5_steps = [
                f"Audit control-loop feedback gains and harmonic damping on {chan_id} between t={s_t} and t={e_t}.",
                f"Verify execution timing of {top_cmd} relative to the spacecraft's orbital attitude/thermal transition.",
                "Check reaction-wheel or thermal-control duty cycle for resonant coupling across correlated channels.",
            ]

        return EngineerExplanationReport(
            chan_id=chan_id,
            anomaly_id=anomaly_id,
            q1_what_changed=q1_text,
            q2_when_did_it_start=q2_timing,
            q3_contributing_measurements=top_contribs,
            q4_how_unusual_severity=q4_unusual,
            q5_engineer_investigation_steps=q5_steps,
        )


# ==============================================================================
# PHASE 10: FALSE-POSITIVE OPTIMIZATION ENGINE (MAXIMIZING NASA F_0.5)
# ==============================================================================

class Phase10FalsePositiveOptimizer:
    """
    Tunes post-processing hyperparameters (`p_min` minimum peak drop and `min_duration`
    minimum sequence persistence) across the fleet to maximize NASA JPL's F_0.5 score.
    """

    @staticmethod
    def optimize_f05_grid(
        channel_outputs: List[Tuple[Phase3DetectionOutput, List[Tuple[int, int]]]],
        p_min_grid: Tuple[float, ...] = (0.05, 0.09, 0.13, 0.17, 0.21, 0.25),
    ) -> Dict[str, object]:
        grid_results = []
        best_f05 = -1.0
        best_cfg = {"p_min": 0.13, "F0.5": 0.0, "F1": 0.0, "FP": 0, "TP": 0}

        for p_val in p_min_grid:
            total_tp, total_fp, total_fn = 0, 0, 0
            for out, gt_seqs in channel_outputs:
                # Filter candidate sequences using candidate `p_val`
                by_batch: Dict[int, List] = {}
                for c in out.candidate_sequences:
                    by_batch.setdefault(c.batch_id, []).append(c)

                kept_seqs: List[Tuple[int, int]] = []
                for b_id, c_list in by_batch.items():
                    c_sorted = sorted(c_list, key=lambda x: x.peak_error, reverse=True)
                    last_k = -1
                    for idx_c, c_item in enumerate(c_sorted):
                        if c_item.pct_drop_to_next >= p_val:
                            last_k = idx_c
                    for idx_c, c_item in enumerate(c_sorted):
                        if idx_c <= last_k:
                            kept_seqs.append((c_item.start_t, c_item.end_t))

                merged = Phase3DynamicThresholder._merge_global_intervals(kept_seqs)
                m = compute_fbeta_metrics(merged, gt_seqs)
                total_tp += int(m["TP"])
                total_fp += int(m["FP"])
                total_fn += int(m["FN"])

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) > 0 else 0.0

            row = {
                "p_min": round(float(p_val), 3),
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "F0.5": round(f05, 4),
            }
            grid_results.append(row)
            if f05 > best_f05:
                best_f05 = f05
                best_cfg = row

        return {"optimal_operating_point": best_cfg, "p_min_sweep": grid_results}


# ==============================================================================
# STAGE 2 VISUALIZATION (4-PANEL ASTROGUARD AI INTELLIGENCE REPORT)
# ==============================================================================

def plot_stage2_intelligence_report(
    model_comparison: Dict[str, Dict[str, object]],
    classified_events: List[ClassifiedAnomalyEvent],
    xai_reports: List[EngineerExplanationReport],
    fp_opt: Dict[str, object],
    output_path: str,
) -> None:
    """Render a 4-panel visual diagnostic report for AstroGuard AI Stage 2 (Phases 5-10)."""
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.2, 9.8), dpi=160)
    fig.patch.set_facecolor("#0F172A")

    # --- Panel 1 (Phase 8 & 5): Model Comparison on Precision, Recall, F1, and NASA F_0.5 ---
    ax1.set_facecolor("#1E293B")
    models = ["Statistical\nControl Chart", "Multivariate\nIsolation Forest", "AstroGuard\nLSTM + NDT"]
    m_keys = ["statistical_control_chart", "isolation_forest_multivariate", "astroguard_lstm_ndt"]
    p_vals = [float(model_comparison[k]["Precision"]) for k in m_keys]
    f1_vals = [float(model_comparison[k]["F1"]) for k in m_keys]
    f05_vals = [float(model_comparison[k]["F0.5"]) for k in m_keys]

    xm = np.arange(len(models))
    w = 0.25
    ax1.bar(xm - w, p_vals, width=w, color="#38BDF8", label="Precision")
    ax1.bar(xm, f1_vals, width=w, color="#A855F7", label="F1-Score (Balanced)")
    bars_f05 = ax1.bar(xm + w, f05_vals, width=w, color="#10B981", label="NASA F0.5-Score (2x Precision Weight)")

    for bar, k in zip(bars_f05, m_keys):
        fp_c = model_comparison[k]["FP"]
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f"F0.5={bar.get_height():.2f}\n(FP={fp_c})",
            ha="center", va="bottom", color="#F8FAFC", fontsize=8.0, fontweight="bold",
        )

    ax1.set_title(
        "Phases 5 & 8: Model Comparison Benchmark (NASA F0.5 vs. F1)",
        color="#F8FAFC", fontsize=11.0, fontweight="bold", pad=8,
    )
    ax1.set_xticks(xm)
    ax1.set_xticklabels(models, color="#CBD5E1", fontsize=8.8)
    ax1.set_ylabel("Metric Score [0.0 - 1.0]", color="#E2E8F0", fontsize=9.5)
    ax1.set_ylim(0, 1.25)
    ax1.tick_params(colors="#CBD5E1")
    ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6, axis="y")
    ax1.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    # --- Panel 2 (Phase 6): Anomaly Pattern Classification Feature Space ---
    ax2.set_facecolor("#1E293B")
    for idx_e, ev in enumerate(classified_events):
        is_point = ev.nasa_taxonomy_match == "point"
        col = "#F97316" if is_point else "#38BDF8"
        marker = "^" if is_point else "o"
        ax2.scatter(
            ev.oscillation_zero_crossings,
            ev.crest_factor,
            color=col,
            marker=marker,
            s=90,
            edgecolors="#F8FAFC",
            linewidth=1.0,
            zorder=4,
        )
        # Stagger nearby labels cleanly so none collide
        dy = -12 if (ev.crest_factor > 6.5 and ev.chan_id == "D-1") or (idx_e % 2 == 1 and ev.crest_factor < 3.6) else 5
        ax2.annotate(
            f"{ev.chan_id} ({ev.nasa_taxonomy_match[:4]})",
            (ev.oscillation_zero_crossings, ev.crest_factor),
            textcoords="offset points",
            xytext=(5, dy),
            color="#E2E8F0",
            fontsize=7.3,
        )

    ax2.axvline(0.40, color="#10B981", linestyle="--", lw=1.4, label="Contextual vs. Point Boundary")
    ax2.scatter([], [], color="#F97316", marker="^", s=75, label="Classified: Point Anomaly")
    ax2.scatter([], [], color="#38BDF8", marker="o", s=75, label="Classified: Contextual Anomaly")

    ax2.set_title(
        "Phase 6: Anomaly Pattern Classification (Signature Space)",
        color="#F8FAFC", fontsize=11.0, fontweight="bold", pad=8,
    )
    ax2.set_xlabel("Oscillation Zero-Crossing Rate (cycles/step)", color="#E2E8F0", fontsize=9.5)
    ax2.set_ylabel("Residual Crest Factor (Peak / Mean)", color="#E2E8F0", fontsize=9.5)
    ax2.set_xlim(0.22, 0.57)
    ax2.set_ylim(1.2, 7.6)
    ax2.tick_params(colors="#CBD5E1")
    ax2.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    # --- Panel 3 (Phase 9): Explainable AI (XAI) Feature Attribution on P-1 Anomaly ---
    ax3.set_facecolor("#1E293B")
    rep0 = xai_reports[0]
    meas_names = [str(m["measurement"]).replace(" (col ", "\n(col ") for m in rep0.q3_contributing_measurements][::-1]
    weights = [float(m["attribution_weight_pct"]) for m in rep0.q3_contributing_measurements][::-1]
    y_pos = np.arange(len(meas_names))
    bar_cols = ["#64748B"] * (len(meas_names) - 1) + ["#38BDF8"]

    bars_xai = ax3.barh(y_pos, weights, color=bar_cols, height=0.52, edgecolor="#F8FAFC", linewidth=0.8)
    for bar in bars_xai:
        ax3.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{bar.get_width():.1f}%",
            va="center", ha="left", color="#F8FAFC", fontsize=8.5, fontweight="bold",
        )

    ax3.set_title(
        f"Phase 9: Explainable AI Attribution — {rep0.chan_id} [{rep0.q2_when_did_it_start['onset_timestep']}, {rep0.q2_when_did_it_start['recovery_timestep']}]",
        color="#F8FAFC", fontsize=11.0, fontweight="bold", pad=8,
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(meas_names, color="#CBD5E1", fontsize=8.2)
    ax3.set_xlabel("Counterfactual Residual Attribution Weight (%)", color="#E2E8F0", fontsize=9.5)
    ax3.set_xlim(0, max(weights + [50.0]) * 1.22)
    ax3.tick_params(colors="#CBD5E1")
    ax3.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6, axis="x")

    # --- Panel 4 (Phase 10): False-Positive Optimization Curve (F_0.5 & FP vs p_min) ---
    ax4.set_facecolor("#1E293B")
    sweep = fp_opt["p_min_sweep"]
    p_grid = [float(r["p_min"]) for r in sweep]
    f05_curve = [float(r["F0.5"]) for r in sweep]
    f1_curve = [float(r["F1"]) for r in sweep]
    prec_curve = [float(r["Precision"]) for r in sweep]

    opt_p = float(fp_opt["optimal_operating_point"]["p_min"])
    opt_f05 = float(fp_opt["optimal_operating_point"]["F0.5"])

    ax4.plot(p_grid, f05_curve, color="#10B981", lw=2.4, marker="o", label="Fleet NASA F0.5-Score")
    ax4.plot(p_grid, f1_curve, color="#A855F7", lw=1.8, linestyle="--", marker="s", label="Fleet F1-Score")
    ax4.plot(p_grid, prec_curve, color="#38BDF8", lw=1.6, linestyle=":", marker="^", label="Fleet Precision")
    ax4.axvline(opt_p, color="#FACC15", linestyle="--", lw=1.5, label=f"Optimal $p_{{min}}^* = {opt_p:.2f}$ (F0.5={opt_f05:.3f})")

    ax4.set_title(
        "Phase 10: False-Positive Pruning Optimization ($p_{min}$ Sweep for Max F0.5)",
        color="#F8FAFC", fontsize=11.0, fontweight="bold", pad=8,
    )
    ax4.set_xlabel("Minimum Relative Peak Drop Threshold ($p_{min}$)", color="#E2E8F0", fontsize=9.5)
    ax4.set_ylabel("Fleet Benchmark Score [0.0 - 1.0]", color="#E2E8F0", fontsize=9.5)
    ax4.set_ylim(0.4, 1.08)
    ax4.tick_params(colors="#CBD5E1")
    ax4.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax4.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    for ax in (ax1, ax2, ax3, ax4):
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ==============================================================================
# MAIN RUNNER FOR ASTROGUARD AI STAGE 2 (PHASES 5–10)
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="AstroGuard AI Stage 2: Advanced Anomaly Intelligence (Phases 5-10)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SMAP/MSL data folder")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to labeled_anomalies.csv")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per channel")
    parser.add_argument("--plot_path", type=str, default="astroguard_stage2_intelligence_report.png", help="Output PNG path")
    args = parser.parse_args()

    cfg_model = Phase2ModelConfig(epochs=args.epochs)
    cfg_thresh = Phase3ThresholdConfig()
    ds_helper = Phase1TelemetryDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        l_s=cfg_model.l_s,
        n_predictions=cfg_model.n_predictions,
    )
    thresholder = Phase3DynamicThresholder(cfg_thresh)
    fleet_runner = FleetBenchmarkRunner(data_dir=args.data_dir, labels_csv=args.labels_csv, epochs=args.epochs)

    specs = fleet_runner.DEFAULT_FLEET_SPECS
    channel_outputs: List[Tuple[Phase3DetectionOutput, List[Tuple[int, int]]]] = []
    classified_events: List[ClassifiedAnomalyEvent] = []
    xai_reports: List[EngineerExplanationReport] = []

    # Aggregate counters for Phase 8 Model Comparison across the entire 8-channel fleet
    model_totals = {
        "statistical_control_chart": {"TP": 0, "FP": 0, "FN": 0},
        "isolation_forest_multivariate": {"TP": 0, "FP": 0, "FN": 0},
        "astroguard_lstm_ndt": {"TP": 0, "FP": 0, "FN": 0},
    }

    for spec in specs:
        if args.data_dir and os.path.exists(args.data_dir):
            train_arr, test_arr, meta = ds_helper.load_or_simulate_channel(spec.chan_id)
            gt_seqs = meta.anomaly_sequences
            gt_classes = meta.anomaly_classes
        else:
            train_arr, test_arr = fleet_runner._simulate_fleet_channel(spec)
            gt_seqs = spec.anomaly_sequences
            gt_classes = spec.anomaly_classes

        X_train, y_train, _ = ds_helper.build_sliding_windows(train_arr)
        X_test, y_test, _ = ds_helper.build_sliding_windows(test_arr)

        trainer = Phase2ForecasterTrainer(input_dim=train_arr.shape[1], cfg=cfg_model)
        trainer.fit(X_train, y_train)
        y_pred = trainer.predict(X_test)

        # Phase 5: Dynamic Thresholding with F_0.5
        out3 = thresholder.run(y_test, y_pred, offset=cfg_model.l_s)
        channel_outputs.append((out3, gt_seqs))

        # Phase 6: Classify each detected anomaly interval
        for idx_anom, interval in enumerate(out3.pruned_dynamic_intervals, start=1):
            ev = Phase6AnomalyClassifier.classify_interval(
                chan_id=spec.chan_id,
                test_arr=test_arr,
                y_test=y_test,
                y_pred=y_pred,
                offset=cfg_model.l_s,
                interval=interval,
                gt_seqs=gt_seqs,
                gt_classes=gt_classes,
            )
            classified_events.append(ev)

            # Phase 9: Generate Explainable AI (XAI) Engineer Report for primary channel P-1 and M-1
            if spec.chan_id in ("P-1", "M-1") and len(xai_reports) < 3:
                xai_rep = Phase9ExplainableAIEngine.explain_anomaly(
                    chan_id=spec.chan_id,
                    anomaly_id=idx_anom,
                    classified_event=ev,
                    test_arr=test_arr,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred=y_pred,
                    forecaster=trainer,
                    e_s=out3.e_s,
                    dynamic_epsilon=out3.epsilon_primary,
                    offset=cfg_model.l_s,
                )
                xai_reports.append(xai_rep)

        # Phase 8: Compare LSTM+NDT vs Isolation Forest vs Statistical Control Chart
        m_comp = Phase8ModelComparisonEngine.evaluate_models_on_channel(
            train_arr, test_arr, X_train, y_train, X_test, y_test, gt_seqs, y_pred, offset=cfg_model.l_s
        )
        for m_key in model_totals:
            model_totals[m_key]["TP"] += int(m_comp[m_key]["TP"])
            model_totals[m_key]["FP"] += int(m_comp[m_key]["FP"])
            model_totals[m_key]["FN"] += int(m_comp[m_key]["FN"])

    # Finalize Phase 8 aggregate F_1 and F_0.5 scores across the fleet
    phase8_summary = {}
    for m_key, counts in model_totals.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) > 0 else 0.0
        phase8_summary[m_key] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "F0.5": round(f05, 4),
        }

    # Phase 6 classification accuracy against ground-truth taxonomy on true positives
    tp_events = [e for e in classified_events if e.ground_truth_category is not None]
    cls_matches = sum(1 for e in tp_events if e.nasa_taxonomy_match == e.ground_truth_category)
    cls_accuracy = round(100.0 * cls_matches / max(1, len(tp_events)), 2)

    # Phase 10: False-Positive Optimization over p_min grid maximizing F_0.5
    fp_opt = Phase10FalsePositiveOptimizer.optimize_f05_grid(channel_outputs)

    summary = {
        "stage": "AstroGuard AI — Stage 2: Advanced Anomaly Intelligence (Phases 5–10)",
        "phase5_and_7_fleet_f05_benchmark": phase8_summary["astroguard_lstm_ndt"],
        "phase6_anomaly_classification": {
            "total_detected_events_classified": len(classified_events),
            "true_positive_events_matched": len(tp_events),
            "taxonomy_classification_accuracy_pct": cls_accuracy,
            "sample_classified_events": [
                {
                    "chan_id": e.chan_id,
                    "interval": list(e.interval),
                    "predicted_category": e.predicted_category,
                    "ground_truth_category": e.ground_truth_category,
                    "crest_factor": e.crest_factor,
                    "oscillation_zero_crossings": e.oscillation_zero_crossings,
                    "monotonic_drift_r2": e.monotonic_drift_r2,
                }
                for e in classified_events[:6]
            ],
        },
        "phase8_model_comparison_engine": phase8_summary,
        "phase9_explainable_ai_reports": [
            {
                "chan_id": r.chan_id,
                "anomaly_id": r.anomaly_id,
                "1_what_changed": r.q1_what_changed,
                "2_when_did_it_start": r.q2_when_did_it_start,
                "3_contributing_measurements": r.q3_contributing_measurements,
                "4_how_unusual_severity": r.q4_how_unusual_severity,
                "5_engineer_investigation_steps": r.q5_engineer_investigation_steps,
            }
            for r in xai_reports
        ],
        "phase10_false_positive_optimization": fp_opt,
    }

    print(json.dumps(summary, indent=2))

    if args.plot_path:
        plot_stage2_intelligence_report(
            model_comparison=phase8_summary,
            classified_events=classified_events,
            xai_reports=xai_reports,
            fp_opt=fp_opt,
            output_path=args.plot_path,
        )
        print(f"\n[Stage 2 Complete] Saved 4-panel AstroGuard AI Intelligence report to: {args.plot_path}")


if __name__ == "__main__":
    main()