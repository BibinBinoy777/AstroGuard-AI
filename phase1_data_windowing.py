#!/usr/bin/env python3
"""
Phase 1: NASA SMAP & MSL Telemetry Data Exploration, Validation & Sliding-Window Engine
=======================================================================================
Project 3 — Spacecraft Anomaly Detection (NASA JPL Telemanom Architecture)
Dataset: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Reference: https://github.com/khundman/telemanom

What Phase 1 Builds & Verifies:
  1. Loads multivariate `.npy` spacecraft channels (Column 0 = continuous telemetry,
     Columns 1..d-1 = binary encoded spacecraft command states) and parses `labeled_anomalies.csv`.
  2. Performs automated data-quality & integrity checks (NaN/Inf scan, [-1, 1] normalization
     check, command sparsity analysis, and anomaly class breakdown).
  3. Builds the vectorized sliding-window tensor generator:
         X: (N, l_s=250, d)  -->  y: (N, n_predictions=10)
  4. Locks down and unit-tests the exact coordinate mapping (`raw_t = window_idx + l_s`)
     between sliding-window predictions and `labeled_anomalies.csv` ground-truth intervals.
  5. Exports a 3-panel Phase 1 EDA & Sliding-Window Alignment figure.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# 1. PHASE 1 DATA STRUCTURES & CONFIGURATION
# ==============================================================================

@dataclass
class ChannelMetadata:
    """Metadata and data-quality profile for a single spacecraft telemetry channel."""
    chan_id: str
    spacecraft: str
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    num_command_features: int
    telemetry_min_max: Tuple[float, float]
    has_nan_or_inf: bool
    command_activation_rate: float
    anomaly_sequences: List[Tuple[int, int]]
    anomaly_classes: List[str]


# ==============================================================================
# 2. DATA INGESTION, INTEGRITY VALIDATION & WINDOWING ENGINE
# ==============================================================================

class Phase1TelemetryDataset:
    """
    Phase 1 module for loading, validating, and windowing NASA SMAP & Curiosity (MSL)
    multivariate spacecraft telemetry streams.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        labels_csv: Optional[str] = None,
        l_s: int = 250,
        n_predictions: int = 10,
    ):
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.l_s = l_s
        self.n_predictions = n_predictions

    def load_or_simulate_channel(
        self, chan_id: str = "P-1"
    ) -> Tuple[np.ndarray, np.ndarray, ChannelMetadata]:
        """
        Loads real NASA `.npy` arrays if `data_dir` is present, or generates a realistic
        multivariate SMAP/MSL telemetry channel with matching structure for local verification.
        """
        if self.data_dir and os.path.exists(self.data_dir):
            train_path = os.path.join(self.data_dir, "train", f"{chan_id}.npy")
            test_path = os.path.join(self.data_dir, "test", f"{chan_id}.npy")
            train_arr = np.load(train_path).astype(np.float32)
            test_arr = np.load(test_path).astype(np.float32)

            spacecraft = "SMAP"
            anomaly_seqs: List[Tuple[int, int]] = []
            anomaly_classes: List[str] = []

            if self.labels_csv and os.path.exists(self.labels_csv):
                df = pd.read_csv(self.labels_csv)
                row = df[df["chan_id"] == chan_id]
                if not row.empty:
                    spacecraft = str(row.iloc[0].get("spacecraft", "SMAP"))
                    raw_seqs = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
                    anomaly_seqs = [(int(s[0]), int(s[1])) for s in raw_seqs]
                    if "class" in row.columns:
                        anomaly_classes = ast.literal_eval(row.iloc[0]["class"])
                    else:
                        anomaly_classes = ["point/contextual"] * len(anomaly_seqs)
        else:
            train_arr, test_arr, anomaly_seqs, anomaly_classes = self._simulate_smap_channel()
            spacecraft = "SMAP (Simulated Verification Stream)"

        # Compute data quality metrics
        combined_telemetry = np.concatenate([train_arr[:, 0], test_arr[:, 0]])
        has_nan_inf = bool(np.isnan(train_arr).any() or np.isnan(test_arr).any() or
                           np.isinf(train_arr).any() or np.isinf(test_arr).any())
        cmd_rate = (
            float(np.mean(test_arr[:, 1:] > 0))
            if test_arr.shape[1] > 1
            else 0.0
        )

        meta = ChannelMetadata(
            chan_id=chan_id,
            spacecraft=spacecraft,
            train_shape= (int(train_arr.shape[0]), int(train_arr.shape[1])),
            test_shape=(int(test_arr.shape[0]), int(test_arr.shape[1])),
            num_command_features=max(0, int(test_arr.shape[1]) - 1),
            telemetry_min_max=(round(float(np.min(combined_telemetry)), 4),
                               round(float(np.max(combined_telemetry)), 4)),
            has_nan_or_inf=has_nan_inf,
            command_activation_rate=round(cmd_rate, 4),
            anomaly_sequences=anomaly_seqs,
            anomaly_classes=anomaly_classes,
        )
        return train_arr, test_arr, meta

    @staticmethod
    def _simulate_smap_channel(
        n_train: int = 2872, n_test: int = 8505, n_commands: int = 24, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[str]]:
        """
        Generates a synthetic channel matching exact NASA SMAP `P-1` dimensions:
          - train shape: (2872, 25) -> 1 telemetry stream + 24 binary command features
          - test shape:  (8505, 25) -> with 3 labeled anomaly sequences
        """
        rng = np.random.default_rng(seed)

        def _make_array(length: int) -> np.ndarray:
            t = np.arange(length, dtype=np.float32)
            cmds = (rng.random((length, n_commands)) < 0.018).astype(np.float32)
            # Command-driven thermal/power response + orbital harmonics normalized to [-1, 1]
            impulse = cmds[:, 0] * 0.35 - cmds[:, 1] * 0.28 + cmds[:, 2] * 0.20
            kernel = np.exp(-np.arange(25, dtype=np.float32) / 6.0)
            cmd_response = np.convolve(impulse, kernel, mode="same")
            raw_sig = (
                0.50 * np.sin(2 * np.pi * t / 210.0)
                + 0.22 * np.cos(2 * np.pi * t / 52.5)
                + cmd_response
                + rng.normal(0.0, 0.03, size=length)
            )
            norm_sig = np.clip(raw_sig / (np.max(np.abs(raw_sig)) + 1e-6), -1.0, 1.0).astype(np.float32)
            return np.column_stack([norm_sig, cmds])

        train_arr = _make_array(n_train)
        test_arr = _make_array(n_test)

        # Match NASA SMAP P-1 ground-truth anomaly intervals:
        # [2149, 2349] (contextual), [4536, 4844] (contextual), [3539, 3779] (point)
        anomaly_seqs = [(2149, 2349), (3539, 3779), (4536, 4844)]
        anomaly_classes = ["contextual", "point", "contextual"]

        # Inject contextual harmonic drift in [2149, 2349]
        t1 = np.arange(2349 - 2149, dtype=np.float32)
        test_arr[2149:2349, 0] = np.clip(
            test_arr[2149:2349, 0] + 0.55 * np.sin(2 * np.pi * t1 / 14.0), -1.0, 1.0
        )
        # Inject sharp point spike burst in [3539, 3779]
        test_arr[3539:3779, 0] = np.clip(test_arr[3539:3779, 0] - 0.78, -1.0, 1.0)
        # Inject uncommanded power drop in [4536, 4844]
        t3 = np.arange(4844 - 4536, dtype=np.float32)
        test_arr[4536:4844, 0] = np.clip(
            test_arr[4536:4844, 0] + 0.62 * np.cos(2 * np.pi * t3 / 18.0), -1.0, 1.0
        )

        return train_arr, test_arr, anomaly_seqs, anomaly_classes

    def build_sliding_windows(
        self, arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized sliding-window builder using stride tricks / index broadcasting.
        Returns:
          X: (N, l_s, d)          -- multivariate lookback window [i, i + l_s)
          y: (N, n_predictions)   -- target telemetry horizon [i + l_s, i + l_s + n_predictions)
          target_start_indices: (N,) -- exact raw timestep index `t = i + l_s` for the 1st predicted step
        """
        T, d = arr.shape
        N = T - self.l_s - self.n_predictions + 1
        if N <= 0:
            raise ValueError(
                f"Array length T={T} is shorter than l_s + n_predictions ({self.l_s + self.n_predictions})."
            )

        # Build windows cleanly and deterministically
        X = np.empty((N, self.l_s, d), dtype=np.float32)
        y = np.empty((N, self.n_predictions), dtype=np.float32)
        for i in range(N):
            X[i] = arr[i : i + self.l_s]
            y[i] = arr[i + self.l_s : i + self.l_s + self.n_predictions, 0]

        target_start_indices = np.arange(self.l_s, self.l_s + N, dtype=np.int64)
        return X, y, target_start_indices

    def build_ground_truth_mask(
        self, test_length: int, anomaly_sequences: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates binary anomaly masks in both:
          1. Raw test timestep coordinates: shape (test_length,)
          2. Sliding-window evaluation coordinates: shape (N_test,), shifted by `l_s`
        """
        raw_mask = np.zeros(test_length, dtype=np.int32)
        for s, e in anomaly_sequences:
            s_clamped = max(0, min(test_length - 1, s))
            e_clamped = max(0, min(test_length - 1, e))
            raw_mask[s_clamped : e_clamped + 1] = 1

        N_test = test_length - self.l_s - self.n_predictions + 1
        window_mask = raw_mask[self.l_s : self.l_s + N_test]
        return raw_mask, window_mask


# ==============================================================================
# 3. AUTOMATED UNIT TESTS FOR PHASE 1 TENSOR SHAPES & ALIGNMENT
# ==============================================================================

def verify_phase1_invariants(
    dataset: Phase1TelemetryDataset,
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    t_indices: np.ndarray,
) -> Dict[str, bool]:
    """Run deterministic assertions to verify zero data leakage and exact index alignment."""
    l_s = dataset.l_s
    n_pred = dataset.n_predictions

    # 1. Shape invariants
    expected_n_train = train_arr.shape[0] - l_s - n_pred + 1
    expected_n_test = test_arr.shape[0] - l_s - n_pred + 1
    assert X_train.shape == (expected_n_train, l_s, train_arr.shape[1]), "X_train shape mismatch!"
    assert y_train.shape == (expected_n_train, n_pred), "y_train shape mismatch!"
    assert X_test.shape == (expected_n_test, l_s, test_arr.shape[1]), "X_test shape mismatch!"
    assert y_test.shape == (expected_n_test, n_pred), "y_test shape mismatch!"

    # 2. Coordinate alignment invariant: y_test[i, 0] must equal test_arr[t_indices[i], 0]
    sample_idx = 42
    raw_t = int(t_indices[sample_idx])
    assert raw_t == sample_idx + l_s, "Offset index formula mismatch!"
    assert np.allclose(y_test[sample_idx], test_arr[raw_t : raw_t + n_pred, 0]), (
        "Target horizon values do not match raw telemetry at t = i + l_s!"
    )

    # 3. Zero lookahead leakage invariant: last lookback step in X_test[i] is raw_t - 1
    assert np.allclose(X_test[sample_idx, -1, :], test_arr[raw_t - 1, :]), (
        "Lookback window leaked future timestep!"
    )

    return {
        "shape_invariants_passed": True,
        "coordinate_alignment_passed": True,
        "zero_lookahead_leakage_passed": True,
    }


# ==============================================================================
# 4. PHASE 1 VISUALIZATION (3-PANEL EDA & WINDOW ALIGNMENT DIAGRAM)
# ==============================================================================

def plot_phase1_eda(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    meta: ChannelMetadata,
    l_s: int,
    n_predictions: int,
    output_path: str,
) -> None:
    """Render a 3-panel Phase 1 diagnostic chart inspecting telemetry, commands, and windowing."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13.5, 9.2), dpi=160, gridspec_kw={"height_ratios": [1.15, 0.9, 1.1]}
    )
    fig.patch.set_facecolor("#0F172A")

    # --- Panel 1: Test Telemetry Stream + Ground-Truth Labeled Anomaly Windows ---
    ax1.set_facecolor("#1E293B")
    t_test = np.arange(len(test_arr))
    ax1.plot(t_test, test_arr[:, 0], color="#38BDF8", lw=0.85, alpha=0.9, label="Normalized Telemetry ($y_t$, col 0)")
    ax1.axvline(l_s, color="#FACC15", linestyle="--", lw=1.3, label=f"First Forecast Start ($t = l_s = {l_s}$)")

    for i, ((s, e), cls_name) in enumerate(zip(meta.anomaly_sequences, meta.anomaly_classes)):
        color = "#EF4444" if "contextual" in cls_name else "#F97316"
        ax1.axvspan(
            s, e, color=color, alpha=0.32,
            label=f"Anomaly {i+1}: [{s}, {e}] ({cls_name})"
        )

    ax1.set_title(
        f"Phase 1A: Spacecraft Test Telemetry Stream & Expert Labels — Channel {meta.chan_id} ({meta.spacecraft})",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax1.set_ylabel("Telemetry [-1, 1]", color="#E2E8F0", fontsize=9.5)
    ax1.set_xlim(0, len(test_arr))
    ax1.tick_params(colors="#CBD5E1")
    ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", ncol=2, facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    # --- Panel 2: Binary Encoded Command Channels Heatmap / Raster (Columns 1..d-1) ---
    ax2.set_facecolor("#1E293B")
    num_cmds_to_show = min(12, test_arr.shape[1] - 1)
    cmd_slice = test_arr[:, 1 : 1 + num_cmds_to_show].T
    ax2.imshow(
        cmd_slice, aspect="auto", cmap="GnBu", interpolation="nearest",
        extent=[0, len(test_arr), num_cmds_to_show + 0.5, 0.5]
    )
    ax2.set_title(
        f"Phase 1B: Encoded Spacecraft Command States (Showing Top {num_cmds_to_show} of {meta.num_command_features} Binary Channels)",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax2.set_ylabel("Command Channel", color="#E2E8F0", fontsize=9.5)
    ax2.set_xlabel("Mission Timestep (t)", color="#E2E8F0", fontsize=9.5)
    ax2.tick_params(colors="#CBD5E1")

    # --- Panel 3: Zoomed Sliding-Window Mechanics (l_s = 250 -> n_predictions = 10) ---
    ax3.set_facecolor("#1E293B")
    win_idx = 1880  # Zoom right before Anomaly 1
    lookback_start = win_idx
    lookback_end = win_idx + l_s
    horizon_end = lookback_end + n_predictions
    zoom_pad = 40

    t_zoom = np.arange(lookback_start - zoom_pad, horizon_end + zoom_pad)
    ax3.plot(
        t_zoom, test_arr[lookback_start - zoom_pad : horizon_end + zoom_pad, 0],
        color="#64748B", lw=1.0, linestyle=":", label="Surrounding Stream"
    )
    ax3.plot(
        np.arange(lookback_start, lookback_end),
        test_arr[lookback_start:lookback_end, 0],
        color="#38BDF8", lw=1.8, label=f"Input Lookback Window $X_i$ ($l_s = {l_s}$ steps: [{lookback_start}, {lookback_end}))"
    )
    ax3.plot(
        np.arange(lookback_end, horizon_end),
        test_arr[lookback_end:horizon_end, 0],
        color="#10B981", lw=2.4, marker="o", markersize=4,
        label=f"Target Horizon $y_i$ ($n_{{pred}} = {n_predictions}$ steps: [{lookback_end}, {horizon_end}))"
    )
    ax3.axvspan(lookback_start, lookback_end, color="#38BDF8", alpha=0.12)
    ax3.axvspan(lookback_end, horizon_end, color="#10B981", alpha=0.24)
    ax3.axvline(lookback_end, color="#10B981", linestyle="--", lw=1.4)

    ax3.set_title(
        f"Phase 1C: Sliding-Window Tensor Alignment Zoom (Window i={win_idx} → Target Starts at t = i + l_s = {lookback_end})",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax3.set_xlabel("Mission Timestep (t)", color="#E2E8F0", fontsize=9.5)
    ax3.set_ylabel("Telemetry [-1, 1]", color="#E2E8F0", fontsize=9.5)
    ax3.set_xlim(lookback_start - zoom_pad, horizon_end + zoom_pad)
    ax3.tick_params(colors="#CBD5E1")
    ax3.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: NASA SMAP/MSL Data Exploration & Windowing")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SMAP/MSL data folder")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to labeled_anomalies.csv")
    parser.add_argument("--chan_id", type=str, default="P-1", help="Channel ID (default: P-1)")
    parser.add_argument("--l_s", type=int, default=250, help="Lookback window length (default: 250)")
    parser.add_argument("--n_predictions", type=int, default=10, help="Prediction horizon (default: 10)")
    parser.add_argument("--plot_path", type=str, default="phase1_eda_window_alignment.png", help="Output PNG path")
    args = parser.parse_args()

    dataset = Phase1TelemetryDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        l_s=args.l_s,
        n_predictions=args.n_predictions,
    )

    train_arr, test_arr, meta = dataset.load_or_simulate_channel(args.chan_id)
    X_train, y_train, _ = dataset.build_sliding_windows(train_arr)
    X_test, y_test, t_indices = dataset.build_sliding_windows(test_arr)
    raw_mask, window_mask = dataset.build_ground_truth_mask(len(test_arr), meta.anomaly_sequences)

    checks = verify_phase1_invariants(
        dataset, train_arr, test_arr, X_train, y_train, X_test, y_test, t_indices
    )

    report = {
        "phase": "Phase 1 — Data Exploration, Validation & Sliding-Window Generator",
        "channel_metadata": {
            "chan_id": meta.chan_id,
            "spacecraft": meta.spacecraft,
            "raw_train_shape": list(meta.train_shape),
            "raw_test_shape": list(meta.test_shape),
            "num_command_channels": meta.num_command_features,
            "telemetry_range": list(meta.telemetry_min_max),
            "has_nan_or_inf": meta.has_nan_or_inf,
            "command_activation_rate": meta.command_activation_rate,
            "ground_truth_anomalies": [
                {"interval": list(seq), "class": cls_name, "length_steps": seq[1] - seq[0]}
                for seq, cls_name in zip(meta.anomaly_sequences, meta.anomaly_classes)
            ],
        },
        "windowed_tensors": {
            "lookback_l_s": args.l_s,
            "horizon_n_predictions": args.n_predictions,
            "X_train_shape": list(X_train.shape),
            "y_train_shape": list(y_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_test_shape": list(y_test.shape),
            "first_test_target_timestep": int(t_indices[0]),
            "last_test_target_timestep": int(t_indices[-1]),
            "anomalous_test_windows_count": int(np.sum(window_mask)),
            "anomalous_test_windows_pct": round(100.0 * float(np.mean(window_mask)), 2),
        },
        "automated_unit_tests": checks,
    }

    print(json.dumps(report, indent=2))

    if args.plot_path:
        plot_phase1_eda(train_arr, test_arr, meta, args.l_s, args.n_predictions, args.plot_path)
        print(f"\n[Phase 1 Complete] Saved 3-panel EDA & Window Alignment chart to: {args.plot_path}")


if __name__ == "__main__":
    main()
