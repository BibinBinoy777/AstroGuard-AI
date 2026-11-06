#!/usr/bin/env python3
"""
Phase 2: Stacked Multivariate LSTM Forecaster, Baseline Comparison & Residual Diagnostics
=========================================================================================
Project 3 — Spacecraft Anomaly Detection (NASA JPL Telemanom Architecture)
Dataset: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Reference: https://github.com/khundman/telemanom

What Phase 2 Builds & Verifies:
  1. Connects directly to `phase1_data_windowing.py` to ingest validated 3D sliding windows:
         X_train: (2613, 250, 25) -> y_train: (2613, 10)
         X_test:  (8246, 250, 25) -> y_test:  (8246, 10)
  2. Implements a Naive EMA/Persistence Baseline (`NaiveTelemetryBaseline`) to establish a
     lower-bound benchmark for multi-step forecasting error.
  3. Implements the 2-Layer Stacked Multivariate LSTM (`TelemanomLSTM` in PyTorch with
     80 -> 80 hidden units, Dropout 0.3, Adam lr=1e-3, gradient clipping, and early stopping
     on a chronological 20% validation split), plus a pure-NumPy Mini-Batch Gradient Descent
     Command-Aware Autoregressive network fallback so epoch-by-epoch convergence can be
     verified in CPU/sandbox environments without PyTorch.
  4. Quantifies:
       - Training vs. Validation MSE convergence & best-epoch checkpointing
       - Forecast horizon error profile across steps k = 1..10
       - Residual separation ratio (Anomalous Window MSE vs. Normal Window MSE)
  5. Exports a 3-panel Phase 2 Forecasting & Convergence Diagnostic chart.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase1_data_windowing import Phase1TelemetryDataset

# Optional PyTorch import (used automatically on Kaggle / Colab / local GPU environments)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==============================================================================
# 1. PHASE 2 MODEL HYPERPARAMETERS
# ==============================================================================

@dataclass
class Phase2ModelConfig:
    """Hyperparameters matching NASA JPL's Telemanom (KDD 2018) LSTM specification."""
    l_s: int = 250
    n_predictions: int = 10
    hidden_units: Tuple[int, int] = (80, 80)
    dropout: float = 0.30
    batch_size: int = 64
    epochs: int = 35
    patience: int = 10
    learning_rate: float = 1e-3
    validation_split: float = 0.20
    grad_clip_norm: float = 1.0
    seed: int = 42


# ==============================================================================
# 2. BASELINE FORECASTER (NAIVE PERSISTENCE / EMA DRIFT BASELINE)
# ==============================================================================

class NaiveTelemetryBaseline:
    """
    Baseline forecaster that predicts the next `n_predictions` steps using an
    exponentially weighted moving average of the last 15 telemetry readings
    (ignoring spacecraft command channels). Used to prove that the multivariate
    model genuinely learns orbital + command dynamics rather than trivial persistence.
    """

    def __init__(self, n_predictions: int = 10, lookback_span: int = 15):
        self.n_predictions = n_predictions
        self.lookback_span = lookback_span

    def predict(self, X: np.ndarray) -> np.ndarray:
        recent_telemetry = X[:, -self.lookback_span :, 0]  # shape (N, span)
        weights = np.exp(np.linspace(-2.0, 0.0, self.lookback_span, dtype=np.float32))
        weights /= np.sum(weights)
        level = recent_telemetry @ weights  # shape (N,)
        return np.repeat(level[:, None], self.n_predictions, axis=1)


# ==============================================================================
# 3. STACKED MULTIVARIATE LSTM (PYTORCH) & COMMAND-AWARE SGD FALLBACK
# ==============================================================================

if TORCH_AVAILABLE:
    class TelemanomLSTM(nn.Module):
        """
        Production 2-Layer Stacked LSTM architecture from Hundman et al. (KDD 2018):
          Input (B, 250, d) -> LSTM(d, 80, batch_first=True) -> Dropout(0.3)
                            -> LSTM(80, 80, batch_first=True) -> Dropout(0.3)
                            -> Linear(80, 10)
        """

        def __init__(self, input_dim: int, cfg: Phase2ModelConfig):
            super().__init__()
            u1, u2 = cfg.hidden_units
            self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=u1, batch_first=True)
            self.drop1 = nn.Dropout(cfg.dropout)
            self.lstm2 = nn.LSTM(input_size=u1, hidden_size=u2, batch_first=True)
            self.drop2 = nn.Dropout(cfg.dropout)
            self.fc = nn.Linear(u2, cfg.n_predictions)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            final_state = self.drop2(out[:, -1, :])
            return self.fc(final_state)


class Phase2ForecasterTrainer:
    """
    Trains and evaluates the Multivariate Sequence Forecaster with chronological
    validation splitting, epoch-by-epoch loss tracking, early stopping, and
    checkpoint serialization.
    """

    def __init__(self, input_dim: int, cfg: Phase2ModelConfig):
        self.input_dim = input_dim
        self.cfg = cfg
        self.backend = (
            f"PyTorch TelemanomLSTM ({'CUDA' if torch.cuda.is_available() else 'CPU'})"
            if TORCH_AVAILABLE
            else "NumPy Command-Aware Temporal MLP (Mini-Batch Adam Optimizer)"
        )
        if TORCH_AVAILABLE:
            torch.manual_seed(cfg.seed)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TelemanomLSTM(input_dim, cfg).to(self.device)
        else:
            self.device = None
            self.model = None
            # Pure-NumPy 2-Layer Temporal MLP with Adam optimizer for sandbox verification
            self.lag_steps = 45
            self.W1: Optional[np.ndarray] = None
            self.b1: Optional[np.ndarray] = None
            self.W2: Optional[np.ndarray] = None
            self.b2: Optional[np.ndarray] = None

    def _extract_temporal_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract recent full-resolution telemetry lags + multi-scale command summaries
        for the pure-NumPy fallback network when PyTorch is unavailable.
        """
        N = len(X)
        # Recent 45 telemetry steps: (N, 45)
        recent_telem = X[:, -self.lag_steps :, 0]
        # Longer-range orbital phase anchors at lags 60, 90, 120, 150, 180, 210, 240: (N, 7)
        anchor_indices = [-60, -90, -120, -150, -180, -210, -240]
        orbital_anchors = X[:, anchor_indices, 0]
        # Recent command pulses over last 25 steps: (N, 25 * (d-1)) summarized into 3 windows
        if X.shape[2] > 1:
            cmd_w1 = np.mean(X[:, -8:, 1:], axis=1)
            cmd_w2 = np.mean(X[:, -20:-8, 1:], axis=1)
            cmd_w3 = np.mean(X[:, -45:-20, 1:], axis=1)
            return np.hstack([recent_telem, orbital_anchors, cmd_w1, cmd_w2, cmd_w3]).astype(np.float32)
        return np.hstack([recent_telem, orbital_anchors]).astype(np.float32)

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, checkpoint_path: Optional[str] = None
    ) -> Dict[str, object]:
        """
        Train with a strict chronological 80/20 split (never shuffling across train/val boundary)
        and track epoch-by-epoch MSE loss + early stopping.
        """
        split_idx = int(len(X_train) * (1.0 - self.cfg.validation_split))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}
        best_val_mse = float("inf")
        best_epoch = 1

        if TORCH_AVAILABLE:
            train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
            criterion = nn.MSELoss()
            best_state = None
            patience_left = self.cfg.patience

            for epoch in range(1, self.cfg.epochs + 1):
                self.model.train()
                batch_losses = []
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    preds = self.model(bx)
                    loss = criterion(preds, by)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                    optimizer.step()
                    batch_losses.append(loss.item())

                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        val_losses.append(criterion(self.model(bx), by).item())

                tr_mse = float(np.mean(batch_losses))
                va_mse = float(np.mean(val_losses))
                history["train_mse"].append(round(tr_mse, 6))
                history["val_mse"].append(round(va_mse, 6))

                if va_mse < best_val_mse:
                    best_val_mse = va_mse
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.cfg.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if best_state is not None:
                self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                if checkpoint_path:
                    torch.save(best_state, checkpoint_path)
        else:
            # Pure-NumPy 2-Layer Neural Network trained with Mini-Batch Adam + Dropout
            rng = np.random.default_rng(self.cfg.seed)
            F_tr = self._extract_temporal_features(X_tr)
            F_val = self._extract_temporal_features(X_val)
            in_features = F_tr.shape[1]
            hidden_dim = self.cfg.hidden_units[0]
            out_dim = self.cfg.n_predictions

            self.W1 = rng.normal(0.0, np.sqrt(2.0 / in_features), size=(in_features, hidden_dim)).astype(np.float32)
            self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
            self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim)).astype(np.float32)
            self.b2 = np.zeros((1, out_dim), dtype=np.float32)

            # Adam moments
            mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
            mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
            mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
            mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
            step_t = 0
            best_weights = None
            patience_left = self.cfg.patience

            for epoch in range(1, self.cfg.epochs + 1):
                perm = rng.permutation(len(F_tr))
                batch_losses = []
                for start in range(0, len(F_tr), self.cfg.batch_size):
                    idx = perm[start : start + self.cfg.batch_size]
                    xb = F_tr[idx]
                    yb = y_tr[idx]
                    B = float(len(xb))
                    step_t += 1

                    # Forward pass (Tanh activation matching LSTM state nonlinearity + Dropout)
                    z1 = xb @ self.W1 + self.b1
                    h1 = np.tanh(z1)
                    if self.cfg.dropout > 0:
                        mask = (rng.random(h1.shape) >= self.cfg.dropout).astype(np.float32) / (1.0 - self.cfg.dropout)
                        h1_drop = h1 * mask
                    else:
                        h1_drop = h1
                    preds = h1_drop @ self.W2 + self.b2

                    diff = preds - yb
                    loss = float(np.mean(diff ** 2))
                    batch_losses.append(loss)

                    # Backward pass
                    dpreds = (2.0 / (B * out_dim)) * diff
                    dW2 = h1_drop.T @ dpreds + 1e-4 * self.W2
                    db2 = np.sum(dpreds, axis=0, keepdims=True)
                    dh1 = dpreds @ self.W2.T
                    if self.cfg.dropout > 0:
                        dh1 = dh1 * mask
                    dz1 = dh1 * (1.0 - h1 ** 2)
                    dW1 = xb.T @ dz1 + 1e-4 * self.W1
                    db1 = np.sum(dz1, axis=0, keepdims=True)

                    # Adam parameter updates
                    lr_t = self.cfg.learning_rate * (np.sqrt(1.0 - beta2 ** step_t) / (1.0 - beta1 ** step_t))
                    mW1 = beta1 * mW1 + (1.0 - beta1) * dW1
                    vW1 = beta2 * vW1 + (1.0 - beta2) * (dW1 ** 2)
                    self.W1 -= lr_t * mW1 / (np.sqrt(vW1) + eps_adam)

                    mb1 = beta1 * mb1 + (1.0 - beta1) * db1
                    vb1 = beta2 * vb1 + (1.0 - beta2) * (db1 ** 2)
                    self.b1 -= lr_t * mb1 / (np.sqrt(vb1) + eps_adam)

                    mW2 = beta1 * mW2 + (1.0 - beta1) * dW2
                    vW2 = beta2 * vW2 + (1.0 - beta2) * (dW2 ** 2)
                    self.W2 -= lr_t * mW2 / (np.sqrt(vW2) + eps_adam)

                    mb2 = beta1 * mb2 + (1.0 - beta1) * db2
                    vb2 = beta2 * vb2 + (1.0 - beta2) * (db2 ** 2)
                    self.b2 -= lr_t * mb2 / (np.sqrt(vb2) + eps_adam)

                # Validation evaluation (deterministic without dropout)
                val_preds = np.tanh(F_val @ self.W1 + self.b1) @ self.W2 + self.b2
                tr_mse = float(np.mean(batch_losses))
                va_mse = float(np.mean((val_preds - y_val) ** 2))
                history["train_mse"].append(round(tr_mse, 6))
                history["val_mse"].append(round(va_mse, 6))

                if va_mse < best_val_mse:
                    best_val_mse = va_mse
                    best_epoch = epoch
                    best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                    patience_left = self.cfg.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if best_weights is not None:
                self.W1, self.b1, self.W2, self.b2 = best_weights
                if checkpoint_path:
                    np.savez(checkpoint_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

        return {
            "backend": self.backend,
            "epochs_trained": len(history["train_mse"]),
            "best_epoch": best_epoch,
            "best_val_mse": round(best_val_mse, 6),
            "final_train_mse": history["train_mse"][-1],
            "history": history,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict `n_predictions` future telemetry steps for each window in `X`."""
        if TORCH_AVAILABLE:
            self.model.eval()
            ds = TensorDataset(torch.from_numpy(X))
            loader = DataLoader(ds, batch_size=self.cfg.batch_size * 2, shuffle=False)
            preds = []
            with torch.no_grad():
                for (bx,) in loader:
                    preds.append(self.model(bx.to(self.device)).cpu().numpy())
            return np.vstack(preds)

        F = self._extract_temporal_features(X)
        return (np.tanh(F @ self.W1 + self.b1) @ self.W2 + self.b2).astype(np.float32)


# ==============================================================================
# 4. PHASE 2 DIAGNOSTIC VISUALIZATION (3-PANEL TRAINING & FORECAST REPORT)
# ==============================================================================

def plot_phase2_diagnostics(
    train_info: Dict[str, object],
    y_test: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    t_indices: np.ndarray,
    window_anomaly_mask: np.ndarray,
    anomaly_sequences: List[Tuple[int, int]],
    chan_id: str,
    output_path: str,
) -> None:
    """Render a 3-panel Phase 2 diagnostic chart for training convergence, forecast tracking, and horizon error."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13.5, 9.5), dpi=160, gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]}
    )
    fig.patch.set_facecolor("#0F172A")

    history = train_info["history"]
    epochs = np.arange(1, len(history["train_mse"]) + 1)
    best_ep = int(train_info["best_epoch"])
    best_val = float(train_info["best_val_mse"])

    # --- Panel 1: Training vs Validation Loss Curve & Best Checkpoint ---
    ax1.set_facecolor("#1E293B")
    ax1.plot(epochs, history["train_mse"], color="#38BDF8", lw=2.0, marker="o", markersize=4, label="Train MSE Loss")
    ax1.plot(epochs, history["val_mse"], color="#10B981", lw=2.0, marker="s", markersize=4, label="Validation MSE Loss (20% Split)")
    ax1.axvline(best_ep, color="#FACC15", linestyle="--", lw=1.4, label=f"Best Checkpoint (Epoch {best_ep}, Val MSE = {best_val:.4f})")

    ax1.set_title(
        f"Phase 2A: Multivariate Sequence Forecaster Convergence — Channel {chan_id}",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax1.set_xlabel("Training Epoch", color="#E2E8F0", fontsize=9.5)
    ax1.set_ylabel("Mean Squared Error (MSE)", color="#E2E8F0", fontsize=9.5)
    ax1.tick_params(colors="#CBD5E1")
    ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    # --- Panel 2: Zoomed Forecast Tracking Across Normal -> Anomaly Transition ---
    ax2.set_facecolor("#1E293B")
    # Focus around Anomaly 1 [2149, 2349] to show normal tracking vs anomalous divergence
    s_anom, e_anom = anomaly_sequences[0] if anomaly_sequences else (2149, 2349)
    zoom_start_t = max(int(t_indices[0]), s_anom - 320)
    zoom_end_t = min(int(t_indices[-1]), e_anom + 220)
    mask_zoom = (t_indices >= zoom_start_t) & (t_indices <= zoom_end_t)

    t_z = t_indices[mask_zoom]
    ax2.plot(t_z, y_test[mask_zoom, 0], color="#94A3B8", lw=1.2, label="Actual Telemetry ($y_t$)")
    ax2.plot(t_z, y_pred_model[mask_zoom, 0], color="#38BDF8", lw=1.5, label="Multivariate Forecaster ($\hat{y}_{t+1}$)")
    ax2.plot(t_z, y_pred_baseline[mask_zoom, 0], color="#F43F5E", lw=1.1, linestyle=":", alpha=0.8, label="Naive EMA Baseline")
    ax2.axvspan(s_anom, e_anom, color="#EF4444", alpha=0.28, label=f"Ground-Truth Anomaly Window [{s_anom}, {e_anom}]")

    ax2.set_title(
        f"Phase 2B: Forecast Tracking on Normal Orbital Telemetry vs. Divergence Inside Anomaly Window [{s_anom}, {e_anom}]",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax2.set_xlabel("Mission Timestep (t)", color="#E2E8F0", fontsize=9.5)
    ax2.set_ylabel("Normalized Telemetry", color="#E2E8F0", fontsize=9.5)
    ax2.set_xlim(zoom_start_t, zoom_end_t)
    ax2.tick_params(colors="#CBD5E1")
    ax2.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(loc="upper right", ncol=2, facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    # --- Panel 3: 10-Step Forecast Horizon Error Profile (Normal Regime vs Anomalous Regime) ---
    ax3.set_facecolor("#1E293B")
    horizons = np.arange(1, y_test.shape[1] + 1)
    normal_idx = window_anomaly_mask == 0
    anom_idx = window_anomaly_mask == 1

    mae_model_norm = np.mean(np.abs(y_test[normal_idx] - y_pred_model[normal_idx]), axis=0)
    mae_base_norm = np.mean(np.abs(y_test[normal_idx] - y_pred_baseline[normal_idx]), axis=0)
    mae_model_anom = np.mean(np.abs(y_test[anom_idx] - y_pred_model[anom_idx]), axis=0)

    ax3.plot(horizons, mae_model_norm, color="#10B981", lw=2.2, marker="o", label="Multivariate Forecaster MAE (Normal Regime)")
    ax3.plot(horizons, mae_base_norm, color="#F59E0B", lw=1.8, linestyle="--", marker="^", label="Naive EMA Baseline MAE (Normal Regime)")
    ax3.plot(horizons, mae_model_anom, color="#EF4444", lw=2.2, marker="s", label="Multivariate Forecaster MAE (Anomalous Regime — High Separation!)")

    ax3.set_title(
        "Phase 2C: Multi-Step Forecast Horizon Profile (Steps k = 1..10) & Normal vs. Anomaly Error Separation",
        color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=8,
    )
    ax3.set_xlabel("Forecast Horizon Step Ahead (k)", color="#E2E8F0", fontsize=9.5)
    ax3.set_ylabel("Mean Absolute Error (MAE)", color="#E2E8F0", fontsize=9.5)
    ax3.set_xticks(horizons)
    ax3.tick_params(colors="#CBD5E1")
    ax3.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

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
    parser = argparse.ArgumentParser(description="Phase 2: NASA Telemanom LSTM Forecaster & Baseline Evaluation")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SMAP/MSL data folder")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to labeled_anomalies.csv")
    parser.add_argument("--chan_id", type=str, default="P-1", help="Channel ID (default: P-1)")
    parser.add_argument("--epochs", type=int, default=35, help="Max training epochs (default: 35)")
    parser.add_argument("--plot_path", type=str, default="phase2_lstm_forecasting_diagnostics.png", help="Output PNG path")
    parser.add_argument("--save_preds", type=str, default=None, help="Optional .npz path to save predictions for Phase 3")
    args = parser.parse_args()

    cfg = Phase2ModelConfig(epochs=args.epochs)

    # 1. Load validated Phase 1 sliding-window tensors
    dataset = Phase1TelemetryDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        l_s=cfg.l_s,
        n_predictions=cfg.n_predictions,
    )
    train_arr, test_arr, meta = dataset.load_or_simulate_channel(args.chan_id)
    X_train, y_train, _ = dataset.build_sliding_windows(train_arr)
    X_test, y_test, t_indices = dataset.build_sliding_windows(test_arr)
    _, window_anomaly_mask = dataset.build_ground_truth_mask(len(test_arr), meta.anomaly_sequences)

    # 2. Evaluate Naive EMA Persistence Baseline
    baseline = NaiveTelemetryBaseline(n_predictions=cfg.n_predictions)
    y_pred_base = baseline.predict(X_test)

    # 3. Train Multivariate Sequence Forecaster
    trainer = Phase2ForecasterTrainer(input_dim=train_arr.shape[1], cfg=cfg)
    train_info = trainer.fit(X_train, y_train)
    y_pred_model = trainer.predict(X_test)

    # 4. Quantify Normal vs Anomalous Regime Error Separation
    normal_mask = window_anomaly_mask == 0
    anom_mask = window_anomaly_mask == 1

    model_norm_mse = float(np.mean((y_test[normal_mask] - y_pred_model[normal_mask]) ** 2))
    model_anom_mse = float(np.mean((y_test[anom_mask] - y_pred_model[anom_mask]) ** 2))
    base_norm_mse = float(np.mean((y_test[normal_mask] - y_pred_base[normal_mask]) ** 2))

    model_norm_mae = float(np.mean(np.abs(y_test[normal_mask] - y_pred_model[normal_mask])))
    model_anom_mae = float(np.mean(np.abs(y_test[anom_mask] - y_pred_model[anom_mask])))
    base_norm_mae = float(np.mean(np.abs(y_test[normal_mask] - y_pred_base[normal_mask])))

    mse_improvement_over_baseline_pct = round(100.0 * (base_norm_mse - model_norm_mse) / (base_norm_mse + 1e-12), 2)
    anomaly_to_normal_mse_ratio = round(model_anom_mse / (model_norm_mse + 1e-12), 2)

    if args.save_preds:
        np.savez(
            args.save_preds,
            y_test=y_test,
            y_pred_model=y_pred_model,
            y_pred_base=y_pred_base,
            t_indices=t_indices,
            window_anomaly_mask=window_anomaly_mask,
        )

    report = {
        "phase": "Phase 2 — Stacked Multivariate LSTM Forecaster & Baseline Comparison",
        "channel": meta.chan_id,
        "training_summary": {
            "backend": train_info["backend"],
            "epochs_trained": train_info["epochs_trained"],
            "best_epoch": train_info["best_epoch"],
            "best_val_mse": train_info["best_val_mse"],
            "final_train_mse": train_info["final_train_mse"],
        },
        "test_forecasting_benchmark": {
            "normal_regime_mse_model": round(model_norm_mse, 5),
            "normal_regime_mse_naive_baseline": round(base_norm_mse, 5),
            "mse_reduction_over_baseline_pct": mse_improvement_over_baseline_pct,
            "normal_regime_mae_model": round(model_norm_mae, 4),
            "normal_regime_mae_naive_baseline": round(base_norm_mae, 4),
            "anomalous_regime_mse_model": round(model_anom_mse, 5),
            "anomalous_regime_mae_model": round(model_anom_mae, 4),
            "anomaly_vs_normal_mse_separation_ratio": f"{anomaly_to_normal_mse_ratio}x",
        },
    }

    print(json.dumps(report, indent=2))

    if args.plot_path:
        plot_phase2_diagnostics(
            train_info=train_info,
            y_test=y_test,
            y_pred_model=y_pred_model,
            y_pred_baseline=y_pred_base,
            t_indices=t_indices,
            window_anomaly_mask=window_anomaly_mask,
            anomaly_sequences=meta.anomaly_sequences,
            chan_id=meta.chan_id,
            output_path=args.plot_path,
        )
        print(f"\n[Phase 2 Complete] Saved 3-panel Forecasting & Convergence chart to: {args.plot_path}")


if __name__ == "__main__":
    main()
