# AstroGuard AI — Intelligent Spacecraft Health Monitoring & Anomaly Investigation Platform

**Official Repository:** <https://github.com/BibinBinoy777/AstroGuard-AI>

**AstroGuard AI** is a complete 5-stage, 27-phase spacecraft telemetry intelligence and anomaly investigation platform expanding [NASA JPL's Telemanom framework (Hundman et al., KDD 2018)](https://github.com/khundman/telemanom) across the [NASA SMAP Satellite & MSL Curiosity Rover Telemetry Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl).


📊 Dashboard & Visual Reports


<img width="1835" height="876" alt="Screenshot (285)" src="https://github.com/user-attachments/assets/fbdbc8ab-8ac5-4084-b38e-cf2c1a70035a" />

<img width="1846" height="889" alt="Screenshot (286)" src="https://github.com/user-attachments/assets/1d9158f7-b573-4509-99f0-adab54c544fb" />



## 1. Complete 5-Stage (27-Phase) Architecture & Benchmarks

| Stage | Phases | Core Modules | Key Verified Engineering Outcome | 
| ----- | ----- | ----- | ----- | 
| **Stage 1: Core ML Engine** | Phases 1–4 | `phase1_data_windowing.py`, `phase2_lstm_forecaster.py` | `(250, d) -> (10,)` sliding windows with zero lookahead leakage; **`63.3%` MSE reduction** over naive EMA baseline and **`16.09x`** anomaly-to-normal error separation | 
| **Stage 2: Advanced Anomaly Intelligence** | Phases 5–10 | `phase3_dynamic_thresholding.py`, `astroguard_stage2_intelligence.py` | Batch Non-Parametric Dynamic Thresholding + $p_{\min}$ pruning achieving **`0.9091` NASA** $F_{0.5}$ (`92.3%` Precision), **`92.9%`** Point vs. Contextual classification accuracy, and 5-Question Explainable AI attribution | 
| **Stage 3: Intelligent Monitoring Platform** | Phases 11–15 | `phase4_fleet_benchmark_and_api.py`, `astroguard_stage3_monitoring_platform.html` | Interactive Mission-Control web platform with dual real-time oscilloscopes, live fault injection, interactive $F_{0.5}$ pruning tuner, incident alert manager, and FastAPI endpoint | 
| **Stage 4: Advanced AI Capabilities** | Phases 16–21 | `astroguard_stage4_advanced_ai.py` | Pre-threshold equipment degradation indicator ($R^2 = 0.90$), Page-Hinkley/KS concept drift detection, cross-channel dependency matrix, AI Investigation Assistant, and **`9.37x` faster** [CVPR 2021 Transfer Learning](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Baireddy_Spacecraft_Time-Series_Anomaly_Detection_Using_Transfer_Learning_CVPRW_2021_paper.pdf) | 
| **Stage 5: Product Qualification** | Phases 22–27 | `astroguard_stage5_qualification.py` | `8/8` automated unit/integration test suites (`100%` pass rate), `6/6` API security guards verified, **`242,137` windows/sec** batch load throughput, and SHA-256 artifact manifest | 

## 2. Visual Diagnostic & Dashboard Gallery

### Stage 1A — Telemetry Exploration, Command States & Sliding-Window Alignment (Phases 1–2)

### Stage 1B — Stacked Multivariate Sequence Forecaster & Residual Separation (Phases 3–4)

### Stage 2A — EWMA Smoothing, Dynamic Threshold Optimization & Pruning Ablation (Phase 5)

### Stage 2B — Model Comparison ($F_{0.5}$ vs. $F_1$), Pattern Classification & 5-Question Explainable AI (Phases 6–10)

### Stage 3 — Multi-Channel Spacecraft Fleet Benchmark & Real-Time FastAPI Service (Phases 11–15)

### Stage 4 — Predictive Equipment Degradation, Cross-Channel Coupling & CVPR 2021 Transfer Learning (Phases 16–21)

### Stage 5 — Automated Invariant Testing, API Security Guards & Load Qualification (Phases 22–27)

## 3. Quickstart & Reproducibility Commands

```
# 1. Stage 1: Verify data integrity, window alignment & train the multivariate forecaster
python3 phase1_data_windowing.py --chan_id P-1
python3 phase2_lstm_forecaster.py --chan_id P-1 --save_preds phase2_predictions.npz

# 2. Stage 2: Run Dynamic Thresholding, Model Comparison (LSTM vs Isolation Forest vs Control Chart), XAI & F0.5 Optimization
python3 phase3_dynamic_thresholding.py --preds_npz phase2_predictions.npz
python3 astroguard_stage2_intelligence.py

# 3. Stage 3 & 4: Run Fleet Benchmark, Predictive Degradation, Transfer Learning & Model Monitoring
python3 phase4_fleet_benchmark_and_api.py --export_docker_dir ./docker_bundle
python3 astroguard_stage4_advanced_ai.py

# 4. Stage 5: Run Full 27-Phase Qualification Suite (Unit Tests, Security Audit, Load Benchmark)
python3 astroguard_stage5_qualification.py


```

## 4. Scientific References

* **Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018).** *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD '18. [GitHub Repository](https://github.com/khundman/telemanom)

* **Baireddy, S., Desai, S. R., Mathur, J., Kidambi, S., & Delp, E. J. (2021).** *Spacecraft Time-Series Anomaly Detection Using Transfer Learning.* IEEE/CVF CVPRW 2021 (AI4Space). [CVF Open Access PDF](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Baireddy_Spacecraft_Time-Series_Anomaly_Detection_Using_Transfer_Learning_CVPRW_2021_paper.pdf)

* **NASA Anomaly Detection Dataset (SMAP & MSL):** [Kaggle Dataset Mirror](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
