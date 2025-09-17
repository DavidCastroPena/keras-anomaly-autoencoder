# Autoencoder Anomaly Detection Wrapper

> Detect forbidden vehicles in pedestrian-only spaces using a TensorFlow/Keras autoencoder with pixel- and frame-level anomaly signals.
<img width="971" height="517" alt="Image" src="https://github.com/user-attachments/assets/65dbc1a7-035d-49f9-805a-12560f4967e3" />
---
## Problem Context

Busy pedestrian walkways (e.g., university paths, plazas) occasionally see **forbidden vehicles** (bicycles, skateboards, carts, cars) enter areas intended for foot traffic only. This raises collision risk and perceived insecurity.

**Objective:** Improve pedestrian safety by **automatically flagging video frames** where a forbidden vehicle appears, so operators can intervene or log incidents.

**Operational constraints:**
- **Missing a vehicle (false negative)** is costlier than a false alarm.
- Alerts should be **actionable** (not too frequent) and **fast** (frame-level decisions in real time).
- 🗺Pixel-level localization (heatmaps) helps triage by showing *where* the anomaly occurs.

---

## 🧩 Decision Problem

For each frame:
- **Binary frame decision:** **Normal** (pedestrians only) vs **Anomalous** (vehicle present).
- **Pixel decision (supporting):** Which regions are anomalous (error heatmap / binary mask) to guide review.

This is implemented as **thresholding** an anomaly score:
- Per-pixel: threshold on a normalized error map (`pixel_tau`).
- Per-frame: threshold on reconstruction error (`frame_thr`).

---

## Dataset

We evaluate on the **UCSD Anomaly Detection Dataset (Ped1 & Ped2)**:
- Outdoor pedestrian walkway footage at UC San Diego.
- **Normal:** pedestrians walking.
- **Anomalies:** bicycles, skateboards, vehicles, or motion in unusual areas.
- Frame-level ground truth annotations support ROC/PR analysis.

---

## 🛠️ Method (Technical Overview)

We wrap a trained **autoencoder (AE)** with additional TensorFlow ops to expose anomaly signals:

- **Reconstruction:** `recon = AE(x)`  
- **Per-pixel error:** `err = |x - recon|`
- **Per-image normalization:** `err_norm = err / (max(err) + 1e-8)` → values in `[0,1]`  
- **Binary pixel mask:** `mask = 1[err_norm ≥ pixel_tau]`
- **Frame score:** `frame_score = mean((x - recon)²)` (MSE)
- **Frame flag:** `frame_flag = 1[frame_score ≥ frame_thr]` (optional)

All are packaged in one Keras `Model(inputs, [recon, err_norm, mask, frame_score, frame_flag])`.

---

## Results (UCSD)

**Reported:**
- **ROC AUC:** `0.742`
- **Average Precision (AUPRC):** `0.9257`
- **Frames with GT found:** `1530` | **Positives:** `1276` | **Negatives:** `254`
- **Threshold for ~2% alert rate:** `0.0031929`
- **Best threshold (Youden’s J):** `0.001468`
- **Cost-minimizing threshold:** `0.000742` (Min expected cost = `223.0`)
- **Confusion at best threshold (given):** `(FP=31, FN=223, TN=0, TP=1276)`

**Interpretation:**
- **AUPRC ~0.93** → very strong when prioritizing *precision–recall*, which is typical for safety monitoring with class imbalance.
- **ROC AUC ~0.74** → moderate separation on a balanced error trade-off curve (ROC is less informative with imbalance).
- **Best-J operating point**:  
  - **Precision ~0.924** → most alerts are true vehicles.  
  - **Recall ~0.823** → ~82% of anomalous frames are caught; ~18% missed.
- **2% alert-rate threshold (`0.00319`)**: practical if operator bandwidth is tight; **expect lower recall** (safer on false alarms, riskier on misses).
- **Cost-minimizing threshold (`0.000742`)**: under cost model, this is optimal; it typically **increases recall** at the expense of more alerts.

---

## Decision in Light of the Metrics

**Safety-first policy (recommended for pedestrian areas):**
- Operate near the **cost-minimizing** or **slightly lower** threshold than Best-J (e.g., around `0.0007–0.0015`).  
- **Why:** Higher recall (fewer missed vehicles) is preferred even if false alarms rise—misses carry higher safety risk.

**Bandwidth-limited policy (control room overwhelmed):**
- Use the **2% alert-rate** threshold (`~0.00319`) to cap alerts.  
- **Trade-off:** Fewer false alarms, but **more missed vehicles**—use when staffing is tight or as a triage layer before human-in-the-loop review.

**What the confusion values suggest:**  
At the provided best-threshold summary, **TP=1276** and **FN=223** ⇒ the system catches most anomalies but still misses some. **FP=31** indicates alerts are usually trustworthy. If your deployment can tolerate more alerts, move the threshold down to cut **FN**.

---

## How to Improve

- **Modeling**
  - Stronger encoder/decode
