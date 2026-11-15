# Bitcoin Flow Intelligence — Transaction Anomaly & Security Scanner

**Live Deployment:** [https://bitcoinanomolydetetction-b8zmcjj77eqqydleg9w3wa.streamlit.app/](https://bitcoinanomolydetetction-b8zmcjj77eqqydleg9w3wa.streamlit.app/)

## About

**Bitcoin Flow Intelligence** is an institutional-grade Bitcoin transaction anomaly and security screening platform built on the **Elliptic++** dataset (203,769 transactions and 234,355 directed money-flow edges). It combines unsupervised look-alike burst detection, behavior-family risk calibration, and multi-hop graph analysis to surface illicit Bitcoin flows (peeling chains, mixer bursts, and structural anomalies) with plain-English explanations.

### Core Platform Capabilities
- **Secure Portal & Live Global Crypto Desk:** Salted `PBKDF2-HMAC-SHA256` user authentication paired with a live global Bitcoin spot, perpetual derivatives, mempool fee, and crypto news terminal.
- **Live CSV Transaction Scanner (`Scan Your CSV`):** Upload any Bitcoin transaction CSV (full 182-feature Elliptic++ format or simplified transaction columns) to score transactions in real time using the trained hybrid ML pipeline.
- **Command Center:** High-level telemetry across 54,194 test-window transactions, alert volume trends, and amount-vs-risk distributions.
- **Ranked Flows & Case Briefs:** Interactive investigator queue ranked by anomaly score (0–100) with 1-hop ego-network flow graphs and downloadable case briefs.
- **Suspicious Chains:** Multi-hop contagion clusters tracking connected high-risk transactions and BTC velocity per window.
- **Model & Drift Lab:** Interactive alert-budget simulator, stratified Top-500 verification audit, concept-drift tracking (window 43 dark-market shutdown), and label-efficiency curves.

---

## Key Results (Untouched Test Period: Time Windows 35–49)

| Model | P@50 | P@100 | P@500 | PR-AUC | Flag Rate | Labels Used |
|---|---:|---:|---:|---:|---:|---:|
| Random guessing (reference) | 0.040 | 0.080 | 0.078 | 0.065 | 1.10% | 0 |
| Amount threshold (mandatory baseline) | 0.000 | 0.000 | 0.000 | 0.044 | 1.19% | 0 |
| Isolation Forest (classic outliers) | 0.000 | 0.000 | 0.000 | 0.036 | 6.02% | 0 |
| Look-alike burst (unsupervised, no labels) | 0.760 | 0.750 | 0.514 | 0.328 | 0.71% | 0 |
| Family risk only (previous model) | 0.000 | 0.000 | 0.053 | 0.171 | 2.67% | 500 |
| **FINAL MODEL — Hybrid: look-alike + family (w=0.7)** | **0.500** | **0.680** | **0.618** | **0.380** | **0.55%** | **500** |

- **5-seed stability:** `0.405 ± 0.026` PR-AUC (~**5.8×** better than random guessing, and beats the mandatory amount baseline which performs worse than random).
- **Stratified top-500 verification sample (100 flows):** `55.0%` precision (`55/100`, 95% Wilson CI: `[45.2%, 64.4%]`) vs. `0.0%` (`0/100`, `[0.0%, 3.7%]`) for the amount baseline.
- **No score ties:** 15,756 distinct scores across 16,670 labelled test transactions (compared with 8 distinct scores in the old cluster-only model).
- **Concept drift finding:** From time window 43 onward (the dark-market shutdown documented in Elliptic), illicit volume drops sharply and all models degrade until recovery around window 48. This is tracked explicitly per time window in the dashboard.

---

## How It Works

1. **Normal-behaviour modelling (Look-alike burst, unsupervised):**  
   Normal Bitcoin traffic is diverse. Automated criminal operations (mixers, peel chains, ransomware cash-outs) produce bursts of near-identical transactions within the same time window. We measure local crowding in log-scaled, PCA-reduced feature space and rank it within each time window.
2. **Light investigator feedback (Family risk):**  
   Transactions are grouped into 50 behaviour families (`KMeans`). Sampling 10 past labels per family (~500 total, 1.7% of training labels) provides a smoothed illicit prior per family.
3. **Hybrid scoring & flow chains:**  
   Percentiles calibrated strictly on past data (`Time step <= 34`) are blended (`w=0.7` selected on validation windows `25–34`), linked via `txs_edgelist.csv` to identify connected suspicious chains, and paired with plain-English explanations.

---

## Project Structure

```
├── app/
│   └── app.py              # Multi-page Streamlit investigator dashboard
├── src/
│   ├── config.py           # Paths, time splits, and model constants
│   ├── data.py             # Elliptic++ loader with Parquet caching
│   ├── features.py         # Log transform, per-step ranking, graph adjacency
│   ├── metrics.py          # Tie-fair P@K, PR-AUC, flag rate, Wilson CI
│   ├── models.py           # Look-alike burst, family risk, percentile calibration
│   └── pipeline.py         # End-to-end runner generating all results/ files
├── results/                # Generated scores, metrics, chains, verification tables
├── notebooks/              # Exploratory notebooks
└── requirements.txt
```

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the pipeline** (regenerates all artifacts in `results/` from `data/elliptic_bitcoin_dataset/`)
   ```bash
   python src/pipeline.py
   ```
3. **Launch the investigator dashboard**
   ```bash
   streamlit run app/app.py
   ```
4. **Run unit & artifact tests**
   ```bash
   python -m unittest discover tests -v
   ```

---

## Dataset

Uses the public **Elliptic++** dataset (`txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv` in `data/elliptic_bitcoin_dataset/`): 203,769 Bitcoin transactions and 234,355 directed money-flow edges across 49 time windows.
- **Train:** windows 1–24 (model fitting)  
- **Validation:** windows 25–34 (hyperparameter selection)  
- **Test:** windows 35–49 (untouched final evaluation)

