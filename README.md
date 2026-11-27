# Bitcoin Flow Intelligence — Transaction Anomaly & Security Scanner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bitcoinanomolydetetction-b8zmcjj77eqqydleg9w3wa.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Web Application:** [https://bitcoinanomolydetetction-b8zmcjj77eqqydleg9w3wa.streamlit.app/](https://bitcoinanomolydetetction-b8zmcjj77eqqydleg9w3wa.streamlit.app/)  
**Research Paper:** [`Bitcoin_Anomaly_Detection_IEEE_Paper.docx`](Bitcoin_Anomaly_Detection_IEEE_Paper.docx) *(IEEE Two-Column Academic Conference Format)*

---

## About

**Bitcoin Flow Intelligence** is an institutional-grade Bitcoin transaction anomaly and security screening platform built on the **Elliptic++** dataset (203,769 transactions and 234,355 directed money-flow edges across 182 node and graph features). 

It combines unsupervised topological look-alike burst detection, active-learning behavior-family risk calibration, and multi-hop graph decomposition to detect illicit Bitcoin flows (peeling chains, mixer bursts, and automated bot structurings) with plain-English forensic explanations.

### Core Platform Capabilities
- **Secure Portal & Live Global Crypto Desk:** Salted `PBKDF2-HMAC-SHA256` authentication (120,000 rounds) supporting login via Username, Work Email, or Full Name, paired with a real-time global Bitcoin spot, perpetual derivatives, mempool fee, and crypto news wire.
  - *Default Demo Login:* Username `admin` | Password `Admin1234` (or 1-click **Instant Demo Sandbox**).
- **Live CSV Transaction Scanner (`Scan Your CSV`):** Stream and score any Bitcoin transaction CSV (up to **1 GB / 1M+ rows**) in memory-safe 15,000-row chunks. Features **Adaptive Schema & Multivariate Anomaly Mode** to automatically map external blockchain headers (`tx_hash`, `amount`, `fee`, `input_count`, etc.) and detect structural outliers. Includes a built-in **Download 500-Tx Test CSV** button.
- **Command Center:** Telemetry across 54,194 test-window transactions, operational risk tier distributions, alert volume trends, and amount-vs-risk distributions.
- **Ranked Flows & Case Briefs:** Interactive investigator queue sorted by anomaly score (0–100) with 1-hop ego-network flow graphs and downloadable SAR briefs.
- **Suspicious Chains:** Multi-hop peeling-chain graph reconstruction isolating connected high-risk transactions and tracking BTC dissipation per hop.
- **Model & Drift Lab:** Interactive alert-budget simulator, stratified Top-500 verification audit with 95% Wilson confidence intervals, concept-drift tracking (window 43 dark-market shutdown), and label-efficiency curves.

---

## Benchmark Performance (Untouched Test Period: Windows 35–49)

Evaluated under strict chronological anti-leakage partitioning (historical windows 1–34 for training and calibration; future windows 35–49 for blind evaluation across 67,504 test flows and 16,670 labeled test flows):

| Model Architecture | Precision@50 | Precision@100 | Precision@500 | PR-AUC | Test Accuracy | Labels Used |
|---|---:|---:|---:|---:|---:|---:|
| Random Guessing (Base Rate) | 4.0% | 8.0% | 7.8% | 0.065 | 93.50% | 0 |
| Amount Threshold (Mandatory Baseline) | 0.0% | 0.0% | 0.0% | 0.044 | 93.50% | 0 |
| Isolation Forest (Classic Outliers) | 0.0% | 0.0% | 0.0% | 0.036 | 92.80% | 0 |
| Look-Alike Burst (Unsupervised, Zero Labels) | **76.0%** | **75.0%** | 51.4% | 0.328 | 93.60% | 0 |
| Family Risk Only (Clustering Baseline) | 0.0% | 0.0% | 5.3% | 0.171 | 93.50% | 500 |
| **FINAL MODEL — Hybrid: Look-Alike + Family (w=0.7)** | **50.0%** | **68.0%** | **61.8%** | **0.405\*** | **93.69%** | **500** |

*\*PR-AUC is the 5-seed mean (`0.405 ± 0.026`), representing an **8.6× lift** over the mandatory amount baseline. Single final test run achieves **0.380 PR-AUC** and **0.788 ROC-AUC**.*

### Key Empirical Findings
- **Failure of Volume Sorting:** Ranking transactions by transferred Bitcoin volume yields **0.0% precision** across top-ranked alerts because launderers actively structure transactions below regulatory thresholds.
- **Power of Zero-Label Topological Bursts:** Unsupervised k-NN manifold density achieves **75.0% Precision@100** without using any training labels.
- **Stratified Audit with 95% Wilson Confidence Intervals:** Across 100 sampled flows from the Top 500 alerts, the champion model achieved **55.0% verified precision** (`55/100`, 95% CI: `[45.2%, 64.4%]`)—peaking at **74.3% precision** in Ranks 51–200—vs. **0.0%** (`0/100`, `[0.0%, 3.7%]`) for the amount baseline.
- **Peeling-Chain Discovery:** Graph connected-component decomposition identified **16 high-risk peeling chains**, including Chain #0 routing 184.2+ BTC across 34 interconnected transactions.

---

## How It Works

```
Raw On-Chain Flows (182 Features)
               │
               ▼
┌──────────────────────────────────────┐
│ Heavy-Tail Transform & Standardization│ ──► sign(X) · ln(1 + |X|) + StandardScaler
└──────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  50-Dimensional Latent Manifold (PCA)│ ──► Retains 88.4% variance
└──────────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐ ┌──────────────────────┐
│ Signal 1:    │ │ Signal 2:            │
│ Look-Alike   │ │ Behavioral Families  │
│ Burst (k-NN) │ │ (KMeans, K=50)       │
│ Zero-Label   │ │ Laplace Bayes Prior  │
└──────────────┘ └──────────────────────┘
      │                 │
      └────────┬────────┘
               ▼
┌──────────────────────────────────────┐
│ Calibrated Empirical Percentile Blend│ ──► S_final = 0.7 · P_burst + 0.3 · P_family
└──────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Graph Peeling Chains & XAI Explainer │ ──► Connected Components + Case SAR Briefs
└──────────────────────────────────────┘
```

1. **Unsupervised Normal-Behavior Manifold (Look-Alike Burst):**  
   Measures local crowding in 50-D PCA space using a within-window k-nearest-neighbor index ($k=10$). Automated laundering bots produce dense clusters of near-identical structural clones.
2. **Label-Efficient Behavioral Priors (Family Risk):**  
   Partitions transactions into $K=50$ behavioral archetypes. An active-learning budget of only 10 verified past flows per cluster (500 labels total, 0.37% of training data) provides an empirical-Bayes smoothed illicit prior.
3. **Percentile Calibration & Fusion:**  
   Both signals are calibrated against historical empirical percentiles and blended ($w=0.7$). An empirical quantile cutoff ($\tau = 0.9516$) isolates the High-Risk escalation tier.
4. **Multi-Hop Peeling Chains:**  
   Connected-component traversal on high-risk directed payment edges isolates multi-hop laundering chains shedding change across intermediate hops.

---

## Project Structure

```
├── app/
│   └── app.py                             # Multi-page Streamlit forensic compliance dashboard
├── src/
│   ├── config.py                          # Global paths, time splits, and model hyperparameters
│   ├── data.py                            # Elliptic++ loader with binary Parquet caching
│   ├── features.py                        # Signed-log transform, ego-network features, graph adjacency
│   ├── metrics.py                         # Tie-fair P@K, PR-AUC, flag rate, Wilson CI calculation
│   ├── models.py                          # Look-alike burst, family risk, streaming CSV inference engine
│   └── pipeline.py                        # End-to-end reproducible pipeline runner
├── results/                               # Generated model bundles, metrics, chains, verification summaries
├── tests/
│   └── test_core.py                       # Automated unit and artifact integrity test suite
├── Bitcoin_Anomaly_Detection_IEEE_Paper.docx # Formal IEEE two-column research paper
├── generate_ieee_docx.py                  # IEEE paper generator script
├── test_bitcoin_transactions.csv          # 500-tx full 182-feature sample CSV for testing the scanner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/nnaina00789/bitcoin_anomoly_detetction.git
cd bitcoin-anomoly-detection
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Application
Start the Streamlit compliance dashboard:
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.  
Log in with `admin` / `Admin1234` or click **Launch Instant Demo Sandbox**.

### 3. Run Automated Tests
Verify pipeline integrity, metrics calculations, and model scorers:
```bash
python -m unittest discover tests -v
```

### 4. Regenerate Model Artifacts (Optional)
To re-run the full training, validation, and evaluation pipeline from raw data:
```bash
python src/pipeline.py
```

---

## Testing the Live CSV Scanner

1. Navigate to the **Scan Your CSV** page in the dashboard.
2. Click **Download 500-Tx Test CSV** (or use [`test_bitcoin_transactions.csv`](test_bitcoin_transactions.csv) from the repo).
3. Drag and drop the CSV into the uploader.
4. The streaming engine will score the transactions in real-time, displaying risk tiers (High / Medium / Low), triggered patterns (Twin Burst, Risky Group, Rare Shape), and plain-English reasons with one-click export.

---

## Dataset Citation

This project uses the public **Elliptic++** dataset:
```bibtex
@inproceedings{elliptic_plus_plus,
  title={Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Karl and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E.},
  booktitle={KDD Workshop on Anomaly Detection in Finance},
  year={2019}
}
```
