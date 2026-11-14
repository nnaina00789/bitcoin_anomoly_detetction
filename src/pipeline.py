"""End-to-end pipeline: python src/pipeline.py

Regenerates EVERY file in results/ from the raw data, so the dashboard always
shows numbers that match the model. Takes about 2-3 minutes.

Steps
1. Load data, build the money-flow graph.
2. Validation (learn on steps 1-24, judge on 25-34) -> choose the model settings.
3. Final run (learn on steps 1-34, test on 35-49, never seen before).
4. Compare against the mandatory amount-threshold baseline and others.
5. Simulate the investigator verification of a sample of the ranked list.
6. Build flows (who sent / received), suspicious chains and plain-English reasons.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import (AMOUNT, FLAG_QUANTILE, LABELS_PER_CLUSTER, N_CLUSTERS,
                    RESULTS_DIR, SEED, TEST_START, TIME, TRAIN_END, VAL_START)
from data import load, load_edges
from features import build_graph, signed_log
from metrics import evaluate, precision_at_k, wilson_ci, n_ties
from models import (blend, families, family_risk, global_density, lookalike_burst,
                    point_unusualness, prepare, to_percentile)

WEIGHTS = (0.3, 0.5, 0.7)
SEEDS = range(5)


def log(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model fitting for one period
# --------------------------------------------------------------------------- #
def fit_period(tx, feats, fit_end):
    t = tx[TIME].to_numpy()
    fit = t <= fit_end
    X = tx[feats].fillna(tx.loc[fit, feats].median()).to_numpy("float32")
    Xl, Zp = prepare(X, fit)
    raw = {
        "point": point_unusualness(Xl, fit),
        "global_density": global_density(Zp, fit),
        "lookalike": lookalike_burst(Zp, t),
    }
    return {"fit": fit, "raw": raw, "cluster": families(Zp, fit)}


def candidates(P, y, per, seed):
    """All candidate scores for one period (percentiles calibrated on the past)."""
    fit, raw = P["fit"], P["raw"]
    pct = {k: to_percentile(v, v[fit]) for k, v in raw.items()}
    risk, verified = family_risk(P["cluster"], y, fit & ~np.isnan(y), per, seed)
    fam = to_percentile(risk, risk[fit])
    c = {"Isolation Forest (classic outliers)": pct["point"],
         "Look-alike burst (no labels)": pct["lookalike"],
         "Global density (no labels)": pct["global_density"],
         "Family risk only (old model)": risk}
    for w in WEIGHTS:
        c[f"Hybrid: look-alike + family (w={w})"] = blend(pct["lookalike"], fam, w)
        c[f"Hybrid: global density + family (w={w})"] = blend(pct["global_density"], fam, w)
    return c, risk, verified, pct


def mean_prauc(P, y, ev, per=LABELS_PER_CLUSTER):
    out = {}
    for s in SEEDS:
        c, *_ = candidates(P, y, per, s)
        for k, v in c.items():
            out.setdefault(k, []).append(average_precision_score(y[ev], v[ev]))
    return {k: (np.mean(v), np.std(v)) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Verification protocol (requirement 2)
# --------------------------------------------------------------------------- #
BANDS = [(1, 50, 30), (51, 200, 35), (201, 500, 35)]   # (from rank, to rank, sample size)


def verification_sample(test, score_col, method, rng):
    """Stratified sample of 100 flows from the labelled Top-500 ranked list."""
    ranked = (test[test["y"].notna()]
              .sort_values(score_col, ascending=False, kind="stable")
              .reset_index(drop=True))
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    parts = []
    for lo, hi, n in BANDS:
        band = ranked.iloc[lo - 1:hi]
        parts.append(band.sample(n=n, random_state=rng).assign(band=f"Rank {lo}-{hi}"))
    s = pd.concat(parts)
    s["method"] = method
    s["outcome"] = s["y"].map({1.0: "Confirmed illicit", 0.0: "Confirmed legitimate"})
    return s[["method", "band", "rank", "txId", TIME, AMOUNT, score_col, "outcome"]].rename(
        columns={score_col: "score"})


def summarise_verification(sample):
    rows = []
    for (method, band), g in list(sample.groupby(["method", "band"])) + \
            [((m, "All sampled"), g) for m, g in sample.groupby("method")]:
        hits = (g["outcome"] == "Confirmed illicit").sum()
        checked = (g["outcome"] != "Could not be confirmed").sum()
        lo, hi = wilson_ci(hits, checked)
        rows.append({"method": method, "band": band, "sampled": len(g), "confirmable": int(checked),
                     "confirmed_illicit": int(hits),
                     "precision": round(hits / checked, 3) if checked else np.nan,
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plain-English reasons
# --------------------------------------------------------------------------- #
def reasons_for(r):
    out = []
    if r["lookalike_pct_step"] >= 0.90:
        out.append(f"Has more near-identical 'twin' transactions at the same time than "
                   f"{r['lookalike_pct_step']:.0%} of others. Bulk, automated activity looks like this.")
    if r["family_checked"] > 0 and r["family_illicit"] > 0:
        out.append(f"Belongs to behaviour group #{int(r['cluster'])}: investigators checked "
                   f"{int(r['family_checked'])} similar past cases and {int(r['family_illicit'])} were criminal.")
    if r["flagged_neighbours"] > 0:
        out.append(f"Directly sends to or receives from {int(r['flagged_neighbours'])} other flagged transaction(s).")
    if r["point_pct"] >= 0.97:
        out.append("Its overall shape is rare compared with normal traffic.")
    if r["amount_pct"] >= 0.99:
        out.append(f"Moves an unusually large amount ({r[AMOUNT]:,.2f} BTC).")
    if not out:
        out.append("Ranked low: looks like ordinary traffic.")
    return " | ".join(out)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    tx, feats = load()
    edges = load_edges()
    y = tx["y"].to_numpy()
    t = tx[TIME].to_numpy()
    A = build_graph(tx, edges)
    log(f"Loaded {len(tx):,} transactions, {A.nnz:,} money-flow links, {len(feats)} features")

    # ---------------- 1. Validation: choose settings honestly ---------------- #
    Pv = fit_period(tx, feats, VAL_START - 1)
    ev_val = (t >= VAL_START) & (t <= TRAIN_END) & ~np.isnan(y)
    val = mean_prauc(Pv, y, ev_val)
    val_df = (pd.DataFrame([{"model": k, "val_PR-AUC": round(m, 3), "val_std": round(s, 3)}
                            for k, (m, s) in val.items()])
              .sort_values("val_PR-AUC", ascending=False))
    hybrids = val_df[val_df["model"].str.startswith("Hybrid")]
    chosen = hybrids.iloc[0]["model"]
    val_df["chosen"] = val_df["model"] == chosen
    val_df.to_csv(RESULTS_DIR / "validation.csv", index=False)
    log(f"Validation done. Chosen: {chosen}")

    # ---------------- 2. Final run on the untouched test period -------------- #
    P = fit_period(tx, feats, TRAIN_END)
    fit = P["fit"]
    test_mask = t >= TEST_START
    ev = test_mask & ~np.isnan(y)
    cands, risk, verified, pct = candidates(P, y, LABELS_PER_CLUSTER, SEED)
    final = cands[chosen]
    thr = np.quantile(final[fit], FLAG_QUANTILE)
    thr_med = np.quantile(final[fit], 0.95)

    amount = tx[AMOUNT].fillna(0).to_numpy()
    rng = np.random.default_rng(SEED)
    rand = rng.random(len(tx))
    rows = [
        (evaluate("Random guessing (reference)", y[ev], rand[ev], np.quantile(rand[fit], FLAG_QUANTILE)), 0),
        (evaluate("Amount threshold (mandatory baseline)", y[ev], amount[ev], np.quantile(amount[fit], FLAG_QUANTILE)), 0),
    ]
    for name in ["Isolation Forest (classic outliers)", "Look-alike burst (no labels)",
                 "Family risk only (old model)"]:
        s = cands[name]
        rows.append((evaluate(name, y[ev], s[ev], np.quantile(s[fit], FLAG_QUANTILE)),
                     len(verified) if "Family" in name else 0))
    rows.append((evaluate(f"FINAL MODEL - {chosen}", y[ev], final[ev], thr), len(verified)))
    metrics = pd.DataFrame([{**r, "labels_used": n} for r, n in rows])
    test_seeds = mean_prauc(P, y, ev)
    metrics["PR-AUC_5seed_mean"] = [np.nan, np.nan, np.nan, np.nan,
                                    round(test_seeds["Family risk only (old model)"][0], 3),
                                    round(test_seeds[chosen][0], 3)]
    metrics["PR-AUC_5seed_std"] = [np.nan, np.nan, np.nan, np.nan,
                                   round(test_seeds["Family risk only (old model)"][1], 3),
                                   round(test_seeds[chosen][1], 3)]
    metrics["distinct_scores"] = [n_ties(rand[ev]), n_ties(amount[ev]),
                                  n_ties(cands["Isolation Forest (classic outliers)"][ev]),
                                  n_ties(cands["Look-alike burst (no labels)"][ev]),
                                  n_ties(cands["Family risk only (old model)"][ev]), n_ties(final[ev])]
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    log("Test metrics:\n" + metrics.to_string(index=False))

    # ---------------- 3. How many investigator checks are needed? ------------ #
    w = float(chosen.split("w=")[1].rstrip(")"))
    base_key = "lookalike" if "look-alike" in chosen else "global_density"
    budget = []
    for per in [2, 4, 10, 20, 50]:
        prs, n = [], 0
        for s in SEEDS:
            r_, v_ = family_risk(P["cluster"], y, fit & ~np.isnan(y), per, s)
            sc = blend(pct[base_key], to_percentile(r_, r_[fit]), w)
            prs.append(average_precision_score(y[ev], sc[ev]))
            n = len(v_)
        budget.append({"labels_per_group": per, "total_labels": n,
                       "PR-AUC mean": round(np.mean(prs), 3), "std": round(np.std(prs), 3)})
    pd.DataFrame(budget).to_csv(RESULTS_DIR / "label_budget.csv", index=False)

    # ---------------- 4. Per-time-step (drift) report ----------------------- #
    per_step = []
    for s in range(TEST_START, t.max() + 1):
        m = (t == s) & ~np.isnan(y)
        ys = y[m]
        if ys.sum() == 0:
            continue
        per_step.append({"time_step": s, "known_illicit": int(ys.sum()), "known_total": int(m.sum()),
                         "illicit_share": round(ys.mean(), 3),
                         "PR-AUC final": round(average_precision_score(ys, final[m]), 3),
                         "PR-AUC amount": round(average_precision_score(ys, amount[m]), 3),
                         "flag_rate_%": round((final[t == s] >= thr).mean() * 100, 2)})
    pd.DataFrame(per_step).to_csv(RESULTS_DIR / "per_step.csv", index=False)

    # ---------------- 5. Build the review table for the test period ---------- #
    U = ((A + A.T) > 0).astype("float32").tocsr()
    flagged_all = (final >= thr).astype("float32")
    fam = (verified.groupby("cluster")["y"].agg(family_checked="size", family_illicit="sum")
           .reset_index())
    out = pd.DataFrame({
        "txId": tx["txId"], TIME: t, AMOUNT: tx[AMOUNT], "fees": tx["fees"],
        "num_input_addresses": tx["num_input_addresses"],
        "num_output_addresses": tx["num_output_addresses"],
        "y": y, "cluster": P["cluster"],
        "risk_score": np.round(final * 100, 1),
        "lookalike_pct": pct[base_key], "point_pct": pct["point"],
        "family_risk": risk, "amount_pct": to_percentile(amount, amount[fit]),
        "n_senders": np.asarray(A.sum(axis=0)).ravel().astype(int),
        "n_receivers": np.asarray(A.sum(axis=1)).ravel().astype(int),
        "flagged_neighbours": (U @ flagged_all).astype(int),
        "flagged": final >= thr,
    })
    out["lookalike_pct_step"] = out.groupby(TIME)["lookalike_pct"].rank(pct=True)
    out["iso_score"] = P["raw"]["point"]
    out = out.merge(fam, on="cluster", how="left").fillna({"family_checked": 0, "family_illicit": 0})
    out["risk_level"] = np.select([final >= thr, final >= thr_med], ["High", "Medium"], "Low")
    out["status"] = out["y"].map({1.0: "Confirmed illicit", 0.0: "Confirmed legitimate"}).fillna(
        "Not yet investigated")

    test = out[test_mask].copy()
    # Suspicious chains = groups of flagged transactions linked by money flow
    fidx = np.flatnonzero(test_mask & (final >= thr))
    sub = U[fidx][:, fidx]
    _, comp = connected_components(sub, directed=False)
    chain = pd.Series(comp, index=tx["txId"].to_numpy()[fidx])
    sizes = chain.map(chain.value_counts())
    test["chain_id"] = test["txId"].map(chain)
    test["chain_size"] = test["txId"].map(sizes).fillna(0).astype(int)
    test["why_flagged"] = test.apply(reasons_for, axis=1)
    test = test.sort_values(["risk_score", "lookalike_pct"], ascending=False)
    test["rank"] = np.arange(1, len(test) + 1)
    test.to_csv(RESULTS_DIR / "scores.csv", index=False)

    ch = (test[test["chain_size"] > 1].groupby("chain_id")
          .agg(transactions=("txId", "size"), first_step=(TIME, "min"), last_step=(TIME, "max"),
               total_BTC=(AMOUNT, "sum"), top_risk=("risk_score", "max"),
               avg_risk=("risk_score", "mean"),
               confirmed_illicit=("status", lambda s: (s == "Confirmed illicit").sum()),
               confirmed_legit=("status", lambda s: (s == "Confirmed legitimate").sum()))
          .sort_values(["transactions", "avg_risk"], ascending=False).reset_index())
    ch.to_csv(RESULTS_DIR / "chains.csv", index=False)

    ids = set(test["txId"])
    fe = edges[edges["txId1"].isin(ids) | edges["txId2"].isin(ids)]
    fe.to_csv(RESULTS_DIR / "flow_edges.csv", index=False)

    ci = (pd.DataFrame({"cluster": P["cluster"]}).value_counts().rename("transactions").reset_index()
          .merge(fam, on="cluster", how="left").fillna(0)
          .assign(family_risk=lambda d: d["cluster"].map(pd.Series(risk, index=P["cluster"]).groupby(level=0).first()))
          .sort_values("family_risk", ascending=False))
    ci.to_csv(RESULTS_DIR / "cluster_info.csv", index=False)

    # ---------------- 6. Verification of a sample of the ranked list --------- #
    test_amt = test.assign(amount_score=test[AMOUNT].fillna(0))
    vs = pd.concat([verification_sample(test, "risk_score", "Our model", SEED),
                    verification_sample(test_amt, "amount_score", "Amount threshold (baseline)", SEED)])
    vs.to_csv(RESULTS_DIR / "verification_sample.csv", index=False)
    vsum = summarise_verification(vs)
    vsum.to_csv(RESULTS_DIR / "verification_summary.csv", index=False)
    log("Verification:\n" + vsum.to_string(index=False))

    # ---------------- 7. Headline numbers for the dashboard ----------------- #
    fm = metrics.iloc[-1]
    bm = metrics.iloc[1]
    rm = metrics.iloc[0]
    our_all = vsum[(vsum["method"] == "Our model") & (vsum["band"] == "All sampled")].iloc[0]
    base_all = vsum[(vsum["method"] != "Our model") & (vsum["band"] == "All sampled")].iloc[0]
    summary = {
        "chosen_model": chosen, "weight_lookalike": w,
        "val_PR-AUC": float(val_df.iloc[0]["val_PR-AUC"]) if val_df.iloc[0]["chosen"] else
        float(val_df[val_df["chosen"]]["val_PR-AUC"].iloc[0]),
        "test_transactions": int(test_mask.sum()), "test_known": int(ev.sum()),
        "test_known_illicit": int(y[ev].sum()),
        "flagged": int((final[test_mask] >= thr).sum()),
        "flag_rate_%": float(fm["flag_rate_%"]), "recall_at_flag_%": float(fm["recall_at_flag_%"]),
        "labels_used": int(len(verified)),
        "PR-AUC_final": float(fm["PR-AUC"]), "PR-AUC_baseline": float(bm["PR-AUC"]),
        "PR-AUC_random": float(rm["PR-AUC"]),
        "PR-AUC_final_5seed": [float(fm["PR-AUC_5seed_mean"]), float(fm["PR-AUC_5seed_std"])],
        "P@100_final": float(fm["P@100"]), "P@100_baseline": float(bm["P@100"]),
        "P@500_final": float(fm["P@500"]), "P@500_baseline": float(bm["P@500"]),
        "verified_precision_final": [float(our_all["precision"]), float(our_all["ci_low"]),
                                     float(our_all["ci_high"]), int(our_all["confirmable"]),
                                     int(our_all["sampled"])],
        "verified_precision_baseline": [float(base_all["precision"]), float(base_all["ci_low"]),
                                        float(base_all["ci_high"]), int(base_all["confirmable"]),
                                        int(base_all["sampled"])],
        "chains": int(len(ch)), "threshold": float(thr),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- 8. Save lightweight model bundle + sample CSV --------- #
    med = tx.loc[fit, feats].median()
    X_fit = tx[feats].fillna(med).to_numpy("float32")
    Xl_fit = signed_log(X_fit)
    sc_fit = StandardScaler().fit(Xl_fit[fit])
    Z_fit = sc_fit.transform(Xl_fit)
    pca_fit = PCA(n_components=20, random_state=SEED).fit(Z_fit[fit])
    Zp_fit = pca_fit.transform(Z_fit).astype("float32")
    iso_fit = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xl_fit[fit])
    km_fit = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(Zp_fit[fit])
    risk_map = pd.Series(risk, index=P["cluster"]).groupby(level=0).first().to_dict()
    rs = np.random.default_rng(SEED)
    idx_fit = rs.choice(np.flatnonzero(fit), size=min(5000, int(fit.sum())), replace=False)
    bundle = {
        "feats": feats, "train_medians": med.to_dict(),
        "scaler": sc_fit, "pca": pca_fit, "iso": iso_fit, "km": km_fit,
        "family_risk_map": risk_map, "prior": float(verified["y"].mean()),
        "ref_lookalike": np.sort(P["raw"]["lookalike"][idx_fit]),
        "ref_point": np.sort(P["raw"]["point"][idx_fit]),
        "ref_family": np.sort(risk[idx_fit]),
        "ref_amount": np.sort(amount[idx_fit]),
        "w": w, "thr": float(thr), "thr_med": float(thr_med),
    }
    joblib.dump(bundle, RESULTS_DIR / "model_bundle.joblib", compress=3)

    # 60-row sample CSV from test period (mix of high/medium/low) for live scanner demo
    sample_idx = np.concatenate([
        test.head(20)["txId"].to_numpy(),
        test[test["risk_level"] == "Medium"].sample(20, random_state=SEED)["txId"].to_numpy(),
        test[test["risk_level"] == "Low"].sample(20, random_state=SEED)["txId"].to_numpy(),
    ])
    tx[tx["txId"].isin(sample_idx)][["txId", TIME] + feats].to_csv(
        RESULTS_DIR / "sample_scan.csv", index=False
    )
    log("Done. All files written to results/")


if __name__ == "__main__":
    main()

