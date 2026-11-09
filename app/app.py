import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIME, AMOUNT = "Time step", "total_BTC"
st.set_page_config(page_title="Bitcoin Flow Investigator", page_icon="🔎", layout="wide")

@st.cache_data
@st.cache_data
def load():
    s = pd.read_csv(ROOT / "results" / "scores.csv")
    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    c = pd.read_csv(ROOT / "results" / "cluster_info.csv")[["cluster", "n_verified"]]
    b = pd.read_csv(ROOT / "results" / "label_budget.csv")
    s["status"] = s["y"].map({1: "Illicit (verified)", 0: "Licit (verified)"}).fillna("Unverified")
    return s[s[TIME] >= 35], m, c, b

scores, metrics, cinfo, budget = load()

st.title("🔎 Bitcoin Unusual Flow Investigator")
st.caption("Cluster-then-verify risk scoring · Elliptic++ dataset · test period: time steps 35–49")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
t0, t1 = st.sidebar.slider("Time steps", 35, 49, (35, 49))
top_n = st.sidebar.slider("Top N flows to review", 10, 500, 100, step=10)
all_status = ["Illicit (verified)", "Licit (verified)", "Unverified"]
statuses = st.sidebar.multiselect("Label status", all_status, default=all_status)

view = scores[scores[TIME].between(t0, t1) & scores["status"].isin(statuses)]
ranked = view.sort_values(["score", TIME], ascending=False).head(top_n)
lab = ranked[ranked["y"].notna()]

# ---- KPI cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions in view", f"{len(view):,}")
c2.metric("Flows in review queue", f"{len(ranked):,}")
c3.metric("Verified illicit in queue", int(lab["y"].sum()))
c4.metric("Precision (verified only)", f"{lab['y'].mean():.1%}" if len(lab) else "–")

tab1, tab2, tab3 = st.tabs(["📋 Ranked flows", "🧩 Clusters", "📊 Model comparison"])

with tab1:
    st.dataframe(
        ranked[["txId", TIME, AMOUNT, "cluster", "score", "status"]],
        hide_index=True,
        column_config={"score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f")},
    )
    st.download_button("⬇️ Download queue as CSV", ranked.to_csv(index=False), "review_queue.csv")

with tab2:
    cl = (view.groupby("cluster")
              .agg(transactions=("txId", "size"), risk=("score", "first"))
              .sort_values("risk", ascending=False).reset_index())
    cl = cl.merge(cinfo, on="cluster", how="left")
    st.caption("Risk = smoothed illicit rate among n_verified training-period transactions. "
               "Fewer verified labels means a less reliable estimate.")
    st.dataframe(
        cl, hide_index=True,
        column_config={"risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.3f")},
    )
    st.caption("Flows in the same cluster share a risk score; within a cluster, the most recent appear first.")
    pick = st.selectbox("Inspect a cluster", cl["cluster"])
    sub = view[view["cluster"] == pick]
    st.write(f"**Cluster {pick}:** {len(sub):,} transactions in view")
    st.bar_chart(sub.groupby(TIME).size(), x_label="Time step", y_label="Transactions")
    st.dataframe(sub[["txId", TIME, AMOUNT, "status"]].head(200), hide_index=True)

with tab3:
    st.dataframe(metrics, hide_index=True)
    st.subheader("How much verification is needed?")
    st.line_chart(budget, x="total_labels", y="PR-AUC mean",
                  x_label="Verified labels", y_label="PR-AUC")
    st.markdown(
        "- **Random reference** is the floor. Anything below it is worse than guessing.\n"
        "- **Amount rule and Isolation Forest** fall below random: illicit flows are not extreme outliers.\n"
        "- **Cluster-then-verify** reaches about 3× random PR-AUC (0.197 ± 0.012 over 5 seeds) "
        "with 500 verified training labels (1.7%). Gains flatten beyond that."
    )