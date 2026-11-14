"""Bitcoin Suspicious Activity Finder — AML & Flow Intelligence Dashboard.

Run with:  streamlit run app/app.py
"""
import hashlib
import hmac
import html
import json
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from models import score_dataframe  # noqa: E402

RES = ROOT / "results"
TIME, AMOUNT = "Time step", "total_BTC"

LEVEL_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
STATUS_ICON = {
    "Confirmed illicit": "Illicit",
    "Confirmed legitimate": "Legitimate",
    "Not yet investigated": "Unlabelled",
}
LEVEL_STYLE = {
    "High": ("#450A0A", "#FCA5A5", "#EF4444"),
    "Medium": ("#451A03", "#FCD34D", "#F59E0B"),
    "Low": ("#052E16", "#86EFAC", "#22C55E"),
    "Outside": ("#1E293B", "#94A3B8", "#475569"),
}

st.set_page_config(
    page_title="Bitcoin Flow Intelligence",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 14% 18%, rgba(14, 165, 233, 0.34) 0%, transparent 45%),
        radial-gradient(circle at 86% 22%, rgba(245, 158, 11, 0.22) 0%, transparent 42%),
        radial-gradient(circle at 78% 84%, rgba(99, 102, 241, 0.32) 0%, transparent 48%),
        radial-gradient(circle at 22% 82%, rgba(16, 185, 129, 0.16) 0%, transparent 45%),
        linear-gradient(135deg, #070D1F 0%, #0F1E3D 48%, #15193B 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(11, 17, 32, 0.96) 100%);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-top: 3px solid #38BDF8;
    border-radius: 12px;
    padding: 15px 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.55);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(17, 25, 46, 0.78), rgba(9, 14, 29, 0.88)) !important;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(56, 189, 248, 0.26) !important;
    border-radius: 20px !important;
    padding: 24px 26px !important;
    box-shadow:
        0 30px 65px rgba(0, 0, 0, 0.65),
        0 0 45px rgba(56, 189, 248, 0.10),
        inset 0 1px 0 rgba(255, 255, 255, 0.14);
    transition: transform 0.45s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.45s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.45s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.48) !important;
    box-shadow:
        0 36px 75px rgba(0, 0, 0, 0.75),
        0 0 55px rgba(56, 189, 248, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.20);
}
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 0 0 !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.40) !important;
    border-radius: 12px !important;
}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
    border: 1px solid rgba(125, 211, 252, 0.35) !important;
    color: #F8FAFC !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.32) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: #7DD3FC !important;
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.45) !important;
    transform: translateY(-1px);
}
.card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.90) 0%, rgba(11, 17, 32, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-top: 2px solid #38BDF8;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 155px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}
.card h4 { margin: 0 0 8px 0; color: #38BDF8; font-size: 0.98rem; font-weight: 600; }
.card p { margin: 0; color: #CBD5E1; font-size: 0.88rem; line-height: 1.5; }
.page-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(11, 19, 38, 0.88) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-left: 4px solid #38BDF8;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
}
.page-banner-kicker {
    color: #38BDF8;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.page-banner-title {
    color: #F8FAFC;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-banner-sub {
    color: #94A3B8;
    font-size: 0.88rem;
    margin: 4px 0 0 0;
}
.page-banner-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 13px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.32);
    color: #6EE7B7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    font-weight: 600;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060A15 0%, #0A1122 100%);
    border-right: 1px solid #1E293B;
}
[data-testid="stHeaderActionElements"] { display: none; }
.vault-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px 0 16px 0;
    perspective: 1000px;
    position: relative;
}
.vault-aura {
    position: absolute;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.25), rgba(99, 102, 241, 0.14), transparent 70%);
    filter: blur(22px);
}
.vault-orbit-outer {
    position: relative;
    width: 164px;
    height: 164px;
    border-radius: 50%;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.16), inset 0 0 24px rgba(56, 189, 248, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
    animation: vaultFloat 6s cubic-bezier(0.45, 0, 0.55, 1) infinite;
}
.vault-orbit-inner {
    width: 128px;
    height: 128px;
    border-radius: 50%;
    border: 1px dashed rgba(245, 158, 11, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    transform: translateZ(18px);
}
.vault-coin {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.25), rgba(15, 23, 42, 0.92));
    border: 1.5px solid rgba(245, 158, 11, 0.65);
    box-shadow:
        0 16px 35px rgba(0, 0, 0, 0.6),
        0 0 28px rgba(245, 158, 11, 0.28),
        inset 0 2px 6px rgba(255, 255, 255, 0.22);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.55rem;
    font-weight: 800;
    color: #FBBF24;
    text-shadow: 0 2px 18px rgba(251, 191, 36, 0.55);
    transform: translateZ(34px);
}
.vault-title {
    text-align: center;
    font-size: 2.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #F8FAFC 0%, #BAE6FD 55%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 18px 0 8px 0;
}
@keyframes vaultFloat {
    0%, 100% { transform: translateY(0px) scale(1); }
    50%      { transform: translateY(-7px) scale(1.02); }
}
</style>
""", unsafe_allow_html=True)


def render_page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-banner">'
        f'<div>'
        f'<div class="page-banner-kicker">{html.escape(kicker)}</div>'
        f'<h1 class="page-banner-title">{html.escape(title)}</h1>'
        f'<p class="page-banner-sub">{html.escape(subtitle)}</p>'
        f'</div>'
        f'<div class="page-banner-status"><span class="status-dot"></span>ENGINE ONLINE · 182-FEATURE HYBRID</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load():
    need = ["scores.csv", "metrics.csv", "summary.json"]
    if not all((RES / f).exists() for f in need):
        return None
    d = {
        "scores": pd.read_csv(RES / "scores.csv"),
        "metrics": pd.read_csv(RES / "metrics.csv"),
        "summary": json.loads((RES / "summary.json").read_text()),
        "edges": pd.read_csv(RES / "flow_edges.csv"),
        "chains": pd.read_csv(RES / "chains.csv"),
        "vsum": pd.read_csv(RES / "verification_summary.csv"),
        "vsample": pd.read_csv(RES / "verification_sample.csv"),
        "per_step": pd.read_csv(RES / "per_step.csv"),
        "budget": pd.read_csv(RES / "label_budget.csv"),
        "validation": pd.read_csv(RES / "validation.csv"),
    }
    s = d["scores"]
    s["Level"] = s["risk_level"].map(LEVEL_ICON)
    s["Status"] = s["status"].map(STATUS_ICON)
    s["Pattern"] = np.select(
        [
            s["chain_size"] > 1,
            s["lookalike_pct_step"] >= 0.90,
            (s["family_checked"] > 0) & (s["family_illicit"] > 0),
            s["point_pct"] >= 0.97,
        ],
        ["Linked Chain", "Twin Burst", "Risky Group", "Rare Shape"],
        default="Normal",
    )
    s["Flow"] = s["n_senders"].astype(str) + " in → " + s["n_receivers"].astype(str) + " out"
    s["Main reason"] = s["why_flagged"].str.split(" | ").str[0]
    return d


def build_sar_text(r):
    action = {"High": "Escalate & Freeze Pending Review", "Medium": "Enhanced Due Diligence (EDD)",
              "Low": "Routine Monitoring"}[r["risk_level"]]
    chain_line = f"Connected to Suspicious Chain #{int(r['chain_id'])} ({int(r['chain_size'])} linked nodes)" \
        if r["chain_size"] > 1 else "Standalone flow (not part of a multi-node flagged chain)"
    reasons = "\n".join(f"- {x}" for x in r["why_flagged"].split(" | "))
    return (
        f"# Case Brief — Tx {int(r['txId'])}\n"
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- **Risk Tier:** {LEVEL_ICON[r['risk_level']]} (Anomaly Score: {r['risk_score']:.1f}/100)\n"
        f"- **Detected Pattern:** {r['Pattern']}\n"
        f"- **Recommended Action:** {action}\n"
        f"- **Time Window:** {int(r[TIME])} | **Amount:** {r[AMOUNT]:,.4f} BTC | **Fee:** {r['fees']:.6f} BTC\n"
        f"- **Graph Context:** {r['Flow']} | {chain_line}\n\n"
        f"## Triggered Detection Signals\n{reasons}\n"
    )


@st.cache_resource
def load_bundle():
    p = RES / "model_bundle.joblib"
    return joblib.load(p) if p.exists() else None


D = load()
if D is None:
    st.error("Results not found. Run `python src/pipeline.py` first.")
    st.stop()
S, SUM = D["scores"], D["summary"]
BY_ID = S.set_index("txId")


# --------------------------------------------------------------------------- #
# Dark-mode SVG flow & chain graphs
# --------------------------------------------------------------------------- #
def _node_meta(tx_id):
    if tx_id in BY_ID.index:
        r = BY_ID.loc[tx_id]
        bg, fg, border = LEVEL_STYLE[r["risk_level"]]
        tag = "Confirmed Illicit" if r["status"] == "Confirmed illicit" else f"Risk {r['risk_score']:.0f}/100"
        sub = f"{r[AMOUNT]:,.3f} BTC · {tag}"
        tip = f"Tx {tx_id} | Window {int(r[TIME])} | Risk {r['risk_score']:.0f}/100 | {r[AMOUNT]:,.4f} BTC"
        return bg, fg, border, sub, tip
    bg, fg, border = LEVEL_STYLE["Outside"]
    return bg, fg, border, "Earlier window", f"Tx {tx_id} (before window 35)"


def _svg_node(x, y, w, h, tx_id, centre=False):
    bg, fg, border, sub, tip = _node_meta(tx_id)
    sw = "3" if centre else "1.5"
    return (
        f'<g><title>{html.escape(tip)}</title>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{bg}" stroke="{border}" stroke-width="{sw}"/>'
        f'<text x="{x + 12}" y="{y + 20}" font-family="system-ui,sans-serif" '
        f'font-size="12" font-weight="700" fill="{fg}">Tx {tx_id}</text>'
        f'<text x="{x + 12}" y="{y + 37}" font-family="system-ui,sans-serif" '
        f'font-size="11" fill="{fg}" opacity="0.9">{html.escape(sub)}</text></g>'
    )


def _curve(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    return (f'<path d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" '
            f'fill="none" stroke="#38BDF8" stroke-opacity="0.55" stroke-width="1.8" '
            f'marker-end="url(#arr)"/>')


def _svg_wrap(width, height, body):
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="background:#0F172A;border:1px solid #1E293B;border-radius:12px;">'
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/></marker></defs>'
        f'{body}</svg>'
    )


def render_tx_flow(tx_id, max_side=6):
    e = D["edges"]
    senders = e.loc[e["txId2"] == tx_id, "txId1"].tolist()
    receivers = e.loc[e["txId1"] == tx_id, "txId2"].tolist()
    s_show, r_show = senders[:max_side], receivers[:max_side]
    n_rows = max(len(s_show), len(r_show), 1)
    nw, nh, gap = 180, 48, 14
    height = max(145, n_rows * (nh + gap) + 24)
    width = 680
    cx, cy = (width - nw) / 2, (height - nh) / 2

    parts = []
    s_top = (height - (len(s_show) * (nh + gap) - gap)) / 2 if s_show else 0
    for i, sid in enumerate(s_show):
        sy = s_top + i * (nh + gap)
        parts.append(_curve(20 + nw, sy + nh / 2, cx, cy + nh / 2))
        parts.append(_svg_node(20, sy, nw, nh, sid))

    r_top = (height - (len(r_show) * (nh + gap) - gap)) / 2 if r_show else 0
    rx = width - nw - 20
    for i, rid in enumerate(r_show):
        ry = r_top + i * (nh + gap)
        parts.append(_curve(cx + nw, cy + nh / 2, rx, ry + nh / 2))
        parts.append(_svg_node(rx, ry, nw, nh, rid))

    parts.append(_svg_node(cx, cy, nw, nh, tx_id, centre=True))
    components.html(_svg_wrap(width, height, "".join(parts)), height=height + 8)
    if len(senders) > max_side or len(receivers) > max_side:
        st.caption(f"Showing up to {max_side} senders/receivers (total: {len(senders)} in, {len(receivers)} out).")


def render_chain_flow(members_df, max_nodes=24):
    sub = members_df.sort_values([TIME, "risk_score"], ascending=[True, False]).head(max_nodes)
    id_set = set(sub["txId"].tolist())
    e = D["edges"]
    e = e[e["txId1"].isin(id_set) & e["txId2"].isin(id_set)]

    steps = sorted(sub[TIME].unique())
    nw, nh, col_gap, row_gap = 170, 48, 55, 14
    by_step = {s: sub.loc[sub[TIME] == s, "txId"].tolist() for s in steps}
    max_rows = max(len(v) for v in by_step.values())
    width = max(680, len(steps) * (nw + col_gap) + 40)
    height = max(170, max_rows * (nh + row_gap) + 30)

    pos = {}
    for ci, s in enumerate(steps):
        col_ids = by_step[s]
        x = 20 + ci * (nw + col_gap)
        top = (height - (len(col_ids) * (nh + row_gap) - row_gap)) / 2
        for ri, tid in enumerate(col_ids):
            pos[tid] = (x, top + ri * (nh + row_gap))

    parts = []
    for a, b in e.itertuples(index=False):
        if a in pos and b in pos:
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            parts.append(_curve(x1 + nw, y1 + nh / 2, x2, y2 + nh / 2))
    for tid, (x, y) in pos.items():
        parts.append(_svg_node(x, y, nw, nh, tid))
    disp_h = min(height, 430)
    components.html(_svg_wrap(width, height, "".join(parts)), height=disp_h + 8)


def legend():
    st.caption("🔴 High risk  ·  🟠 Medium risk  ·  🟢 Low risk  ·  ⚪ Earlier window  ·  Hover any node for details.")


# --------------------------------------------------------------------------- #
# Page 1: Command Center
# --------------------------------------------------------------------------- #
def page_home():
    render_page_header(
        "Executive Telemetry",
        "Command Center",
        "Overview of scanned Bitcoin traffic, high-risk alert patterns, and detection architecture.",
    )

    vp, vb = SUM["verified_precision_final"], SUM["verified_precision_baseline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Scanned", f"{SUM['test_transactions']:,}", "Time windows 35–49", delta_color="off")
    c2.metric("High-Risk Alerts", f"{SUM['flagged']:,}", f"{SUM['flag_rate_%']:.2f}% flag rate", delta_color="off")
    c3.metric("Verified Hit Rate", f"{vp[0]:.0%}", f"vs {vb[0]:.0%} amount baseline")
    c4.metric("Lift Over Random", f"{SUM['PR-AUC_final'] / SUM['PR-AUC_random']:.1f}×",
              f"PR-AUC {SUM['PR-AUC_final']:.3f} vs {SUM['PR-AUC_random']:.3f}")

    st.write("")
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Traffic & Alert Volume by Window")
            view_mode = st.segmented_control(
                "View", ["Alerts Only", "Alert Rate (%)", "Total Traffic"],
                default="Alerts Only", label_visibility="collapsed",
            )
            step_vol = S.groupby(TIME).agg(Total=("txId", "size"), Flagged=("flagged", "sum")).reset_index()
            step_vol["Rate_%"] = (step_vol["Flagged"] / step_vol["Total"] * 100).round(2)
            tips = [
                alt.Tooltip(f"{TIME}:O", title="Window"),
                alt.Tooltip("Total:Q", title="Total Txs", format=","),
                alt.Tooltip("Flagged:Q", title="High-Risk Alerts", format=","),
                alt.Tooltip("Rate_%:Q", title="Alert Rate (%)", format=".2f"),
            ]
            x_enc = alt.X(f"{TIME}:O", title="Time Window (~2 weeks each)", axis=alt.Axis(labelAngle=0))

            if view_mode == "Alert Rate (%)":
                ch = alt.Chart(step_vol).mark_area(
                    line={"color": "#38BDF8", "strokeWidth": 2.5},
                    point={"color": "#38BDF8", "size": 55},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[alt.GradientStop(color="rgba(56,189,248,0.35)", offset=0),
                               alt.GradientStop(color="rgba(56,189,248,0.02)", offset=1)],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                ).encode(x=x_enc, y=alt.Y("Rate_%:Q", title="Alert Rate (%)"), tooltip=tips)
            else:
                y_col = "Flagged" if view_mode != "Total Traffic" else "Total"
                y_title = "High-Risk Alerts" if y_col == "Flagged" else "Total Transactions"
                bar_color = "#EF4444" if y_col == "Flagged" else "#38BDF8"
                bars = alt.Chart(step_vol).mark_bar(
                    color=bar_color, cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
                ).encode(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_title), tooltip=tips)
                labels = bars.mark_text(dy=-8, fontSize=11, color="#E2E8F0").encode(
                    text=alt.Text(f"{y_col}:Q", format=","),
                )
                ch = bars + labels
            st.altair_chart(ch.properties(height=300), use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Amount (BTC) vs. Model Risk")
            lvl_pick = st.segmented_control(
                "Risk filter", ["All", "🔴 High", "🟠 Medium", "🟢 Low"],
                default="All", label_visibility="collapsed",
            )
            lvl_map = {"🔴 High": "High", "🟠 Medium": "Medium", "🟢 Low": "Low"}
            if lvl_pick in lvl_map:
                sub = S[S["risk_level"] == lvl_map[lvl_pick]]
                samp = sub.sample(n=min(800, len(sub)), random_state=42).copy()
            else:
                # Balanced sample so High-risk flows are never buried under Low-risk bulk
                parts = [
                    S[S["risk_level"] == lv].sample(
                        n=min(400, int((S["risk_level"] == lv).sum())), random_state=42
                    )
                    for lv in ["Low", "Medium", "High"]
                ]
                samp = pd.concat(parts)

            samp["BTC_clip"] = samp[AMOUNT].clip(lower=1e-4)
            amt_99 = float(S[AMOUNT].quantile(0.99))
            pts = alt.Chart(samp).mark_circle(size=42, opacity=0.75).encode(
                x=alt.X("BTC_clip:Q", scale=alt.Scale(type="log"), title="Transaction Amount (BTC, log scale)"),
                y=alt.Y("risk_score:Q", scale=alt.Scale(domain=[0, 102]), title="Risk Score (0–100)"),
                color=alt.Color(
                    "risk_level:N", legend=None,
                    scale=alt.Scale(domain=["High", "Medium", "Low"], range=["#EF4444", "#F59E0B", "#22C55E"]),
                ),
                tooltip=[
                    alt.Tooltip("txId:Q", title="TxID"),
                    alt.Tooltip(f"{TIME}:Q", title="Window"),
                    alt.Tooltip(f"{AMOUNT}:Q", title="Amount (BTC)", format=",.4f"),
                    alt.Tooltip("risk_score:Q", title="Risk Score", format=".1f"),
                    alt.Tooltip("Level:N", title="Tier"),
                    alt.Tooltip("Pattern:N", title="Pattern"),
                ],
            )
            rule = alt.Chart(pd.DataFrame({"x": [amt_99]})).mark_rule(
                strokeDash=[5, 4], color="#94A3B8", strokeWidth=1.5,
            ).encode(x="x:Q")
            st.altair_chart((pts + rule).properties(height=300), use_container_width=True)
            st.caption("Dashed line = 99th-percentile BTC amount cutoff. High-risk flows cluster at ordinary BTC sizes.")

    st.divider()
    b_left, b_right = st.columns([2, 1.5], vertical_alignment="center")
    section = b_left.segmented_control(
        "Bottom section", ["Top 5 Urgent Alerts", "How It Works"],
        default="Top 5 Urgent Alerts", label_visibility="collapsed",
    )
    with b_right:
        q1, q2 = st.columns(2)
        q1.page_link(PAGES["queue"], label="Full Queue", icon=":material/arrow_forward:", use_container_width=True)
        q2.page_link(PAGES["chains"], label=f"Chains ({SUM['chains']})", icon=":material/hub:", use_container_width=True)

    if section == "How It Works":
        a, b, c = st.columns(3, gap="medium")
        with a:
            st.markdown('<div class="card"><h4>1. Normal Behaviour Model</h4>'
                        '<p>Legitimate users send varied amounts with diverse input/output shapes. '
                        'We learn this baseline diversity from past traffic (windows 1–34).</p></div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="card"><h4>2. Look-Alike Burst Detection</h4>'
                        '<p>Automated criminal tools create dense bursts of twin transactions in the same window. '
                        'Per-window crowding surfaces them without labels.</p></div>', unsafe_allow_html=True)
        with c:
            st.markdown(f'<div class="card"><h4>3. Graph & Feedback ({SUM["labels_used"]} labels)</h4>'
                        f'<p>50 behaviour families calibrated with ~10 past checks each, combined with '
                        f'money-flow graph links ({SUM["chains"]} connected chains).</p></div>', unsafe_allow_html=True)
    else:
        top5 = S.head(5)[["rank", "Level", "txId", TIME, AMOUNT, "risk_score", "Pattern", "Flow", "Main reason"]]
        st.dataframe(
            top5, hide_index=True, use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank"),
                "Level": st.column_config.TextColumn("Tier"),
                "txId": st.column_config.TextColumn("Transaction ID"),
                TIME: st.column_config.NumberColumn("Window"),
                AMOUNT: st.column_config.NumberColumn("Amount (BTC)", format="%.4f"),
                "risk_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=100, format="%.1f"),
                "Main reason": st.column_config.TextColumn("Why Flagged", width="large"),
            },
        )


# --------------------------------------------------------------------------- #
# Page 2: Ranked Flows (Investigator Workbench)
# --------------------------------------------------------------------------- #
def page_queue():
    render_page_header(
        "Investigator Workbench",
        "Ranked Flows",
        "Statistically unusual Bitcoin flows ranked from #1 downward, plus the stratified Top-500 verification audit.",
    )

    tab_rank, tab_audit = st.tabs(["Ranked Anomaly List", "Verified Sample Audit (Top 500)"])

    with tab_rank:
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([2, 2, 2, 2.4])
            top_k = f1.selectbox("Scope", ["Top 50", "Top 100", "Top 500", "All", "🔴 High Only", "🟠 Medium Only", "🟢 Low Only"])
            t0, t1 = f2.slider("Time windows", 35, 49, (35, 49))
            sort_by = f3.selectbox("Sort by", ["Anomaly score", "Look-alike burst", "Amount (BTC)", "Chain size"])

        v = S[S[TIME].between(t0, t1)]
        tier_map = {"🔴 High Only": "High", "🟠 Medium Only": "Medium", "🟢 Low Only": "Low"}
        if top_k in tier_map:
            v = v[v["risk_level"] == tier_map[top_k]]

        sort_col = {"Anomaly score": "risk_score", "Look-alike burst": "lookalike_pct_step",
                    "Amount (BTC)": AMOUNT, "Chain size": "chain_size"}[sort_by]
        v = v.sort_values([sort_col, "risk_score"], ascending=False)
        if top_k.startswith("Top "):
            v = v.head(int(top_k.split()[1]))

        pick_map = {
            int(r["txId"]): f"#{int(r['rank'])} · {int(r['txId'])} ({r['risk_score']:.0f}/100 · {r[AMOUNT]:.2f} BTC)"
            for _, r in v.head(500).iterrows()
        }
        picked_tx = f4.selectbox(
            "Jump to Transaction", [None] + list(pick_map.keys()),
            format_func=lambda x: "All in scope (type or pick)..." if x is None else pick_map[x],
        )
        if picked_tx is not None:
            v = v[v["txId"] == picked_tx]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flows Shown", f"{len(v):,}")
        m2.metric("Mean Anomaly Score", f"{v['risk_score'].mean():.1f} / 100" if len(v) else "–")
        m3.metric("In Connected Chains", f"{int((v['chain_size'] > 1).sum()):,}")
        m4.metric("Median BTC", f"{v[AMOUNT].median():.4f}" if len(v) else "–")

        if v.empty:
            st.info("No flows match these filters.")
            return

        cols = ["rank", "Level", "txId", TIME, AMOUNT, "risk_score", "Pattern", "Flow", "Main reason"]
        event = st.dataframe(
            v[cols].head(1000), hide_index=True, use_container_width=True, height=350,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "rank": st.column_config.NumberColumn("Rank"),
                "Level": st.column_config.TextColumn("Tier"),
                "txId": st.column_config.TextColumn("Transaction ID"),
                TIME: st.column_config.NumberColumn("Window"),
                AMOUNT: st.column_config.NumberColumn("Amount (BTC)", format="%.4f"),
                "risk_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=100, format="%.1f"),
                "Main reason": st.column_config.TextColumn("Why Flagged", width="large"),
            },
        )

        e1, e2 = st.columns(2)
        sar_bundle = "\n---\n\n".join(build_sar_text(row) for _, row in v.head(25).iterrows())
        e1.download_button("Export Top 25 Case Briefs (.md)", sar_bundle, "top25_case_briefs.md",
                           "text/markdown", icon=":material/description:", use_container_width=True)
        e2.download_button("Export Ranked Flows (.csv)", v[cols].to_csv(index=False),
                           "ranked_flows.csv", "text/csv", icon=":material/download:", use_container_width=True)

        rows = event.selection.rows
        if not rows:
            st.info("Click any row above to inspect its signal breakdown and money-flow graph.")
        else:
            detail(v.iloc[rows[0]])

    with tab_audit:
        st.subheader("Stratified Verification Sample (100 Flows from Top 500)")
        st.write("Stratified 100-flow audit sampled across rank bands (`1–50`, `51–200`, `201–500`) of the "
                 "labelled Top-500 ranked list for both our detector and the mandatory **Amount Threshold** baseline.")
        vs = D["vsum"].rename(columns={
            "method": "Method", "band": "Rank Band", "sampled": "Sampled",
            "confirmable": "Confirmable", "confirmed_illicit": "Confirmed Illicit",
            "precision": "Precision", "ci_low": "95% CI Low", "ci_high": "95% CI High",
        })
        st.dataframe(
            vs, hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ["Precision", "95% CI Low", "95% CI High"]},
        )
        st.subheader("Sampled Transactions")
        vpick = st.segmented_control("Filter sample", ["Our model", "Amount threshold (baseline)"],
                                     default="Our model", label_visibility="collapsed")
        vsub = D["vsample"][D["vsample"]["method"] == vpick]
        st.dataframe(vsub.drop(columns=["method"]), hide_index=True, use_container_width=True, height=340)


def detail(r):
    tx_id = int(r["txId"])
    st.divider()
    st.subheader(f"Flow Inspector · Tx {tx_id}")

    a, b, c, d = st.columns(4)
    a.metric("Anomaly Score", f"{r['risk_score']:.1f} / 100")
    b.metric("Risk Tier", LEVEL_ICON[r["risk_level"]])
    c.metric("Amount Moved", f"{r[AMOUNT]:,.4f} BTC")
    d.metric("Global Rank", f"#{int(r['rank']):,}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Why It Was Surfaced")
        for reason in r["why_flagged"].split(" | "):
            st.markdown(f"- {reason}")

        sig_df = pd.DataFrame({
            "Signal": ["Look-alike burst", "Behaviour group", "Shape rarity", "BTC amount"],
            "Score": [round(r["lookalike_pct_step"] * 100, 1), round(r["family_risk"] * 100, 1),
                      round(r["point_pct"] * 100, 1), round(r["amount_pct"] * 100, 1)],
        })
        bar = alt.Chart(sig_df).mark_bar(cornerRadiusEnd=5, color="#38BDF8").encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Signal Percentile (0–100)"),
            y=alt.Y("Signal:N", sort="-x", title=None),
            tooltip=["Signal", "Score"],
        ).properties(height=155)
        st.altair_chart(bar, use_container_width=True)

        st.markdown(
            f"**Metadata:** Window **{int(r[TIME])}** · Inputs **{int(r['num_input_addresses'])}** · "
            f"Outputs **{int(r['num_output_addresses'])}** · Fee **{r['fees']:.6f} BTC**"
        )
        if r["chain_size"] > 1:
            st.warning(f"Linked to **Suspicious Chain #{int(r['chain_id'])}** ({int(r['chain_size'])} flagged nodes).")

        e1, e2 = st.columns(2)
        case_json = json.dumps({
            "txId": tx_id, "time_window": int(r[TIME]), "anomaly_score": float(r["risk_score"]),
            "risk_tier": r["risk_level"], "pattern": r["Pattern"],
            "total_BTC": float(r[AMOUNT]), "reasons": r["why_flagged"].split(" | "),
            "chain_id": None if pd.isna(r["chain_id"]) else int(r["chain_id"]),
        }, indent=2)
        e1.download_button("Case JSON", case_json, f"case_{tx_id}.json", "application/json",
                           icon=":material/code:", use_container_width=True)
        e2.download_button("SAR Brief (.md)", build_sar_text(r), f"SAR_{tx_id}.md", "text/markdown",
                           icon=":material/description:", use_container_width=True)

    with right:
        st.markdown("##### Money-Flow Neighbourhood")
        e = D["edges"]
        if not ((e["txId2"] == tx_id).any() or (e["txId1"] == tx_id).any()):
            st.info("No direct transaction links in the dataset graph.")
        else:
            render_tx_flow(tx_id)
            legend()


# --------------------------------------------------------------------------- #
# Page 3: Suspicious Chains
# --------------------------------------------------------------------------- #
def page_chains():
    render_page_header(
        "Multi-Hop Contagion",
        "Suspicious Chains",
        "Connected clusters of high-risk transactions passing funds directly to each other.",
    )
    ch = D["chains"].copy()
    if ch.empty:
        st.info("No multi-transaction chains found.")
        return
    ch["duration"] = (ch["last_step"] - ch["first_step"] + 1).clip(lower=1)
    ch["velocity_btc"] = (ch["total_BTC"] / ch["duration"]).round(3)

    a, b, c = st.columns(3)
    a.metric("Chains Detected", f"{len(ch):,}")
    b.metric("Largest Chain", f"{int(ch['transactions'].max())} nodes")
    c.metric("Chains With Confirmed Illicit", f"{int((ch['confirmed_illicit'] > 0).sum())}")

    show = ch.rename(columns={
        "chain_id": "Chain", "transactions": "Nodes", "first_step": "Start", "last_step": "End",
        "duration": "Windows", "velocity_btc": "BTC / Window",
        "total_BTC": "Total BTC", "top_risk": "Max Risk", "avg_risk": "Avg Risk",
        "confirmed_illicit": "Criminal", "confirmed_legit": "Legitimate",
    })
    event = st.dataframe(
        show, hide_index=True, use_container_width=True, height=290,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Avg Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Total BTC": st.column_config.NumberColumn(format="%.3f"),
            "Max Risk": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    if not event.selection.rows:
        st.info("Click any chain above to inspect its flow graph.")
        return

    cid = ch.iloc[event.selection.rows[0]]["chain_id"]
    members = S[S["chain_id"] == cid].sort_values([TIME, "risk_score"], ascending=[True, False])
    st.subheader(f"Chain #{int(cid)} · {len(members)} connected transactions")
    render_chain_flow(members)
    legend()
    st.download_button(f"Export Chain #{int(cid)} CSV", members.to_csv(index=False),
                       f"chain_{int(cid)}.csv", "text/csv")
    st.dataframe(
        members[["rank", "Level", "txId", TIME, AMOUNT, "risk_score", "Pattern", "Main reason"]],
        hide_index=True, use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "Level": st.column_config.TextColumn("Tier"),
            "txId": st.column_config.TextColumn("Transaction ID"),
            TIME: st.column_config.NumberColumn("Window"),
            AMOUNT: st.column_config.NumberColumn("Amount (BTC)", format="%.4f"),
            "risk_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=100, format="%.1f"),
            "Main reason": st.column_config.TextColumn("Why Flagged", width="large"),
        },
    )


# --------------------------------------------------------------------------- #
# Page 4: Model & Drift Lab
# --------------------------------------------------------------------------- #
def page_proof():
    render_page_header(
        "Backtesting & Validation",
        "Model & Drift Lab",
        "Evaluation on untouched test windows 35–49, alert-budget trade-offs, and drift resilience.",
    )

    st.subheader("1. Alert Budget Simulator")
    with st.container(border=True):
        pct_cut = st.slider(
            "Alert threshold (flag top % of traffic)", 0.2, 10.0, 1.0, step=0.2,
            help="Slide to see the trade-off between investigator workload and criminals caught.",
        )
        cut_score = float(np.quantile(S["risk_score"], 1 - pct_cut / 100))
        sim_flag = S["risk_score"] >= cut_score
        sim_lab = S[sim_flag & S["y"].notna()]
        tot_illicit = int((S["y"] == 1).sum())
        caught = int((sim_lab["y"] == 1).sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Risk Cutoff", f"{cut_score:.1f} / 100")
        s2.metric("Alerts Generated", f"{int(sim_flag.sum()):,}", f"{sim_flag.mean():.2%} of traffic", delta_color="off")
        s3.metric("Known Precision", f"{sim_lab['y'].mean():.1%}" if len(sim_lab) else "–")
        s4.metric("Criminals Caught", f"{caught} / {tot_illicit}", f"{caught / max(tot_illicit, 1):.1%} recall")

    st.divider()
    st.subheader("2. Detection Accuracy & Verification")
    c_left, c_right = st.columns(2)

    with c_left:
        st.caption("**Top-K Precision** — Share of top alerts that are truly illicit")
        m = D["metrics"].copy()
        pick = m[m["model"].str.contains("FINAL|Amount|Random")].copy()
        pick["Model"] = np.where(
            pick["model"].str.contains("FINAL"), "Hybrid Model",
            np.where(pick["model"].str.contains("Amount"), "Amount Baseline", "Random"),
        )
        topk = pick.melt("Model", ["P@50", "P@100", "P@500"], var_name="Cutoff", value_name="Precision")
        lift = alt.Chart(topk).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Cutoff:N", sort=["P@50", "P@100", "P@500"], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Precision:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.85])),
            xOffset="Model:N",
            color=alt.Color(
                "Model:N",
                legend=alt.Legend(orient="top", title=None),
                scale=alt.Scale(
                    domain=["Hybrid Model", "Amount Baseline", "Random"],
                    range=["#38BDF8", "#64748B", "#334155"],
                ),
            ),
            tooltip=["Model", "Cutoff", alt.Tooltip("Precision:Q", format=".1%")],
        ).properties(height=270)
        st.altair_chart(lift, use_container_width=True)

    with c_right:
        st.caption("**Stratified Audit** — Confirmed illicit rate across 100 sampled alerts")
        vs = D["vsum"].copy()
        vs["Method"] = np.where(vs["method"].str.contains("Our"), "Hybrid Model", "Amount Baseline")
        band_order = ["Rank 1-50", "Rank 51-200", "Rank 201-500", "All sampled"]
        audit = alt.Chart(vs).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("band:N", sort=band_order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("precision:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.85])),
            xOffset="Method:N",
            color=alt.Color(
                "Method:N",
                legend=alt.Legend(orient="top", title=None),
                scale=alt.Scale(domain=["Hybrid Model", "Amount Baseline"], range=["#38BDF8", "#64748B"]),
            ),
            tooltip=[
                "Method",
                alt.Tooltip("band:N", title="Band"),
                alt.Tooltip("confirmed_illicit:Q", title="Confirmed Illicit"),
                alt.Tooltip("sampled:Q", title="Sampled"),
                alt.Tooltip("precision:Q", title="Precision", format=".1%"),
            ],
        ).properties(height=270)
        st.altair_chart(audit, use_container_width=True)

    with st.expander("View full model benchmark table"):
        nice_m = m[["model", "P@50", "P@100", "P@500", "PR-AUC", "recall_at_flag_%", "labels_used"]].rename(columns={
            "model": "Model",
            "recall_at_flag_%": "Recall @ Flag %",
            "labels_used": "Labels Used",
        })
        st.dataframe(
            nice_m, hide_index=True, use_container_width=True,
            column_config={
                "P@50": st.column_config.NumberColumn(format="percent"),
                "P@100": st.column_config.NumberColumn(format="percent"),
                "P@500": st.column_config.NumberColumn(format="percent"),
                "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                "Recall @ Flag %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    st.divider()
    st.subheader("3. Concept Drift & Label Efficiency")
    d1, d2 = st.columns([3, 2])

    with d1:
        st.caption("**Performance Over Time** — Dark-market shutdown at window 43 causes severe drift")
        ps = D["per_step"].rename(columns={
            "time_step": "Window", "PR-AUC final": "Hybrid Model", "PR-AUC amount": "Amount Baseline",
        })
        long = ps.melt("Window", ["Hybrid Model", "Amount Baseline"], var_name="Approach", value_name="PR-AUC")
        color_scale = alt.Scale(domain=["Hybrid Model", "Amount Baseline"], range=["#38BDF8", "#64748B"])
        area = alt.Chart(long[long["Approach"] == "Hybrid Model"]).mark_area(
            opacity=0.16, color="#38BDF8"
        ).encode(x="Window:O", y="PR-AUC:Q")
        line = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Window:O", title="Time Window"),
            y=alt.Y("PR-AUC:Q", title="PR-AUC"),
            color=alt.Color("Approach:N", legend=alt.Legend(orient="top", title=None), scale=color_scale),
            tooltip=["Window", "Approach", alt.Tooltip("PR-AUC:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [43]})).mark_rule(
            strokeDash=[5, 4], strokeWidth=1.8, color="#EF4444"
        ).encode(x="x:O")
        st.altair_chart((area + line + rule).properties(height=265), use_container_width=True)

    with d2:
        st.caption(f"**Label Efficiency** — Gains plateau at ~{SUM['labels_used']} labels (1.7% of data)")
        b = D["budget"].copy()
        b_area = alt.Chart(b).mark_area(opacity=0.18, color="#38BDF8", clip=True).encode(
            x=alt.X("total_labels:Q", title="Verified Past Labels"),
            y=alt.Y("PR-AUC mean:Q", title="PR-AUC", scale=alt.Scale(domain=[0.32, 0.45])),
            y2=alt.datum(0.32),
        )
        b_line = alt.Chart(b).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5, color="#38BDF8").encode(
            x=alt.X("total_labels:Q", title="Verified Past Labels"),
            y=alt.Y("PR-AUC mean:Q", title="PR-AUC", scale=alt.Scale(domain=[0.32, 0.45])),
            tooltip=[
                alt.Tooltip("total_labels:Q", title="Labels Used"),
                alt.Tooltip("PR-AUC mean:Q", title="PR-AUC", format=".3f"),
            ],
        )
        sweet = alt.Chart(pd.DataFrame({"x": [SUM["labels_used"]]})).mark_rule(
            strokeDash=[4, 4], color="#22C55E", strokeWidth=1.5
        ).encode(x="x:Q")
        st.altair_chart((b_area + b_line + sweet).properties(height=265), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 5: Scan Your CSV (Live ML Inference)
# --------------------------------------------------------------------------- #
def page_scan():
    render_page_header(
        "Live ML Screening Engine",
        "Scan Your CSV",
        "Upload any Bitcoin transaction CSV to run the trained 182-feature anomaly-detection pipeline live.",
    )

    bundle = load_bundle()
    if bundle is None:
        st.error("Model bundle not found (`results/model_bundle.joblib`). Run `python src/pipeline.py` first.")
        return

    sample_path = RES / "sample_scan.csv"
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        up = c1.file_uploader(
            "Upload transaction CSV (full 182-feature Elliptic++ format or simplified columns like `txId`, `Time step`, `total_BTC`, `fees`, `num_input_addresses`, `num_output_addresses`)",
            type=["csv"],
        )
        c2.write("")
        run_demo = c2.button("Run Demo on Sample (60 Txs)", icon=":material/play_arrow:", use_container_width=True)
        if sample_path.exists():
            c2.download_button("Download Sample CSV", sample_path.read_bytes(),
                               "sample_bitcoin_txs.csv", "text/csv", icon=":material/download:",
                               use_container_width=True)

    if up is not None:
        st.session_state["scan_df"] = pd.read_csv(up)
    elif run_demo and sample_path.exists():
        st.session_state["scan_df"] = pd.read_csv(sample_path)

    df_in = st.session_state.get("scan_df")
    if df_in is None:
        st.info("Upload a CSV above or click **Run Demo on Sample (60 Txs)** to test the live ML scanner.")
        return

    with st.spinner("Running trained scaler, PCA, Isolation Forest, K-Means & look-alike burst detector..."):
        scored, n_feats = score_dataframe(df_in, bundle)

    scored["Tier"] = scored["risk_level"].map(LEVEL_ICON)
    scored.insert(0, "Rank", np.arange(1, len(scored) + 1))

    if n_feats == len(bundle["feats"]):
        st.success(f"Scanned **{len(scored):,}** transactions using all **{n_feats}** trained features.")
    else:
        st.info(f"Scanned **{len(scored):,}** transactions (**{n_feats}/{len(bundle['feats'])}** Elliptic++ features "
                "matched; remaining features imputed with training medians).")

    n_hi = int((scored["risk_level"] == "High").sum())
    n_med = int((scored["risk_level"] == "Medium").sum())
    n_lo = int((scored["risk_level"] == "Low").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions Scanned", f"{len(scored):,}")
    k2.metric("🔴 High Risk", f"{n_hi:,}", f"{n_hi / max(len(scored), 1):.1%}", delta_color="off")
    k3.metric("🟠 Medium Watchlist", f"{n_med:,}", f"{n_med / max(len(scored), 1):.1%}", delta_color="off")
    k4.metric("🟢 Low / Normal", f"{n_lo:,}", f"{n_lo / max(len(scored), 1):.1%}", delta_color="off")

    ch_l, ch_r = st.columns(2)
    with ch_l:
        with st.container(border=True):
            st.caption("**Risk Tier Distribution**")
            counts = pd.DataFrame({"Tier": ["High", "Medium", "Low"], "Count": [n_hi, n_med, n_lo]})
            bar = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=50).encode(
                x=alt.X("Tier:N", sort=["High", "Medium", "Low"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Transactions"),
                color=alt.Color("Tier:N", legend=None,
                                scale=alt.Scale(domain=["High", "Medium", "Low"], range=["#EF4444", "#F59E0B", "#22C55E"])),
                tooltip=["Tier", "Count"],
            ).properties(height=210)
            st.altair_chart(bar, use_container_width=True)

    with ch_r:
        with st.container(border=True):
            st.caption("**Detected Patterns**")
            pats = scored["Pattern"].value_counts().reset_index()
            pats.columns = ["Pattern", "Count"]
            pbar = alt.Chart(pats).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#38BDF8", size=46).encode(
                x=alt.X("Pattern:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Transactions"),
                tooltip=["Pattern", "Count"],
            ).properties(height=210)
            st.altair_chart(pbar, use_container_width=True)

    tier_pick = st.segmented_control(
        "Filter by tier", ["All", "🔴 High", "🟠 Medium", "🟢 Low"],
        default="All", label_visibility="collapsed",
    )
    tier_map = {"🔴 High": "High", "🟠 Medium": "Medium", "🟢 Low": "Low"}
    view_df = scored[scored["risk_level"] == tier_map[tier_pick]] if tier_pick in tier_map else scored

    show_cols = ["Rank", "Tier", "txId", TIME, AMOUNT, "risk_score", "Pattern", "Why Flagged"]
    st.dataframe(
        view_df[show_cols], hide_index=True, use_container_width=True, height=380,
        column_config={
            "txId": st.column_config.TextColumn("Transaction ID"),
            TIME: st.column_config.NumberColumn("Window"),
            AMOUNT: st.column_config.NumberColumn("Amount (BTC)", format="%.4f"),
            "risk_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=100, format="%.1f"),
            "Why Flagged": st.column_config.TextColumn("Why Flagged", width="large"),
        },
    )
    st.download_button("Download Scored Results (.csv)", view_df.to_csv(index=False),
                       "scanned_transactions.csv", "text/csv")


PROFILE_STORE = ROOT / ".profiles.json"
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30


def _hash_pw(pw: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return f"{salt}${digest}"


def _verify_pw(pw: str, stored: str) -> bool:
    if "$" in stored:
        salt, expected = stored.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
        return hmac.compare_digest(actual, expected)
    legacy = hashlib.sha256(pw.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, stored)


def _validate_password(pw: str) -> str | None:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isalpha() for c in pw) or not any(c.isdigit() for c in pw):
        return "Password must include at least one letter and one number."
    return None


def _load_profiles() -> dict:
    if "profiles" not in st.session_state:
        data = {}
        if PROFILE_STORE.exists():
            try:
                data = json.loads(PROFILE_STORE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        st.session_state["profiles"] = data
    return st.session_state["profiles"]


def _save_profiles(profiles: dict) -> None:
    st.session_state["profiles"] = profiles
    try:
        PROFILE_STORE.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    except OSError:
        pass


def page_login():
    _, center_col, _ = st.columns([0.9, 1.35, 0.9])

    with center_col:
        st.markdown(
            '<div class="vault-wrap">'
            '<div class="vault-aura"></div>'
            '<div class="vault-orbit-outer">'
            '<div class="vault-orbit-inner">'
            '<div class="vault-coin">&#8383;</div>'
            '</div>'
            '</div>'
            '<div class="vault-title">Bitcoin Transaction Anomaly &amp; Security Scanner</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            tab_create, tab_signin = st.tabs(["Create Profile", "Sign In"])

            with tab_create:
                with st.form("create_profile_form"):
                    name = st.text_input("Full Name", placeholder="e.g. Naina Sharma")
                    username = st.text_input("Email or Username", placeholder="e.g. naina@btc-scanner.io")
                    role = st.selectbox(
                        "Role",
                        [
                            "Transaction Investigator",
                            "Blockchain Security Engineer",
                            "Exchange / Wallet Operator",
                            "Individual User",
                        ],
                    )
                    c_pw1, c_pw2 = st.columns(2)
                    password = c_pw1.text_input("Password", type="password", placeholder="8+ chars, alphanumeric")
                    confirm_pw = c_pw2.text_input("Confirm Password", type="password", placeholder="Repeat password")
                    submitted = st.form_submit_button(
                        "Create Profile & Open CSV Scanner", type="primary", use_container_width=True
                    )
                    if submitted:
                        uname = username.strip().lower()
                        pw_err = _validate_password(password)
                        if not name.strip() or not uname:
                            st.error("Please enter both your full name and a username or email.")
                        elif pw_err:
                            st.error(pw_err)
                        elif password != confirm_pw:
                            st.error("Passwords do not match.")
                        else:
                            profiles = _load_profiles()
                            if uname in profiles:
                                st.error("A profile with that username already exists. Switch to Sign In.")
                            else:
                                profile = {
                                    "name": name.strip(),
                                    "username": uname,
                                    "role": role,
                                    "password_hash": _hash_pw(password),
                                }
                                profiles[uname] = profile
                                _save_profiles(profiles)
                                st.session_state["user_profile"] = {
                                    "name": profile["name"],
                                    "username": uname,
                                    "role": role,
                                    "session_id": secrets.token_hex(4).upper(),
                                }
                                st.session_state["login_attempts"] = 0
                                st.rerun()

            with tab_signin:
                lock_until = st.session_state.get("lockout_until", 0.0)
                remaining = int(lock_until - time.time())
                if remaining > 0:
                    st.warning(f"Too many failed sign-in attempts. Please wait {remaining}s before trying again.")
                with st.form("signin_form"):
                    login_user = st.text_input("Work Email or Username", key="login_user")
                    login_pw = st.text_input("Password", type="password", key="login_pw")
                    signin_btn = st.form_submit_button(
                        "Sign In to Scanner", type="primary", use_container_width=True, disabled=remaining > 0
                    )
                    if signin_btn:
                        profiles = _load_profiles()
                        uname = login_user.strip().lower()
                        prof = profiles.get(uname)
                        if prof and _verify_pw(login_pw, prof["password_hash"]):
                            st.session_state["user_profile"] = {
                                "name": prof["name"],
                                "username": uname,
                                "role": prof["role"],
                                "session_id": secrets.token_hex(4).upper(),
                            }
                            st.session_state["login_attempts"] = 0
                            st.session_state.pop("lockout_until", None)
                            st.rerun()
                        else:
                            attempts = st.session_state.get("login_attempts", 0) + 1
                            st.session_state["login_attempts"] = attempts
                            if attempts >= MAX_LOGIN_ATTEMPTS:
                                st.session_state["lockout_until"] = time.time() + LOCKOUT_SECONDS
                                st.session_state["login_attempts"] = 0
                                st.error(f"Account temporarily locked for {LOCKOUT_SECONDS}s after {MAX_LOGIN_ATTEMPTS} failed attempts.")
                            else:
                                left_tries = MAX_LOGIN_ATTEMPTS - attempts
                                st.error(f"Invalid credentials. {left_tries} attempt(s) remaining before temporary lockout.")

            if st.button("Launch Instant Demo Sandbox", icon=":material/verified_user:", use_container_width=True):
                st.session_state["user_profile"] = {
                    "name": "Demo Analyst",
                    "username": "demo",
                    "role": "Sandbox Session",
                    "session_id": secrets.token_hex(4).upper(),
                }
                st.rerun()


if not st.session_state.get("user_profile"):
    st.navigation([st.Page(page_login, title="Sign In", default=True)], position="hidden").run()
else:
    user = st.session_state["user_profile"]
    sid = user.get("session_id", "ACTIVE")
    PAGES = {
        "scan": st.Page(page_scan, title="Scan Your CSV", icon=":material/upload_file:", default=True),
        "home": st.Page(page_home, title="Command Center", icon=":material/dashboard:", url_path="overview"),
        "queue": st.Page(page_queue, title="Ranked Flows", icon=":material/format_list_numbered:", url_path="queue"),
        "chains": st.Page(page_chains, title="Suspicious Chains", icon=":material/hub:", url_path="chains"),
        "proof": st.Page(page_proof, title="Model & Drift Lab", icon=":material/monitoring:", url_path="lab"),
    }
    with st.sidebar:
        with st.container(border=True):
            st.markdown(f"**{html.escape(user['name'])}**")
            st.caption(f"{html.escape(user['role'])} · ID `{sid}`")
        if st.button("Sign Out", icon=":material/logout:", use_container_width=True):
            st.session_state.pop("user_profile", None)
            st.rerun()
    nav = st.navigation(list(PAGES.values()))
    nav.run()
