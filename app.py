# -*- coding: utf-8 -*-
"""
Behçet Nomogram Calculator — Standalone
========================================
Major Organ Involvement (vasculitis/uveitis) risk prediction
from CBC components (Neutrophil, Monocyte, Lymphocyte).

Model: Firth penalized logistic regression, n=179, AUC 0.921.
No data is stored — calculations run entirely in the browser session.
"""

import streamlit as st
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# ============================================================
# MODEL COEFFICIENTS (Firth penalized logistic, n=179)
# P(complicated) = 1 / (1 + exp(-(b0 + b1·NEU + b2·MONO + b3·LYMPH)))
# All cell counts in 10^3/µL (×10^9/L)
# ============================================================
COEF = {
    "intercept": -4.631548,
    "NEU":        1.413616,
    "MONO":       7.276231,
    "LYMPH":     -2.823528,
}
MODEL_AUC = 0.921
MODEL_N = 179
RISK_CUTOFF = 0.575       # Youden J optimal
CUTOFF_SENS = 80.9
CUTOFF_SPEC = 91.1

# Variable ranges (for nomogram axes) — from training data
RANGES = {
    "NEU":   {"min": 2.26, "max": 14.51, "median": 5.11},
    "MONO":  {"min": 0.18, "max": 1.33,  "median": 0.50},
    "LYMPH": {"min": 1.07, "max": 5.99,  "median": 2.23},
}

# ============================================================
# UNIT CONVERSION
# Model trained on 10^3/µL. Convert user input to this base.
# ============================================================
UNIT_FACTORS = {
    "10³/µL  (×10⁹/L)": 1.0,        # base unit — no conversion
    "cells/µL  (/mm³)": 0.001,      # 4500 /µL → 4.5 ×10³/µL
    "10⁹/L": 1.0,                   # identical to 10³/µL
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Behçet Nomogram Calculator",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1a4d5c;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #5a7684; font-style: italic;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;
    }
    .risk-low  { background: linear-gradient(135deg,#d4f1d4,#b8e6b8); border-left: 6px solid #4caf50; }
    .risk-mid  { background: linear-gradient(135deg,#fff4d4,#ffe9a8); border-left: 6px solid #ff9800; }
    .risk-high { background: linear-gradient(135deg,#ffd4d4,#ffb8b8); border-left: 6px solid #d32f2f; }
    .prob-text { font-size: 3.2rem; font-weight: 700; color: #1a4d5c; line-height: 1; }
    .risk-label { font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem; }
    .disclaimer {
        background: #fff3e0; border-left: 4px solid #e65100; padding: 1rem;
        border-radius: 6px; margin-top: 2rem; font-size: 0.85rem; color: #5d4037;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🩸 Behçet Nomogram Calculator</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Complicated Behçet\'s disease (vasculitis / uveitis) '
    'risk prediction from CBC components</div>',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About this calculator"):
    st.markdown(f"""
    This tool estimates the probability that a patient with Behçet's disease has
    **Major Organ Involvement** (vascular and/or ocular involvement) versus
    **purely mucocutaneous disease**, based on three complete blood count (CBC)
    components: neutrophil, monocyte, and lymphocyte counts.

    **Model:** Firth penalized logistic regression (n = {MODEL_N}, AUC = {MODEL_AUC}).

    **How to use:**
    1. Select the unit your lab reports (default 10³/µL)
    2. Enter neutrophil, monocyte, and lymphocyte counts
    3. Read the predicted probability and the nomogram below

    This calculator does **not** diagnose Behçet's disease — it predicts whether
    established Behçet's disease is complicated. No data is stored.
    """)

# ============================================================
# UNIT SELECTION + INPUTS
# ============================================================
st.markdown("### Enter CBC values")

unit = st.selectbox(
    "Unit (applies to all three counts)",
    list(UNIT_FACTORS.keys()),
    index=0,
    help="Model is calibrated for 10³/µL. Other units are auto-converted."
)
factor = UNIT_FACTORS[unit]

# Adjust input defaults/ranges to the chosen unit so the fields feel natural
disp = 1.0 / factor   # multiply base value by this to show in chosen unit

c1, c2, c3 = st.columns(3)
with c1:
    neu_in = st.number_input(
        "Neutrophil (NEU)",
        min_value=0.0, max_value=100000.0,
        value=round(5.0 * disp, 3), step=round(0.1 * disp, 3),
        format="%.3f" if factor != 1.0 else "%.2f"
    )
with c2:
    mono_in = st.number_input(
        "Monocyte (MONO)",
        min_value=0.0, max_value=100000.0,
        value=round(0.5 * disp, 3), step=round(0.01 * disp, 4),
        format="%.3f" if factor != 1.0 else "%.2f"
    )
with c3:
    lymph_in = st.number_input(
        "Lymphocyte (LYMPH)",
        min_value=0.001, max_value=100000.0,
        value=round(2.0 * disp, 3), step=round(0.1 * disp, 3),
        format="%.3f" if factor != 1.0 else "%.2f"
    )

# Convert to base unit (10^3/µL)
neu = neu_in * factor
mono = mono_in * factor
lymph = lymph_in * factor

# Guard against zero lymphocyte
if lymph < 0.001:
    st.error("Lymphocyte count cannot be zero.")
    st.stop()

# Plausibility hint if values fall outside training range
warnings = []
for name, val in [("Neutrophil", neu), ("Monocyte", mono), ("Lymphocyte", lymph)]:
    key = {"Neutrophil": "NEU", "Monocyte": "MONO", "Lymphocyte": "LYMPH"}[name]
    r = RANGES[key]
    if val < r["min"] * 0.5 or val > r["max"] * 1.5:
        warnings.append(f"{name} ({val:.2f} ×10³/µL) is outside the model's training range "
                        f"({r['min']}–{r['max']}). Check the unit selection.")
if warnings:
    for w in warnings:
        st.warning("⚠️ " + w)

# ============================================================
# CALCULATE
# ============================================================
logit = (COEF["intercept"] + COEF["NEU"] * neu +
         COEF["MONO"] * mono + COEF["LYMPH"] * lymph)
prob = 1.0 / (1.0 + math.exp(-max(-30, min(30, logit))))

# Risk category
if prob < 0.30:
    rc, rl, emoji = "risk-low", "LOW RISK", "🟢"
    interp = ("The inflammatory profile is consistent with **purely mucocutaneous "
              "Behçet's disease**. Vascular/ocular involvement is unlikely.")
elif prob < 0.70:
    rc, rl, emoji = "risk-mid", "INTERMEDIATE RISK", "🟡"
    interp = ("Indeterminate range. **Clinical evaluation and imaging** are advised; "
              "this score alone is not decisive.")
else:
    rc, rl, emoji = "risk-high", "HIGH RISK", "🔴"
    interp = ("The inflammatory profile is highly consistent with **complicated "
              "Behçet's disease**. Detailed systemic assessment is advised.")

st.markdown(f"""
<div class="result-box {rc}">
    <div style="font-size:1rem;color:#555;margin-bottom:0.5rem;">
        Predicted probability of complicated Behçet's disease
    </div>
    <div class="prob-text">{prob:.1%}</div>
    <div class="risk-label">{emoji} {rl}</div>
    <div style="margin-top:1rem;color:#444;">{interp}</div>
</div>
""", unsafe_allow_html=True)

# Cut-off reference
above = "above" if prob >= RISK_CUTOFF else "below"
st.caption(
    f"Optimal cut-off (Youden J): {RISK_CUTOFF:.2f} · "
    f"sensitivity {CUTOFF_SENS:.0f}%, specificity {CUTOFF_SPEC:.0f}%. "
    f"This patient is **{above}** the cut-off."
)

# ============================================================
# NOMOGRAM
# ============================================================
st.markdown("### Nomogram")

def build_nomogram(highlight=None):
    """highlight: dict of {NEU, MONO, LYMPH} in base units to mark on axes."""
    predictors = [
        ("NEU",   COEF["NEU"],   RANGES["NEU"]),
        ("MONO",  COEF["MONO"],  RANGES["MONO"]),
        ("LYMPH", COEF["LYMPH"], RANGES["LYMPH"]),
    ]
    # points scaling: strongest contributor spans 100 points
    ranges_pts = []
    for name, b, r in predictors:
        rng = abs(b * (r["max"] - r["min"]))
        ranges_pts.append(rng)
    max_range = max(ranges_pts)
    scale = 100.0 / max_range
    total_max = sum(ranges_pts) * scale

    n_axes = 1 + len(predictors) + 2
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 0.85 * n_axes + 1.2))
    plt.subplots_adjust(hspace=1.3, left=0.16, right=0.96, top=0.90, bottom=0.06)

    def nice_ticks(vmin, vmax, target=5):
        rng = vmax - vmin
        raw = rng / target
        mag = 10 ** np.floor(np.log10(raw))
        for m in [1, 2, 2.5, 5, 10]:
            step = m * mag
            if rng / step <= target + 2:
                break
        start = np.ceil(vmin / step) * step
        ticks = list(np.arange(start, vmax + step * 0.5, step))
        ticks = [t for t in ticks if vmin - step*0.01 <= t <= vmax + step*0.01]
        if not ticks or abs(ticks[0] - vmin) > step * 0.25:
            ticks = [vmin] + ticks
        if abs(ticks[-1] - vmax) > step * 0.25:
            ticks = ticks + [vmax]
        return ticks, step

    def draw(ax, xmax_pts, ticks_pts, labels, title, color, mark=None, fs=9):
        ax.set_xlim(-2, 102)
        ax.set_ylim(0, 1)
        line_end = max(ticks_pts) if ticks_pts else xmax_pts
        ax.hlines(0.5, 0, line_end, color=color, lw=1.6)
        for t, l in zip(ticks_pts, labels):
            ax.vlines(t, 0.4, 0.6, color=color, lw=1.2)
            ax.text(t, -0.18, l, ha="center", va="top", fontsize=fs)
        if mark is not None:
            ax.plot(mark, 0.5, "v", color="#c0392b", markersize=11, zorder=5)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center",
                      fontsize=10.5, labelpad=18, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # 1) Points axis
    pts_ticks = list(np.arange(0, 101, 10))
    draw(axes[0], 100, pts_ticks, [str(int(t)) for t in pts_ticks],
         "Points", "#2E86AB", fs=9)

    # 2) Predictor axes
    total_patient_pts = 0
    for i, (name, b, r) in enumerate(predictors):
        ax = axes[i + 1]
        max_pts = abs(b * (r["max"] - r["min"])) * scale
        xt, step = nice_ticks(r["min"], r["max"],
                              target=max(2, min(5, int(max_pts / 20))))
        if step >= 1:
            labels = [f"{t:.0f}" for t in xt]
        else:
            dec = max(1, int(-np.floor(np.log10(step))))
            labels = [f"{t:.{dec}f}" for t in xt]
        if b > 0:
            pts = [(t - r["min"]) * b * scale for t in xt]
        else:
            pts = [(r["max"] - t) * abs(b) * scale for t in xt]

        # patient marker
        mark = None
        if highlight is not None:
            xval = highlight[name]
            xval = max(r["min"], min(r["max"], xval))  # clamp to axis
            if b > 0:
                mark = (xval - r["min"]) * b * scale
            else:
                mark = (r["max"] - xval) * abs(b) * scale
            # accumulate real (unclamped) points
            if b > 0:
                total_patient_pts += (highlight[name] - r["min"]) * b * scale
            else:
                total_patient_pts += (r["max"] - highlight[name]) * abs(b) * scale

        direction = " ↑" if b > 0 else " ↓"
        draw(ax, max_pts, pts, labels, name + direction, "#333333", mark=mark, fs=9)

    # 3) Total Points
    tgt = 8
    raw = total_max / tgt
    mag = 10 ** np.floor(np.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        tstep = m * mag
        if total_max / tstep <= tgt + 2:
            break
    tot_ticks = list(np.arange(0, total_max + tstep * 0.5, tstep))
    tp_mark = total_patient_pts if highlight is not None else None
    # clamp marker to axis
    tp_mark_disp = None if tp_mark is None else max(0, min(total_max, tp_mark))
    axtp = axes[-2]
    axtp.set_xlim(-2, total_max * 1.02 + 2)
    axtp.set_ylim(0, 1)
    axtp.hlines(0.5, 0, max(tot_ticks), color="#A23B72", lw=1.6)
    for t in tot_ticks:
        axtp.vlines(t, 0.4, 0.6, color="#A23B72", lw=1.2)
        axtp.text(t, -0.18, f"{t:.0f}", ha="center", va="top", fontsize=9)
    if tp_mark_disp is not None:
        axtp.plot(tp_mark_disp, 0.5, "v", color="#c0392b", markersize=11, zorder=5)
    axtp.set_yticks([]); axtp.set_xticks([])
    axtp.set_ylabel("Total Points", rotation=0, ha="right", va="center",
                    fontsize=10.5, labelpad=18, fontweight="bold")
    for sp in axtp.spines.values():
        sp.set_visible(False)

    # 4) Predicted probability
    intercept = COEF["intercept"]
    min_contrib = sum(
        b * r["min"] if b > 0 else b * r["max"]
        for _, b, r in predictors
    )

    def tp_to_prob(tp):
        eta = intercept + min_contrib + tp / scale
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, eta))))

    def prob_to_tp(p):
        lp = math.log(p / (1 - p))
        return (lp - intercept - min_contrib) * scale

    # Total Points ekseninin kapsadığı gerçek olasılık aralığı
    p_lo = tp_to_prob(0)
    p_hi = tp_to_prob(total_max)

    # Olasılık ekseni KENDİ genişliğini kullansın (Total Points'e sıkışmasın).
    # Etiket olasılıklarının gerçek TP konumlarını al, sonra bu [tp_min, tp_max]
    # aralığını 0..total_max fiziksel genişliğe lineer ölçekle → tüm eksen dolu.
    all_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    visible = [p for p in all_levels if p_lo <= p <= p_hi]
    if len(visible) < 2:
        visible = [p_lo, p_hi]

    tp_positions = [prob_to_tp(p) for p in visible]
    tp_span_lo, tp_span_hi = min(tp_positions), max(tp_positions)
    span = (tp_span_hi - tp_span_lo) or 1.0

    def remap(tp):
        """Gerçek TP konumunu, olasılık ekseninin tüm genişliğine (0..total_max) yay."""
        return (tp - tp_span_lo) / span * total_max

    pticks = [remap(prob_to_tp(p)) for p in visible]
    plabels = [f"{p:.2f}" for p in visible]

    # de-collide
    if pticks:
        min_gap = total_max * 0.045
        ft, fl = [pticks[0]], [plabels[0]]
        for t, l in zip(pticks[1:], plabels[1:]):
            if t - ft[-1] >= min_gap:
                ft.append(t); fl.append(l)
        if pticks[-1] != ft[-1]:
            if pticks[-1] - ft[-1] >= min_gap * 0.6:
                ft.append(pticks[-1]); fl.append(plabels[-1])
            else:
                ft[-1] = pticks[-1]; fl[-1] = plabels[-1]
        pticks, plabels = ft, fl

    # Hasta marker'ını da aynı remap ile yerleştir
    if tp_mark_disp is not None:
        prob_mark = remap(max(tp_span_lo, min(tp_span_hi, total_patient_pts)))
    else:
        prob_mark = None

    axp = axes[-1]
    axp.set_xlim(-2, total_max * 1.02 + 2)
    axp.set_ylim(0, 1)
    axp.hlines(0.5, 0, max(pticks) if pticks else total_max, color="#F18F01", lw=1.6)
    for t, l in zip(pticks, plabels):
        axp.vlines(t, 0.4, 0.6, color="#F18F01", lw=1.2)
        axp.text(t, -0.18, l, ha="center", va="top", fontsize=9)
    if prob_mark is not None:
        axp.plot(prob_mark, 0.5, "v", color="#c0392b", markersize=11, zorder=5)
    axp.set_yticks([]); axp.set_xticks([])
    axp.set_ylabel("Predicted\nProbability\n(Complicated)", rotation=0,
                   ha="right", va="center", fontsize=10, labelpad=18, fontweight="bold")
    for sp in axp.spines.values():
        sp.set_visible(False)

    fig.suptitle(
        f"Nomogram — Major Organ Involvement in Behçet's Disease\n"
        f"(AUC = {MODEL_AUC}, n = {MODEL_N}, Firth penalized regression)",
        fontsize=11, y=0.98
    )
    return fig

fig = build_nomogram(highlight={"NEU": neu, "MONO": mono, "LYMPH": lymph})
st.pyplot(fig)
st.caption("🔻 Red markers show this patient's position on each axis.")

# Download nomogram
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
st.download_button("🖼️ Download nomogram (300 DPI)", buf.getvalue(),
                   "nomogram.png", "image/png")

# ============================================================
# DISCLAIMER
# ============================================================
st.markdown("""
<div class="disclaimer">
<b>⚠️ For research and educational use only.</b> This calculator should not be used
as the sole basis for clinical decisions. The model was derived from a single-center
cohort and requires external validation. Each patient must be evaluated individually,
with appropriate imaging and laboratory work-up when systemic involvement is suspected.
This tool does not replace medical advice.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Behçet Nomogram Calculator · NEU + MONO + LYMPH model · "
    "Firth penalized logistic regression · No data stored."
)
