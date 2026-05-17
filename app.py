# -*- coding: utf-8 -*-
"""
Behçet Risk Hesaplayıcı — Standalone Web Uygulaması
======================================================
Komplike Behçet (Vaskülit ± Üveit) için risk tahmini.

Bu uygulama, 180 hastalık bir kohortta türetilen lojistik regresyon
modellerine dayanır. Saf mukokutanöz vs komplike Behçet ayrımı için
tasarlanmıştır.

Deploy etmek için:
  - Streamlit Cloud'a yükle: https://share.streamlit.io
  - Veya: streamlit run risk_calculator.py
"""

import streamlit as st
import numpy as np
import math

# ====== SAYFA AYARLARI ======
st.set_page_config(
    page_title="Behçet Risk Hesaplayıcı",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====== STİL ======
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a4d5c;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5a7684;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #d4f1d4 0%, #b8e6b8 100%); border-left: 6px solid #4caf50; }
    .risk-mid { background: linear-gradient(135deg, #fff4d4 0%, #ffe9a8 100%); border-left: 6px solid #ff9800; }
    .risk-high { background: linear-gradient(135deg, #ffd4d4 0%, #ffb8b8 100%); border-left: 6px solid #d32f2f; }
    .prob-text {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a4d5c;
        line-height: 1;
    }
    .risk-label {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .info-box {
        background: #f0f7f9;
        border-left: 4px solid #1a4d5c;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .disclaimer {
        background: #fff3e0;
        border-left: 4px solid #e65100;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 2rem;
        font-size: 0.85rem;
        color: #5d4037;
    }
</style>
""", unsafe_allow_html=True)

# ====== MODEL KATSAYILARI (180 hastalık kohorttan türetilen) ======
# Lojistik regresyon: P(Komplike) = 1 / (1 + exp(-(intercept + β·x)))
MODELS = {
    "SIRI (önerilen)": {
        "auc": 0.918,
        "intercept": -5.987041,
        "coefs": {"SIRI": 5.075754},
        "description": "En yüksek spesifite (%93.3) — pratik klinik kullanım",
        "cutoff": 1.30,
        "sens": 74.2,
        "spec": 93.3
    },
    "AISI": {
        "auc": 0.913,
        "intercept": -4.696455,
        "coefs": {"AISI": 0.013837},
        "description": "En yüksek Youden J — sensitivite-spesifite dengesi",
        "cutoff": 322.51,
        "sens": 84.3,
        "spec": 85.6
    },
}

# ====== BAŞLIK ======
st.markdown('<div class="main-title">🩺 Behçet Risk Hesaplayıcı</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Komplike Behçet Hastalığı (Vaskülit / Üveit) Olasılık Tahmini</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ Hesaplayıcı hakkında"):
    st.markdown("""
    Bu hesaplayıcı, **180 hastalık tek merkezli bir kohorttan** türetilen
    lojistik regresyon modellerine dayanır. **Saf mukokutanöz Behçet** ile
    **komplike Behçet** (vaskülit ve/veya üveit tutulumu olan) hastaları
    ayırt etmek için tasarlanmıştır.

    **Kullanım:**
    1. Modeli seçin (SIRI önerilen)
    2. Hastanın hemogram değerlerini girin
    3. Otomatik hesaplanan olasılığı okuyun

    **Eğitim verisi:** Yalnızca Behçet tanısı olan hastalardır.
    Bu hesaplayıcı **Behçet'i tanımlamaz** — hastalığın komplike olup
    olmadığını tahmin eder.
    """)

# ====== GİRİŞ ALANLARI ======
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Model Seçimi")
    model_name = st.selectbox(
        "Hangi modeli kullanmak istersiniz?",
        list(MODELS.keys()),
        help="SIRI en yüksek spesifite sağlar. AISI sensitivite-spesifite dengesinde en iyisidir."
    )
    model = MODELS[model_name]

    st.markdown(f"""
    <div class="info-box">
    <b>Model performansı:</b><br>
    AUC = <b>{model['auc']:.3f}</b><br>
    {model['description']}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Hasta Verileri")
st.caption("Hemogram değerlerini girin (×10³/µL hücre sayıları, PLT için ×10³/µL)")

# Hücre sayıları her zaman istenir
c1, c2, c3, c4 = st.columns(4)
with c1:
    neu = st.number_input("Nötrofil (NEU)", min_value=0.0, max_value=30.0,
                          value=4.5, step=0.1, help="×10³/µL")
with c2:
    lymph = st.number_input("Lenfosit (LYMPH)", min_value=0.1, max_value=10.0,
                            value=2.0, step=0.1, help="×10³/µL — 0 olamaz")
with c3:
    mono = st.number_input("Monosit (MONO)", min_value=0.0, max_value=3.0,
                           value=0.5, step=0.01, help="×10³/µL")
with c4:
    plt_val = st.number_input("Trombosit (PLT)", min_value=50, max_value=1000,
                              value=280, step=10, help="×10³/µL")

# İndeksleri hesapla
try:
    siri = (neu * mono) / lymph
    aisi = (plt_val * neu * mono) / lymph
except ZeroDivisionError:
    st.error("Lenfosit 0 olamaz.")
    st.stop()

# Hesaplanan indeksleri göster (sadece kullandığımız iki indeks)
st.markdown("#### Hesaplanan İnflamatuar İndeksler")
ic1, ic2 = st.columns(2)
ic1.metric("SIRI", f"{siri:.2f}")
ic2.metric("AISI", f"{aisi:.0f}")

# ====== HESAPLAMA ======
all_indices = {'SIRI': siri, 'AISI': aisi}

logit = model['intercept']
for var, coef in model['coefs'].items():
    if var in all_indices:
        logit += coef * all_indices[var]
    else:
        st.error(f"Değişken eksik: {var}")
        st.stop()

probability = 1 / (1 + math.exp(-logit))

# Risk kategorisi
if probability < 0.3:
    risk_class = "risk-low"
    risk_label = "DÜŞÜK RİSK"
    risk_emoji = "🟢"
    interpretation = (
        "Hastanın inflamatuar profili **saf mukokutanöz Behçet** ile uyumlu. "
        "Vaskülit veya üveit tutulumu olasılığı düşük."
    )
elif probability < 0.7:
    risk_class = "risk-mid"
    risk_label = "ORTA RİSK"
    risk_emoji = "🟡"
    interpretation = (
        "Belirsiz aralık. **Klinik değerlendirme ve görüntüleme** ile "
        "doğrulama önerilir. Tek başına bu skor karar verici değildir."
    )
else:
    risk_class = "risk-high"
    risk_label = "YÜKSEK RİSK"
    risk_emoji = "🔴"
    interpretation = (
        "Hastanın inflamatuar profili **komplike Behçet** (vaskülit/üveit) "
        "ile yüksek derecede uyumlu. **Detaylı sistemik değerlendirme** önerilir."
    )

# ====== SONUÇ GÖSTERİMİ ======
st.markdown("---")
st.markdown(f"""
<div class="result-box {risk_class}">
    <div style="font-size: 1rem; color: #555; margin-bottom: 0.5rem;">
        Komplike Behçet Olasılığı
    </div>
    <div class="prob-text">{probability:.1%}</div>
    <div class="risk-label">{risk_emoji} {risk_label}</div>
    <div style="margin-top: 1rem; color: #444;">{interpretation}</div>
</div>
""", unsafe_allow_html=True)

# Olasılık çubuğu
st.markdown("##### Risk Spektrumu")
bar_html = f"""
<div style="position: relative; height: 30px; background: linear-gradient(to right, #4caf50 0%, #ff9800 50%, #d32f2f 100%); border-radius: 15px; overflow: hidden;">
    <div style="position: absolute; left: {probability*100:.1f}%; top: -5px; width: 4px; height: 40px; background: #1a4d5c; transform: translateX(-2px);"></div>
    <div style="position: absolute; left: {probability*100:.1f}%; top: -25px; transform: translateX(-50%); font-weight: 600; color: #1a4d5c;">▼ {probability:.1%}</div>
</div>
<div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.85rem; color: #666;">
    <span>%0 (Mukokutanöz)</span>
    <span>%50</span>
    <span>%100 (Komplike)</span>
</div>
"""
st.markdown(bar_html, unsafe_allow_html=True)

# Cut-off karşılaştırması
if model.get('cutoff'):
    primary_marker = list(model['coefs'].keys())[0]
    if primary_marker in all_indices:
        cur_val = all_indices[primary_marker]
        comparison = "≥" if cur_val >= model['cutoff'] else "<"
        st.markdown(f"""
        <div class="info-box">
        <b>Cut-off karşılaştırması:</b><br>
        Hastanın {primary_marker} değeri: <b>{cur_val:.2f}</b><br>
        Modelin optimal cut-off değeri (Youden J): <b>{model['cutoff']:.2f}</b><br>
        {primary_marker} {comparison} {model['cutoff']:.2f} →
        Bu cut-off'ta sensitivite <b>{model['sens']:.1f}%</b>,
        spesifite <b>{model['spec']:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

# ====== DISCLAIMER ======
st.markdown("""
<div class="disclaimer">
<b>⚠️ Önemli Uyarı:</b><br>
Bu araç yalnızca <b>araştırma ve eğitim amaçlıdır</b>. Klinik karar verme süreçlerinde
tek başına kullanılmamalıdır. Modelin türetildiği kohort tek merkezlidir ve harici
validasyon gerektirir. Her hastanın klinik değerlendirmesi bireysel olarak yapılmalı,
sistemik tutulum şüphesinde uygun görüntüleme ve laboratuvar tetkikleri istenmelidir.
<br><br>
Bu hesaplayıcı tıbbi tavsiye yerine geçmez.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Behçet Risk Hesaplayıcı v1.0 · https://behcetrisc.streamlit.app/")
