# -*- coding: utf-8 -*-
"""
Behçet Hastalığı Klinik İstatistik Paneli v2.0
==================================================
Saf Mukokutanöz vs. Komplike Behçet (Vaskülit / Üveit) Analizi

Özellikler:
 - Grup bazlı normallik (Shapiro-Wilk her grupta ayrı)
 - Varyans homojenliği (Levene), gerekirse Welch t/ANOVA
 - Post-hoc: Tukey HSD (parametrik) veya Dunn (non-parametrik)
 - Çoklu karşılaştırma düzeltmesi: Benjamini-Hochberg (FDR)
 - Etki büyüklükleri: Cohen's d, rank-biserial r
 - Hem Mean±SD hem Median (IQR) raporlama
 - Tablo 1 (demografik) + Tablo 2 (biyomarker)
 - Strip plot + anlamlılık parantezleri (Figure 1 tarzı)
 - Spearman korelasyon heatmap (Figure 3 tarzı)
 - Çoklu ROC eğrisi + AUC + optimal cut-off (Youden J)
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Opsiyonel paketler (kullanıcının yüklemesi gerekir):
try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.oneway import anova_oneway
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.metrics import roc_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Firth logistic regression — manuel implementasyon
# Firth (1993): Bias reduction in maximum likelihood estimates
# Jeffreys prior ile penalize edilmiş log-likelihood
HAS_FIRTH = True

def firth_logistic_regression(X, y, max_iter=100, tol=1e-6):
    """
    Firth penalized logistic regression.

    Parameters
    ----------
    X : ndarray (n, p) — design matrix (constant DAHİL)
    y : ndarray (n,)   — binary outcome
    max_iter, tol      — yakınsama parametreleri

    Returns
    -------
    dict: beta, se, ci_lo, ci_hi, pvals, loglik, fitted_prob
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)

    for it in range(max_iter):
        eta = X @ beta
        # Sayısal stabilite için clip
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        # Fisher information
        WX = X * W[:, None]
        I = X.T @ WX
        # Inverse (regularize a bit if singular)
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I)
        # Hat matrix diagonal: H_ii = W_i * x_i^T I^-1 x_i
        H_diag = np.einsum('ij,jk,ik->i', WX, I_inv, X)
        # Firth-corrected score
        U = X.T @ (y - mu + H_diag * (0.5 - mu))
        # Update
        delta = I_inv @ U
        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Standart hatalar (penalized info matrix'in inversi)
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu)
    I = X.T @ (X * W[:, None])
    try:
        I_inv = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        I_inv = np.linalg.pinv(I)
    se = np.sqrt(np.maximum(np.diag(I_inv), 0))
    # Wald CI ve p-değerleri
    z = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    # Penalized log-likelihood
    eps = 1e-12
    ll = np.sum(y * np.log(mu + eps) + (1-y) * np.log(1 - mu + eps))
    # Jeffreys penalty: 0.5 * log|I|
    try:
        sign, logdet = np.linalg.slogdet(I)
        ll_pen = ll + 0.5 * logdet
    except Exception:
        ll_pen = ll
    return {
        'beta': beta, 'se': se,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'pvals': pvals,
        'loglik': ll, 'loglik_pen': ll_pen,
        'fitted_prob': mu
    }

# ====== SAYFA AYARLARI ======
st.set_page_config(page_title="Behçet İstatistik Paneli v2", layout="wide")
st.title("🩺 Behçet Hastalığı: Klinik Veri Analiz Paneli v2.0")
st.caption("Saf Mukokutanöz Behçet vs. Komplike Behçet (Vaskülit ± Üveit)")

# Uyarı: paket eksikse
missing = []
if not HAS_POSTHOC: missing.append("scikit-posthocs")
if not HAS_STATSMODELS: missing.append("statsmodels")
if not HAS_SKLEARN: missing.append("scikit-learn")
if missing:
    st.warning(f"Eksik paketler: `{', '.join(missing)}`. Kurmak için: `pip install {' '.join(missing)}`")

# ====== YARDIMCI FONKSİYONLAR ======

def cohens_d(g1, g2):
    """Bağımsız iki grup için Cohen's d (pooled SD)."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled > 0 else np.nan

def rank_biserial_r(g1, g2):
    """Mann-Whitney U için etki büyüklüğü (rank-biserial korelasyon)."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    try:
        u, _ = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        return 1 - (2*u) / (n1*n2)
    except ValueError:
        return np.nan

def fmt_p(p):
    if pd.isna(p): return "—"
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def fmt_descr(data, mode='auto', is_normal=True):
    """Tanımlayıcı istatistik. mode: 'mean', 'median', 'auto' (dağılıma göre)."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) == 0: return "—"
    if mode == 'auto':
        mode = 'mean' if is_normal else 'median'
    if mode == 'mean':
        sd = np.std(data, ddof=1) if len(data) > 1 else 0
        return f"{np.mean(data):.2f} ± {sd:.2f}"
    else:
        q1, q3 = np.percentile(data, [25, 75])
        return f"{np.median(data):.2f} [{q1:.2f}–{q3:.2f}]"

def shapiro_per_group(df, col, group_col):
    """Her grupta Shapiro-Wilk; herhangi biri p<0.05 ise dağılım non-normal."""
    p_vals = {}
    for g in df[group_col].dropna().unique():
        d = df[df[group_col]==g][col].dropna()
        if len(d) >= 3:
            try:
                _, p = stats.shapiro(d)
                p_vals[g] = p
            except Exception:
                p_vals[g] = np.nan
    if not p_vals:
        return False, {}
    valid = [p for p in p_vals.values() if not pd.isna(p)]
    is_normal = (min(valid) > 0.05) if valid else False
    return is_normal, p_vals

def levene_test(*groups):
    """Levene testi (median-bazlı, daha robust)."""
    valid_groups = [np.asarray(g)[~np.isnan(g)] for g in groups]
    valid_groups = [g for g in valid_groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return np.nan, np.nan
    try:
        return stats.levene(*valid_groups, center='median')
    except Exception:
        return np.nan, np.nan

def add_significance_bracket(ax, x1, x2, y, p_value, h=0.02):
    """Strip/box plot üzerine anlamlılık paranteziekler."""
    if pd.isna(p_value) or p_value >= 0.05:
        return
    if p_value < 0.001: label = "p < 0.001"
    elif p_value < 0.01: label = f"p = {p_value:.3f}"
    else: label = f"p = {p_value:.3f}"
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    ax.text((x1+x2)/2, y+h, label, ha='center', va='bottom', fontsize=10)


# ====== DELONG TESTİ (iki bağımlı ROC eğrisinin karşılaştırılması) ======
# DeLong, DeLong & Clarke-Pearson (1988); Sun & Xu (2014) hızlı algoritma
def _compute_midrank(x):
    """Bağlı gözlemler için midrank (DeLong yardımcı fonksiyonu)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2

def _fast_delong(predictions, label_1_count):
    """Hızlı DeLong kovaryans hesaplama."""
    m = label_1_count
    n = predictions.shape[1] - m
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    if k > 1:
        sx = np.cov(v01); sy = np.cov(v10)
    else:
        sx = np.array([[np.var(v01.flatten(), ddof=1)]])
        sy = np.array([[np.var(v10.flatten(), ddof=1)]])
    delong_cov = sx / m + sy / n
    return aucs, delong_cov

def delong_test(y_true, score1, score2):
    """
    DeLong testi: iki bağımlı ROC eğrisinin AUC'lerini karşılaştırır.
    Aynı hastalar üzerinde iki farklı belirteç için.
    Returns: (auc1, auc2, z, p)
    """
    y_true = np.asarray(y_true).astype(int)
    score1 = np.asarray(score1, dtype=float)
    score2 = np.asarray(score2, dtype=float)
    order = np.argsort(-y_true)  # pozitifler önce
    label_1_count = int(y_true.sum())
    preds = np.vstack((score1[order], score2[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    var = float((l @ cov @ l.T).item())
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)

def effective_scores(y_true, raw_scores):
    """AUC<0.5 olduğunda skorları ters çevirir (LENFOSİT gibi inverse prediktörler için)."""
    if not HAS_SKLEARN:
        return raw_scores
    fpr, tpr, _ = roc_curve(y_true, raw_scores)
    return -np.asarray(raw_scores) if auc(fpr, tpr) < 0.5 else np.asarray(raw_scores)


# ====== YARDIMCI: TABLO -> PNG (yayın için 300 DPI) ======
def df_to_png_bytes(df, title=None, max_col_width=None):
    """
    Pandas DataFrame'i 300 DPI yüksek-çözünürlüklü PNG bytes'a çevirir.
    Makale için doğrudan yapıştırılabilir.
    """
    n_rows, n_cols = df.shape
    # Boyutu içeriğe göre ayarla
    col_width = 1.5
    fig_w = max(6, n_cols * col_width)
    fig_h = max(1.2, 0.45 * (n_rows + 1) + (0.4 if title else 0))

    fig_t, ax_t = plt.subplots(figsize=(fig_w, fig_h))
    ax_t.axis('off')

    if title:
        ax_t.set_title(title, fontsize=12, fontweight='bold', pad=10, loc='left')

    # Tablo verisi
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([str(v) for v in row.values])

    table = ax_t.table(
        cellText=cell_text,
        colLabels=df.columns.tolist(),
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Başlık satırı stillemesi
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')

    # Alternatif satır renklendirmesi
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#f5f5f5')

    if max_col_width:
        table.auto_set_column_width(col=list(range(n_cols)))

    plt.tight_layout()
    buf_t = io.BytesIO()
    fig_t.savefig(buf_t, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig_t)
    return buf_t.getvalue()


def download_buttons_for_table(df, base_filename, label_prefix="", title=None):
    """
    Bir DataFrame için iki indirme butonu üretir: CSV + 300 DPI PNG.
    Yan yana iki sütunda gösterir.
    """
    c1, c2 = st.columns(2)
    with c1:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            f"📥 {label_prefix}CSV indir",
            csv,
            f"{base_filename}.csv",
            "text/csv",
            key=f"csv_{base_filename}"
        )
    with c2:
        try:
            png_bytes = df_to_png_bytes(df, title=title)
            st.download_button(
                f"🖼️ {label_prefix}PNG indir (300 DPI)",
                png_bytes,
                f"{base_filename}.png",
                "image/png",
                key=f"png_{base_filename}"
            )
        except Exception:
            pass


def download_fig_300dpi(fig, base_filename, label="Şekli PNG indir (300 DPI)"):
    """Matplotlib figure için 300 DPI PNG indirme butonu."""
    buf_f = io.BytesIO()
    fig.savefig(buf_f, format='png', dpi=300, bbox_inches='tight',
                  facecolor='white')
    st.download_button(
        f"🖼️ {label}",
        buf_f.getvalue(),
        f"{base_filename}.png",
        "image/png",
        key=f"fig_{base_filename}"
    )


# ====== DOSYA YÜKLEME ======
uploaded = st.file_uploader(
    "Veri dosyasını seçin (xlsx, xls, csv, sav)",
    type=["xlsx", "xls", "csv", "sav"]
)

if not uploaded:
    st.info("Lütfen bir veri dosyası yükleyin.")
    st.stop()

@st.cache_data
def load_data(file, ext):
    if ext == 'csv':
        return pd.read_csv(file)
    elif ext == 'sav':
        import pyreadstat
        df, _ = pyreadstat.read_sav(file)
        return df
    else:
        return pd.read_excel(file)

ext = uploaded.name.split('.')[-1].lower()
try:
    df = load_data(uploaded, ext)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

# Sütun adlarını temizle
df.columns = df.columns.str.strip()

# Sayısal dönüşüm
numeric_cols = ['SEDİM', 'CRP', 'NEU', 'PLT', 'LENFOSİT', 'MONOSİT',
                'YAŞ', 'HASTALIK SÜRESİ(yıl)', 'Kolşisin', 'Biyolojik',
                'DMARD', 'VASKÜLİT', 'UVEİT']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

# SIRI / SII hesapla (mevcut sütunlarla tutarlılık kontrolü)
if all(c in df.columns for c in ['NEU', 'MONOSİT', 'LENFOSİT']):
    df['SIRI'] = (df['NEU'] * df['MONOSİT']) / df['LENFOSİT']
if all(c in df.columns for c in ['NEU', 'PLT', 'LENFOSİT']):
    df['SII'] = (df['NEU'] * df['PLT']) / df['LENFOSİT']

# ─── EK İNFLAMATUAR İNDEKSLER ───
# NLR: Neutrophil-to-Lymphocyte Ratio
if all(c in df.columns for c in ['NEU', 'LENFOSİT']):
    df['NLR'] = df['NEU'] / df['LENFOSİT']
# PLR: Platelet-to-Lymphocyte Ratio
if all(c in df.columns for c in ['PLT', 'LENFOSİT']):
    df['PLR'] = df['PLT'] / df['LENFOSİT']
# NMLR: (Neutrophil + Monocyte) / Lymphocyte
if all(c in df.columns for c in ['NEU', 'MONOSİT', 'LENFOSİT']):
    df['NMLR'] = (df['NEU'] + df['MONOSİT']) / df['LENFOSİT']
# AISI: Aggregate Inflammation Systemic Index = (PLT × NEU × MONO) / LYMPH
if all(c in df.columns for c in ['NEU', 'PLT', 'MONOSİT', 'LENFOSİT']):
    df['AISI'] = (df['PLT'] * df['NEU'] * df['MONOSİT']) / df['LENFOSİT']

# Not: dNLR ve dPLR hesaplaması, lökosit (WBC) sayımı gerektirdiğinden
# bu veride hesaplanmıyor. Veriye WBC sütunu eklenirse manuel olarak hesaplanabilir.

# NaN satırları (VASKÜLİT veya UVEİT eksik olanları) çıkar
before = len(df)
df = df.dropna(subset=['VASKÜLİT', 'UVEİT']).reset_index(drop=True)
dropped = before - len(df)

# ====== GRUP TANIMI: 4 ALT GRUP + İKİLİ ======
def define_4group(row):
    v = row['VASKÜLİT'] == 1
    u = row['UVEİT'] == 1
    if v and u: return 'Kombine (V+Ü)'
    elif v:     return 'Sadece Vaskülit'
    elif u:     return 'Sadece Üveit'
    else:       return 'Saf Mukokutanöz'

df['Grup'] = df.apply(define_4group, axis=1)
df['Birlesik_Grup'] = df['Grup'].apply(
    lambda x: 'Saf Mukokutanöz' if x == 'Saf Mukokutanöz' else 'Komplike Behçet'
)

# Grup sırası (görsel tutarlılık için)
GROUP_ORDER_4 = ['Saf Mukokutanöz', 'Sadece Vaskülit', 'Sadece Üveit', 'Kombine (V+Ü)']
GROUP_ORDER_2 = ['Saf Mukokutanöz', 'Komplike Behçet']

# ====== KENAR ÇUBUĞU ======
st.sidebar.header("⚙️ Analiz Ayarları")

analysis_mode = st.sidebar.radio(
    "Karşılaştırma Modu",
    ["İkili (Komplike vs Saf Mukokutanöz)", "4-Grup Detaylı"],
    index=0
)

alpha = st.sidebar.number_input("α (anlamlılık eşiği)", value=0.05,
                                 min_value=0.001, max_value=0.1, step=0.005)

use_fdr = st.sidebar.checkbox(
    "Çoklu karşılaştırma düzeltmesi (Benjamini-Hochberg FDR)",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Grup Dağılımı")
st.sidebar.write("**4 Alt Grup:**")
st.sidebar.write(df['Grup'].value_counts().reindex(GROUP_ORDER_4))
st.sidebar.write("**İkili:**")
st.sidebar.write(df['Birlesik_Grup'].value_counts().reindex(GROUP_ORDER_2))

if dropped > 0:
    st.sidebar.warning(f"{dropped} satır eksik VASKÜLİT/UVEİT nedeniyle çıkarıldı.")

# Hedef parametreler — klinik mantıkta sıralı:
# 1) Akut faz reaktanları → 2) Hücre sayıları → 3) Basit oranlar → 4) Kompozit indeksler
target_cols = [
    'SEDİM', 'CRP',                                  # akut faz
    'NEU', 'PLT', 'LENFOSİT', 'MONOSİT',             # hücre sayıları
    'NLR', 'PLR',                                    # basit oranlar
    'NMLR', 'SIRI', 'SII', 'AISI'                    # kompozit indeksler
]
target_cols = [c for c in target_cols if c in df.columns]

# Aktif grup kolonu
group_col = 'Birlesik_Grup' if analysis_mode.startswith("İkili") else 'Grup'
group_order = GROUP_ORDER_2 if analysis_mode.startswith("İkili") else GROUP_ORDER_4

# ====== SEKMELER ======
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Veri Önizleme",
    "👥 Tablo 1: Demografik",
    "🧪 Tablo 2: Biyomarker",
    "📈 Strip Plot",
    "🔥 Korelasyon",
    "🎯 ROC Analizi",
    "🔬 Lojistik Regresyon"
])

# ────────────────────────────────────────────────
# TAB 1: VERİ ÖNİZLEME
# ────────────────────────────────────────────────
with tab1:
    st.subheader("Veri Önizleme")
    st.write(f"**Toplam:** {len(df)} hasta, {len(df.columns)} sütun")

    # SIRI/SII tutarlılık
    orig_siri_col = next((c for c in df.columns if 'SIRI' in c and c != 'SIRI'), None)
    if orig_siri_col:
        orig = pd.to_numeric(df[orig_siri_col], errors='coerce')
        diff = (df['SIRI'] - orig).abs()
        n_diff = (diff > 0.01).sum()
        if n_diff > 0:
            st.warning(f"⚠️ SIRI: {n_diff} satırda orijinalle uyuşmazlık var. Yeniden hesaplanan değerler kullanılıyor.")
        else:
            st.success("✅ SIRI değerleri orijinalle tutarlı.")

    st.dataframe(df[['HASTA_ADI', 'YAŞ', 'Grup', 'Birlesik_Grup'] +
                    [c for c in target_cols if c in df.columns]].head(20))

# ────────────────────────────────────────────────
# TAB 2: DEMOGRAFİK (TABLO 1)
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Tablo 1 — Demografik ve Klinik Özellikler")

    demo_rows = []
    demo_continuous = []
    if 'YAŞ' in df.columns: demo_continuous.append('YAŞ')
    if 'HASTALIK SÜRESİ(yıl)' in df.columns: demo_continuous.append('HASTALIK SÜRESİ(yıl)')

    for col in demo_continuous:
        is_normal, _ = shapiro_per_group(df, col, group_col)
        row = {'Değişken': col, 'Dağılım': 'Normal' if is_normal else 'Non-Normal'}
        groups_data = []
        for g in group_order:
            d = df[df[group_col]==g][col].dropna()
            row[g] = fmt_descr(d, mode='auto', is_normal=is_normal)
            groups_data.append(d)

        # Test
        if len(groups_data) == 2:
            if is_normal:
                # Levene → Welch'e geç gerekirse
                _, p_lev = levene_test(*groups_data)
                eq_var = (p_lev > 0.05) if not pd.isna(p_lev) else True
                _, p = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=eq_var)
                test = "t-test" if eq_var else "Welch t"
            else:
                _, p = stats.mannwhitneyu(groups_data[0], groups_data[1])
                test = "MWU"
        else:
            if is_normal:
                _, p = stats.f_oneway(*groups_data)
                test = "ANOVA"
            else:
                _, p = stats.kruskal(*groups_data)
                test = "Kruskal-W"

        row['Test'] = test
        row['p'] = fmt_p(p)
        demo_rows.append(row)

    # Cinsiyet tahmini (HASTA_ADI'ndan, sadece informatif)
    # Kategorik: ilaç kullanımı
    cat_vars = []
    if 'Kolşisin' in df.columns: cat_vars.append('Kolşisin')
    if 'Biyolojik' in df.columns: cat_vars.append('Biyolojik')
    if 'DMARD' in df.columns: cat_vars.append('DMARD')

    for col in cat_vars:
        row = {'Değişken': col + ' (kullanım, n %)', 'Dağılım': '—'}
        # Chi-kare
        try:
            ct = pd.crosstab(df[col], df[group_col])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            test = "χ²"
        except Exception:
            p = np.nan
            test = "—"

        for g in group_order:
            sub = df[df[group_col]==g][col].dropna()
            n_pos = int((sub == 1).sum())
            total = len(sub)
            pct = (n_pos / total * 100) if total > 0 else 0
            row[g] = f"{n_pos}/{total} ({pct:.1f}%)"

        row['Test'] = test
        row['p'] = fmt_p(p)
        demo_rows.append(row)

    demo_df = pd.DataFrame(demo_rows)

    def hl_p(v):
        try:
            if v == "<0.001": return 'background-color: #D4EFDF'
            return 'background-color: #D4EFDF' if float(v) < alpha else ''
        except: return ''

    st.dataframe(demo_df.style.map(hl_p, subset=['p']), use_container_width=True)

    download_buttons_for_table(
        demo_df, "tablo1_demografik",
        label_prefix="Tablo 1 — ",
        title="Tablo 1. Demografik ve Klinik Özellikler"
    )

# ────────────────────────────────────────────────
# TAB 3: BİYOMARKER (TABLO 2)
# ────────────────────────────────────────────────
with tab3:
    st.subheader(f"Tablo 2 — İnflamatuar Biyomarker Karşılaştırması ({analysis_mode})")

    summary_rows = []
    raw_pvals = []  # FDR için

    for col in target_cols:
        is_normal, shap_pvals = shapiro_per_group(df, col, group_col)
        groups_data = [df[df[group_col]==g][col].dropna().values for g in group_order]

        # Levene
        _, p_lev = levene_test(*groups_data)
        eq_var = (p_lev > 0.05) if not pd.isna(p_lev) else True

        # Omnibus test
        try:
            if len(group_order) == 2:
                if is_normal:
                    _, p_omni = stats.ttest_ind(groups_data[0], groups_data[1],
                                                 equal_var=eq_var)
                    test = "t-test" if eq_var else "Welch t"
                else:
                    _, p_omni = stats.mannwhitneyu(groups_data[0], groups_data[1])
                    test = "MWU"
            else:
                if is_normal and eq_var:
                    _, p_omni = stats.f_oneway(*groups_data)
                    test = "ANOVA"
                elif is_normal and not eq_var and HAS_STATSMODELS:
                    res = anova_oneway(groups_data, use_var='unequal')
                    p_omni = res.pvalue
                    test = "Welch ANOVA"
                else:
                    _, p_omni = stats.kruskal(*groups_data)
                    test = "Kruskal-W"
        except Exception as e:
            p_omni = np.nan
            test = "—"

        # Etki büyüklüğü (sadece ikili modda)
        eff_str = "—"
        if len(group_order) == 2:
            mk = df[df[group_col]=='Saf Mukokutanöz'][col].dropna().values
            kp = df[df[group_col]=='Komplike Behçet'][col].dropna().values
            if is_normal:
                d = cohens_d(kp, mk)
                eff_str = f"d = {d:.2f}" if not pd.isna(d) else "—"
            else:
                r = rank_biserial_r(kp, mk)
                eff_str = f"r = {r:.2f}" if not pd.isna(r) else "—"

        # Trend
        means = [np.mean(g) if len(g)>0 else np.nan for g in groups_data]
        if len(means) == 2 and not pd.isna(p_omni) and p_omni < alpha:
            trend = "↑" if means[1] > means[0] else "↓"
        else:
            trend = "↔"

        row = {
            'Parametre': col,
            'Dağılım': 'Normal' if is_normal else 'Non-Normal',
            'Levene p': fmt_p(p_lev),
        }
        for i, g in enumerate(group_order):
            row[g] = fmt_descr(groups_data[i], mode='auto', is_normal=is_normal)
        row['Test'] = test
        row['p'] = fmt_p(p_omni)
        row['Etki'] = eff_str
        row['Trend'] = trend
        summary_rows.append(row)
        raw_pvals.append(p_omni)

    summary_df = pd.DataFrame(summary_rows)

    # FDR düzeltmesi
    if use_fdr and HAS_STATSMODELS:
        valid_mask = ~pd.isna(raw_pvals)
        p_arr = np.array(raw_pvals, dtype=float)
        p_arr_valid = p_arr[valid_mask]
        if len(p_arr_valid) > 0:
            _, p_adj, _, _ = multipletests(p_arr_valid, alpha=alpha, method='fdr_bh')
            full_adj = np.full_like(p_arr, np.nan, dtype=float)
            full_adj[valid_mask] = p_adj
            summary_df['p (FDR)'] = [fmt_p(p) for p in full_adj]

    def hl_p_full(v):
        try:
            if v == "<0.001": return 'background-color: #D4EFDF'
            return 'background-color: #D4EFDF' if float(v) < alpha else ''
        except: return ''

    cols_to_color = ['p']
    if 'p (FDR)' in summary_df.columns: cols_to_color.append('p (FDR)')

    st.dataframe(
        summary_df.style.map(hl_p_full, subset=cols_to_color),
        use_container_width=True
    )

    # Post-hoc (sadece 4-grup modunda + anlamlı omnibus)
    if len(group_order) > 2:
        st.markdown("---")
        st.subheader("📐 Post-hoc Analizler")
        st.caption("Sadece omnibus p < α olan parametreler için ikili karşılaştırma.")

        for i, col in enumerate(target_cols):
            p_omni = raw_pvals[i]
            if pd.isna(p_omni) or p_omni >= alpha:
                continue
            is_normal = summary_rows[i]['Dağılım'] == 'Normal'

            st.markdown(f"**{col}** (omnibus p = {fmt_p(p_omni)})")
            sub = df.dropna(subset=[col, group_col])

            if is_normal and HAS_STATSMODELS:
                tuk = pairwise_tukeyhsd(sub[col], sub[group_col], alpha=alpha)
                tuk_df = pd.DataFrame(data=tuk._results_table.data[1:],
                                       columns=tuk._results_table.data[0])
                st.dataframe(tuk_df)
            elif not is_normal and HAS_POSTHOC:
                dunn = sp.posthoc_dunn(sub, val_col=col, group_col=group_col,
                                        p_adjust='holm')
                st.dataframe(dunn.style.map(
                    lambda v: 'background-color: #D4EFDF' if isinstance(v, (int,float)) and v < alpha else ''
                ))
            else:
                st.info("Post-hoc için gerekli paket eksik.")

    download_buttons_for_table(
        summary_df, "tablo2_biyomarker",
        label_prefix="Tablo 2 — ",
        title="Tablo 2. İnflamatuar Biyomarker Karşılaştırması"
    )

# ────────────────────────────────────────────────
# TAB 4: STRIP PLOT (Figure 1 tarzı)
# ────────────────────────────────────────────────
with tab4:
    st.subheader("Strip Plot — Grup Karşılaştırmaları")
    st.caption("Mean ± SD çizgili, anlamlılık parantezleri otomatik eklenir.")

    sel_params = st.multiselect("Görselleştirilecek parametreler",
                                 target_cols, default=target_cols[:4])

    if sel_params:
        n_plots = len(sel_params)
        ncols = 2
        nrows = (n_plots + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5*nrows))
        axes = np.atleast_2d(axes).flatten()

        palette = ['#4A90E2', '#F5A623', '#7ED321', '#D0021B']

        # Negatif olamayacak biyolojik parametreler (eksen 0'dan başlasın)
        non_negative_params = {'SEDİM', 'CRP', 'NEU', 'PLT', 'LENFOSİT',
                               'MONOSİT', 'NLR', 'PLR', 'NMLR', 'SIRI',
                               'SII', 'AISI', 'YAŞ', 'HASTALIK SÜRESİ(yıl)'}

        for idx, param in enumerate(sel_params):
            ax = axes[idx]
            sub = df.dropna(subset=[param, group_col])

            # Bu parametrenin dağılımı normal mi? Buna göre overlay seç
            is_normal_param, _ = shapiro_per_group(sub, param, group_col)
            overlay_label = "Mean ± SD" if is_normal_param else "Median [IQR]"

            # Strip plot
            sns.stripplot(data=sub, x=group_col, y=param, order=group_order,
                          ax=ax, palette=palette[:len(group_order)],
                          jitter=0.25, alpha=0.7, size=4)

            # Merkez (mean/median) ve dispersiyon (SD/IQR) çizgileri
            for i, g in enumerate(group_order):
                d = sub[sub[group_col]==g][param].dropna()
                if len(d) == 0: continue
                if is_normal_param:
                    center = d.mean()
                    lower, upper = center - d.std(), center + d.std()
                else:
                    q1, med, q3 = np.percentile(d, [25, 50, 75])
                    center, lower, upper = med, q1, q3
                # Negatif olamayacak parametrelerde alt sınırı 0'da kırp
                if param in non_negative_params:
                    lower = max(0, lower)
                ax.hlines(center, i-0.3, i+0.3, colors='black', linewidth=2)
                ax.vlines(i, lower, upper, colors='black', linewidth=1.5)
                ax.hlines([lower, upper], i-0.1, i+0.1, colors='black', linewidth=1)

            ax.set_title(f"{param}  ({overlay_label})", fontsize=12, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='x', rotation=15)

            # Negatif olamayacak parametrelerde y-ekseni alt sınırını 0'a sabitle
            if param in non_negative_params:
                cur_bottom, cur_top = ax.get_ylim()
                ax.set_ylim(bottom=max(0, cur_bottom * 0.95), top=cur_top)

            # Anlamlılık parantezleri (ikili karşılaştırmalar)
            y_max = sub[param].max()
            y_range = sub[param].max() - sub[param].min()
            y_step = y_range * 0.08

            comparisons = []
            for i in range(len(group_order)):
                for j in range(i+1, len(group_order)):
                    comparisons.append((i, j))

            level = 0
            for (i, j) in comparisons:
                g1 = sub[sub[group_col]==group_order[i]][param].dropna()
                g2 = sub[sub[group_col]==group_order[j]][param].dropna()
                if len(g1) < 3 or len(g2) < 3: continue
                is_normal_pair, _ = shapiro_per_group(
                    sub[sub[group_col].isin([group_order[i], group_order[j]])],
                    param, group_col)
                if is_normal_pair:
                    _, p = stats.ttest_ind(g1, g2, equal_var=False)
                else:
                    _, p = stats.mannwhitneyu(g1, g2)
                if not pd.isna(p) and p < alpha:
                    y_pos = y_max + y_step * (1 + level)
                    add_significance_bracket(ax, i, j, y_pos, p, h=y_step*0.3)
                    level += 1
            # Y aralığını parantezlere göre genişlet
            if level > 0:
                ax.set_ylim(top=y_max + y_step * (level + 2))

        # Boş subplot'ları gizle
        for k in range(idx+1, len(axes)):
            axes[k].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # İndirme
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Strip plot PNG indir (300 DPI)", buf.getvalue(),
                           "stripplot.png", "image/png",
                           key="fig_stripplot")

# ────────────────────────────────────────────────
# TAB 5: SPEARMAN KORELASYON HEATMAP
# ────────────────────────────────────────────────
with tab5:
    st.subheader("Spearman Korelasyon Heatmap")

    extra_vars = []
    for c in ['YAŞ', 'HASTALIK SÜRESİ(yıl)']:
        if c in df.columns: extra_vars.append(c)

    all_corr_vars = extra_vars + target_cols
    selected_vars = st.multiselect("Korelasyona dahil edilecek değişkenler",
                                    all_corr_vars, default=all_corr_vars)

    subset = st.radio("Hangi hasta grubu üzerinde?",
                      ["Tüm hastalar", "Saf Mukokutanöz", "Komplike Behçet"],
                      horizontal=True)

    if subset == "Tüm hastalar":
        df_corr = df[selected_vars].dropna()
    else:
        df_corr = df[df['Birlesik_Grup']==subset][selected_vars].dropna()

    if len(df_corr) >= 5 and len(selected_vars) >= 2:
        corr = df_corr.corr(method='spearman')

        # Alt üçgen maskesi
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(max(8, len(selected_vars)*0.9),
                                         max(7, len(selected_vars)*0.8)))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={'shrink':0.7},
                    annot_kws={'size':10}, ax=ax)
        ax.set_title(f"Spearman Korelasyon Heatmap ({subset}, n={len(df_corr)})",
                     fontsize=13, pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        # P-değerleri tablosu (anlamlı olanlar için)
        with st.expander("📋 Anlamlı korelasyonlar (p < α)"):
            sig_rows = []
            for i in range(len(selected_vars)):
                for j in range(i+1, len(selected_vars)):
                    a, b = selected_vars[i], selected_vars[j]
                    if a in df_corr.columns and b in df_corr.columns:
                        try:
                            r, p = stats.spearmanr(df_corr[a], df_corr[b])
                            if p < alpha:
                                sig_rows.append({
                                    'Değişken 1': a, 'Değişken 2': b,
                                    'r (Spearman)': f"{r:.3f}", 'p': fmt_p(p)
                                })
                        except Exception:
                            pass
            if sig_rows:
                sig_df_sorted = pd.DataFrame(sig_rows).sort_values('p').reset_index(drop=True)
                st.dataframe(sig_df_sorted)
                download_buttons_for_table(
                    sig_df_sorted, "anlamli_korelasyonlar",
                    label_prefix="Anlamlı korelasyonlar — ",
                    title="Anlamlı Spearman Korelasyonları (p < α)"
                )
            else:
                st.info("Anlamlı korelasyon bulunamadı.")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Heatmap PNG indir (300 DPI)", buf.getvalue(),
                           "correlation_heatmap.png", "image/png",
                           key="fig_heatmap")
    else:
        st.warning("Yeterli veri yok (en az 5 hasta + 2 değişken gerekli).")

# ────────────────────────────────────────────────
# TAB 6: ROC ANALİZİ (Figure 2 tarzı)
# ────────────────────────────────────────────────
with tab6:
    st.subheader("ROC Eğri Analizi — Tanısal Performans")

    if not HAS_SKLEARN:
        st.error("scikit-learn paketi gerekli: `pip install scikit-learn`")
    else:
        st.caption("Komplike Behçet'i Saf Mukokutanöz'den ayırt etme gücü.")

        # Pozitif sınıf: Komplike Behçet
        df_roc = df.dropna(subset=target_cols + ['Birlesik_Grup'])
        y_true = (df_roc['Birlesik_Grup'] == 'Komplike Behçet').astype(int)

        sel_roc = st.multiselect("ROC için parametreler",
                                  target_cols, default=target_cols)

        if sel_roc and len(y_true.unique()) == 2:
            fig, ax = plt.subplots(figsize=(8, 7))
            colors = plt.cm.tab10(np.linspace(0, 1, len(sel_roc)))

            roc_results = []
            for i, param in enumerate(sel_roc):
                scores = df_roc[param].values
                fpr, tpr, thresholds = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)

                # AUC < 0.5 → parametre TERS YÖNLÜ prediktör
                # (örn. düşük LENFOSİT komplike hastalığı öngörür)
                if roc_auc < 0.5:
                    # Skoru ters çevirerek gerçek AUC'yi al
                    fpr, tpr, thr_flip = roc_curve(y_true, -scores)
                    roc_auc = auc(fpr, tpr)
                    thresholds = -thr_flip  # eşiği orijinal birime geri çevir
                    direction = "↓"  # düşük değer komplikeyi öngörür
                    direction_label = "düşük değer → Komplike"
                else:
                    direction = "↑"  # yüksek değer komplikeyi öngörür
                    direction_label = "yüksek değer → Komplike"

                # Youden J ile optimal cut-off
                j_scores = tpr - fpr
                opt_idx = np.argmax(j_scores)
                opt_thr = thresholds[opt_idx]
                opt_sens = tpr[opt_idx]
                opt_spec = 1 - fpr[opt_idx]

                # Eğri etiketi: ters yönlüleri belirt
                label_suffix = " ↓" if direction == "↓" else ""
                ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f"{param}{label_suffix} (AUC = {roc_auc:.3f})")

                # Cut-off yorumu: ters yönlüde "≤", normal yönde "≥"
                cutoff_op = "≤" if direction == "↓" else "≥"

                roc_results.append({
                    'Parametre': param,
                    'Yön': direction,
                    'AUC': f"{roc_auc:.3f}",
                    'Cut-off': f"{cutoff_op} {opt_thr:.2f}",
                    'Sensitivite': f"{opt_sens:.2%}",
                    'Spesifite': f"{opt_spec:.2%}",
                    'Youden J': f"{j_scores[opt_idx]:.3f}",
                    'Yorum': direction_label
                })

            ax.plot([0,1], [0,1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
            ax.set_title("Multiple ROC Curves\n(Komplike vs Saf Mukokutanöz Behçet)",
                         fontsize=13)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Optimal Cut-off Değerleri (Youden J)")
            roc_df = pd.DataFrame(roc_results).sort_values('AUC', ascending=False).reset_index(drop=True)
            st.dataframe(roc_df, use_container_width=True)

            download_buttons_for_table(
                roc_df, "roc_sonuclari",
                label_prefix="ROC sonuçları — ",
                title="ROC Analizi — Optimal Cut-off Değerleri (Youden J)"
            )

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            st.download_button("🖼️ ROC eğrisi PNG indir (300 DPI)", buf.getvalue(),
                               "roc_curves.png", "image/png",
                               key="fig_roc_curves")

            # ─── DeLong testi: AUC'ler arası karşılaştırma ───
            st.markdown("---")
            st.subheader("📐 DeLong Testi — AUC'ler arası anlamlı fark var mı?")
            st.caption("İki bağımlı ROC eğrisini istatistiksel olarak karşılaştırır "
                       "(aynı hastalar, farklı belirteçler). p < 0.05 → AUC'ler "
                       "anlamlı derecede farklı.")

            # Her belirteç için "etkin" skorları (ters yönlüleri çevrilmiş) hazırla
            score_map = {}
            for param in sel_roc:
                raw = df_roc[param].values
                score_map[param] = effective_scores(y_true.values, raw)

            # Çift yönlü DeLong matrisi
            n = len(sel_roc)
            pval_matrix = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(i+1, n):
                    try:
                        _, _, _, p = delong_test(
                            y_true.values,
                            score_map[sel_roc[i]],
                            score_map[sel_roc[j]]
                        )
                        pval_matrix[i, j] = p
                        pval_matrix[j, i] = p
                    except Exception:
                        pass

            pval_df = pd.DataFrame(pval_matrix, index=sel_roc, columns=sel_roc)

            # Renkli görselleştirme: anlamlı farkları vurgula
            def fmt_cell(v):
                if pd.isna(v): return "—"
                if v < 0.001: return "<0.001"
                return f"{v:.3f}"

            display_df = pval_df.map(fmt_cell)

            def hl_sig(v):
                try:
                    if v == "—": return ''
                    p = 0.0005 if v == "<0.001" else float(v)
                    if p < 0.001: return 'background-color: #2E86AB; color: white'
                    if p < 0.01:  return 'background-color: #A4C9E1'
                    if p < 0.05:  return 'background-color: #D4EFDF'
                    return ''
                except: return ''

            st.dataframe(display_df.style.map(hl_sig), use_container_width=True)
            st.caption("🟦 koyu mavi: p<0.001 | 🟨 açık mavi: p<0.01 | 🟩 yeşil: p<0.05")

            # DeLong matrisi için indirme (index'i sütun yap)
            delong_export = display_df.reset_index().rename(columns={'index': 'Parametre'})
            download_buttons_for_table(
                delong_export, "delong_pmatrix",
                label_prefix="DeLong p-matrisi — ",
                title="DeLong Testi — Çift Yönlü AUC Karşılaştırma Matrisi"
            )

            # En yüksek AUC'li belirteçle diğerlerini özetle
            best_idx = roc_df['AUC'].astype(float).idxmax() if len(roc_df) > 0 else None
            if best_idx is not None and best_idx in roc_df.index:
                best_param = roc_df.loc[best_idx, 'Parametre']
                st.markdown(f"**En yüksek AUC: {best_param}**. Diğer belirteçlerle DeLong karşılaştırması:")
                rows = []
                for p in sel_roc:
                    if p == best_param: continue
                    try:
                        a1, a2, z, pv = delong_test(
                            y_true.values,
                            score_map[best_param],
                            score_map[p]
                        )
                        rows.append({
                            'Karşılaştırma': f"{best_param} vs {p}",
                            f'AUC ({best_param})': f"{a1:.3f}",
                            f'AUC ({p})': f"{a2:.3f}",
                            'ΔAUC': f"{a1-a2:+.3f}",
                            'z': f"{z:.3f}",
                            'p': fmt_cell(pv)
                        })
                    except Exception:
                        pass
                if rows:
                    delong_compare_df = pd.DataFrame(rows)
                    st.dataframe(delong_compare_df, use_container_width=True)
                    download_buttons_for_table(
                        delong_compare_df, "delong_karsilastirma",
                        label_prefix="DeLong karşılaştırma — ",
                        title=f"DeLong Testi — {best_param} vs Diğer Belirteçler"
                    )

# ────────────────────────────────────────────────
# YAN PANEL NOTLARI
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# TAB 7: LOJİSTİK REGRESYON + NOMOGRAM
# ────────────────────────────────────────────────
with tab7:
    st.subheader("🔬 Lojistik Regresyon — Covariate Düzeltmesi + Nomogram")

    if not HAS_STATSMODELS or not HAS_SKLEARN:
        st.error("Bu sekme için `statsmodels` ve `scikit-learn` gerekli.")
    else:
        import statsmodels.api as sm

        st.info(
            "📑 **Bu sekmede sırasıyla:** "
            "Ana belirteç + covariate seçimi → OR tablosu → Forest plot → "
            "Düzeltilmiş ROC → **Nomogram** → **📏 Kalibrasyon eğrisi** → "
            "**🔄 Bootstrap iç validasyon** → **👥 Subgroup analizi** → "
            "İnteraktif risk hesaplayıcı. Hepsini görmek için aşağı kaydırın."
        )

        st.markdown(
            "**Bağımlı değişken:** Komplike Behçet (1) vs Saf Mukokutanöz (0)"
        )
        st.caption(
            "Lojistik regresyon ile birden fazla belirteci aynı modele koyarsın; "
            "yaş, hastalık süresi, ilaç gibi karıştırıcıları (covariate) düzelterek "
            "**her belirtecin bağımsız katkısını** ölçer."
        )

        col1, col2 = st.columns(2)
        with col1:
            default_marker = 'SIRI' if 'SIRI' in target_cols else target_cols[0]
            primary_marker = st.selectbox(
                "Ana belirteç",
                target_cols,
                index=target_cols.index(default_marker)
            )

        # Olası covariate'leri tespit et
        # Hücre sayıları (NEU/MONOSİT/LENFOSİT) ve klinik değişkenler birlikte
        covariate_pool = []
        for c in ['NEU', 'MONOSİT', 'LENFOSİT', 'PLT',
                   'YAŞ', 'HASTALIK SÜRESİ(yıl)',
                   'Kolşisin', 'Biyolojik', 'DMARD']:
            if c in df.columns: covariate_pool.append(c)

        with col2:
            covariates = st.multiselect(
                "Eş değişkenler (covariates)",
                covariate_pool,
                default=[c for c in ['YAŞ', 'HASTALIK SÜRESİ(yıl)']
                          if c in covariate_pool]
            )

        # Veriyi hazırla
        predictors = [primary_marker] + [c for c in covariates if c != primary_marker]
        data_lr = df.dropna(subset=predictors + ['Birlesik_Grup']).copy()
        y_lr = (data_lr['Birlesik_Grup'] == 'Komplike Behçet').astype(int)
        X_lr = data_lr[predictors].copy()
        X_lr_const = sm.add_constant(X_lr, has_constant='add')

        # ─── COMPLETE SEPARATION TESPİTİ ───
        separation_warnings = []
        for c in predictors:
            col_data = data_lr[c]
            # Sadece binary/az unique değişkenler için anlamlı
            if col_data.nunique() <= 5:
                ct = pd.crosstab(col_data, y_lr)
                # Herhangi bir hücre 0 mı?
                zero_cells = (ct == 0).sum().sum()
                if zero_cells > 0:
                    # Hangi değer hangi gruba özgü?
                    for val in ct.index:
                        row = ct.loc[val]
                        if (row == 0).any():
                            absent_group = "Komplike Behçet" if row[1] == 0 else "Saf Mukokutanöz"
                            present_count = int(row.sum())
                            separation_warnings.append({
                                'variable': c,
                                'value': val,
                                'absent_group': absent_group,
                                'present_count': present_count,
                                'crosstab': ct
                            })

        if separation_warnings:
            with st.container():
                st.error(
                    "⚠️ **COMPLETE SEPARATION TESPİT EDİLDİ** — Standart lojistik regresyon "
                    "bazı değişkenlerde **şişirilmiş** OR ve CI değerleri verecek. "
                    "Aşağıdaki **Firth penalized regression** seçeneğini kullanmanı öneririm."
                )
                for w in separation_warnings:
                    st.markdown(
                        f"- **{w['variable']} = {w['value']}** olan {w['present_count']} hasta "
                        f"sadece tek bir grupta var (**{w['absent_group']}** grubunda hiç yok). "
                        f"Bu durum OR'nin sonsuza yakın çıkmasına neden olur."
                    )
                    with st.expander(f"📋 {w['variable']} × grup çapraz tablosu"):
                        st.dataframe(w['crosstab'])

        # ─── Yöntem seçimi: Standart MLE vs Firth ───
        use_firth_default = bool(separation_warnings) and HAS_FIRTH
        method_help = (
            "**Standart MLE:** Klasik lojistik regresyon. Complete separation varsa OR şişer.\n\n"
            "**Firth penalized:** Jeffreys prior ile düzenlenmiş; az örnek veya separation durumunda güvenilir."
        )
        method_choice = st.radio(
            "Regresyon yöntemi",
            ["Standart MLE", "Firth penalized"] if HAS_FIRTH else ["Standart MLE"],
            index=1 if use_firth_default else 0,
            horizontal=True,
            help=method_help
        )

        if method_choice == "Firth penalized" and not HAS_FIRTH:
            st.warning("`firthlogist` paketi kurulu değil. `pip install firthlogist` ile kurun.")
            method_choice = "Standart MLE"

        use_firth = (method_choice == "Firth penalized")

        try:
            # ─── Model fit (MLE veya Firth) ───
            if use_firth:
                # Manuel Firth implementasyonu
                firth_res = firth_logistic_regression(X_lr_const.values, y_lr.values)
                params = pd.Series(firth_res['beta'], index=X_lr_const.columns)
                conf = pd.DataFrame({
                    0: firth_res['ci_lo'],
                    1: firth_res['ci_hi']
                }, index=X_lr_const.columns)
                pvals = pd.Series(firth_res['pvals'], index=X_lr_const.columns)
                pred_full = pd.Series(firth_res['fitted_prob'], index=data_lr.index)
                # Pseudo-R²
                ll_model = firth_res['loglik']
                p_null = y_lr.mean()
                ll_null = (y_lr * np.log(p_null + 1e-12) +
                            (1-y_lr) * np.log(1 - p_null + 1e-12)).sum()
                pseudo_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else np.nan
                aic_val = -2 * ll_model + 2 * (len(predictors) + 1)
                llr_p = stats.chi2.sf(2*(ll_model - ll_null), df=len(predictors))
                model_label = "Firth Penalized Regression"
            else:
                model = sm.Logit(y_lr, X_lr_const).fit(disp=False, maxiter=100)
                params = model.params
                conf = model.conf_int()
                pvals = model.pvalues
                pred_full = model.predict(X_lr_const)
                pseudo_r2 = model.prsquared
                aic_val = model.aic
                llr_p = model.llr_pvalue
                model_label = "Standart MLE Logistic Regression"

            st.caption(f"📋 Yöntem: **{model_label}**")

            # ─── Odds Ratio Tablosu ───
            st.markdown("### 📊 Model Çıktısı: Odds Ratios (95% CI)")

            or_rows = []
            for var in params.index:
                if var == 'const': continue
                or_val = np.exp(params[var])
                or_lo = np.exp(conf.loc[var, 0])
                or_hi = np.exp(conf.loc[var, 1])
                # Çok büyük OR'leri uyarı ile göster
                or_str = f"{or_val:.3f}" if or_val < 1000 else f"{or_val:.2e}"
                ci_str = (f"[{or_lo:.3f} – {or_hi:.3f}]"
                          if or_hi < 1000 else f"[{or_lo:.2e} – {or_hi:.2e}]")
                or_rows.append({
                    'Değişken': var,
                    'β (coef)': f"{params[var]:+.3f}",
                    'OR': or_str,
                    '95% CI': ci_str,
                    'p': fmt_p(pvals[var])
                })
            or_df = pd.DataFrame(or_rows)

            def hl_or_p(v):
                try:
                    if v == "<0.001": return 'background-color: #D4EFDF'
                    return 'background-color: #D4EFDF' if float(v) < alpha else ''
                except: return ''

            st.dataframe(or_df.style.map(hl_or_p, subset=['p']),
                         use_container_width=True)

            download_buttons_for_table(
                or_df, "regresyon_OR_tablosu",
                label_prefix="OR tablosu — ",
                title="Lojistik Regresyon — Odds Ratios (95% CI)"
            )

            # Model fit istatistikleri
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("McFadden Pseudo R²", f"{pseudo_r2:.3f}")
            c2.metric("AIC", f"{aic_val:.1f}")
            c3.metric("LLR p-value", fmt_p(llr_p))
            c4.metric("Konkordans (n)", f"{int(y_lr.sum())}/{len(y_lr)}")

            # ─── OR Forest Plot ───
            st.markdown("### 🌳 Forest Plot (Odds Ratio)")
            fig_or, ax_or = plt.subplots(figsize=(9, max(3, 0.5*len(predictors)+2)))
            var_names = [v for v in params.index if v != 'const']
            ors_raw = [np.exp(params[v]) for v in var_names]
            ci_lo_raw = [np.exp(conf.loc[v, 0]) for v in var_names]
            ci_hi_raw = [np.exp(conf.loc[v, 1]) for v in var_names]

            # Görsel için ekstrem CI'leri kırp (1e-3 ile 1e6 arası)
            CLIP_LO, CLIP_HI = 1e-3, 1e6
            ors = [max(CLIP_LO, min(CLIP_HI, o)) for o in ors_raw]
            ci_lo = [max(CLIP_LO, min(CLIP_HI, c)) for c in ci_lo_raw]
            ci_hi = [max(CLIP_LO, min(CLIP_HI, c)) for c in ci_hi_raw]
            clipped_flags = [(o > CLIP_HI or c_hi > CLIP_HI or c_lo < CLIP_LO)
                              for o, c_lo, c_hi in zip(ors_raw, ci_lo_raw, ci_hi_raw)]

            y_pos = np.arange(len(var_names))
            ax_or.errorbar(ors, y_pos,
                            xerr=[np.array(ors)-np.array(ci_lo),
                                  np.array(ci_hi)-np.array(ors)],
                            fmt='o', color='#2E86AB', ecolor='gray',
                            capsize=4, markersize=8)
            ax_or.axvline(1, color='red', linestyle='--', alpha=0.6, label='OR = 1')
            ax_or.set_yticks(y_pos)
            # Etiketlere "*" ekle eğer clipped ise
            labels = [v + (" *" if f else "") for v, f in zip(var_names, clipped_flags)]
            ax_or.set_yticklabels(labels)
            ax_or.set_xscale('log')
            ax_or.set_xlabel("Odds Ratio (log scale)")
            ax_or.set_title("Düzeltilmiş Odds Ratios — 95% CI ile" +
                              ("\n(* CI sınırları görsel için kırpıldı)" if any(clipped_flags) else ""))
            ax_or.legend()
            ax_or.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_or)
            download_fig_300dpi(fig_or, "regresyon_forest_plot",
                                  label="Forest plot PNG indir (300 DPI)")

            # ─── Düzeltilmiş vs Düzeltilmemiş ROC ───
            st.markdown("---")
            st.markdown("### 🎯 Düzeltilmiş vs Düzeltilmemiş ROC")

            # Düzeltilmemiş: sadece ana belirteç tek başına
            if use_firth:
                X_single_const = sm.add_constant(data_lr[[primary_marker]], has_constant='add')
                firth_single = firth_logistic_regression(X_single_const.values, y_lr.values)
                pred_single = pd.Series(firth_single['fitted_prob'], index=data_lr.index)
            else:
                X_single = sm.add_constant(data_lr[[primary_marker]], has_constant='add')
                model_single = sm.Logit(y_lr, X_single).fit(disp=False, maxiter=100)
                pred_single = model_single.predict(X_single)

            fpr1, tpr1, _ = roc_curve(y_lr, pred_single)
            auc1 = auc(fpr1, tpr1)
            fpr2, tpr2, _ = roc_curve(y_lr, pred_full)
            auc2 = auc(fpr2, tpr2)

            fig_adj, ax_adj = plt.subplots(figsize=(8, 7))
            ax_adj.plot(fpr1, tpr1, lw=2.5, color='#E07A5F',
                        label=f"{primary_marker} tek başına (AUC = {auc1:.3f})")
            ax_adj.plot(fpr2, tpr2, lw=2.5, color='#3D405B',
                        label=f"+ covariates ({len(covariates)} ek değişken) AUC = {auc2:.3f}")
            ax_adj.plot([0,1], [0,1], 'k--', alpha=0.4)
            ax_adj.set_xlabel("1 − Spesifite (FPR)", fontsize=12)
            ax_adj.set_ylabel("Sensitivite (TPR)", fontsize=12)
            ax_adj.set_title("Düzeltilmemiş vs Düzeltilmiş ROC", fontsize=13)
            ax_adj.legend(loc='lower right')
            ax_adj.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_adj)
            download_fig_300dpi(fig_adj, "duzeltilmis_ROC",
                                  label="Düzeltilmiş ROC PNG indir (300 DPI)")

            # DeLong testi: tek vs tam model
            try:
                _, _, z_dl, p_dl = delong_test(y_lr.values,
                                                 pred_single.values,
                                                 pred_full.values)
                st.info(
                    f"**DeLong testi (tek belirteç vs düzeltilmiş model):** "
                    f"ΔAUC = {auc2-auc1:+.3f}, z = {z_dl:.3f}, p = **{fmt_p(p_dl)}**"
                )
                if p_dl < alpha:
                    st.success(
                        "Covariate eklemek modelin ayırt etme gücünü ANLAMLI olarak değiştirdi."
                    )
                else:
                    st.warning(
                        f"{primary_marker} büyük ölçüde **bağımsız** bir belirteç — "
                        "yaş/hastalık süresi/ilaç düzeltmesi modeli istatistiksel olarak "
                        "anlamlı derecede iyileştirmedi."
                    )
            except Exception as e:
                st.warning(f"DeLong hesaplanamadı: {e}")

            # ─── NOMOGRAM ───
            st.markdown("---")
            st.markdown("### 📐 Nomogram")
            with st.expander("ℹ️ Nomogram nasıl okunur?"):
                st.markdown("""
                Her satırda hastanın değerini bul → üstteki **Points** ekseninden 
                puanı oku → tüm değişkenlerin puanlarını topla → en alttaki 
                **Total Points** ekseninde toplam puanı işaretle → karşı eksenden 
                **Predicted Probability** (komplike Behçet olasılığı) değerini oku.
                """)

            try:
                # Her değişken için katkı aralığı (params zaten yukarıda tanımlandı)
                intercept = params['const']
                var_info = []
                for v in predictors:
                    x_min, x_max = float(data_lr[v].min()), float(data_lr[v].max())
                    b = float(params[v])
                    is_binary = data_lr[v].nunique() <= 2
                    rng = abs(b * (x_max - x_min))
                    var_info.append({
                        'name': v, 'b': b,
                        'x_min': x_min, 'x_max': x_max,
                        'is_binary': is_binary, 'range': rng
                    })

                max_range = max(vi['range'] for vi in var_info) if var_info else 1
                scale = 100.0 / max_range if max_range > 0 else 1
                total_max = sum(vi['range'] for vi in var_info) * scale

                # Her değişkenin kendi puan aralığı (eksen genişliği için)
                for vi in var_info:
                    vi['max_pts'] = vi['range'] * scale  # bu değişkenin maksimum katkısı (puan)

                # Yardımcı: yuvarlanmış "güzel" tick aralıkları
                def nice_ticks(x_min, x_max, target_n=5):
                    """Güzel yuvarlanmış tick'ler üret + x_min ve x_max'i de dahil et."""
                    rng = x_max - x_min
                    if rng <= 0:
                        return [x_min], 1
                    raw_step = rng / target_n
                    magnitude = 10 ** np.floor(np.log10(raw_step))
                    for mult in [1, 2, 2.5, 5, 10]:
                        step = mult * magnitude
                        if rng / step <= target_n + 2:
                            break
                    start = np.ceil(x_min / step) * step
                    ticks = list(np.arange(start, x_max + step*0.5, step))
                    ticks = [t for t in ticks if x_min - step*0.01 <= t <= x_max + step*0.01]
                    # x_min ve x_max'i ekle (yakın değilse)
                    if not ticks or abs(ticks[0] - x_min) > step * 0.25:
                        ticks = [x_min] + ticks
                    if abs(ticks[-1] - x_max) > step * 0.25:
                        ticks = ticks + [x_max]
                    return ticks, step

                # Şekil: hepsi 0-100 ekseninde, ama her değişken kendi max_pts kadarını kullanır
                n_axes = 1 + len(predictors) + 2
                fig_n, axes_n = plt.subplots(n_axes, 1,
                                               figsize=(14, 0.85*n_axes + 1.5))
                plt.subplots_adjust(hspace=1.4, left=0.18, right=0.96,
                                     top=0.93, bottom=0.05)

                def draw_axis(ax, xmin, xmax, ticks, labels, title,
                                color='black', fontsize=10, axis_end=100):
                    """axis_end: eksenin görsel olarak uzandığı yer (puan)."""
                    ax.set_xlim(-2, axis_end + 2)
                    ax.set_ylim(0, 1)
                    # Sadece kullanılan bölgede çizgi çiz (sağa boş uzantı yok)
                    line_end = max(ticks) if ticks else xmax
                    ax.hlines(0.5, xmin, line_end, color=color, lw=1.5)
                    for t, lbl in zip(ticks, labels):
                        ax.vlines(t, 0.4, 0.6, color=color, lw=1.2)
                        ax.text(t, -0.15, lbl, ha='center', va='top',
                                fontsize=fontsize)
                    ax.set_yticks([]); ax.set_xticks([])
                    ax.set_ylabel(title, rotation=0, ha='right', va='center',
                                   fontsize=10.5, labelpad=20, fontweight='bold')
                    for sp in ax.spines.values(): sp.set_visible(False)

                # 1) Üst eksen: Points (0-100, 10'arlı)
                pts_ticks = list(np.arange(0, 101, 10))
                draw_axis(axes_n[0], 0, 100, pts_ticks,
                           [str(int(t)) for t in pts_ticks], "Points",
                           color='#2E86AB', fontsize=9, axis_end=100)

                # 2) Her değişken için bir eksen — kendi max_pts kadarını kullanır
                # Tüm eksenler "düşük risk → yüksek risk" yönünde (soldan sağa puan artar)
                for i, vi in enumerate(var_info):
                    ax = axes_n[i + 1]
                    if vi['is_binary']:
                        # 0 ve 1 — düşük katkılı sol, yüksek katkılı sağ
                        if vi['b'] > 0:
                            # b>0: değer arttıkça risk artar → 0 sol, 1 sağ
                            pts = [0, vi['max_pts']]
                            labels = ['0', '1']
                        else:
                            # b<0: değer arttıkça risk azalır → 1 sol, 0 sağ
                            pts = [0, vi['max_pts']]
                            labels = ['1', '0']
                        draw_axis(ax, 0, vi['max_pts'], pts, labels,
                                    vi['name'], fontsize=10, axis_end=100)
                    else:
                        # Continuous — tick yoğunluğu eksenin gerçek genişliğine göre
                        if vi['max_pts'] < 20:
                            target_n = 2
                        elif vi['max_pts'] < 40:
                            target_n = 3
                        elif vi['max_pts'] < 70:
                            target_n = 4
                        else:
                            target_n = 5
                        x_ticks, x_step = nice_ticks(vi['x_min'], vi['x_max'],
                                                       target_n=target_n)
                        if x_step >= 1:
                            x_labels = [f"{t:.0f}" for t in x_ticks]
                        else:
                            decimals = max(1, int(-np.floor(np.log10(x_step))))
                            x_labels = [f"{t:.{decimals}f}" for t in x_ticks]

                        # b>0: x_min → 0 puan, x_max → max puan (etiket sırası: küçükten büyüğe)
                        # b<0: x_min → max puan, x_max → 0 puan
                        #      → etiketler GÖRSEL OLARAK ters yönde, ama biz fiziksel yön olarak
                        #        sol=0puan, sağ=maxpuan tutuyoruz. Bu yüzden b<0 ise
                        #        etiketler büyükten küçüğe gider (mantıklı, çünkü sol az risk)
                        if vi['b'] > 0:
                            pts = [(t - vi['x_min']) * vi['b'] * scale
                                   for t in x_ticks]
                        else:
                            pts = [(vi['x_max'] - t) * abs(vi['b']) * scale
                                   for t in x_ticks]

                        if vi['max_pts'] < 20:
                            min_gap = vi['max_pts'] / 3
                        else:
                            min_gap = max(8, vi['max_pts'] / 8)
                        order = np.argsort(pts)
                        pts_s = [pts[k] for k in order]
                        lbl_s = [x_labels[k] for k in order]
                        kept_pts, kept_lbl = [pts_s[0]], [lbl_s[0]]
                        for p, l in zip(pts_s[1:], lbl_s[1:]):
                            if p - kept_pts[-1] >= min_gap:
                                kept_pts.append(p)
                                kept_lbl.append(l)
                        # Eksen başlığına β yönünü ekle (≥/≤ işaretiyle)
                        direction_marker = " ↑" if vi['b'] > 0 else " ↓"
                        draw_axis(ax, 0, vi['max_pts'], kept_pts, kept_lbl,
                                    vi['name'] + direction_marker,
                                    fontsize=9, axis_end=100)

                # 3) Total Points — daha az tick
                tot_target = 8
                tot_step_raw = total_max / tot_target
                tot_mag = 10 ** np.floor(np.log10(tot_step_raw))
                for mult in [1, 2, 2.5, 5, 10]:
                    tot_step = mult * tot_mag
                    if total_max / tot_step <= tot_target + 2: break
                tot_ticks = list(np.arange(0, total_max + tot_step*0.5, tot_step))
                draw_axis(axes_n[-2], 0, total_max, tot_ticks,
                           [f"{t:.0f}" for t in tot_ticks], "Total Points",
                           color='#A23B72', fontsize=9, axis_end=total_max)

                # 4) Predicted Probability — TEMSILI olasılıklar, ÇAKIŞMASIZ
                # Pratik olasılık seviyeleri:
                prob_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
                prob_ticks_raw, prob_labels_raw = [], []
                min_contribs_sum = sum(
                    vi['b'] * vi['x_min'] if vi['b'] > 0 else vi['b'] * vi['x_max']
                    for vi in var_info
                )
                for p in prob_levels:
                    logit_p = np.log(p / (1 - p))
                    lp_total = logit_p - intercept
                    tp = (lp_total - min_contribs_sum) * scale
                    if 0 <= tp <= total_max:
                        prob_ticks_raw.append(tp)
                        prob_labels_raw.append(f"{p:.2f}")

                # Çakışma kontrolü — gevşek (etiketler kısa, yer yetebilir)
                min_gap_prob = total_max * 0.04
                if prob_ticks_raw:
                    prob_ticks = [prob_ticks_raw[0]]
                    prob_labels = [prob_labels_raw[0]]
                    for t, l in zip(prob_ticks_raw[1:], prob_labels_raw[1:]):
                        if t - prob_ticks[-1] >= min_gap_prob:
                            prob_ticks.append(t)
                            prob_labels.append(l)
                else:
                    prob_ticks, prob_labels = [], []

                draw_axis(axes_n[-1], 0, total_max, prob_ticks, prob_labels,
                           "Predicted\nProbability\n(Komplike)",
                           color='#F18F01', fontsize=9, axis_end=total_max)

                fig_n.suptitle(
                    f"Nomogram — Komplike Behçet Risk Tahmini\n"
                    f"(Model AUC = {auc2:.3f}, n = {len(data_lr)}, {model_label})",
                    fontsize=11, y=0.99
                )
                st.pyplot(fig_n)
                download_fig_300dpi(fig_n, "nomogram",
                                      label="Nomogram PNG indir (300 DPI)")
            except Exception as e:
                st.warning(f"Nomogram çizilemedi: {e}")

            # ─── KALİBRASYON EĞRİSİ + HOSMER-LEMESHOW ───
            st.markdown("---")
            st.markdown("# 📏 Kalibrasyon — Modelin Güvenilirliği")
            with st.expander("ℹ️ Kalibrasyon nedir, niye önemli?"):
                st.markdown("""
                **Ayırt etme (discrimination)** ≠ **kalibrasyon**.
                - **AUC** modelin "hangi hastanın komplike olduğunu" doğru sıralayıp sıralamadığını söyler.
                - **Kalibrasyon** ise "%80 risk dedik, gerçekten %80'i komplike çıktı mı?" sorusunu cevaplar.

                Bir model AUC'si yüksek olsa bile **sistematik olarak yüksek/düşük risk** tahmin edebilir.
                Nomogram makalelerinde **kalibrasyon eğrisi mecburidir**.

                - **Diagonal yakın eğri** = iyi kalibrasyon
                - **Hosmer-Lemeshow p > 0.05** = anlamlı sapma yok (iyi)
                - **Brier skoru** (0–1, düşük iyi); 0.25 = chance, 0 = mükemmel
                """)

            try:
                # Tahmin edilen olasılıklar
                pred_probs = np.asarray(pred_full).flatten()
                y_actual = y_lr.values

                # 10 desile böl (Hosmer-Lemeshow standart)
                n_bins = 10
                # Eşit-frekans bins (decile)
                bins = np.unique(np.quantile(pred_probs, np.linspace(0, 1, n_bins+1)))
                if len(bins) < 3:
                    n_bins = max(3, len(bins) - 1)
                    bins = np.unique(np.quantile(pred_probs, np.linspace(0, 1, n_bins+1)))
                bin_idx = np.digitize(pred_probs, bins[1:-1])

                cal_data = []
                hl_stat = 0
                for k in range(n_bins):
                    mask = (bin_idx == k)
                    n_k = mask.sum()
                    if n_k == 0: continue
                    obs_freq = y_actual[mask].mean()
                    exp_freq = pred_probs[mask].mean()
                    obs_count = y_actual[mask].sum()
                    exp_count = pred_probs[mask].sum()
                    cal_data.append({
                        'bin': k+1, 'n': int(n_k),
                        'observed': obs_freq, 'expected': exp_freq,
                        'obs_count': int(obs_count), 'exp_count': exp_count
                    })
                    # Hosmer-Lemeshow chi-square
                    if exp_count > 0 and (n_k - exp_count) > 0:
                        hl_stat += ((obs_count - exp_count)**2 / exp_count +
                                     ((n_k - obs_count) - (n_k - exp_count))**2 /
                                     (n_k - exp_count))
                hl_df = max(1, n_bins - 2)
                hl_p = stats.chi2.sf(hl_stat, df=hl_df)

                # Brier skoru
                brier = np.mean((pred_probs - y_actual)**2)

                # Calibration plot
                fig_cal, ax_cal = plt.subplots(figsize=(8, 7))
                obs = [d['observed'] for d in cal_data]
                exp = [d['expected'] for d in cal_data]
                sizes = [d['n']*8 for d in cal_data]

                # Mükemmel kalibrasyon çizgisi
                ax_cal.plot([0,1], [0,1], 'k--', alpha=0.5, lw=1.5,
                            label='Mükemmel kalibrasyon')
                # Veri noktaları
                ax_cal.scatter(exp, obs, s=sizes, alpha=0.7,
                                c='#2E86AB', edgecolors='black', zorder=3,
                                label='Decile grupları (boyut = n)')
                # Lowess eğrisi (basit polinom uydurma)
                if len(exp) >= 3:
                    try:
                        # 2. derece polinom fit (smoothing için)
                        sort_idx = np.argsort(exp)
                        x_sorted = np.array(exp)[sort_idx]
                        y_sorted = np.array(obs)[sort_idx]
                        coefs = np.polyfit(x_sorted, y_sorted, deg=min(2, len(exp)-1))
                        x_smooth = np.linspace(0, 1, 100)
                        y_smooth = np.polyval(coefs, x_smooth)
                        y_smooth = np.clip(y_smooth, 0, 1)
                        ax_cal.plot(x_smooth, y_smooth, color='#E07A5F',
                                     lw=2, alpha=0.8, label='Lowess fit')
                    except Exception:
                        pass

                ax_cal.set_xlim(-0.02, 1.02)
                ax_cal.set_ylim(-0.02, 1.02)
                ax_cal.set_xlabel("Tahmin edilen olasılık", fontsize=12)
                ax_cal.set_ylabel("Gözlenen olasılık", fontsize=12)
                ax_cal.set_title(
                    f"Kalibrasyon Eğrisi\n"
                    f"Hosmer-Lemeshow χ² = {hl_stat:.2f}, p = {fmt_p(hl_p)} · "
                    f"Brier = {brier:.3f}",
                    fontsize=12
                )
                ax_cal.legend(loc='upper left', fontsize=10)
                ax_cal.grid(alpha=0.3)
                ax_cal.set_aspect('equal')
                plt.tight_layout()
                st.pyplot(fig_cal)

                # Yorumlama
                c1, c2, c3 = st.columns(3)
                c1.metric("Hosmer-Lemeshow p", fmt_p(hl_p),
                            "İyi kalibrasyon" if hl_p > 0.05 else "Sapma var")
                c2.metric("Brier Skoru", f"{brier:.3f}",
                            "İyi" if brier < 0.20 else ("Orta" if brier < 0.25 else "Zayıf"))
                c3.metric("Decile sayısı", str(len(cal_data)))

                # Detaylı tablo
                with st.expander("📋 Kalibrasyon tablosu (decile bazlı)"):
                    cal_df = pd.DataFrame(cal_data)
                    cal_df['observed'] = cal_df['observed'].apply(lambda x: f"{x:.3f}")
                    cal_df['expected'] = cal_df['expected'].apply(lambda x: f"{x:.3f}")
                    cal_df['exp_count'] = cal_df['exp_count'].apply(lambda x: f"{x:.2f}")
                    cal_df.columns = ['Decile', 'n', 'Gözlenen oran',
                                       'Beklenen oran', 'Gözlenen sayı', 'Beklenen sayı']
                    st.dataframe(cal_df, use_container_width=True)
                    download_buttons_for_table(
                        cal_df, "kalibrasyon_decile_tablosu",
                        label_prefix="Decile tablosu — ",
                        title="Kalibrasyon — Decile Bazlı Gözlenen vs Beklenen"
                    )

                # Kalibrasyon eğrisi PNG
                download_fig_300dpi(fig_cal, "kalibrasyon_egrisi",
                                      label="Kalibrasyon eğrisi PNG indir (300 DPI)")

            except Exception as e:
                st.warning(f"Kalibrasyon hesaplanamadı: {e}")

            # ─── BOOTSTRAP İÇ VALİDASYON (Harrell yöntemi) ───
            st.markdown("---")
            st.markdown("# 🔄 Bootstrap İç Validasyon — Optimism-Corrected AUC")
            with st.expander("ℹ️ Bootstrap iç validasyon nedir?"):
                st.markdown("""
                Modelin **overfitting** miktarını ölçer. Harici (dış) validasyon yapılamadığında
                Harrell (1996) tarafından önerilen standart yöntemdir.

                **Adımlar:**
                1. Orijinal veriden AUC hesapla → **AUC_app** (apparent)
                2. B kez bootstrap örneklem al (yerine koyarak), her birinde:
                   - Bootstrap'tan yeni model fit et → AUC_boot
                   - Aynı modeli **orijinal veride** test et → AUC_test
                   - Optimizm = AUC_boot − AUC_test
                3. **Düzeltilmiş AUC = AUC_app − ortalama(optimizm)**

                Makalede: *"Internal validation was performed using 1000 bootstrap resamples
                with the optimism-corrected AUC reported."*
                """)

            cbb1, cbb2 = st.columns([1, 3])
            with cbb1:
                n_boot = st.selectbox(
                    "Bootstrap sayısı",
                    [100, 200, 500, 1000, 2000],
                    index=3,
                    help="1000 standart, 2000 daha kararlı; süre ~30 sn"
                )
            with cbb2:
                run_boot = st.button("▶ Bootstrap iç validasyon çalıştır",
                                       type="primary")

            if run_boot:
                try:
                    # Bootstrap'ta her zaman Firth kullan — MLE separation durumunda
                    # bootstrap iterasyonlarının çoğunda singular matrix verir.
                    # Firth tüm durumlarda kararlı çalışır.
                    boot_method = "Firth"
                    if not use_firth and separation_warnings:
                        st.info(
                            "ℹ️ Veride complete separation var. Bootstrap kararlılığı için "
                            "**Firth yöntemi** ile çalışacak (MLE'de iterasyonların büyük "
                            "kısmı singular matrix nedeniyle başarısız olur)."
                        )
                    elif not use_firth:
                        st.caption("Bootstrap için Firth penalization kullanılıyor (kararlılık için).")

                    progress = st.progress(0, text="Bootstrap çalışıyor...")
                    rng_boot = np.random.default_rng(42)
                    X_arr = X_lr_const.values
                    y_arr = y_lr.values
                    n = len(y_arr)

                    # Apparent AUC (mevcut model üzerinden)
                    fpr_app, tpr_app, _ = roc_curve(y_arr, pred_full)
                    auc_app = auc(fpr_app, tpr_app)

                    optimism_list = []
                    boot_aucs = []
                    test_aucs = []
                    n_skipped_class = 0
                    n_skipped_fit = 0

                    for b in range(n_boot):
                        # Bootstrap örneklem (yerine koyarak)
                        idx_b = rng_boot.choice(n, size=n, replace=True)
                        X_b = X_arr[idx_b]; y_b = y_arr[idx_b]

                        # Bootstrap'ta tek sınıf çıkarsa atla
                        if len(np.unique(y_b)) < 2:
                            n_skipped_class += 1
                            continue

                        # Bootstrap model fit (her durumda Firth — kararlılık için)
                        try:
                            fr = firth_logistic_regression(X_b, y_b, max_iter=100)
                            beta_b = fr['beta']
                            if not np.all(np.isfinite(beta_b)):
                                n_skipped_fit += 1
                                continue

                            # AUC_boot (bootstrap verisinde)
                            pred_b_on_b = 1/(1+np.exp(-np.clip(X_b @ beta_b, -30, 30)))
                            fpr_b, tpr_b, _ = roc_curve(y_b, pred_b_on_b)
                            auc_b = auc(fpr_b, tpr_b)

                            # AUC_test (orijinal veride)
                            pred_b_on_orig = 1/(1+np.exp(-np.clip(X_arr @ beta_b, -30, 30)))
                            fpr_t, tpr_t, _ = roc_curve(y_arr, pred_b_on_orig)
                            auc_t = auc(fpr_t, tpr_t)

                            boot_aucs.append(auc_b)
                            test_aucs.append(auc_t)
                            optimism_list.append(auc_b - auc_t)
                        except Exception:
                            n_skipped_fit += 1
                            continue

                        if (b+1) % max(1, n_boot//20) == 0:
                            progress.progress((b+1)/n_boot,
                                                 text=f"Bootstrap: {b+1}/{n_boot}")

                    progress.empty()

                    # Tanılayıcı bilgi
                    if n_skipped_class > 0 or n_skipped_fit > 0:
                        st.caption(
                            f"ℹ️ {len(optimism_list)}/{n_boot} bootstrap başarılı. "
                            f"Atlanan: {n_skipped_class} (tek sınıf), "
                            f"{n_skipped_fit} (fit hatası)."
                        )

                    if len(optimism_list) < 10:
                        st.error("Yeterli başarılı bootstrap iterasyonu yok.")
                    else:
                        mean_opt = np.mean(optimism_list)
                        auc_corrected = auc_app - mean_opt
                        # %95 CI: bootstrap distribution dan
                        ci_lo_opt = np.percentile(optimism_list, 2.5)
                        ci_hi_opt = np.percentile(optimism_list, 97.5)
                        auc_ci_lo = auc_app - ci_hi_opt
                        auc_ci_hi = auc_app - ci_lo_opt

                        # Sonuçlar
                        st.markdown("#### 📈 Sonuçlar")
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        bc1.metric("Apparent AUC", f"{auc_app:.4f}")
                        bc2.metric("Ortalama Optimizm",
                                     f"{mean_opt:+.4f}",
                                     f"%{mean_opt*100:.2f}")
                        bc3.metric("Düzeltilmiş AUC",
                                     f"{auc_corrected:.4f}",
                                     f"Δ {-mean_opt:+.4f}",
                                     delta_color="inverse")
                        bc4.metric("95% CI (düzeltilmiş)",
                                     f"[{auc_ci_lo:.3f}, {auc_ci_hi:.3f}]")

                        # Yorum
                        if mean_opt < 0.02:
                            st.success(
                                f"✅ Optimizm çok düşük ({mean_opt:+.3f}). "
                                "Model overfitting göstermiyor, düzeltilmiş performans "
                                "apparent ile neredeyse aynı."
                            )
                        elif mean_opt < 0.05:
                            st.info(
                                f"ℹ️ Hafif optimizm var ({mean_opt:+.3f}). "
                                "Düzeltilmiş AUC daha güvenilir bir tahmin."
                            )
                        else:
                            st.warning(
                                f"⚠️ Belirgin optimizm ({mean_opt:+.3f}). "
                                "Model overfitting gösteriyor olabilir; düzeltilmiş "
                                "değer rapor edilmeli."
                            )

                        # Histogram + apparent vs corrected
                        fig_b, axes_b = plt.subplots(1, 2, figsize=(13, 5))

                        # Sol: AUC dağılımları
                        axes_b[0].hist(boot_aucs, bins=30, alpha=0.6,
                                          color='#2E86AB', label='AUC_boot (bootstrap)',
                                          edgecolor='white')
                        axes_b[0].hist(test_aucs, bins=30, alpha=0.6,
                                          color='#E07A5F', label='AUC_test (orijinal)',
                                          edgecolor='white')
                        axes_b[0].axvline(auc_app, color='black', lw=2,
                                             linestyle='--', label=f'Apparent = {auc_app:.3f}')
                        axes_b[0].axvline(auc_corrected, color='#A23B72', lw=2,
                                             label=f'Düzeltilmiş = {auc_corrected:.3f}')
                        axes_b[0].set_xlabel("AUC", fontsize=11)
                        axes_b[0].set_ylabel("Frekans", fontsize=11)
                        axes_b[0].set_title("Bootstrap AUC Dağılımı")
                        axes_b[0].legend(loc='upper left', fontsize=9)
                        axes_b[0].grid(alpha=0.3)

                        # Sağ: Optimizm dağılımı
                        axes_b[1].hist(optimism_list, bins=30, color='#F18F01',
                                          alpha=0.7, edgecolor='white')
                        axes_b[1].axvline(mean_opt, color='red', lw=2,
                                             label=f'Ortalama = {mean_opt:+.4f}')
                        axes_b[1].axvline(0, color='black', lw=1, alpha=0.5)
                        axes_b[1].set_xlabel("Optimizm (AUC_boot − AUC_test)",
                                                fontsize=11)
                        axes_b[1].set_ylabel("Frekans", fontsize=11)
                        axes_b[1].set_title(f"Optimizm Dağılımı (n = {len(optimism_list)} bootstrap)")
                        axes_b[1].legend(fontsize=10)
                        axes_b[1].grid(alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig_b)
                        download_fig_300dpi(fig_b, "bootstrap_validasyon",
                                              label="Bootstrap grafikleri PNG indir (300 DPI)")

                        # Bootstrap özet tablosu (makale Tablo formatında)
                        bootstrap_summary = pd.DataFrame([{
                            'Metrik': 'Apparent AUC',
                            'Değer': f"{auc_app:.4f}"
                        }, {
                            'Metrik': 'Ortalama Optimizm',
                            'Değer': f"{mean_opt:+.4f}"
                        }, {
                            'Metrik': 'Optimism-corrected AUC',
                            'Değer': f"{auc_corrected:.4f}"
                        }, {
                            'Metrik': '95% CI (düzeltilmiş)',
                            'Değer': f"[{auc_ci_lo:.4f}, {auc_ci_hi:.4f}]"
                        }, {
                            'Metrik': 'Bootstrap iter (başarılı)',
                            'Değer': f"{len(optimism_list)}/{n_boot}"
                        }])
                        download_buttons_for_table(
                            bootstrap_summary, "bootstrap_ozet",
                            label_prefix="Bootstrap özeti — ",
                            title="Bootstrap İç Validasyon — Özet Sonuçlar"
                        )

                        # Makaleye yazılacak cümle örneği
                        st.info(
                            f"📝 **Makaleye yazılacak cümle:**\n\n"
                            f"*\"Internal validation was performed using {n_boot} bootstrap "
                            f"resamples. The apparent AUC was {auc_app:.3f}, with a mean "
                            f"optimism of {mean_opt:+.3f}, yielding an optimism-corrected "
                            f"AUC of {auc_corrected:.3f} (95% CI: {auc_ci_lo:.3f}–{auc_ci_hi:.3f}).\"*"
                        )

                except Exception as e:
                    st.error(f"Bootstrap çalıştırılamadı: {e}")
                    st.exception(e)

            # ─── SUBGROUP ANALİZİ ───
            st.markdown("---")
            st.markdown("# 👥 Subgroup Analizi — Alt Gruplarda Model Performansı")
            with st.expander("ℹ️ Subgroup analizi nedir, niye önemli?"):
                st.markdown("""
                Modelin **farklı alt gruplarda tutarlı performans gösterip göstermediğini**
                test eder. Hakemler genellikle sorar: *"Modeliniz genel olarak AUC = 0.92,
                ama 60+ yaş hastalarda da bu kadar iyi mi?"*

                **Her alt grup için hesaplanır:**
                - **n** — alt grup büyüklüğü
                - **Olay sayısı** — komplike Behçet hasta sayısı
                - **AUC** — alt gruba özgü ayırt etme gücü
                - **95% CI** — bootstrap ile (200 iter)

                **Forest plot** ile görsel sunum yapılır. AUC değerleri ve CI'leri birbirine
                yakınsa model **alt gruplar arası tutarlı** demektir.

                Makalede ifade: *"Subgroup analyses showed consistent model performance
                across age groups, sex, and disease duration."*
                """)

            try:
                # Alt grup oluşturulabilecek değişkenler — sadece klinik demografik
                # (Kolşisin/DMARD/Biyolojik çıkarıldı; bunlar tedavi seçimi olduğundan
                #  reverse causality riski taşır)
                subgroup_candidates = []
                for c in ['YAŞ', 'HASTALIK SÜRESİ(yıl)']:
                    if c in data_lr.columns:
                        subgroup_candidates.append(c)

                # Klinik eşikler (literatürde Behçet için sık kullanılan)
                CLINICAL_THRESHOLDS = {
                    'YAŞ': 40,                    # genç (≤40) vs orta-ileri (>40)
                    'HASTALIK SÜRESİ(yıl)': 5,    # erken (≤5) vs uzun süreli (>5)
                }

                if not subgroup_candidates:
                    st.info("Alt grup analizi için uygun değişken bulunamadı.")
                else:
                    sg_col1, sg_col2, sg_col3 = st.columns([2, 1, 1])
                    with sg_col1:
                        sg_vars = st.multiselect(
                            "Alt grup değişkenlerini seçin",
                            subgroup_candidates,
                            default=subgroup_candidates,
                            help="Continuous değişkenler klinik eşiklere göre 2 gruba bölünür"
                        )
                    with sg_col2:
                        split_method = st.radio(
                            "Eşik yöntemi",
                            ["Klinik (önerilen)", "Median"],
                            help=(
                                "Klinik: Yaş için 40, Süre için 5 yıl "
                                "(Behçet literatüründe sık kullanılan eşikler).\n\n"
                                "Median: veri ortancasına göre eşit n'li split."
                            )
                        )
                    with sg_col3:
                        sg_n_boot = st.number_input(
                            "Bootstrap (CI)",
                            min_value=50, max_value=1000, value=200, step=50
                        )

                    if sg_vars:
                        # Alt gruplar için stratified AUC hesapla
                        pred_arr = np.asarray(pred_full).flatten()
                        y_arr_sg = y_lr.values

                        sg_results = []
                        rng_sg = np.random.default_rng(123)

                        def auc_with_ci(y_sub, p_sub, n_boot=200, rng=None):
                            """AUC + bootstrap percentile CI."""
                            if len(np.unique(y_sub)) < 2:
                                return np.nan, np.nan, np.nan
                            fpr_, tpr_, _ = roc_curve(y_sub, p_sub)
                            auc_pt = auc(fpr_, tpr_)
                            aucs_b = []
                            n_s = len(y_sub)
                            for _ in range(n_boot):
                                idx = rng.choice(n_s, size=n_s, replace=True)
                                yb = y_sub[idx]; pb = p_sub[idx]
                                if len(np.unique(yb)) < 2: continue
                                try:
                                    fb, tb, _ = roc_curve(yb, pb)
                                    aucs_b.append(auc(fb, tb))
                                except Exception:
                                    pass
                            if len(aucs_b) < 10:
                                return auc_pt, np.nan, np.nan
                            ci_lo_a = np.percentile(aucs_b, 2.5)
                            ci_hi_a = np.percentile(aucs_b, 97.5)
                            return auc_pt, ci_lo_a, ci_hi_a

                        # Genel (overall) AUC referans için
                        overall_auc, overall_lo, overall_hi = auc_with_ci(
                            y_arr_sg, pred_arr, n_boot=sg_n_boot, rng=rng_sg
                        )
                        sg_results.append({
                            'group_label': '📊 GENEL (tüm hastalar)',
                            'variable': '—',
                            'subgroup': 'All',
                            'n': len(y_arr_sg),
                            'events': int(y_arr_sg.sum()),
                            'event_rate': y_arr_sg.mean(),
                            'auc': overall_auc,
                            'ci_lo': overall_lo,
                            'ci_hi': overall_hi
                        })

                        # Her subgrup değişkeni için stratify et
                        for sgv in sg_vars:
                            col_data = data_lr[sgv]

                            # Eşik seçimi: klinik veya median
                            if split_method.startswith("Klinik") and sgv in CLINICAL_THRESHOLDS:
                                threshold = CLINICAL_THRESHOLDS[sgv]
                                thresh_label = "klinik"
                            else:
                                threshold = float(col_data.median())
                                thresh_label = "median"

                            if sgv == 'YAŞ':
                                groups = [
                                    (f"YAŞ ≤ {threshold:.0f} ({thresh_label})",
                                     col_data <= threshold),
                                    (f"YAŞ > {threshold:.0f} ({thresh_label})",
                                     col_data > threshold),
                                ]
                            elif 'SÜRE' in sgv.upper():
                                groups = [
                                    (f"Süre ≤ {threshold:.0f} yıl ({thresh_label})",
                                     col_data <= threshold),
                                    (f"Süre > {threshold:.0f} yıl ({thresh_label})",
                                     col_data > threshold),
                                ]
                            else:
                                groups = [
                                    (f"{sgv} ≤ {threshold:.1f}",
                                     col_data <= threshold),
                                    (f"{sgv} > {threshold:.1f}",
                                     col_data > threshold),
                                ]

                            for label, mask in groups:
                                mask_arr = mask.values
                                y_sub = y_arr_sg[mask_arr]
                                p_sub = pred_arr[mask_arr]
                                a, lo, hi = auc_with_ci(y_sub, p_sub,
                                                         n_boot=sg_n_boot,
                                                         rng=rng_sg)
                                sg_results.append({
                                    'group_label': label,
                                    'variable': sgv,
                                    'subgroup': label,
                                    'n': int(mask_arr.sum()),
                                    'events': int(y_sub.sum()),
                                    'event_rate': (y_sub.mean()
                                                    if len(y_sub) > 0 else 0),
                                    'auc': a,
                                    'ci_lo': lo,
                                    'ci_hi': hi
                                })

                        # ─── Tablo gösterimi ───
                        sg_df = pd.DataFrame(sg_results)
                        sg_display = sg_df.copy()
                        sg_display['n (event)'] = sg_display.apply(
                            lambda r: f"{r['n']} ({r['events']}, {r['event_rate']:.0%})",
                            axis=1
                        )
                        sg_display['AUC (95% CI)'] = sg_display.apply(
                            lambda r: (f"{r['auc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]"
                                       if not pd.isna(r['auc']) and not pd.isna(r['ci_lo'])
                                       else (f"{r['auc']:.3f}" if not pd.isna(r['auc'])
                                              else "—")),
                            axis=1
                        )
                        table_df = sg_display[['group_label', 'n (event)', 'AUC (95% CI)']].rename(
                            columns={'group_label': 'Alt Grup'}
                        )
                        st.dataframe(table_df, use_container_width=True,
                                      hide_index=True)
                        download_buttons_for_table(
                            table_df, "subgroup_tablosu",
                            label_prefix="Subgroup tablosu — ",
                            title="Subgroup Analizi — Alt Gruplarda AUC Performansı"
                        )

                        # ─── Forest Plot ───
                        valid = sg_df[~sg_df['auc'].isna()].copy()
                        valid_forest = valid[~valid['ci_lo'].isna()].reset_index(drop=True)

                        if len(valid_forest) >= 2:
                            fig_sg, ax_sg = plt.subplots(
                                figsize=(11, max(4, 0.45 * len(valid_forest) + 1.5))
                            )
                            y_pos = np.arange(len(valid_forest))[::-1]  # Üstten alta

                            # Renkleri: overall siyah, diğerleri değişkene göre
                            color_map = {'—': '#000000'}
                            cmap = plt.cm.Set2(np.linspace(0, 1, max(1, len(sg_vars))))
                            for i, sgv in enumerate(sg_vars):
                                color_map[sgv] = cmap[i]
                            colors = [color_map.get(v, '#2E86AB')
                                       for v in valid_forest['variable']]

                            for i, row in valid_forest.iterrows():
                                yp = y_pos[i]
                                err_lo = row['auc'] - row['ci_lo']
                                err_hi = row['ci_hi'] - row['auc']
                                ax_sg.errorbar(row['auc'], yp,
                                                xerr=[[err_lo], [err_hi]],
                                                fmt='o', color=colors[i],
                                                ecolor=colors[i], capsize=4,
                                                markersize=9, lw=2)
                                # Sağda AUC etiketi
                                ax_sg.text(1.02, yp,
                                            f" {row['auc']:.3f}",
                                            va='center', fontsize=9,
                                            fontweight='bold')

                            # Referans çizgi: overall AUC
                            ax_sg.axvline(overall_auc, color='gray',
                                            linestyle='--', alpha=0.6, lw=1,
                                            label=f'Genel AUC = {overall_auc:.3f}')
                            # AUC = 0.5 referans
                            ax_sg.axvline(0.5, color='red', linestyle=':',
                                            alpha=0.5, lw=1, label='AUC = 0.5 (şans)')

                            ax_sg.set_yticks(y_pos)
                            ax_sg.set_yticklabels(valid_forest['group_label'],
                                                   fontsize=10)
                            ax_sg.set_xlabel("AUC (95% CI)", fontsize=11)
                            ax_sg.set_xlim(0.4, 1.05)
                            ax_sg.set_title(
                                "Alt Grup AUC Forest Plot\n"
                                f"({len(valid_forest)} grup, bootstrap CI ile)",
                                fontsize=12
                            )
                            ax_sg.legend(loc='lower left', fontsize=9)
                            ax_sg.grid(axis='x', alpha=0.3)
                            ax_sg.spines['top'].set_visible(False)
                            ax_sg.spines['right'].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig_sg)
                            download_fig_300dpi(fig_sg, "subgroup_forest_plot",
                                                  label="Subgroup forest plot PNG indir (300 DPI)")

                        # ─── Heterojenite testi ───
                        # Her değişken için iki alt grubun AUC'lerini karşılaştır
                        # DeLong veya basit z-testi (overlapping CIs)
                        st.markdown("#### 🔬 Heterojenite Testi (Alt Gruplar Arası)")
                        het_rows = []
                        for sgv in sg_vars:
                            sub_rows = valid_forest[valid_forest['variable'] == sgv]
                            if len(sub_rows) != 2: continue
                            a1, lo1, hi1 = (sub_rows.iloc[0]['auc'],
                                              sub_rows.iloc[0]['ci_lo'],
                                              sub_rows.iloc[0]['ci_hi'])
                            a2, lo2, hi2 = (sub_rows.iloc[1]['auc'],
                                              sub_rows.iloc[1]['ci_lo'],
                                              sub_rows.iloc[1]['ci_hi'])
                            # Basit z-testi: CI'lerden SE türet
                            se1 = (hi1 - lo1) / (2 * 1.96)
                            se2 = (hi2 - lo2) / (2 * 1.96)
                            se_diff = np.sqrt(se1**2 + se2**2)
                            if se_diff > 0:
                                z = (a1 - a2) / se_diff
                                p_het = 2 * (1 - stats.norm.cdf(abs(z)))
                            else:
                                z, p_het = np.nan, np.nan
                            overlap = (lo1 <= a2 <= hi1) or (lo2 <= a1 <= hi2)
                            het_rows.append({
                                'Değişken': sgv,
                                f'AUC {sub_rows.iloc[0]["group_label"]}': f"{a1:.3f}",
                                f'AUC {sub_rows.iloc[1]["group_label"]}': f"{a2:.3f}",
                                'ΔAUC': f"{a1-a2:+.3f}",
                                'CI Overlap': "✅ Var" if overlap else "❌ Yok",
                                'p (yaklaşık)': fmt_p(p_het)
                            })
                        if het_rows:
                            # Sadece ΔAUC ve p'yi standartlaştır, isimleri sadeleştir
                            het_df = pd.DataFrame([
                                {'Değişken': r['Değişken'],
                                 'ΔAUC': r['ΔAUC'],
                                 'CI Çakışıyor': r['CI Overlap'],
                                 'p': r['p (yaklaşık)']}
                                for r in het_rows
                            ])

                            def hl_het(v):
                                try:
                                    if v == "<0.001": return 'background-color: #FADBD8'
                                    return ('background-color: #FADBD8'
                                            if float(v) < 0.05 else
                                            'background-color: #D4EFDF')
                                except: return ''

                            st.dataframe(het_df.style.map(hl_het, subset=['p']),
                                          use_container_width=True)
                            st.caption(
                                "🟩 yeşil: p ≥ 0.05 → alt gruplar arası fark yok "
                                "(model tutarlı) · 🟥 kırmızı: p < 0.05 → anlamlı fark "
                                "(model alt gruplar arası farklı performans gösteriyor)"
                            )
                            download_buttons_for_table(
                                het_df, "heterojenite_tablosu",
                                label_prefix="Heterojenite — ",
                                title="Subgroup Heterojenite Testi"
                            )

                            # Otomatik klinik yorum
                            n_consistent = sum(1 for r in het_rows
                                                 if (r['p (yaklaşık)'] != '<0.001' and
                                                     float(r['p (yaklaşık)']) >= 0.05)
                                                 or r['CI Overlap'] == '✅ Var')
                            if n_consistent == len(het_rows):
                                st.success(
                                    f"✅ Tüm {len(het_rows)} alt grupta model performansı "
                                    "**tutarlı** — heterojenite tespit edilmedi."
                                )
                            elif n_consistent > 0:
                                st.info(
                                    f"ℹ️ {n_consistent}/{len(het_rows)} alt grupta model "
                                    "tutarlı, diğerlerinde sınırlı heterojenite var."
                                )
                            else:
                                st.warning(
                                    "⚠️ Birden fazla alt grupta anlamlı heterojenite var. "
                                    "Modelin alt grup-spesifik kalibrasyonu önerilebilir."
                                )

            except Exception as e:
                st.warning(f"Subgroup analizi çalıştırılamadı: {e}")
                st.exception(e)

            # ─── İNTERAKTİF RİSK HESAPLAYICI ───
            st.markdown("---")
            st.markdown("### 🧮 İnteraktif Risk Hesaplayıcı")
            st.caption("Hasta değerlerini gir, anlık tahmin al.")

            calc_cols = st.columns(min(len(predictors), 4))
            user_vals = {}
            for i, var in enumerate(predictors):
                with calc_cols[i % len(calc_cols)]:
                    is_bin = data_lr[var].nunique() <= 2
                    if is_bin:
                        user_vals[var] = st.selectbox(
                            var, [0, 1],
                            index=0,
                            help=f"OR = {np.exp(params[var]):.2f}"
                        )
                    else:
                        median_val = float(data_lr[var].median())
                        user_vals[var] = st.number_input(
                            var,
                            value=median_val,
                            min_value=float(data_lr[var].min()*0.5),
                            max_value=float(data_lr[var].max()*1.5),
                            step=0.1,
                            help=f"OR = {np.exp(params[var]):.3f} her birim artış için"
                        )

            # Olasılığı hesapla (MLE veya Firth)
            # Linear predictor manuel hesapla, her iki yöntemde de çalışsın
            user_logit_raw = float(intercept)
            for var, val in user_vals.items():
                user_logit_raw += float(params[var]) * val
            user_prob = 1 / (1 + math.exp(-user_logit_raw)) if abs(user_logit_raw) < 700 else (1.0 if user_logit_raw > 0 else 0.0)
            user_logit = user_logit_raw

            c1, c2, c3 = st.columns(3)
            c1.metric("Komplike Olasılığı", f"{user_prob:.1%}")
            c2.metric("Linear Predictor (η)", f"{user_logit:+.2f}")
            risk_cat = ("Düşük" if user_prob < 0.3 else
                         "Orta" if user_prob < 0.7 else "Yüksek")
            c3.metric("Risk Kategorisi", risk_cat)

            # Olasılık çubuğu
            fig_bar, ax_bar = plt.subplots(figsize=(10, 1.5))
            ax_bar.barh(0, user_prob, color='#E07A5F', height=0.5)
            ax_bar.barh(0, 1-user_prob, left=user_prob,
                         color='#E5E5E5', height=0.5)
            ax_bar.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_yticks([])
            ax_bar.set_xticks(np.arange(0, 1.1, 0.1))
            ax_bar.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0,1.1,0.1)])
            ax_bar.set_title(f"Tahmin edilen olasılık: {user_prob:.1%}",
                              fontsize=11)
            for sp in ['top','right','left']: ax_bar.spines[sp].set_visible(False)
            st.pyplot(fig_bar)

        except Exception as e:
            st.error(f"Regresyon çalıştırılamadı: {e}")
            st.exception(e)

st.sidebar.markdown("---")
st.sidebar.info("""
**Analiz Akışı:**
1. Her grupta Shapiro-Wilk → grup-bazlı normallik
2. Levene → varyans homojenliği
3. Normal+eşit varyans: t-test / ANOVA
   Normal+eşit değil: Welch
   Non-normal: MWU / Kruskal-Wallis
4. Post-hoc: Tukey HSD veya Dunn (Holm düzeltmeli)
5. FDR (BH) ile çoklu test düzeltmesi
6. ROC + Youden J ile optimal cut-off
7. DeLong testi → AUC karşılaştırması
8. Lojistik regresyon → covariate düzeltmesi
""")
