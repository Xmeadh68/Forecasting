"""
Dashboard Peramalan Beban Listrik Nasional Panama
Menampilkan hasil pemodelan dari Kaggle (folder hasil_kaggle/) + ramalan langsung dengan Chronos-Bolt.
Jalankan:  streamlit run app.py
"""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard Beban Listrik Panama", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# =============================================================================
# PALET — keluarga warna sejuk (biru–teal–indigo) yang harmonis, gradasi halus
# =============================================================================
BLUE, NAVY, SKY, TEAL, INDIGO = "#1f6fd1", "#0b3c75", "#8ec5ff", "#2a9d8f", "#5b5bd6"
INK, INK2, MUTED, GRID = "#14213d", "#3d4b63", "#7b879c", "#e7edf5"
FAM = {"Baseline": "#b4bfcd", "Klasik": "#93c5fd", "Machine Learning": "#2a9d8f",
       "Deep Learning": "#48c1dc", "Transfer Learning": "#1f6fd1", "Hybrid": "#5b5bd6", "Ensemble": "#0b3c75"}
FAM_ID = {"Baseline": "Pembanding sederhana", "Klasik": "Statistik klasik", "Machine Learning": "Machine learning",
          "Deep Learning": "Deep learning", "Transfer Learning": "Transfer learning", "Hybrid": "Hybrid",
          "Ensemble": "Gabungan (ensemble)"}
RAMP = ["#0b3c75", "#16509a", "#1f6fd1", "#3f8ae0", "#6aa6ea", "#93c0f1", "#bcd8f7", "#dbeafb"]
SEQ = [[0, "#eef5fe"], [0.35, "#9cc5f3"], [0.7, "#3f8ae0"], [1, "#0b3c75"]]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_kaggle")

# =============================================================================
# BAHASA MANUSIA — nama variabel & kelompok
# =============================================================================
GROUP_NAME = {"Riwayat beban (lag)": "Pola beban hari-hari sebelumnya", "Kalender (jam/hari/musim)": "Jam, hari & musim",
              "Libur & sekolah": "Hari libur & masa sekolah", "Suhu": "Suhu udara", "Kelembapan": "Kelembapan udara",
              "Presipitasi": "Curah hujan", "Angin": "Kecepatan angin", "Rezim COVID-19": "Masa COVID-19",
              "Ramalan foundation model": "Ramalan model transfer learning"}
CITY = {"toc": "Tocumen (Kota Panama)", "san": "Kota Santiago", "dav": "Kota David"}
VARN = {"T2M": "Suhu udara", "QV2M": "Kelembapan udara", "TQL": "Curah hujan", "W2M": "Kecepatan angin"}
UNIT = {"T2M": "°C", "QV2M": "%", "TQL": "l/m²", "W2M": "m/s"}
# Kamus variabel data asli (mengikuti lembar 'deskripsi' pada file Excel)
COL_DESC = {"datetime": ("Tanggal dan jam pencatatan (zona waktu Panama, UTC−5)", "tanggal-jam"),
            "nat_demand": ("Beban (kebutuhan) listrik nasional Panama — yang diramal", "MWh"),
            "Holiday_ID": ("Kode jenis hari libur nasional (0 = bukan hari libur)", "kode"),
            "holiday": ("Penanda hari libur (1 = hari libur, 0 = hari biasa)", "ya/tidak"),
            "school": ("Penanda masa sekolah (1 = masa sekolah, 0 = libur sekolah)", "ya/tidak")}
for _v, _vn in VARN.items():
    for _c, _cn in CITY.items():
        COL_DESC[f"{_v}_{_c}"] = (f"{_vn} di {_cn}" + (" (ketinggian 2 meter)" if _v in ("T2M", "QV2M", "W2M") else ""), UNIT[_v])


def col_label(c):
    """Nama kolom data dalam bahasa manusia, untuk pilihan kolom."""
    return f"{COL_DESC[c][0]} [{c}]" if c in COL_DESC else str(c)


def kamus_df():
    return pd.DataFrame([dict(**{"Nama di data": k, "Arti": v[0], "Satuan": v[1]}) for k, v in COL_DESC.items()])


FEAT_NAME = {"hour": "Jam", "dow": "Hari dalam seminggu", "month": "Bulan", "doy_sin": "Musim dalam setahun",
             "doy_cos": "Musim dalam setahun (2)", "weekend": "Akhir pekan", "horizon": "Jarak jam dari saat meramal",
             "Holiday_ID": "Jenis hari libur", "holiday": "Hari libur", "school": "Masa sekolah", "covid": "Masa COVID-19",
             "T2M_mean": "Suhu rata-rata 3 kota", "QV2M_mean": "Kelembapan rata-rata 3 kota",
             "T2M_mean_24h": "Suhu rata-rata 24 jam terakhir", "lag_24": "Beban 1 hari sebelumnya",
             "lag_48": "Beban 2 hari sebelumnya", "lag_72": "Beban 3 hari sebelumnya", "lag_168": "Beban 1 minggu sebelumnya",
             "lag_336": "Beban 2 minggu sebelumnya", "roll24_mean": "Rata-rata beban sehari (H-1)",
             "roll24_std": "Naik-turun beban sehari (H-1)", "roll24_max": "Beban tertinggi sehari (H-1)",
             "roll168_mean": "Rata-rata beban seminggu", "y_at_origin": "Beban tengah malam saat meramal",
             "mean24_origin": "Rata-rata beban kemarin", "lag_168_ratio": "Perbandingan beban minggu lalu",
             "fm": "Ramalan model transfer learning", "fm_minus_lag24": "Ramalan TL dibanding kemarin",
             "fm_minus_lag168": "Ramalan TL dibanding minggu lalu", "fm_ratio_mean24": "Ramalan TL dibanding rata-rata kemarin"}


def nice_feat(f):
    if f in FEAT_NAME:
        return FEAT_NAME[f]
    p = str(f).split("_")
    if len(p) == 2 and p[0] in VARN and p[1] in CITY:
        return f"{VARN[p[0]]} di {CITY[p[1]]}"
    if f == "nat_demand":
        return "Beban listrik nasional"
    return str(f)


def nice_group(g):
    return GROUP_NAME.get(g, g)


def short(n):
    n = str(n)
    if n.startswith("Ensemble"):
        return "Gabungan 3 model terbaik"
    return n.replace(" (zero-shot)", "").replace(" + kovariat", " (+cuaca)").replace("Seasonal Naive (168j)",
                                                                                     "Salin minggu lalu")


# =============================================================================
# GAYA
# =============================================================================
st.markdown(f"""
<style>
.block-container{{padding-top:1rem;padding-bottom:2.5rem;max-width:1440px}}
[data-testid="stHeader"]{{background:transparent}}
.stApp{{background:linear-gradient(180deg,#eef4fb 0%,#f6f9fd 40%,#f6f9fd 100%)}}
/* warna utama dikunci biru (tidak bergantung config.toml) */
span[data-baseweb="tag"]{{background-color:{BLUE} !important}}
[data-baseweb="tab-highlight"]{{background-color:{BLUE} !important}}
button[data-baseweb="tab"][aria-selected="true"] p{{color:{BLUE} !important}}
[data-testid="stTabs"] button p{{font-size:15px;font-weight:650}}
div[role="radiogroup"] label div:first-child{{border-color:{BLUE} !important}}
[data-testid="stSlider"] div[role="slider"]{{background-color:{BLUE} !important}}
/* hero */
.hero{{background:linear-gradient(120deg,#0b3c75 0%,#1f6fd1 55%,#5b8de8 100%);border-radius:20px;padding:22px 28px;
color:#fff;display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
box-shadow:0 10px 30px rgba(11,60,117,.25);margin-bottom:14px;position:relative;overflow:hidden}}
.hero:after{{content:"";position:absolute;right:-60px;top:-80px;width:260px;height:260px;border-radius:50%;
background:rgba(255,255,255,.08)}}
.hero h1{{margin:0;font-size:27px;font-weight:800;letter-spacing:-.4px;color:#fff}}
.hero p{{margin:6px 0 0;font-size:14px;opacity:.9}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}
.chip{{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);padding:4px 11px;border-radius:999px;font-size:12.5px}}
.bolt{{font-size:52px;line-height:1;filter:drop-shadow(0 4px 10px rgba(0,0,0,.25));z-index:1}}
/* KPI */
.kpis{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:6px 0 16px}}
@media(max-width:1250px){{.kpis{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}
@media(max-width:700px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.kpi{{background:#fff;border-radius:16px;padding:14px 16px;border:1px solid #dce7f5;position:relative;overflow:hidden;
box-shadow:0 2px 10px rgba(20,33,61,.05);min-height:112px}}
.kpi:before{{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:linear-gradient(90deg,{NAVY},{BLUE},{SKY})}}
.kpi .ic{{position:absolute;right:12px;top:12px;font-size:20px;opacity:.85}}
.kl{{font-size:11.5px;color:{MUTED};text-transform:uppercase;letter-spacing:.6px;font-weight:700;padding-right:26px}}
.kv{{font-size:24px;font-weight:800;margin:6px 0 2px;line-height:1.15;color:{INK};overflow-wrap:anywhere}}
.kv.sm{{font-size:16.5px}}.ks{{font-size:12px;color:{INK2}}}
.kpi.hl{{background:linear-gradient(135deg,{NAVY} 0%,{BLUE} 100%);border:0}}
.kpi.hl .kl,.kpi.hl .ks{{color:rgba(255,255,255,.85)}}.kpi.hl .kv{{color:#fff}}
.kpi.ok{{background:linear-gradient(135deg,#1d7a70 0%,{TEAL} 100%);border:0}}
.kpi.md{{background:linear-gradient(135deg,#16509a 0%,#3f8ae0 100%);border:0}}
.kpi.hi{{background:linear-gradient(135deg,#b45309 0%,#e08a1e 100%);border:0}}
.kpi.ok .kl,.kpi.ok .ks,.kpi.md .kl,.kpi.md .ks,.kpi.hi .kl,.kpi.hi .ks{{color:rgba(255,255,255,.88)}}
.kpi.ok .kv,.kpi.md .kv,.kpi.hi .kv{{color:#fff}}
/* kartu */
[data-testid="stVerticalBlockBorderWrapper"]{{background:#fff;border:1px solid #dce7f5 !important;border-radius:16px !important;
box-shadow:0 2px 12px rgba(20,33,61,.05)}}
.ct{{font-weight:750;color:{INK};font-size:15.5px;margin:2px 0 0}}
.cs{{color:{MUTED};font-size:12.5px;margin:2px 0 4px}}
.ins{{background:#f1f6fd;border-left:4px solid {BLUE};border-radius:8px;padding:8px 12px;font-size:13px;color:{INK2};
line-height:1.55;margin:2px 0 4px}}.ins b{{color:{INK}}}
.sum{{background:#fff;border:1px solid #dce7f5;border-radius:16px;padding:16px 20px;margin:2px 0 14px;
box-shadow:0 2px 12px rgba(20,33,61,.05)}}
.sum h4{{margin:0 0 8px;color:{INK};font-size:16px}}
.sum ul{{margin:0;padding-left:18px;color:{INK2};font-size:14px;line-height:1.75}}.sum b{{color:{INK}}}
.crit{{display:flex;justify-content:space-between;align-items:center;padding:10px 2px;border-bottom:1px dashed #e3eaf4}}
.crit:last-child{{border-bottom:0}}.crit .l{{color:{INK2};font-size:13.5px}}
.crit .v{{font-weight:750;color:{INK};text-align:right;font-size:15px}}
.crit .v small{{display:block;font-weight:500;color:{MUTED};font-size:11.5px}}
.badge{{display:inline-block;padding:3px 12px;border-radius:999px;font-size:12.5px;font-weight:700}}
.b-ok{{background:#d8f1ed;color:#16675e}}.b-md{{background:#dbeafb;color:#16509a}}.b-hi{{background:#fdebd3;color:#9a4a07}}
.lg{{display:inline-flex;align-items:center;gap:6px;margin:0 12px 4px 0;font-size:12px;color:{INK2}}}
.lg i{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.sec{{display:flex;align-items:center;gap:10px;margin:18px 0 8px}}
.sec h3{{margin:0;font-size:18px;color:{INK}}}.sec span{{color:{MUTED};font-size:13px}}
[data-testid="stExpander"] details{{background:#fff;border:1px solid #dce7f5 !important;border-radius:16px !important;
box-shadow:0 2px 12px rgba(20,33,61,.05)}}
[data-testid="stExpander"] summary p{{font-weight:750;color:{INK};font-size:15.5px}}
.foot{{color:{MUTED};font-size:12px;margin-top:18px;text-align:center}}
</style>""", unsafe_allow_html=True)


# =============================================================================
# UTILITAS
# =============================================================================
def style(fig, h=330, legend=True, unified=True, top_legend=True):
    fig.update_layout(height=h, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=legend,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK2, size=12),
                      margin=dict(l=52, r=18, t=40 if (legend and top_legend) else 14, b=42),
                      hovermode="x unified" if unified else "closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
                      hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor="#b9d3f2"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#d5dfeb", automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#d5dfeb", automargin=True)
    return fig


def show(fig, key):
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def card(h=None):
    """Kartu berbingkai dengan tinggi seragam (jika didukung versi Streamlit)."""
    if h:
        try:
            return st.container(border=True, height=h)
        except TypeError:
            pass
    return st.container(border=True)


def head(t, s=""):
    st.markdown(f'<div class="ct">{t}</div>' + (f'<div class="cs">{s}</div>' if s else ""), unsafe_allow_html=True)


def insight(t):
    st.markdown(f'<div class="ins">💡 {t}</div>', unsafe_allow_html=True)


def section(t, s=""):
    st.markdown(f'<div class="sec"><h3>{t}</h3><span>{s}</span></div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: (ikon, label, nilai, keterangan, varian '' | 'hl' | 'ok' | 'md' | 'hi')"""
    html = ""
    for ic, a, b, d, *v in items:
        html += (f'<div class="kpi {v[0] if v else ""}"><span class="ic">{ic}</span><div class="kl">{a}</div>'
                 f'<div class="kv{" sm" if len(str(b)) > 14 else ""}">{b}</div><div class="ks">{d}</div></div>')
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan, BIAS=np.nan, R2=np.nan)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                MAPE=np.mean(np.abs((y - p) / y)) * 100, BIAS=np.mean(p - y),
                R2=1 - np.sum((y - p) ** 2) / ss_tot if ss_tot > 0 else np.nan)


def tgl(ts, year=True, short_year=False):
    """Tanggal berformat Indonesia, mis. 24 Mei 2019."""
    ts = pd.Timestamp(ts)
    y = f" {ts:%y}" if short_year else (f" {ts.year}" if year else "")
    return f"{ts.day:02d} {BULAN[ts.month - 1]}{y}"


def legend_html(pairs):
    return "".join(f'<span class="lg"><i style="background:{c}"></i>{k}</span>' for k, c in pairs)


# =============================================================================
# DATA (tetap dari folder hasil_kaggle/)
# =============================================================================
@st.cache_data(show_spinner="Memuat hasil pemodelan…")
def load_bundle(folder):
    def csv(name, **kw):
        p = os.path.join(folder, name)
        return pd.read_csv(p, **kw) if os.path.exists(p) else None
    meta = json.load(open(os.path.join(folder, "meta.json")))
    pred = csv("prediksi_model.csv.gz", parse_dates=["origin", "datetime"])
    pred["galat"] = pred["aktual"] - pred["prediksi"]
    pred["ape"] = (pred["galat"].abs() / pred["aktual"]) * 100
    pred["kondisi"] = np.where(pred["origin"] >= pd.Timestamp(meta.get("covid_start", "2020-03-15")), "COVID-19", "Normal")
    pred["bulan"] = pred["datetime"].dt.to_period("M").dt.to_timestamp()
    data = csv("data_historis.csv.gz", parse_dates=["datetime"]).set_index("datetime").sort_index()
    return dict(meta=meta, leader=csv("leaderboard_model.csv"), pred=pred, data=data,
                feat=csv("pengaruh_fitur.csv"), grp=csv("pengaruh_kelompok.csv"), corr=csv("korelasi_eksogen.csv"),
                shap=csv("shap_sampel.csv.gz"), cek=csv("cek_kebocoran.csv"), hfeat=csv("pengaruh_fitur_hybrid.csv"))


NEEDED = ["meta.json", "leaderboard_model.csv", "prediksi_model.csv.gz", "data_historis.csv.gz"]


def find_data_dir():
    """Cari hasil Kaggle di: folder hasil_kaggle/, akar repo, atau di dalam hasil_kaggle.zip."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [DATA_DIR, here]:
        if all(os.path.exists(os.path.join(d, f)) for f in NEEDED):
            return d
    for zp in [os.path.join(here, "hasil_kaggle.zip"), os.path.join(DATA_DIR, "hasil_kaggle.zip")]:
        if os.path.exists(zp):
            import tempfile
            import zipfile
            out = os.path.join(tempfile.gettempdir(), "hasil_kaggle_unzip")
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.endswith("/"):
                        with open(os.path.join(out, os.path.basename(n)), "wb") as fh:
                            fh.write(z.read(n))
            if all(os.path.exists(os.path.join(out, f)) for f in NEEDED):
                return out
    return None


B = None
_dir = find_data_dir()
if _dir is not None:
    B = load_bundle(_dir)

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan tampilan")
    show_eval = st.toggle("Tampilkan evaluasi model (detail teknis)", value=False,
                          help="Menampilkan tab berisi perbandingan lengkap semua model, cek kebocoran data, dan tabel metrik.")
    st.caption("Tab evaluasi berisi angka-angka teknis untuk laporan atau presentasi. Tampilan utama sudah cukup "
               "untuk membaca hasil ramalan.")
    st.divider()
    st.caption("Sumber: hasil pemodelan di Kaggle (folder hasil_kaggle/). Evaluasi memakai data 70% latih, 10% validasi, "
               "20% uji (400 hari), diramal ulang setiap tengah malam.")

if B is None:
    st.error("Hasil Kaggle tidak ditemukan. Letakkan file berikut di folder **hasil_kaggle/**, di akar repo, "
             "atau upload **hasil_kaggle.zip**: " + ", ".join(NEEDED))
    st.stop()

M, LB, PRED, DATA = B["meta"], B["leader"], B["pred"], B["data"]
BEST = M["best"]
SUSPECT = set(M.get("suspect") or [])
FAM_OF = dict(zip(LB.Model, LB.Keluarga))
MODELS = list(LB.sort_values("VAL_MAE").Model)
TEST = PRED[PRED.split == "test"]
LAST = pd.Timestamp(M["last_ts"])
y_all = DATA.nat_demand


def fam_color(n):
    return FAM.get(FAM_OF.get(n, ""), BLUE)


# =============================================================================
# HERO + FILTER
# =============================================================================
st.markdown(f"""
<div class="hero"><div style="z-index:1">
<h1>Peramalan Beban Listrik Nasional Panama</h1>
<p>Berapa kebutuhan listrik 24 jam ke depan, seberapa tepat ramalannya, dan apa saja yang memengaruhinya.</p>
<div class="chips"><span class="chip">📅 Data {tgl(pd.Timestamp(M['data_start']))} – {tgl(LAST)}</span>
<span class="chip">⏱️ {M['n_obs']:,} jam data</span><span class="chip">🧪 Diuji {M['test_days']} hari</span>
<span class="chip">⭐ Model terbaik: {short(BEST)}</span></div></div><div class="bolt">⚡</div></div>
""".replace(f"{M['n_obs']:,}", f"{M['n_obs']:,}".replace(",", ".")), unsafe_allow_html=True)

t_min, t_max = TEST.origin.min().date(), TEST.origin.max().date()
with st.container(border=True):
    f1, f2, f3 = st.columns([2.3, 2, 2])
    sel_model = f1.selectbox("🔎 Pilih model ramalan", MODELS, index=MODELS.index(BEST),
                             format_func=lambda n: f"{short(n)}  ·  {FAM_ID.get(FAM_OF[n], FAM_OF[n])}"
                             + ("  ⭐ terbaik" if n == BEST else "") + ("  ⚠️ diduga bocor" if n in SUSPECT else ""))
    kondisi = f2.radio("🗓️ Periode yang dinilai", ["Semua", "Normal", "Masa COVID-19"], horizontal=True,
                       help="Masa COVID-19 = sejak 15 Maret 2020, saat pembatasan kegiatan mengubah pola pemakaian listrik.")
    rng = f3.date_input("📅 Rentang tanggal uji", (t_min, t_max), min_value=t_min, max_value=t_max)
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        rng = (t_min, t_max)


def filt(P):
    P = P[(P.origin.dt.date >= rng[0]) & (P.origin.dt.date <= rng[1])]
    return P if kondisi == "Semua" else P[P.kondisi == ("COVID-19" if kondisi.startswith("Masa") else "Normal")]


TF = filt(TEST)


@st.cache_data(show_spinner=False)
def score_table(_tf, key):
    rows = [dict(Model=n, Keluarga=FAM_OF.get(n, "-"), **metrics(g.aktual, g.prediksi)) for n, g in _tf.groupby("model")]
    return pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["Model", "Keluarga", "MAE", "RMSE", "MAPE", "BIAS", "R2"])


SC = score_table(TF, (M.get("generated"), BEST, str(rng), kondisi, len(TF)))
if SC.empty:
    st.warning("Tidak ada data pada filter ini. Ubah rentang tanggal atau periode.")
    st.stop()
SCI = SC.set_index("Model")

# ---------- ramalan besok untuk model terpilih + interval dari galat 90 hari terakhir ----------
g90 = TEST[(TEST.model == sel_model) & (TEST.origin >= TEST.origin.max() - pd.Timedelta(days=89))]
qq = g90.groupby("horizon").galat.quantile([0.1, 0.9]).unstack()
FUT = PRED[(PRED.split == "future") & (PRED.model == sel_model)].sort_values("datetime").reset_index(drop=True)
FUT["lo"], FUT["hi"] = FUT.prediksi.values + qq[0.1].values, FUT.prediksi.values + qq[0.9].values
pk = FUT.loc[FUT.prediksi.idxmax()]
daily_peaks = y_all.loc[LAST - pd.Timedelta(days=90):].resample("D").max()
pct = (daily_peaks < pk.prediksi).mean() * 100
status = ("Tinggi", "hi", "b-hi") if pct >= 90 else (("Sedang", "md", "b-md") if pct >= 50 else ("Normal", "ok", "b-ok"))
ms = SCI.loc[sel_model] if sel_model in SCI.index else None
naive_mae = SCI.MAE.get("Seasonal Naive (168j)", np.nan)

if ms is not None:
    better = (1 - ms.MAE / naive_mae) * 100 if naive_mae == naive_mae else np.nan
    kpi_row([("🤖", "Model dipakai", short(sel_model), FAM_ID.get(FAM_OF[sel_model], "")
              + (" · terbaik" if sel_model == BEST else "") + (" · ⚠️ diduga bocor" if sel_model in SUSPECT else ""), "hl"),
             ("🎯", "Ketepatan ramalan", f"{fmt(100 - ms.MAPE, 1)}%", f"MAPE (rata-rata meleset) {fmt(ms.MAPE, 2)}%"),
             ("📏", "Selisih rata-rata (MAE)", f"±{fmt(ms.MAE, 0)} MWh",
              f"RMSE {fmt(ms.RMSE, 1)} MWh · R² {fmt(ms.R2, 3)}"),
             ("📈", "Lebih tepat dari cara sederhana", f"{fmt(better, 0)}%" if better == better else "–",
              "dibanding menyalin beban minggu lalu"),
             ("🔝", "Puncak beban besok", f"{fmt(pk.prediksi, 0)} MWh", f"sekitar pukul {pk.datetime:%H:%M}"),
             ("🚦", "Status beban besok", status[0], f"lebih tinggi dari {pct:.0f}% hari dalam 3 bulan terakhir", status[1])])

# =============================================================================
# TABS
# =============================================================================
tab_names = ["🏠 Ringkasan", "📉 Ramalan vs Kenyataan", "🔍 Variabel yang Berpengaruh", "🚀 Ramalan Langsung"]
if show_eval:
    tab_names.insert(3, "🏆 Evaluasi Model")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# =============================================================================
# RINGKASAN (termasuk eksplorasi data penting)
# =============================================================================
with tabs["🏠 Ringkasan"]:
    # ---------- temuan khusus periode yang dinilai ----------
    CS = pd.Timestamp(M.get("covid_start", "2020-03-15"))
    T_START = pd.Timestamp(next((x["mulai"] for x in M.get("split", []) if x["bagian"] == "Test"), str(TEST.datetime.min())))
    YR = pd.Timedelta(days=364)                                    # 52 minggu → hari dalam seminggu tetap sama
    a19, b19 = CS - YR, LAST - YR

    def prof24(a, b, weekend=None):
        X = DATA.loc[a:b]
        X = X[X.holiday == 0]
        if weekend is not None:
            X = X[(X.index.dayofweek >= 5) == weekend]
        return X.groupby(X.index.hour).nat_demand.mean()

    def wk_gap(a, b):
        return (1 - prof24(a, b, True).mean() / prof24(a, b, False).mean()) * 100

    Gm = TEST[TEST.model == sel_model]
    m_norm = Gm[Gm.kondisi == "Normal"].ape.mean()
    m_cov = Gm[Gm.kondisi == "COVID-19"].ape.mean()
    ok = SC[~SC.Model.isin(SUSPECT)]
    best_here = ok.iloc[0] if len(ok) else SC.iloc[0]
    lbr = LB[~LB.Model.isin(SUSPECT) & (LB.Keluarga != "Baseline")].dropna(subset=["MAPE_NORMAL", "MAPE_COVID"]) \
        if "MAPE_COVID" in LB else pd.DataFrame()
    if len(lbr):
        lbr = lbr.assign(naik=lbr.MAPE_COVID - lbr.MAPE_NORMAL)
        rob, frag = lbr.sort_values("naik").iloc[0], lbr.sort_values("naik").iloc[-1]
    drop = (1 - y_all.loc[CS:LAST].mean() / y_all.loc[a19:b19].mean()) * 100
    g19, g20 = wk_gap(a19, b19), wk_gap(CS, LAST)
    tiles, figp = [], go.Figure()     # tile: (ikon, angka besar, judul, keterangan)

    if kondisi.startswith("Masa"):
        theme = ("#1d7a70", "#2a9d8f", "#e3f4f1")
        judul, sub = "Temuan pada masa COVID-19", f"{tgl(CS, False)} – {tgl(LAST)} · dibandingkan dengan periode yang sama tahun 2019"
        p19, p20 = prof24(a19, b19, False), prof24(CS, LAST, False)
        day19, day20 = p19.loc[9:17].mean(), p20.loc[9:17].mean()
        ngt19, ngt20 = p19.loc[0:5].mean(), p20.loc[0:5].mean()
        Gc = Gm[Gm.kondisi == "COVID-19"]
        first = Gc[Gc.origin < CS + pd.Timedelta(days=14)].ape.mean()
        rest = Gc[Gc.origin >= CS + pd.Timedelta(days=14)].ape.mean()
        tiles += [("📉", f"−{fmt(drop, 0)}%", "Beban listrik turun", "dibanding periode yang sama tahun 2019"),
                  ("🏢", f"−{fmt((1 - day20 / day19) * 100, 0)}% vs −{fmt((1 - ngt20 / ngt19) * 100, 0)}%", "Jam kerja vs malam hari",
                   "kantor & sekolah tutup, rumah tangga tetap memakai listrik"),
                  ("📅", f"{fmt(g19, 0)}% → {fmt(g20, 0)}%", "Beda hari kerja & akhir pekan menyempit",
                   "saat pembatasan, setiap hari terasa seperti akhir pekan"),
                  (("⏰", f"{int(p19.idxmax()):02d}:00 → {int(p20.idxmax()):02d}:00", "Jam puncak hari kerja bergeser",
                    f"puncak {fmt(p19.max(), 0)} → {fmt(p20.max(), 0)} MWh") if p19.idxmax() != p20.idxmax() else
                   ("⏰", f"−{fmt((1 - p20.max() / p19.max()) * 100, 0)}%", f"Puncak hari kerja (tetap pukul {int(p20.idxmax()):02d}:00)",
                    f"turun dari {fmt(p19.max(), 0)} ke {fmt(p20.max(), 0)} MWh")),
                  ("🎯", f"{fmt(first, 1)}% → {fmt(rest, 1)}%", "Meleset: 2 minggu pertama → sesudahnya",
                   f"{short(sel_model)} {'mulai membaca pola baru' if rest < first else 'masih kesulitan'}")]
        if len(lbr):
            tiles.append(("🛡️", short(rob.Model), "Paling tahan perubahan",
                          f"meleset {fmt(rob.MAPE_COVID, 2)}% · paling terganggu: {short(frag.Model)} (+{fmt(frag.naik, 1)} poin)"))
        figp.add_vrect(x0=9, x1=17, fillcolor="#fde9c9", opacity=0.45, line_width=0,
                       annotation_text="jam kerja", annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
        for lab, p, c, d in [("Hari kerja 2019", p19, "#93c0f1", "dot"), ("Hari kerja masa COVID", p20, NAVY, None),
                             ("Akhir pekan 2019", prof24(a19, b19, True), "#9fd6cf", "dot"),
                             ("Akhir pekan masa COVID", prof24(CS, LAST, True), TEAL, None)]:
            figp.add_trace(go.Scatter(x=p.index, y=p.values, name=lab, line=dict(color=c, width=2.8 if d is None else 2, dash=d),
                                      hovertemplate="%{y:,.0f} MWh"))
        chart_t, chart_s = "Pola beban sehari: sebelum vs saat pandemi", "Titik-titik = 2019 · garis tegas = masa COVID-19"
        figp.update_xaxes(title="Jam", dtick=3).update_yaxes(title="MWh")
        is_bar = False

    elif kondisi == "Normal":
        theme = ("#16509a", "#3f8ae0", "#e6f0fc")
        A, Bn = T_START, CS - pd.Timedelta(hours=1)
        judul, sub = "Temuan pada masa normal", f"{tgl(A)} – {tgl(Bn)} · sebelum pembatasan COVID-19"
        grow = (y_all.loc[A:Bn].mean() / y_all.loc[A - YR:Bn - YR].mean() - 1) * 100
        pw, pe = prof24(A, Bn, False), prof24(A, Bn, True)
        Xh = DATA.loc[A:Bn]
        hol = Xh[Xh.holiday == 1].nat_demand.mean()
        wkd = Xh[(Xh.holiday == 0) & (Xh.index.dayofweek < 5)].nat_demand.mean()
        Xd = Xh[Xh.index.hour.isin(range(9, 18))]
        tm = Xd[[c for c in Xd.columns if c.startswith("T2M")]].mean(axis=1)
        slope = np.polyfit(tm, Xd.nat_demand, 1)[0] if len(Xd) > 50 else np.nan
        Gn = Gm[Gm.kondisi == "Normal"]
        tiles += [("📈", f"{'+' if grow > 0 else '−'}{fmt(abs(grow), 1)}%", "Pertumbuhan beban",
                   f"dibanding setahun sebelumnya · rata-rata {fmt(y_all.loc[A:Bn].mean(), 0)} MWh"),
                  ("⏰", f"{int(pw.idxmax()):02d}:00", "Jam puncak hari kerja", f"rata-rata puncak {fmt(pw.max(), 0)} MWh"),
                  ("🏖️", f"−{fmt(wk_gap(A, Bn), 0)}% / −{fmt((1 - hol / wkd) * 100, 0)}%", "Akhir pekan / hari libur",
                   "lebih rendah dibanding hari kerja"),
                  ("🌡️", f"+{fmt(slope, 0)} MWh/°C", "Efek suhu di siang hari", "makin panas, makin banyak AC menyala"),
                  ("🎯", f"{fmt(m_norm, 2)}%", f"Meleset · {short(sel_model)}",
                   f"{fmt((Gn.ape <= 5).mean() * 100, 0)}% jam ramalan meleset kurang dari 5%")]
        own = ok[ok.Keluarga.isin(["Klasik", "Machine Learning", "Deep Learning"])]
        tlo = ok[ok.Keluarga == "Transfer Learning"]
        if len(own) and len(tlo):
            d_ = own.iloc[0].MAPE - tlo.iloc[0].MAPE
            tiles.append(("🤖", f"{fmt(own.iloc[0].MAPE, 2)}% vs {fmt(tlo.iloc[0].MAPE, 2)}%", "Dilatih sendiri vs transfer learning",
                          f"{short(own.iloc[0].Model)} vs {short(tlo.iloc[0].Model)} · "
                          + ("bersaing ketat saat pola stabil" if abs(d_) <= 0.5 else
                             ("model buatan sendiri unggul" if d_ < 0 else "transfer learning unggul"))))
        figp.add_vrect(x0=9, x1=17, fillcolor="#fde9c9", opacity=0.45, line_width=0,
                       annotation_text="jam kerja", annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
        ph = Xh[Xh.holiday == 1].groupby(Xh[Xh.holiday == 1].index.hour).nat_demand.mean()
        for lab, p, c in [("Hari kerja", pw, NAVY), ("Akhir pekan", pe, "#3f8ae0"), ("Hari libur", ph, TEAL)]:
            figp.add_trace(go.Scatter(x=p.index, y=p.values, name=lab, line=dict(color=c, width=2.8), hovertemplate="%{y:,.0f} MWh"))
        chart_t, chart_s = "Pola beban sehari pada masa normal", "Rata-rata beban tiap jam menurut jenis hari"
        figp.update_xaxes(title="Jam", dtick=3).update_yaxes(title="MWh")
        is_bar = False

    else:
        theme = ("#0b3c75", "#5b5bd6", "#ecebfb")
        judul, sub = "Temuan pada seluruh periode uji", f"{tgl(T_START)} – {tgl(LAST)} · masa normal + masa COVID-19"
        mm = Gm.groupby("bulan").ape.mean()
        worst, easy = mm.idxmax(), mm.idxmin()
        tiles += [("🎯", f"{fmt(m_norm, 2)}% → {fmt(m_cov, 2)}%", "Meleset: normal → masa COVID-19", short(sel_model)),
                  ("📆", f"{BULAN[worst.month - 1]} {worst.year}", "Bulan paling sulit diramal", f"meleset {fmt(mm.max(), 2)}%"),
                  ("✅", f"{BULAN[easy.month - 1]} {easy.year}", "Bulan paling mudah diramal", f"meleset {fmt(mm.min(), 2)}%"),
                  ("📉", f"−{fmt(drop, 0)}%", "Dampak pandemi pada beban", f"beda hari kerja–akhir pekan {fmt(g19, 0)}% → {fmt(g20, 0)}%")]
        if len(lbr):
            tiles.append(("🛡️", short(rob.Model), "Paling stabil di kedua kondisi",
                          f"{fmt(rob.MAPE_NORMAL, 2)}% → {fmt(rob.MAPE_COVID, 2)}%"))
        tiles.append(("🏆", short(best_here.Model), "Paling tepat di seluruh periode", f"meleset {fmt(best_here.MAPE, 2)}%"))
        xs = list(range(len(mm)))
        cv = [d >= CS.to_period("M").to_timestamp() for d in mm.index]
        figp.add_trace(go.Bar(x=xs, y=mm.values, marker=dict(color=[TEAL if c else "#6aa6ea" for c in cv], line=dict(color="#fff", width=1)),
                              text=[f"{v:.1f}" for v in mm.values], textposition="outside", cliponaxis=False,
                              customdata=[f"{BULAN[d.month - 1]} {d.year}" for d in mm.index],
                              hovertemplate="%{customdata}: %{y:.2f}%<extra></extra>"))
        if any(cv):
            k0 = cv.index(True)
            figp.add_vline(x=k0 - 0.5, line=dict(color=MUTED, dash="dash"))
            figp.add_annotation(x=k0 - 0.5, y=1.02, yref="paper", text="mulai COVID-19 →", showarrow=False, xanchor="left",
                                yanchor="bottom", font=dict(size=10.5, color="#1d7a70"))
        figp.update_xaxes(tickvals=xs, ticktext=[f"{BULAN[d.month - 1]}<br>{d:%y}" for d in mm.index])
        figp.update_yaxes(title="meleset (%)", range=[0, float(mm.max()) * 1.25])
        chart_t, chart_s = "Rata-rata meleset per bulan", f"{short(sel_model)} · biru = masa normal, teal = masa COVID-19"
        is_bar = True

    c1_, c2_, c3_ = theme
    st.markdown(f"""
<div style="background:linear-gradient(120deg,{c1_} 0%,{c2_} 100%);border-radius:16px;padding:14px 20px;margin:4px 0 12px;
color:#fff;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;box-shadow:0 6px 18px rgba(11,60,117,.18)">
<div><div style="font-size:12px;letter-spacing:.8px;text-transform:uppercase;opacity:.85;font-weight:700">🔎 Temuan periode yang dinilai</div>
<div style="font-size:21px;font-weight:800;margin-top:2px">{judul}</div>
<div style="font-size:13px;opacity:.9;margin-top:2px">{sub}</div></div>
<div style="background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:5px 14px;font-size:12.5px">
{len(tiles)} temuan otomatis · ubah lewat filter “Periode yang dinilai”</div></div>""", unsafe_allow_html=True)

    cA, cB = st.columns([1.2, 1])
    with cA:
        html = "".join(
            f'<div style="background:#fff;border:1px solid #dce7f5;border-radius:14px;padding:12px 14px;display:flex;gap:12px;'
            f'align-items:flex-start;box-shadow:0 2px 10px rgba(20,33,61,.05);min-height:96px">'
            f'<div style="width:42px;height:42px;border-radius:12px;background:{c3_};display:flex;align-items:center;'
            f'justify-content:center;font-size:21px;flex:none">{ic}</div><div style="min-width:0">'
            f'<div style="font-size:{19 if len(big) <= 14 else 15.5}px;font-weight:800;color:{INK};line-height:1.2">{big}</div>'
            f'<div style="font-size:12.8px;font-weight:700;color:{c1_};margin-top:3px">{lab}</div>'
            f'<div style="font-size:12px;color:{MUTED};margin-top:2px;line-height:1.4">{desc}</div></div></div>'
            for ic, big, lab, desc in tiles)
        st.markdown(f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px">{html}</div>',
                    unsafe_allow_html=True)
    with cB:
        with card(400):
            head(chart_t, chart_s)
            show(style(figp, 315, legend=not is_bar, unified=not is_bar), "periode")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ---------- ringkasan utama ----------
    best_now = SC.iloc[0]
    bl = [f"Model <b>{short(BEST)}</b> dipilih sebagai yang terbaik. Ramalannya rata-rata hanya meleset "
          f"<b>{fmt(SCI.MAPE.get(BEST, np.nan), 2)}%</b> (sekitar ±{fmt(SCI.MAE.get(BEST, np.nan), 0)} MWh per jam)."]
    tl, ntl = SC[SC.Keluarga == "Transfer Learning"], SC[SC.Keluarga.isin(["Klasik", "Machine Learning", "Deep Learning"])]
    tl_ok = tl[~tl.Model.isin(SUSPECT)]
    if len(tl_ok) and len(ntl):
        a, b = ntl.iloc[0], tl_ok.iloc[0]
        bl.append(f"Model yang <b>sudah dilatih sebelumnya</b> (transfer learning) seperti <b>{short(b.Model)}</b> meleset "
                  f"{fmt(b.MAPE, 2)}%, sedangkan model yang <b>dilatih dari nol</b> terbaik ({short(a.Model)}) meleset "
                  f"{fmt(a.MAPE, 2)}%.")
    hyb = SC[SC.Keluarga == "Hybrid"]
    hb = M.get("hybrid_base")
    if len(hyb) and hb in SCI.index:
        h0 = hyb.iloc[0]
        d = (1 - h0.MAPE / SCI.loc[hb].MAPE) * 100
        bl.append(f"Menggabungkan {short(hb)} dengan model buatan sendiri (<b>{short(h0.Model)}</b>) "
                  f"{'mengurangi' if d > 0 else 'menambah'} kesalahan sebesar <b>{fmt(abs(d), 0)}%</b>.")
    if SUSPECT:
        bl.append(f"⚠️ <b>{', '.join(short(x) for x in SUSPECT)}</b> hasilnya terlalu sempurna, diduga sudah pernah "
                  f"\"melihat\" data ini saat dilatih oleh pembuatnya. Tetap ditampilkan, tetapi tidak dijadikan model terbaik.")
    with st.expander("📌 Ringkasan singkat — klik untuk menampilkan / menyembunyikan", expanded=True):
        st.markdown('<div class="sum" style="box-shadow:none;border:0;padding:4px 6px;margin:0"><ul>'
                    + "".join(f"<li>{x}</li>" for x in bl) + "</ul></div>", unsafe_allow_html=True)

    # ---------- ramalan besok + jam kritis ----------
    c1, c2 = st.columns([2.35, 1])
    with c1:
        with card(560):
            head("⚡ Ramalan kebutuhan listrik 24 jam ke depan",
                 f"{short(sel_model)} · area biru muda = rentang kemungkinan (80% yakin nilai sebenarnya ada di sini)")
            hist = y_all.loc[LAST - pd.Timedelta(hours=71):LAST]
            lastweek = y_all.reindex(FUT.datetime - pd.Timedelta(days=7)).values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.lo, fill="tonexty", fillcolor="rgba(63,138,224,0.16)",
                                     line=dict(width=0), name="Rentang kemungkinan", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Kenyataan 3 hari terakhir",
                                     line=dict(color="#8a97ab", width=2), hovertemplate="%{y:,.0f} MWh"))
            fig.add_trace(go.Scatter(x=FUT.datetime, y=lastweek, name="Minggu lalu (jam yang sama)",
                                     line=dict(color="#93c5fd", width=1.6, dash="dot"), hovertemplate="%{y:,.0f} MWh"))
            fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.prediksi, name="Ramalan", mode="lines+markers",
                                     marker=dict(size=6, color=BLUE), line=dict(color=BLUE, width=3), hovertemplate="%{y:,.0f} MWh"))
            fig.add_vline(x=LAST, line=dict(color=MUTED, dash="dash"))
            fig.add_annotation(x=pk.datetime, y=pk.prediksi, text=f"Puncak {fmt(pk.prediksi, 0)} MWh", showarrow=True,
                               arrowhead=2, ay=-36, font=dict(color=INK, size=11.5), bgcolor="#fff", bordercolor=BLUE)
            show(style(fig, 360).update_yaxes(title="MWh"), "fut")
            dlw = (FUT.prediksi.sum() / np.nansum(lastweek) - 1) * 100
            insight(f"Besok kebutuhan listrik diperkirakan paling tinggi sekitar <b>pukul {pk.datetime:%H:%M}</b> "
                    f"({fmt(pk.prediksi, 0)} MWh). Total kebutuhan sehari <b>{'lebih tinggi' if dlw > 0 else 'lebih rendah'} "
                    f"{fmt(abs(dlw), 1)}%</b> dibanding hari yang sama minggu lalu.")
    with c2:
        with card(560):
            head("⏰ Jam-jam penting besok", "Bekal untuk mengatur pembangkit")
            ramp = np.diff(np.r_[y_all.loc[LAST], FUT.prediksi.values])
            ri = int(np.argmax(ramp))
            lo_i = FUT.prediksi.idxmin()
            rows = [("🔝 Beban tertinggi", f"{fmt(pk.prediksi, 0)} MWh<small>pukul {pk.datetime:%H:%M}</small>"),
                    ("🔻 Beban terendah", f"{fmt(FUT.loc[lo_i, 'prediksi'], 0)} MWh<small>pukul {FUT.loc[lo_i, 'datetime']:%H:%M}</small>"),
                    ("🚀 Kenaikan paling cepat", f"+{fmt(ramp[ri], 0)} MW/jam<small>menjelang pukul {FUT.datetime.iloc[ri]:%H:%M}</small>"),
                    ("🔋 Total energi sehari", f"{fmt(FUT.prediksi.sum() / 1000, 2)} GWh<small>rata-rata {fmt(FUT.prediksi.mean(), 0)} MWh/jam</small>"),
                    ("⚖️ Kerataan beban", f"{fmt(FUT.prediksi.mean() / pk.prediksi * 100, 0)}%<small>makin tinggi makin merata</small>"),
                    ("🚦 Status", f'<span class="badge {status[2]}">{status[0]}</span>')]
            st.markdown("".join(f'<div class="crit"><span class="l">{a}</span><span class="v">{b}</span></div>' for a, b in rows),
                        unsafe_allow_html=True)
            out = FUT[["datetime", "prediksi", "lo", "hi"]].rename(columns={"datetime": "waktu", "prediksi": "ramalan_MWh",
                                                                             "lo": "batas_bawah", "hi": "batas_atas"})
            st.download_button("⬇️ Unduh ramalan 24 jam", out.round(1).to_csv(index=False).encode(), "ramalan_24jam.csv",
                               "text/csv")
            insight(f"Siapkan tambahan pasokan sebelum pukul <b>{FUT.datetime.iloc[ri]:%H:%M}</b>, saat beban naik paling cepat.")

    # ---------- peringkat · variabel · normal vs covid ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        with card(610):
            head("🏆 Model paling tepat", "Rata-rata meleset (%) · makin pendek makin bagus")
            T = SC.head(8).iloc[::-1]
            fig = go.Figure(go.Bar(x=T.MAPE, y=[short(n) + (" ⚠️" if n in SUSPECT else "") for n in T.Model], orientation="h",
                                   marker=dict(color=[fam_color(n) for n in T.Model], line=dict(color="#fff", width=1),
                                               pattern=dict(shape=["/" if n in SUSPECT else "" for n in T.Model])),
                                   text=[f"{v:.2f}%" for v in T.MAPE], textposition="outside", cliponaxis=False,
                                   textfont=dict(color=INK2, size=11), customdata=[FAM_ID.get(k, k) for k in T.Keluarga],
                                   hovertemplate="<b>%{y}</b><br>%{customdata}<br>meleset %{x:.2f}%<extra></extra>"))
            fig.update_layout(margin=dict(l=10, r=48, t=8, b=30))
            show(style(fig, 330, legend=False, unified=False), "rank")
            st.markdown(legend_html([(FAM_ID[k], c) for k, c in FAM.items() if k in set(T.Keluarga)]), unsafe_allow_html=True)
            r0 = SC[~SC.Model.isin(SUSPECT)].iloc[0]
            insight(f"<b>{short(r0.Model)}</b> paling tepat pada periode ini: ramalannya rata-rata hanya meleset "
                    f"<b>{fmt(r0.MAPE, 2)}%</b>, atau sekitar ±{fmt(r0.MAE, 0)} MWh setiap jam.")
    with c2:
        with card(610):
            head("🧩 Apa yang menentukan ramalan?", "Porsi pengaruh tiap kelompok variabel")
            if B["grp"] is not None:
                g = B["grp"].sort_values("kontribusi_pct", ascending=False).reset_index(drop=True)
                labels = [nice_group(k) for k in g.kelompok]
                cols = [RAMP[min(i, len(RAMP) - 1)] for i in range(len(g))]
                txt = [f"{v:.0f}%" if v >= 5 else "" for v in g.kontribusi_pct]
                fig = go.Figure(go.Pie(labels=labels, values=g.kontribusi_pct, hole=0.62, sort=False, direction="clockwise",
                                       marker=dict(colors=cols, line=dict(color="#fff", width=2)), text=txt,
                                       textinfo="text", textposition="inside", insidetextorientation="horizontal",
                                       textfont=dict(color="#fff", size=12),
                                       hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>", domain=dict(x=[0.1, 0.9], y=[0, 1])))
                fig.update_layout(annotations=[dict(text=f"<b>{g.kontribusi_pct.iloc[0]:.0f}%</b><br><span style='font-size:11px'>"
                                                         f"variabel utama</span>", x=0.5, y=0.5, showarrow=False,
                                                    font=dict(size=20, color=INK))])
                show(style(fig, 250, legend=False, unified=False).update_layout(margin=dict(l=0, r=0, t=6, b=6)), "donut")
                st.markdown(legend_html(list(zip([f"{l} ({v:.1f}%)" for l, v in zip(labels, g.kontribusi_pct)], cols))[:6]),
                            unsafe_allow_html=True)
                gx = g[g.kelompok.isin(["Suhu", "Kelembapan", "Presipitasi", "Angin", "Libur & sekolah"])]
                insight(f"Ramalan paling banyak ditentukan oleh <b>{nice_group(g.kelompok.iloc[0]).lower()}</b> "
                        f"({g.kontribusi_pct.iloc[0]:.0f}%). Dari variabel luar, yang paling berpengaruh adalah "
                        f"<b>{nice_group(gx.kelompok.iloc[0]).lower() if len(gx) else '-'}</b>.")
    with c3:
        with card(610):
            head("🦠 Saat normal vs masa COVID-19", "Rata-rata meleset (%) · 6 model teratas")
            lbn = LB.dropna(subset=["MAPE_NORMAL", "MAPE_COVID"]) if "MAPE_COVID" in LB else pd.DataFrame()
            if len(lbn):
                T = lbn[~lbn.Model.isin(SUSPECT)].sort_values("VAL_MAE").head(6).iloc[::-1]
                fig = go.Figure([go.Bar(name="Kondisi normal", y=[short(n) for n in T.Model], x=T.MAPE_NORMAL, orientation="h",
                                        marker_color=BLUE, hovertemplate="%{x:.2f}%"),
                                 go.Bar(name="Masa COVID-19", y=[short(n) for n in T.Model], x=T.MAPE_COVID, orientation="h",
                                        marker_color=TEAL, hovertemplate="%{x:.2f}%")])
                fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06, margin=dict(l=10, r=14, t=40, b=30))
                show(style(fig, 330, unified=False).update_xaxes(title="meleset (%)"), "regime")
                rob = T.assign(d=T.MAPE_COVID - T.MAPE_NORMAL).sort_values("d").iloc[0]
                worst = T.assign(d=T.MAPE_COVID - T.MAPE_NORMAL).sort_values("d").iloc[-1]
                insight(f"Saat COVID-19 pola pemakaian listrik berubah drastis. <b>{short(rob.Model)}</b> paling stabil, "
                        f"sedangkan <b>{short(worst.Model)}</b> paling terganggu (meleset naik {fmt(worst.d, 1)} poin).")

    # ---------- eksplorasi data (yang penting) ----------
    section("📊 Mengenal pola beban listrik", "Pola historis 2015–2020 yang menjadi dasar ramalan")
    with st.container(border=True):
        head("📈 Perjalanan beban listrik dari tahun ke tahun",
             "Garis tipis = rata-rata harian · garis tebal = rata-rata 30 hari · area teal = masa COVID-19")
        dly = y_all.resample("D").mean()
        fig = go.Figure([go.Scatter(x=dly.index, y=dly.values, name="Rata-rata harian", line=dict(color="#bcd8f7", width=1),
                                    hovertemplate="%{x|%d %b %Y}: %{y:,.0f} MWh<extra></extra>"),
                         go.Scatter(x=dly.index, y=dly.rolling(30, min_periods=7).mean(), name="Rata-rata 30 hari",
                                    line=dict(color=NAVY, width=2.4), hovertemplate="%{y:,.0f} MWh<extra></extra>")])
        cs = pd.Timestamp(M.get("covid_start", "2020-03-15"))
        fig.add_vrect(x0=cs, x1=dly.index.max(), fillcolor=TEAL, opacity=0.12, line_width=0)
        for sp in M.get("split", [])[1:]:
            fig.add_vline(x=pd.Timestamp(sp["mulai"]), line=dict(color="#a7b4c6", dash="dot"))
            fig.add_annotation(x=pd.Timestamp(sp["mulai"]), y=1, yref="paper", showarrow=False, xanchor="left", yanchor="top",
                               text=" mulai data " + ("validasi" if sp["bagian"].startswith("Val") else "uji"),
                               font=dict(size=10.5, color=MUTED))
        show(style(fig, 300).update_yaxes(title="MWh"), "trend")
        yr = y_all.groupby(y_all.index.year).mean()
        g_ = ((yr.get(2019, np.nan) / yr.get(2015, np.nan)) ** (1 / 4) - 1) * 100
        pre = y_all.loc["2019-03-15":"2019-06-26"].mean()
        post = y_all.loc[cs:].mean()
        insight(f"Beban listrik tumbuh sekitar <b>{fmt(g_, 1)}% per tahun</b> sejak 2015. Saat pembatasan COVID-19, beban "
                f"turun <b>{fmt((1 - post / pre) * 100, 0)}%</b> dibanding periode yang sama tahun 2019.")

    c1, c2 = st.columns(2)
    with c1:
        with card(470):
            head("🗓️ Kapan listrik paling banyak dipakai?", "Rata-rata beban per hari dan jam")
            pv = DATA.assign(h=DATA.index.hour, d=DATA.index.dayofweek).pivot_table(index="d", columns="h", values="nat_demand")
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns], y=[HARI[i] for i in pv.index],
                                       colorscale=SEQ, xgap=2, ygap=2, colorbar=dict(title="MWh", thickness=10, len=0.9),
                                       hovertemplate="%{y}, pukul %{x}:00<br>%{z:,.0f} MWh<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 300, legend=False, unified=False).update_xaxes(title="Jam"), "heat")
            mx = np.unravel_index(np.nanargmax(pv.values), pv.shape)
            mn = np.unravel_index(np.nanargmin(pv.values), pv.shape)
            insight(f"Pemakaian tertinggi biasanya hari <b>{HARI[pv.index[mx[0]]]} pukul {pv.columns[mx[1]]:02d}:00</b>, "
                    f"terendah hari <b>{HARI[pv.index[mn[0]]]} pukul {pv.columns[mn[1]]:02d}:00</b>. "
                    f"Akhir pekan jelas lebih rendah dari hari kerja.")
    with c2:
        with card(470):
            head("🏖️ Pola sehari: hari kerja, akhir pekan, dan hari libur", "Rata-rata beban tiap jam")
            prof = DATA.assign(h=DATA.index.hour, tipe=np.where(DATA.holiday == 1, "Hari libur",
                                                                np.where(DATA.index.dayofweek >= 5, "Akhir pekan", "Hari kerja")))
            prof = prof.groupby(["tipe", "h"]).nat_demand.mean().unstack(0)
            fig = go.Figure([go.Scatter(x=prof.index, y=prof[c], name=c, line=dict(color=col, width=2.6),
                                        hovertemplate="%{y:,.0f} MWh") for c, col in
                             [("Hari kerja", NAVY), ("Akhir pekan", "#3f8ae0"), ("Hari libur", TEAL)] if c in prof])
            show(style(fig, 300).update_xaxes(title="Jam", dtick=2).update_yaxes(title="MWh"), "prof")
            if {"Hari kerja", "Hari libur"} <= set(prof.columns):
                dd = (1 - prof["Hari libur"].mean() / prof["Hari kerja"].mean()) * 100
                insight(f"Di hari libur, pemakaian listrik rata-rata <b>{fmt(dd, 0)}% lebih rendah</b> dari hari kerja. "
                        f"Puncak hari kerja terjadi sekitar <b>pukul {int(prof['Hari kerja'].idxmax()):02d}:00</b>.")

# =============================================================================
# RAMALAN VS KENYATAAN
# =============================================================================
with tabs["📉 Ramalan vs Kenyataan"]:
    c1, c2 = st.columns([2, 2])
    msel = c1.multiselect("Bandingkan model", MODELS, default=[sel_model], format_func=short, key="avp_m")
    days = sorted(TF.origin.dt.date.unique())
    dr = c2.select_slider("Perbesar tanggal", options=days, value=(days[max(0, len(days) - 14)], days[-1]),
                          format_func=lambda d: tgl(d, short_year=True))
    W = TF[(TF.origin.dt.date >= dr[0]) & (TF.origin.dt.date <= dr[1])]
    with st.container(border=True):
        head("📉 Ramalan dibanding kenyataan, jam per jam", f"{tgl(dr[0])} – {tgl(dr[1])}")
        act = W.drop_duplicates("datetime").sort_values("datetime")
        fig = go.Figure(go.Scatter(x=act.datetime, y=act.aktual, name="Kenyataan", line=dict(color="#8a97ab", width=2.2),
                                   hovertemplate="%{y:,.0f}"))
        for n in msel:
            d = W[W.model == n].sort_values("datetime")
            fig.add_trace(go.Scatter(x=d.datetime, y=d.prediksi, name=short(n), hovertemplate="%{y:,.0f}",
                                     line=dict(color=BLUE if n == sel_model else fam_color(n), width=2)))
        show(style(fig, 360).update_yaxes(title="MWh"), "avp")
        G = TF[TF.model == sel_model]
        within = (G.ape <= 5).mean() * 100
        insight(f"Dengan <b>{short(sel_model)}</b>, <b>{fmt(within, 0)}%</b> jam ramalannya meleset kurang dari 5%. "
                f"Ramalan cenderung <b>{'terlalu tinggi' if G.galat.mean() < 0 else 'terlalu rendah'}</b> "
                f"rata-rata {fmt(abs(G.galat.mean()), 0)} MWh.")

    c1, c2, c3 = st.columns(3)
    with c1:
        with card(460):
            head("🎯 Seberapa dekat dengan kenyataan?", "Tiap titik = 1 jam · titik di garis putus = tepat")
            smp = G.sample(min(3000, len(G)), random_state=1)
            lo_, hi_ = float(np.nanmin(smp.aktual)), float(np.nanmax(smp.aktual))
            fig = go.Figure([go.Scattergl(x=smp.aktual, y=smp.prediksi, mode="markers", marker=dict(size=4, color=BLUE, opacity=0.3),
                                          hovertemplate="kenyataan %{x:,.0f}<br>ramalan %{y:,.0f}<extra></extra>"),
                             go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", line=dict(color=NAVY, dash="dash", width=1.5),
                                        hoverinfo="skip")])
            show(style(fig, 290, legend=False, unified=False).update_xaxes(title="Kenyataan (MWh)").update_yaxes(title="Ramalan (MWh)"), "sc")
            insight("Makin rapat titik-titik ke garis putus-putus, makin tepat ramalannya.")
    with c2:
        with card(460):
            head("📊 Sebaran kesalahan", "Kanan nol = ramalan terlalu rendah, kiri nol = terlalu tinggi")
            fig = go.Figure(go.Histogram(x=G.galat, nbinsx=60, marker=dict(color="#3f8ae0", line=dict(color="#fff", width=1)),
                                         hovertemplate="selisih %{x}<br>%{y} jam<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=NAVY, dash="dash"))
            show(style(fig, 290, legend=False, unified=False).update_xaxes(title="Selisih (MWh)").update_yaxes(title="jumlah jam"), "hist")
            q = np.nanpercentile(G.galat.abs(), 90)
            insight(f"9 dari 10 jam, selisih ramalan tidak lebih dari <b>±{fmt(q, 0)} MWh</b>.")
    with c3:
        with card(460):
            head("📅 Hari apa paling sulit diramal?", "Rata-rata meleset (%) per hari")
            dw = G.assign(d=G.datetime.dt.dayofweek).groupby("d").ape.mean().reindex(range(7))
            cols = [RAMP[i] for i in pd.Series(dw.values).rank(ascending=False, method="first").fillna(7).astype(int).clip(1, 7).values - 1]
            fig = go.Figure(go.Bar(x=[h[:3] for h in HARI], y=dw.values, marker=dict(color=cols, line=dict(color="#fff", width=1)),
                                   text=[f"{v:.1f}%" for v in dw.values], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
            show(style(fig, 290, legend=False, unified=False).update_yaxes(title="meleset (%)"), "dow")
            insight(f"Hari <b>{HARI[int(dw.idxmax())]}</b> paling sulit diramal, <b>{HARI[int(dw.idxmin())]}</b> paling mudah.")

    with st.container(border=True):
        head("🔎 Hari-hari dengan ramalan paling meleset", f"{short(sel_model)} · cek apakah bertepatan dengan hari libur")
        day = G.groupby("origin").agg(meleset=("ape", "mean"), selisih=("galat", lambda e: e.abs().mean()),
                                      kenyataan=("aktual", "mean"), ramalan=("prediksi", "mean")).reset_index()
        day["tanggal"] = day.origin.dt.date
        day["hari"] = [HARI[pd.Timestamp(d).dayofweek] for d in day.tanggal]
        hol = DATA.holiday.resample("D").max()
        day["hari libur"] = ["✅" if hol.get(pd.Timestamp(d), 0) == 1 else "" for d in day.tanggal]
        worst = day.sort_values("meleset", ascending=False).head(8)
        cA, cB = st.columns([1.3, 1])
        with cA:
            st.dataframe(worst[["tanggal", "hari", "hari libur", "meleset", "selisih", "kenyataan", "ramalan"]], hide_index=True,
                         column_config={"meleset": st.column_config.ProgressColumn("meleset (%)", format="%.1f", min_value=0,
                                                                                   max_value=float(max(worst.meleset.max(), 1))),
                                        "selisih": st.column_config.NumberColumn("selisih (MWh)", format="%.0f"),
                                        "kenyataan": st.column_config.NumberColumn("rata² kenyataan", format="%.0f"),
                                        "ramalan": st.column_config.NumberColumn("rata² ramalan", format="%.0f")})
        with cB:
            fig = go.Figure(go.Scatter(x=day.origin, y=day.meleset, mode="lines", line=dict(color=BLUE, width=1.5), fill="tozeroy",
                                       fillcolor="rgba(63,138,224,0.12)", hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>"))
            show(style(fig, 270, legend=False).update_yaxes(title="meleset per hari (%)"), "dailymape")
        n_hol = (worst["hari libur"] == "✅").sum()
        insight(f"<b>{n_hol} dari 8</b> hari paling meleset adalah hari libur. Hari libur dan perubahan mendadak "
                f"(seperti awal pembatasan COVID-19) memang paling sulit ditebak.")

# =============================================================================
# FAKTOR YANG BERPENGARUH
# =============================================================================
with tabs["🔍 Variabel yang Berpengaruh"]:
    if B["feat"] is None:
        st.warning("File pengaruh_fitur.csv tidak ada pada hasil Kaggle.")
    else:
        FT, GR = B["feat"].copy(), B["grp"].copy()
        st.markdown(f'<div class="sum"><h4>🔍 Cara membaca</h4><ul><li>Pengaruh dihitung dari model <b>{M.get("imp_model")}</b>: '
                    f'seberapa besar ramalan memburuk jika suatu variabel "disembunyikan" dari model.</li>'
                    f'<li><b>Pola beban hari-hari sebelumnya</b> wajar paling dominan karena pemakaian listrik sangat berulang. '
                    f'Untuk melihat variabel luar (cuaca, libur), aktifkan <b>"hanya variabel luar"</b>.</li></ul></div>',
                    unsafe_allow_html=True)
        with st.expander("📖 Kamus variabel — arti setiap kolom pada data", expanded=False):
            st.dataframe(kamus_df(), hide_index=True)
            st.caption("Data cuaca diukur di tiga kota: Tocumen (bandara Kota Panama), Santiago, dan David. "
                       "Variabel turunan seperti 'Beban 1 hari sebelumnya' dibuat dari kolom beban listrik nasional.")
        c1, c2 = st.columns(2)
        with c1:
            with card(560):
                head("📦 Kelompok variabel paling penting", "Tambahan kesalahan (MWh) jika kelompok ini disembunyikan")
                G2 = GR.sort_values("kenaikan_mae_perm")
                cols = [RAMP[min(i, len(RAMP) - 1)] for i in range(len(G2))][::-1]
                fig = go.Figure(go.Bar(x=G2.kenaikan_mae_perm, y=[nice_group(k) for k in G2.kelompok], orientation="h",
                                       marker=dict(color=cols, line=dict(color="#fff", width=1)),
                                       text=[f"+{v:.1f}" for v in G2.kenaikan_mae_perm], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: +%{x:.1f} MWh<extra></extra>"))
                fig.update_layout(margin=dict(l=10, r=44, t=8, b=34))
                show(style(fig, 340, legend=False, unified=False).update_xaxes(title="MWh"), "perm")
                top = GR.sort_values("kenaikan_mae_perm", ascending=False)
                ext = top[~top.kelompok.isin(["Riwayat beban (lag)", "Kalender (jam/hari/musim)", "Rezim COVID-19"])]
                insight(f"Tanpa informasi <b>{nice_group(top.kelompok.iloc[0]).lower()}</b>, ramalan meleset {fmt(top.kenaikan_mae_perm.iloc[0], 0)} MWh "
                        f"lebih jauh. Dari variabel luar, <b>{nice_group(ext.kelompok.iloc[0]).lower() if len(ext) else '-'}</b> "
                        f"paling penting.")
        with c2:
            with card(560):
                cc1, cc2 = st.columns([1, 1])
                only_ex = cc1.toggle("Hanya variabel luar", value=True)
                n_top = cc2.slider("Jumlah variabel", 5, min(25, len(FT)), 10)
                F2 = FT[FT.kelompok != "Riwayat beban (lag)"] if only_ex else FT
                F2 = F2.sort_values("nilai", ascending=False).head(n_top).iloc[::-1]
                head("🏅 Variabel satu per satu", "Makin panjang batang, makin besar pengaruhnya")
                fig = go.Figure(go.Bar(x=F2.nilai, y=[nice_feat(f) for f in F2.fitur], orientation="h",
                                       marker=dict(color=F2.nilai, colorscale=SEQ[1:], line=dict(color="#fff", width=1)),
                                       customdata=[nice_group(k) for k in F2.kelompok],
                                       hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
                fig.update_layout(margin=dict(l=10, r=16, t=8, b=30))
                fig.update_xaxes(showticklabels=False)
                show(style(fig, 360, legend=False, unified=False), "feat")
                insight(f"Variabel tunggal paling berpengaruh: <b>{nice_feat(F2.fitur.iloc[-1])}</b>.")

        if B["hfeat"] is not None:
            with st.container(border=True):
                HF = B["hfeat"].head(12).iloc[::-1]
                head(f"🤝 Apa yang dilengkapi model buatan sendiri? (model hybrid, dasar {short(M.get('hybrid_base'))})",
                     "Informasi yang tidak dimiliki model transfer learning dan ditambahkan oleh model buatan sendiri")
                fig = go.Figure(go.Bar(x=HF.nilai, y=[nice_feat(f) for f in HF.fitur], orientation="h",
                                       marker=dict(color=HF.nilai, colorscale=[[0, "#c9c9f5"], [1, INDIGO]], line=dict(color="#fff", width=1)),
                                       customdata=[nice_group(k) for k in HF.kelompok],
                                       hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
                fig.update_layout(margin=dict(l=10, r=16, t=8, b=30))
                fig.update_xaxes(showticklabels=False)
                show(style(fig, 340, legend=False, unified=False), "hfeat")
                hx = B["hfeat"][~B["hfeat"].fitur.str.startswith("fm")]
                insight(f"Model buatan sendiri paling banyak mengoreksi ramalan transfer learning dengan informasi "
                        f"<b>{nice_feat(hx.fitur.iloc[0]) if len(hx) else '-'}</b> dan "
                        f"<b>{nice_feat(hx.fitur.iloc[1]) if len(hx) > 1 else '-'}</b>.")

        c1, c2 = st.columns(2)
        with c1:
            with card(560):
                city = st.selectbox("Lokasi pengukuran suhu", list(CITY), format_func=lambda k: CITY[k], key="effcity")
                head(f"🌡️ Makin panas, makin banyak listrik dipakai? ({CITY[city]})",
                     "Rata-rata beban pada tiap suhu, dipisah siang (08–18) dan malam")
                X = DATA.assign(tb=(DATA[f"T2M_{city}"] * 2).round() / 2,
                                w=np.where(DATA.index.hour.isin(range(8, 19)), "Siang", "Malam"))
                E = X.groupby(["w", "tb"]).nat_demand.agg(["mean", "count"]).reset_index()
                E = E[E["count"] >= 30]
                fig = go.Figure([go.Scatter(x=E[E.w == w].tb, y=E[E.w == w]["mean"], name=w, mode="lines+markers",
                                            line=dict(color=c, width=2.6), marker=dict(size=6), hovertemplate="%{y:,.0f} MWh")
                                 for w, c in [("Siang", BLUE), ("Malam", TEAL)]])
                show(style(fig, 330).update_xaxes(title="Suhu (°C)").update_yaxes(title="Rata-rata beban (MWh)"), "effT")
                Es = E[E.w == "Siang"]
                slope = np.polyfit(Es.tb, Es["mean"], 1)[0] if len(Es) > 3 else np.nan
                insight(f"Di siang hari, setiap kenaikan suhu <b>1°C</b> diikuti kenaikan beban sekitar "
                        f"<b>{fmt(slope, 0)} MWh</b>, karena pemakaian pendingin ruangan (AC) bertambah.")
        with c2:
            with card(560):
                head("🔗 Seberapa erat hubungan tiap variabel dengan beban?", "−1 = berlawanan · 0 = tidak berhubungan · +1 = searah")
                if B["corr"] is not None:
                    C = B["corr"].sort_values("Pearson", key=abs).tail(10)
                    fig = go.Figure(go.Bar(x=C.Pearson, y=[nice_feat(v) for v in C.variabel], orientation="h",
                                           marker=dict(color=C.Pearson, colorscale=[[0, TEAL], [0.5, "#dbeafb"], [1, NAVY]],
                                                       cmin=-1, cmax=1, line=dict(color="#fff", width=1)),
                                           text=[f"{v:+.2f}" for v in C.Pearson], textposition="outside", cliponaxis=False,
                                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
                    fig.update_layout(margin=dict(l=10, r=40, t=8, b=30))
                    show(style(fig, 400, legend=False, unified=False).update_xaxes(range=[-1, 1]), "corr")
                    c0 = B["corr"].iloc[B["corr"].Pearson.abs().argmax()]
                    insight(f"<b>{nice_feat(c0.variabel)}</b> paling erat hubungannya dengan beban listrik "
                            f"(korelasi {c0.Pearson:+.2f}).")

# =============================================================================
# EVALUASI MODEL (opsional, disembunyikan secara default)
# =============================================================================
if show_eval:
    with tabs["🏆 Evaluasi Model"]:
        c1, c2, c3 = st.columns([3, 1, 1])
        fam_sel = c1.multiselect("Jenis model", list(FAM), default=[k for k in FAM if k in set(LB.Keluarga)],
                                 format_func=lambda k: FAM_ID.get(k, k))
        met = c2.selectbox("Ukuran", ["MAPE", "MAE", "RMSE"], help="MAPE = rata-rata meleset (%), MAE = selisih rata-rata (MWh), "
                                                                   "RMSE = seperti MAE tapi lebih menghukum kesalahan besar")
        c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        c3.download_button("⬇️ Tabel lengkap", LB.to_csv(index=False).encode(), "leaderboard_model.csv", "text/csv")
        S2, P2 = SC[SC.Keluarga.isin(fam_sel)], TF[TF.keluarga.isin(fam_sel)]
        if S2.empty:
            st.warning("Pilih minimal satu jenis model.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                with card(580):
                    head(f"Perbandingan {met} semua model", "Periode sesuai filter di atas")
                    T = S2.sort_values(met).iloc[::-1]
                    fig = go.Figure(go.Bar(x=T[met], y=[short(n) + (" ⚠️" if n in SUSPECT else "") for n in T.Model], orientation="h",
                                           marker=dict(color=[fam_color(n) for n in T.Model], line=dict(color="#fff", width=1),
                                                       pattern=dict(shape=["/" if n in SUSPECT else "" for n in T.Model])),
                                           text=[f"{v:.2f}" for v in T[met]], textposition="outside", cliponaxis=False,
                                           hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>"))
                    fig.update_layout(margin=dict(l=10, r=46, t=8, b=30))
                    show(style(fig, 400, legend=False, unified=False).update_xaxes(title=met), "cmp")
                    st.markdown(legend_html([(FAM_ID[k], c) for k, c in FAM.items() if k in set(T.Keluarga)]), unsafe_allow_html=True)
            with c2:
                with card(580):
                    head("Kesalahan menurut jarak waktu ramalan", "Jam ke-1 s.d. ke-24 setelah ramalan dibuat · 6 model teratas")
                    fig = go.Figure()
                    for n in S2.Model[:6]:
                        g = P2[P2.model == n].groupby("horizon").galat.apply(lambda e: e.abs().mean())
                        fig.add_trace(go.Scatter(x=g.index, y=g.values, name=short(n), mode="lines",
                                                 line=dict(color=fam_color(n), width=3 if n == sel_model else 1.8,
                                                           dash=None if n == sel_model else "dot"), hovertemplate="%{y:.1f} MWh"))
                    show(style(fig, 400).update_xaxes(title="jam ke-", dtick=2).update_yaxes(title="selisih (MWh)"), "hz")
                    insight("Makin jauh jam yang diramal dari tengah malam, biasanya makin besar kesalahannya.")

            c1, c2 = st.columns(2)
            with c1:
                with card(560):
                    head("Cek kebocoran data (model transfer learning)",
                         "Model jujur memburuk di hari libur & awal lockdown (12–28 Mar 2020) karena tidak punya info itu")
                    CK = B["cek"]
                    if CK is None or CK.empty:
                        st.caption("File cek_kebocoran.csv tidak ada.")
                    else:
                        CK = CK.sort_values("MAPE_hari_biasa").iloc[::-1]
                        fig = go.Figure([go.Bar(name=lab, y=[short(n) + (" ⚠️" if n in SUSPECT else "") for n in CK.Model], x=CK[col],
                                                orientation="h", marker_color=c, hovertemplate="%{x:.2f}%")
                                         for lab, col, c in [("Hari biasa", "MAPE_hari_biasa", NAVY),
                                                             ("Hari libur", "MAPE_hari_libur", "#3f8ae0"),
                                                             ("Awal lockdown", "MAPE_awal_lockdown", TEAL)]])
                        fig.update_layout(barmode="group", bargap=0.25, margin=dict(l=10, r=14, t=40, b=30))
                        show(style(fig, 380, unified=False).update_xaxes(title="meleset (%)"), "cek")
                        insight("⚠️ = diduga bocor: tetap sangat tepat saat lockdown padahal tidak diberi informasi apa pun "
                                "tentang COVID-19, dan jauh lebih tepat dari semua model lain.")
            with c2:
                with card(560):
                    hb = M.get("hybrid_base")
                    head("Hybrid dibanding model dasarnya", f"Model dasar: {short(hb) if hb else '–'}")
                    names = ([hb] if hb in set(LB.Model) else []) + list(LB[LB.Keluarga == "Hybrid"].Model)
                    if len(names) < 2:
                        st.caption("Model hybrid tidak ada pada hasil ini.")
                    else:
                        T = LB.set_index("Model").loc[names]
                        fig = go.Figure([go.Bar(name=k, x=[short(n) for n in T.index], y=T[c], marker_color=col,
                                                text=[f"{v:.2f}" for v in T[c]], textposition="outside", cliponaxis=False,
                                                hovertemplate="%{y:.2f}%")
                                         for k, c, col in [("Seluruh uji", "TEST_MAPE", INDIGO), ("Normal", "MAPE_NORMAL", BLUE),
                                                           ("Masa COVID-19", "MAPE_COVID", TEAL)] if c in T])
                        fig.update_layout(barmode="group", bargap=0.25)
                        show(style(fig, 380, unified=False).update_yaxes(title="meleset (%)"), "hybcmp")
                        insight("Batang lebih pendek dari model dasar berarti penggabungan berhasil mengurangi kesalahan.")

            with st.container(border=True):
                head("Peta kesalahan bulanan", "Bulan mana yang sulit diramal? Makin gelap = makin meleset (%)")
                H = P2.groupby(["model", "bulan"]).ape.mean().unstack()
                H = H.loc[[m for m in S2.Model if m in H.index]]
                fig = go.Figure(go.Heatmap(z=H.values, x=[f"{BULAN[d.month - 1]} {d:%y}" for d in H.columns],
                                           y=[short(n) for n in H.index], colorscale=SEQ, xgap=2, ygap=2,
                                           colorbar=dict(title="%", thickness=10), text=np.round(H.values, 1),
                                           texttemplate="%{text}", textfont=dict(size=10),
                                           hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(margin=dict(l=10, r=20, t=8, b=30))
                show(style(fig, max(280, 30 * len(H) + 80), legend=False, unified=False), "heatm")

            with st.container(border=True):
                head("Tabel evaluasi lengkap", "VAL = data validasi (dasar memilih model terbaik) · TEST = data uji 400 hari")
                tb = LB[LB.Keluarga.isin(fam_sel)].copy()
                tb.insert(0, "Peringkat", tb.VAL_MAE.rank(method="first").astype(int))
                tb["Model"] = tb.Model.map(short)
                tb["Keluarga"] = tb.Keluarga.map(lambda k: FAM_ID.get(k, k))
                st.dataframe(tb.sort_values("Peringkat"), hide_index=True,
                             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tb.columns if tb[c].dtype.kind == "f"})

# =============================================================================
# RAMALAN LANGSUNG
# =============================================================================
CHRONOS = {"Chronos-Bolt Tiny (paling cepat)": "amazon/chronos-bolt-tiny",
           "Chronos-Bolt Mini": "amazon/chronos-bolt-mini",
           "Chronos-Bolt Small (disarankan)": "amazon/chronos-bolt-small",
           "Chronos-Bolt Base (paling tepat, lebih lambat)": "amazon/chronos-bolt-base"}


@st.cache_resource(show_spinner="Menyiapkan model (hanya sekali)…")
def load_chronos(name):
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return BaseChronosPipeline.from_pretrained(name, device_map="cpu")


def chronos_forecast(name, ctx, h):
    import torch
    q, _ = load_chronos(name).predict_quantiles(torch.tensor(ctx, dtype=torch.float32)[None, :], h,
                                                quantile_levels=[0.1, 0.5, 0.9])
    q = q[0].float().numpy()
    return q[:, 1], q[:, 0], q[:, 2]


def seasonal_naive(s, h, step):
    season = int(pd.Timedelta("7D") / step)
    if season * 2 > len(s):
        season = max(1, int(pd.Timedelta("1D") / step))
    v = s.values
    return np.array([v[len(v) - season + (i % season)] for i in range(h)])


def read_any(up):
    if up.name.lower().endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(up)
        return xl.parse("data" if "data" in xl.sheet_names else xl.sheet_names[0])
    return pd.read_csv(up, sep=None, engine="python")


def prep_series(df, tcol, ycol):
    s = df[[tcol, ycol]].copy()
    s[tcol] = pd.to_datetime(s[tcol], errors="coerce")
    s[ycol] = pd.to_numeric(s[ycol], errors="coerce")
    s = s.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol, keep="last").set_index(tcol)[ycol]
    step = s.index.to_series().diff().median()
    full = pd.date_range(s.index[0], s.index[-1], freq=step)
    n_gap = int(len(full) - s.reindex(full).notna().sum())
    return s.reindex(full).interpolate(limit_direction="both"), step, n_gap


with tabs["🚀 Ramalan Langsung"]:
    st.markdown('<div class="sum"><h4>🚀 Coba ramal dengan data Anda sendiri</h4><ul>'
                '<li>Unggah data beban terbaru (CSV/Excel yang berisi kolom waktu dan kolom beban).</li>'
                '<li>Model <b>Chronos-Bolt</b> (buatan Amazon, sudah dilatih dengan jutaan data deret waktu) langsung meramal '
                'tanpa perlu dilatih ulang.</li><li>Nyalakan <b>mode uji</b> untuk melihat seberapa tepat ramalannya pada data Anda.</li>'
                '</ul></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2.5], gap="medium")
    with left:
        with st.container(border=True):
            head("⚙️ Pengaturan")
            how = st.radio("Data", ["Unggah file", "Pakai data contoh"], horizontal=True)
            raw_df = None
            if how == "Unggah file":
                up = st.file_uploader("File CSV / Excel", type=["csv", "xlsx", "xls"], key="live_up")
                if up is not None:
                    try:
                        raw_df = read_any(up)
                    except Exception as e:
                        st.error(f"File tidak bisa dibaca: {e}")
            else:
                raw_df = DATA.reset_index()
            tcol = ycol = None
            if raw_df is not None:
                cols = list(raw_df.columns)
                t_guess = next((c for c in cols if any(k in str(c).lower() for k in ("date", "time", "tanggal", "waktu"))), cols[0])
                num = [c for c in cols if pd.api.types.is_numeric_dtype(raw_df[c])]
                y_guess = "nat_demand" if "nat_demand" in cols else (num[0] if num else cols[-1])
                tcol = st.selectbox("Kolom tanggal & jam", cols, index=cols.index(t_guess))
                ycol = st.selectbox("Kolom yang ingin diramal", cols, index=cols.index(y_guess),
                                    help="Biasanya nat_demand (beban listrik nasional). Arti setiap kolom ada di Kamus variabel.")
            mlabel = st.selectbox("Model", list(CHRONOS), index=2)
            h = int(st.number_input("Berapa langkah ke depan?", 1, 168, 24, help="Paling tepat sampai 64 langkah"))
            ctx_len = st.select_slider("Berapa banyak data lampau dipakai?", [168, 336, 512, 1024, 2048], value=2048)
            backtest = st.toggle("Mode uji (sembunyikan data terakhir lalu bandingkan)", value=False)
            run = st.button("🚀 Ramalkan sekarang", type="primary", disabled=raw_df is None)
    with right:
        if run and raw_df is not None:
            try:
                s, step, n_gap = prep_series(raw_df, tcol, ycol)
                if len(s) < 2 * h + 24:
                    raise ValueError("data terlalu pendek untuk jumlah langkah ini")
                actual, s_in = (s.iloc[-h:], s.iloc[:-h]) if backtest else (None, s)
                ctx = s_in.values[-ctx_len:].astype(np.float32)
                fidx = pd.date_range(s_in.index[-1] + step, periods=h, freq=step)
                naive = seasonal_naive(s_in, h, step)
                try:
                    with st.spinner("Sedang meramal…"):
                        med, lo, hi = chronos_forecast(CHRONOS[mlabel], ctx, h)
                    used = mlabel.split(" (")[0]
                except Exception as e:
                    st.warning(f"Model Chronos belum tersedia di server ({type(e).__name__}). Sementara memakai cara sederhana "
                               "(menyalin pola minggu lalu).")
                    sd = float(np.nanstd(s_in.diff(int(pd.Timedelta("1D") / step) or 1).values[-ctx_len:]))
                    med, lo, hi, used = naive, naive - 1.28 * sd, naive + 1.28 * sd, "Salin minggu lalu"
                st.session_state["live"] = dict(s=s_in, fidx=fidx, med=med, lo=lo, hi=hi, used=used, actual=actual,
                                                naive=naive, n_gap=n_gap, step=step, ycol=ycol)
            except Exception as e:
                st.error(f"Data tidak bisa diproses: {e}")
        R = st.session_state.get("live")
        if R is None:
            with st.container(border=True):
                head("Belum ada ramalan", "Pilih data lalu klik 'Ramalkan sekarang'. Contoh isi file yang bisa diunggah:")
                st.dataframe(pd.DataFrame({"datetime": ["2020-06-26 01:00", "2020-06-26 02:00", "…"],
                                           "nat_demand": [1012.5, 978.1, None]}), hide_index=True)
                st.caption("Nama kolom boleh berbeda — Anda bisa memilih kolomnya di panel pengaturan.")
            with st.expander("📖 Kamus variabel data contoh", expanded=False):
                st.dataframe(kamus_df(), hide_index=True)
        else:
            pk_i = int(np.argmax(R["med"]))
            items = [("🤖", "Model", R["used"], f"jarak data {pd.Timedelta(R['step'])}", "hl"),
                     ("🔝", "Puncak ramalan", fmt(R["med"].max(), 0), f"{tgl(R['fidx'][pk_i])} {R['fidx'][pk_i]:%H:%M}"),
                     ("📊", "Rata-rata ramalan", fmt(R["med"].mean(), 0), f"{len(R['med'])} langkah ke depan")]
            if R["actual"] is not None:
                m1, m0 = metrics(R["actual"].values, R["med"]), metrics(R["actual"].values, R["naive"])
                bt = m1["MAE"] < m0["MAE"]
                items += [("🎯", "Meleset (mode uji)", f"{fmt(m1['MAPE'], 2)}%", f"selisih rata-rata {fmt(m1['MAE'], 0)}", "ok" if bt else "hi"),
                          ("📏", "Cara sederhana", f"{fmt(m0['MAPE'], 2)}%",
                           f"model {'lebih tepat' if bt else 'kurang tepat'} {fmt(abs(1 - m1['MAE'] / m0['MAE']) * 100, 0)}%")]
            kpi_row(items)
            with st.container(border=True):
                head("📈 Hasil ramalan", (f"{R['n_gap']} data kosong diisi otomatis · " if R["n_gap"] else "") +
                     "area biru muda = rentang kemungkinan")
                hist = R["s"].iloc[-min(len(R["s"]), 7 * 24):]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["lo"], fill="tonexty", fillcolor="rgba(63,138,224,0.16)",
                                         line=dict(width=0), name="Rentang kemungkinan", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Data lampau", line=dict(color="#8a97ab", width=1.8),
                                         hovertemplate="%{y:,.1f}"))
                if R["actual"] is not None:
                    fig.add_trace(go.Scatter(x=R["actual"].index, y=R["actual"].values, name="Kenyataan (disembunyikan)",
                                             line=dict(color=TEAL, width=2.4), hovertemplate="%{y:,.1f}"))
                    fig.add_trace(go.Scatter(x=R["fidx"], y=R["naive"], name="Cara sederhana", hovertemplate="%{y:,.1f}",
                                             line=dict(color="#93c5fd", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["med"], name="Ramalan", mode="lines+markers", marker=dict(size=5),
                                         line=dict(color=BLUE, width=3), hovertemplate="%{y:,.1f}"))
                fig.add_vline(x=R["s"].index[-1], line=dict(color=MUTED, dash="dash"))
                show(style(fig, 420).update_yaxes(title=str(R["ycol"])), "live")
                out = pd.DataFrame({"waktu": R["fidx"], "ramalan": R["med"], "batas_bawah": R["lo"], "batas_atas": R["hi"]})
                if R["actual"] is not None:
                    out["kenyataan"] = R["actual"].values
                st.download_button("⬇️ Unduh hasil (CSV)", out.round(2).to_csv(index=False).encode(), "ramalan_langsung.csv", "text/csv")
                insight(f"Puncak diperkirakan pada <b>{tgl(R['fidx'][pk_i])} pukul {R['fidx'][pk_i]:%H:%M}</b> sebesar "
                        f"<b>{fmt(R['med'].max(), 0)}</b>.")

st.markdown('<div class="foot">Ramalan diuji pada 400 hari terakhir (24 Mei 2019 – 27 Jun 2020), dibuat ulang setiap tengah malam '
            'untuk 24 jam berikutnya · Model transfer learning dipakai tanpa pelatihan ulang · Dibuat dengan Streamlit & Plotly</div>',
            unsafe_allow_html=True)
