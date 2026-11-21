"""
Dashboard Peramalan Beban Listrik Nasional Panama
- Hasil semua model dari Kaggle (klasik, ML, DL, transfer learning) — evaluasi 70:10:20 rolling-origin
- Ramalan langsung dari data upload memakai foundation model Chronos-Bolt (CPU)

Jalankan:  streamlit run app.py
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard Beban Listrik Panama", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# =============================================================================
# TEMA
# =============================================================================
BLUE, BLUE_D, BLUE_L = "#1f6fd1", "#16509a", "#dbe9fb"
INK, INK2, MUTED, GRID = "#16202e", "#3b4757", "#7a8699", "#e6ecf3"
FAM = {"Baseline": "#9a9993", "Klasik": "#2a78d6", "Machine Learning": "#eb6834",
       "Deep Learning": "#1baf7a", "Transfer Learning": "#eda100", "Hybrid": "#6250d6", "Ensemble": "#e87ba4"}
GRP_COL = {"Riwayat beban (lag)": "#2a78d6", "Kalender (jam/hari/musim)": "#1baf7a", "Libur & sekolah": "#7fd3b4",
           "Suhu": "#eb6834", "Kelembapan": "#f4a37f", "Presipitasi": "#eda100", "Angin": "#c9b26b",
           "Rezim COVID-19": "#9a9993", "Ramalan foundation model": "#6250d6"}
SEQ = [[0, "#eef5fe"], [0.5, "#5598e7"], [1, "#104281"]]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "hasil_kaggle")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1450px}
[data-testid="stHeader"]{background:transparent}
.hero{display:flex;align-items:center;gap:16px;margin-bottom:4px}
.logo{width:54px;height:54px;border-radius:14px;background:linear-gradient(135deg,#1f6fd1,#16509a);
display:flex;align-items:center;justify-content:center;font-size:30px;box-shadow:0 4px 14px rgba(31,111,209,.35)}
.hero h1{font-size:28px;margin:0;color:#16202e;letter-spacing:-.4px;font-weight:800}
.hero p{margin:2px 0 0;color:#5b6778;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:8px 0 14px}
.kpi{background:linear-gradient(135deg,#1f6fd1 0%,#16509a 100%);border-radius:14px;padding:14px 16px;color:#fff;
box-shadow:0 4px 14px rgba(22,80,154,.22);min-height:104px}
.kpi.alt{background:linear-gradient(135deg,#0f8a63 0%,#0b6b4d 100%)}
.kpi.warn{background:linear-gradient(135deg,#d9622b 0%,#b2471a 100%)}
.kl{font-size:12px;opacity:.85;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.kv{font-size:25px;font-weight:800;margin:5px 0 2px;line-height:1.15;overflow-wrap:anywhere}
.kv.sm{font-size:17px}.ks{font-size:12px;opacity:.88}
.ct{font-weight:700;color:#16202e;font-size:15.5px;margin:0 0 1px}
.cs{color:#7a8699;font-size:12.5px;margin:0 0 6px}
.insight{background:#fff;border-left:5px solid #1f6fd1;border-radius:10px;padding:12px 16px;margin:2px 0 12px;
color:#3b4757;font-size:14px;line-height:1.65;box-shadow:0 1px 4px rgba(16,32,46,.06)}
.insight b{color:#16202e}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700}
.b-ok{background:#dcf5ea;color:#0b6b4d}.b-hi{background:#fde6da;color:#a63d10}.b-md{background:#fff3cf;color:#8a6300}
.crit{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #edf1f6}
.crit:last-child{border-bottom:0}.crit .l{color:#5b6778;font-size:13px}.crit .v{font-weight:700;color:#16202e;text-align:right}
.crit .v small{display:block;font-weight:500;color:#7a8699;font-size:11.5px}
.lg{display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:#3b4757}
.lg i{width:10px;height:10px;border-radius:3px;display:inline-block}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:#b9d3f2 !important;border-radius:14px !important;
box-shadow:0 1px 6px rgba(16,32,46,.05)}
[data-testid="stTabs"] button p{font-size:15px;font-weight:600}
.foot{color:#7a8699;font-size:12px;margin-top:14px;text-align:center}
</style>""", unsafe_allow_html=True)


# =============================================================================
# UTILITAS
# =============================================================================
def style(fig, h=340, legend=True, unified=True):
    fig.update_layout(height=h, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=legend,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK2, size=12),
                      margin=dict(l=50, r=16, t=28, b=40), hovermode="x unified" if unified else "closest",
                      legend=dict(orientation="h", y=1.13, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor="#b9d3f2"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#cfd8e3")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#cfd8e3")
    return fig


def show(fig, key):
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def card_title(t, s=""):
    st.markdown(f'<div class="ct">{t}</div>' + (f'<div class="cs">{s}</div>' if s else ""), unsafe_allow_html=True)


def kpi_row(items):
    """items: (label, nilai, keterangan[, varian 'alt'|'warn'])"""
    html = ""
    for it in items:
        a, b, d = it[:3]
        cls = it[3] if len(it) > 3 else ""
        html += (f'<div class="kpi {cls}"><div class="kl">{a}</div>'
                 f'<div class="kv{" sm" if len(str(b)) > 15 else ""}">{b}</div><div class="ks">{d}</div></div>')
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def short(n):
    return "Ensemble (top-3)" if str(n).startswith("Ensemble") else str(n)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan, BIAS=np.nan)
    return dict(MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                MAPE=np.mean(np.abs((y - p) / y)) * 100, BIAS=np.mean(p - y))


def fam_color(n, fam_of):
    return FAM.get(fam_of.get(n, ""), BLUE)


# =============================================================================
# DATA HASIL KAGGLE
# =============================================================================
NEEDED = ["meta.json", "leaderboard_model.csv", "prediksi_model.csv.gz", "data_historis.csv.gz"]


@st.cache_data(show_spinner="Memuat hasil Kaggle…")
def load_bundle(files: dict):
    def csv(name, **kw):
        if name not in files:
            return None
        return pd.read_csv(io.BytesIO(files[name]), compression="gzip" if name.endswith(".gz") else None, **kw)
    meta = json.loads(files["meta.json"])
    pred = csv("prediksi_model.csv.gz", parse_dates=["origin", "datetime"])
    pred["galat"] = pred["aktual"] - pred["prediksi"]
    pred["ape"] = (pred["galat"].abs() / pred["aktual"]) * 100
    pred["kondisi"] = np.where(pred["origin"] >= pd.Timestamp(meta.get("covid_start", "2020-03-15")),
                               "COVID-19", "Normal")
    pred["bulan"] = pred["datetime"].dt.to_period("M").dt.to_timestamp()
    data = csv("data_historis.csv.gz", parse_dates=["datetime"]).set_index("datetime").sort_index()
    return dict(meta=meta, leader=csv("leaderboard_model.csv"), pred=pred, data=data,
                fut=csv("ramalan_24jam_kedepan.csv", parse_dates=["datetime"]),
                feat=csv("pengaruh_fitur.csv"), grp=csv("pengaruh_kelompok.csv"),
                corr=csv("korelasi_eksogen.csv"), shap=csv("shap_sampel.csv.gz"),
                cek=csv("cek_kebocoran.csv"), hfeat=csv("pengaruh_fitur_hybrid.csv"))


def files_from_dir(d):
    return {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}


def files_from_zip(b):
    z = zipfile.ZipFile(io.BytesIO(b))
    return {os.path.basename(n): z.read(n) for n in z.namelist() if not n.endswith("/")}


with st.sidebar:
    st.markdown("### ⚙️ Sumber hasil Kaggle")
    src = st.radio("Pilih sumber", ["Folder bawaan (hasil_kaggle/)", "Upload hasil_kaggle.zip"], label_visibility="collapsed")
    files = None
    if src.startswith("Upload"):
        up = st.file_uploader("hasil_kaggle.zip dari tab Output Kaggle", type=["zip"])
        if up is not None:
            files = files_from_zip(up.getvalue())
    elif os.path.isdir(DEFAULT_DIR):
        files = files_from_dir(DEFAULT_DIR)

B = None
if files is not None:
    miss = [f for f in NEEDED if f not in files]
    if miss:
        st.sidebar.error("File tidak lengkap: " + ", ".join(miss))
    else:
        B = load_bundle(files)

# =============================================================================
# HEADER + FILTER GLOBAL
# =============================================================================
sub = "Peramalan per jam 24 jam ke depan · non-transfer learning vs transfer learning"
if B is not None:
    M = B["meta"]
    sub = (f"Data {pd.Timestamp(M['data_start']):%d %b %Y} – {pd.Timestamp(M['last_ts']):%d %b %Y} · "
           f"{M['n_obs']:,} jam · evaluasi 70:10:20 rolling-origin harian").replace(",", ".")
st.markdown(f'<div class="hero"><div class="logo">⚡</div><div><h1>Dashboard Peramalan Beban Listrik Nasional Panama</h1>'
            f'<p>{sub}</p></div></div>', unsafe_allow_html=True)

if B is not None:
    LB, PRED, DATA = B["leader"], B["pred"], B["data"]
    BEST = M["best"]
    FAM_OF = dict(zip(LB.Model, LB.Keluarga))
    MODELS = list(LB.sort_values("VAL_MAE").Model)
    SUSPECT = set(M.get("suspect") or [])
    TEST = PRED[PRED.split == "test"]
    t_min, t_max = TEST.origin.min().date(), TEST.origin.max().date()

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2.2, 1.6, 2.2, 1.2])
        sel_model = f1.selectbox("🔎 Model", MODELS, index=MODELS.index(BEST),
                                 format_func=lambda n: f"{short(n)}  ·  {FAM_OF[n]}" + ("  ⭐" if n == BEST else "")
                                 + ("  ⚠️ dicurigai bocor" if n in SUSPECT else ""))
        kondisi = f2.radio("Kondisi periode uji", ["Semua", "Normal", "COVID-19"], horizontal=True)
        rng = f3.date_input("📅 Rentang tanggal uji", (t_min, t_max), min_value=t_min, max_value=t_max)
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            rng = (t_min, t_max)
        f4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        f4.download_button("⬇️ Leaderboard", LB.to_csv(index=False).encode(), "leaderboard_model.csv", "text/csv")

    def filt(P):
        P = P[(P.origin.dt.date >= rng[0]) & (P.origin.dt.date <= rng[1])]
        return P if kondisi == "Semua" else P[P.kondisi == kondisi]

    TF = filt(TEST)

    # skor tiap model pada filter aktif
    @st.cache_data(show_spinner=False)
    def score_table(_tf, key):
        rows = [dict(Model=n, Keluarga=FAM_OF.get(n, "-"), **metrics(g.aktual, g.prediksi)) for n, g in _tf.groupby("model")]
        return pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
    SC = score_table(TF, (M.get("generated"), BEST, str(rng), kondisi, len(TF)))

    # ---------- interval prediksi utk model terpilih (galat 90 hari uji terakhir) ----------
    def pi_bounds(model):
        g = TEST[(TEST.model == model) & (TEST.origin >= TEST.origin.max() - pd.Timedelta(days=89))]
        q = g.groupby("horizon").galat.quantile([0.1, 0.9]).unstack()
        return q[0.1].values, q[0.9].values

    FUT = PRED[(PRED.split == "future") & (PRED.model == sel_model)].sort_values("datetime")
    lo_q, hi_q = pi_bounds(sel_model)
    FUT = FUT.assign(lo=FUT.prediksi.values + lo_q, hi=FUT.prediksi.values + hi_q)

    # ---------- KPI ----------
    ms = SC.set_index("Model").loc[sel_model] if sel_model in set(SC.Model) else None
    naive = SC.set_index("Model").MAE.get("Seasonal Naive (168j)", np.nan)
    pk = FUT.loc[FUT.prediksi.idxmax()]
    last = pd.Timestamp(M["last_ts"])
    daily_peaks = DATA.nat_demand.loc[last - pd.Timedelta(days=90):].resample("D").max()
    pct = (daily_peaks < pk.prediksi).mean() * 100
    status = ("Tinggi", "warn") if pct >= 90 else (("Sedang", "") if pct >= 50 else ("Normal", "alt"))
    if ms is not None:
        kpi_row([("Model dipilih", short(sel_model), FAM_OF[sel_model] + (" · terbaik" if sel_model == BEST else "")
                  + (" · ⚠️ dicurigai bocor" if sel_model in SUSPECT else ""), "warn" if sel_model in SUSPECT else ""),
                 ("Akurasi (100 − MAPE)", f"{fmt(100 - ms.MAPE, 2)}%", f"MAPE {fmt(ms.MAPE, 2)}%"),
                 ("MAE", f"{fmt(ms.MAE)} MWh", f"RMSE {fmt(ms.RMSE)} MWh"),
                 ("Lebih baik dari naive", f"{fmt((1 - ms.MAE / naive) * 100)}%" if naive == naive else "–",
                  "pengurangan MAE vs Seasonal Naive"),
                 ("Puncak beban besok", f"{fmt(pk.prediksi, 0)} MWh", f"pukul {pk.datetime:%H:%M} · {pk.datetime:%d %b %Y}"),
                 ("Status beban besok", status[0], f"puncak > {pct:.0f}% puncak harian 90 hari terakhir", status[1])])
    else:
        st.warning("Model terpilih tidak punya data pada filter ini.")


def need_bundle():
    st.info("Hasil Kaggle belum dimuat. Letakkan folder **hasil_kaggle/** di samping app.py atau upload "
            "**hasil_kaggle.zip** lewat sidebar (ikon › di kiri atas). Tab **Ramalan Langsung** tetap bisa dipakai.")


tabs = st.tabs(["🏠 Ringkasan", "🏆 Evaluasi Model", "📉 Aktual vs Prediksi", "🔍 Variabel Berpengaruh",
                "📈 Eksplorasi Data", "🚀 Ramalan Langsung"])

# =============================================================================
# TAB 1 — RINGKASAN
# =============================================================================
with tabs[0]:
    if B is None:
        need_bundle()
    else:
        # ---------- insight otomatis ----------
        tl = SC[SC.Keluarga == "Transfer Learning"]
        ntl = SC[SC.Keluarga.isin(["Klasik", "Machine Learning", "Deep Learning"])]
        bul = [f"Model terbaik hasil seleksi validasi: <b>{short(BEST)}</b>. Pada filter aktif, model paling akurat adalah "
               f"<b>{short(SC.Model.iloc[0])}</b> (MAPE {fmt(SC.MAPE.iloc[0], 2)}%)."]
        if len(tl) and len(ntl):
            a, b = ntl.iloc[0], tl.iloc[0]
            win = "transfer learning" if b.MAPE < a.MAPE else "non-transfer learning"
            bul.append(f"Non-TL terbaik <b>{a.Model}</b> ({fmt(a.MAPE, 2)}%) vs TL terbaik <b>{b.Model}</b> "
                       f"({fmt(b.MAPE, 2)}%) → unggul: <b>{win}</b>.")
        lbn = LB.dropna(subset=["MAPE_NORMAL", "MAPE_COVID"]) if "MAPE_COVID" in LB else pd.DataFrame()
        if len(lbn):
            r = lbn.set_index("Model").loc[BEST] if BEST in set(lbn.Model) else lbn.iloc[0]
            rob = lbn.assign(d=lbn.MAPE_COVID - lbn.MAPE_NORMAL).sort_values("MAPE_COVID").iloc[0]
            bul.append(f"Saat pembatasan COVID-19, galat model terbaik naik dari {fmt(r.MAPE_NORMAL, 2)}% menjadi "
                       f"<b>{fmt(r.MAPE_COVID, 2)}%</b>; model paling tahan perubahan pola: <b>{short(rob.Model)}</b> "
                       f"({fmt(rob.MAPE_COVID, 2)}%).")
        if SUSPECT:
            bul.append(f"⚠️ <b>{', '.join(short(x) for x in SUSPECT)}</b> dicurigai sudah melihat data ini saat pretraining "
                       f"(galat tetap sangat kecil saat awal lockdown). Tetap ditampilkan, tetapi tidak dipilih sebagai model terbaik.")
        hyb = SC[SC.Keluarga == "Hybrid"]
        hb = M.get("hybrid_base")
        if len(hyb) and hb in set(SC.Model):
            h0, b0 = hyb.iloc[0], SC.set_index("Model").loc[hb]
            d = (1 - h0.MAPE / b0.MAPE) * 100
            bul.append(f"Model hybrid <b>{h0.Model}</b> {'menurunkan' if d > 0 else 'menaikkan'} MAPE peramal dasarnya "
                       f"({short(hb)}) dari {fmt(b0.MAPE, 2)}% ke <b>{fmt(h0.MAPE, 2)}%</b> ({fmt(abs(d))}%).")
        if B["grp"] is not None:
            g = B["grp"].sort_values("kontribusi_pct", ascending=False)
            gx = g[g.kelompok != "Riwayat beban (lag)"]
            bul.append(f"Variabel eksternal paling berpengaruh: <b>{gx.kelompok.iloc[0]}</b> "
                       f"({fmt(gx.kontribusi_pct.iloc[0])}%), setelah riwayat beban & kalender.")
        st.markdown('<div class="insight">💡 <b>Insight utama</b><br>' + "<br>".join("• " + x for x in bul) + "</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns([2.3, 1])
        with c1:
            with st.container(border=True):
                card_title("Ramalan 24 jam ke depan", f"{short(sel_model)} · pita = interval 80% dari galat 90 hari uji terakhir")
                hist = DATA.nat_demand.loc[last - pd.Timedelta(hours=71):last]
                lastweek = DATA.nat_demand.reindex(FUT.datetime - pd.Timedelta(days=7))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.lo, fill="tonexty", fillcolor="rgba(31,111,209,0.16)",
                                         line=dict(width=0), name="Interval 80%", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Aktual 72 jam terakhir",
                                         line=dict(color="#5b6778", width=2), hovertemplate="%{y:,.0f} MWh"))
                fig.add_trace(go.Scatter(x=FUT.datetime, y=lastweek.values, name="Minggu lalu (jam sama)",
                                         line=dict(color="#9a9993", width=1.4, dash="dot"), hovertemplate="%{y:,.0f} MWh"))
                fig.add_trace(go.Scatter(x=FUT.datetime, y=FUT.prediksi, name="Ramalan", mode="lines+markers",
                                         marker=dict(size=6), line=dict(color=BLUE, width=3), hovertemplate="%{y:,.0f} MWh"))
                fig.add_vline(x=last, line=dict(color=MUTED, dash="dash"))
                fig.add_annotation(x=pk.datetime, y=pk.prediksi, text=f"Puncak {fmt(pk.prediksi, 0)}", showarrow=True,
                                   arrowhead=2, ay=-34, font=dict(color=INK, size=11), bgcolor="#fff", bordercolor=BLUE)
                show(style(fig, 360).update_yaxes(title="MWh"), "fut")
        with c2:
            with st.container(border=True):
                card_title("Jam kritis besok", "Informasi operasional dari ramalan")
                ramp = np.diff(np.r_[DATA.nat_demand.loc[last], FUT.prediksi.values])
                ri = int(np.argmax(ramp))
                lo_i = FUT.prediksi.idxmin()
                badge = {"Tinggi": "b-hi", "Sedang": "b-md", "Normal": "b-ok"}[status[0]]
                rows = [("Beban puncak", f"{fmt(pk.prediksi, 0)} MWh<small>pukul {pk.datetime:%H:%M}</small>"),
                        ("Beban terendah", f"{fmt(FUT.loc[lo_i, 'prediksi'], 0)} MWh<small>pukul {FUT.loc[lo_i, 'datetime']:%H:%M}</small>"),
                        ("Kenaikan tercepat", f"+{fmt(ramp[ri], 0)} MW/jam<small>menjelang {FUT.datetime.iloc[ri]:%H:%M}</small>"),
                        ("Total energi", f"{fmt(FUT.prediksi.sum() / 1000, 2)} GWh<small>rata-rata {fmt(FUT.prediksi.mean(), 0)} MWh/jam</small>"),
                        ("Load factor", f"{fmt(FUT.prediksi.mean() / pk.prediksi * 100)}%<small>rata-rata ÷ puncak</small>"),
                        ("Status", f'<span class="badge {badge}">{status[0]}</span>')]
                st.markdown("".join(f'<div class="crit"><span class="l">{a}</span><span class="v">{b}</span></div>'
                                    for a, b in rows), unsafe_allow_html=True)
                out = FUT[["datetime", "prediksi", "lo", "hi"]].rename(
                    columns={"prediksi": "ramalan_MWh", "lo": "batas_bawah", "hi": "batas_atas"})
                st.download_button("⬇️ Download ramalan 24 jam", out.round(1).to_csv(index=False).encode(),
                                   "ramalan_24jam.csv", "text/csv")

        c1, c2, c3 = st.columns([1.25, 1, 1.15])
        with c1:
            with st.container(border=True):
                card_title("Peringkat model (MAPE)", "Filter aktif · makin pendek makin akurat")
                T = SC.head(10).iloc[::-1]
                fig = go.Figure(go.Bar(x=T.MAPE, y=[short(n) + (" ⚠️" if n in SUSPECT else "") for n in T.Model], orientation="h",
                                       marker=dict(color=[fam_color(n, FAM_OF) for n in T.Model],
                                                   line=dict(color="#fff", width=1.5),
                                                   pattern=dict(shape=["/" if n in SUSPECT else "" for n in T.Model])),
                                       text=[f"{v:.2f}%" for v in T.MAPE], textposition="outside", cliponaxis=False,
                                       textfont=dict(color=INK2, size=11), customdata=T.Keluarga,
                                       hovertemplate="<b>%{y}</b><br>%{customdata}<br>MAPE %{x:.2f}%<extra></extra>"))
                fig.update_layout(margin=dict(l=150, r=45, t=10, b=30))
                show(style(fig, 330, legend=False, unified=False), "rank")
                st.markdown("".join(f'<span class="lg"><i style="background:{c}"></i>{k}</span>'
                                    for k, c in FAM.items() if k in set(SC.Keluarga)), unsafe_allow_html=True)
        with c2:
            with st.container(border=True):
                card_title("Kontribusi variabel", f"Model {M.get('imp_model')} · per kelompok")
                if B["grp"] is not None:
                    g = B["grp"].sort_values("kontribusi_pct", ascending=False)
                    fig = go.Figure(go.Pie(labels=g.kelompok, values=g.kontribusi_pct, hole=0.58, sort=False,
                                           marker=dict(colors=[GRP_COL.get(k, BLUE) for k in g.kelompok],
                                                       line=dict(color="#fff", width=2)),
                                           textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                                           hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"))
                    fig.update_layout(legend=dict(orientation="v", y=0.5, x=1.0, font=dict(size=10.5)),
                                      annotations=[dict(text="Pengaruh", x=0.5, y=0.5, showarrow=False,
                                                        font=dict(size=13, color=INK2))], margin=dict(l=0, r=0, t=10, b=10))
                    show(style(fig, 330, unified=False).update_layout(margin=dict(l=0, r=0, t=10, b=10)), "donut")
        with c3:
            with st.container(border=True):
                card_title("Kondisi normal vs COVID-19", "MAPE uji per kondisi · 6 model teratas")
                if len(lbn):
                    T = lbn.sort_values("VAL_MAE").head(6)
                    fig = go.Figure([go.Bar(name="Normal", x=[short(n) for n in T.Model], y=T.MAPE_NORMAL,
                                            marker_color="#2a78d6", hovertemplate="%{y:.2f}%"),
                                     go.Bar(name="COVID-19", x=[short(n) for n in T.Model], y=T.MAPE_COVID,
                                            marker_color="#eb6834", hovertemplate="%{y:.2f}%")])
                    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
                    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
                    show(style(fig, 330).update_yaxes(title="MAPE (%)"), "regime")

# =============================================================================
# TAB 2 — EVALUASI MODEL
# =============================================================================
with tabs[1]:
    if B is None:
        need_bundle()
    else:
        c1, c2 = st.columns([3, 1])
        fam_sel = c1.multiselect("Keluarga model", list(FAM), default=[k for k in FAM if k in set(LB.Keluarga)])
        met = c2.selectbox("Metrik", ["MAPE", "MAE", "RMSE"])
        S2 = SC[SC.Keluarga.isin(fam_sel)]
        P2 = TF[TF.keluarga.isin(fam_sel)]
        if S2.empty:
            st.warning("Pilih minimal satu keluarga model.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                with st.container(border=True):
                    card_title(f"Perbandingan {met} semua model", "Periode uji sesuai filter atas")
                    T = S2.sort_values(met).iloc[::-1]
                    unit = "%" if met == "MAPE" else ""
                    fig = go.Figure(go.Bar(x=T[met], y=[short(n) for n in T.Model], orientation="h",
                                           marker=dict(color=[fam_color(n, FAM_OF) for n in T.Model],
                                                       line=dict(color="#fff", width=1.5)),
                                           text=[f"{v:.2f}{unit}" for v in T[met]], textposition="outside",
                                           cliponaxis=False, textfont=dict(size=11, color=INK2),
                                           hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>"))
                    fig.update_layout(margin=dict(l=160, r=50, t=10, b=35))
                    show(style(fig, max(320, 34 * len(T)), legend=False, unified=False).update_xaxes(title=met), "cmp")
            with c2:
                with st.container(border=True):
                    card_title("Galat menurut horizon", "MAE jam ke-1 s.d. ke-24 · 6 model teratas")
                    fig = go.Figure()
                    for n in S2.Model[:6]:
                        g = P2[P2.model == n].groupby("horizon").galat.apply(lambda e: e.abs().mean())
                        fig.add_trace(go.Scatter(x=g.index, y=g.values, name=short(n), mode="lines",
                                                 line=dict(color=fam_color(n, FAM_OF), width=3 if n == sel_model else 1.6,
                                                           dash=None if n == sel_model else "dot"),
                                                 hovertemplate="%{y:.1f} MWh"))
                    fig.update_layout(legend=dict(y=1.22, font=dict(size=10)))
                    show(style(fig, max(320, 34 * len(S2))).update_xaxes(title="Jam ke-", dtick=2)
                         .update_yaxes(title="MAE (MWh)"), "hz")

            with st.container(border=True):
                card_title("Peta galat bulanan (MAPE %)", "Bulan mana yang sulit diramal? Warna makin gelap = galat makin besar")
                H = P2.groupby(["model", "bulan"]).ape.mean().unstack()
                H = H.loc[[m for m in S2.Model if m in H.index]]
                fig = go.Figure(go.Heatmap(z=H.values, x=[d.strftime("%b %y") for d in H.columns],
                                           y=[short(n) for n in H.index], colorscale=SEQ, xgap=2, ygap=2,
                                           colorbar=dict(title="MAPE %", thickness=10),
                                           text=np.round(H.values, 1), texttemplate="%{text}", textfont=dict(size=10),
                                           hovertemplate="%{y}<br>%{x}<br>MAPE %{z:.2f}%<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(margin=dict(l=160, r=20, t=10, b=35))
                show(style(fig, max(280, 30 * len(H) + 80), legend=False, unified=False), "heatm")

            c1, c2 = st.columns([1.2, 1])
            with c1:
                with st.container(border=True):
                    card_title("Cek kebocoran data pretraining (foundation model)",
                               "Model jujur memburuk di hari libur & awal lockdown (12–28 Mar 2020) karena tak punya info itu")
                    CK = B["cek"]
                    if CK is None or CK.empty:
                        st.caption("File cek_kebocoran.csv tidak ada pada hasil ini.")
                    else:
                        CK = CK.sort_values("MAPE_hari_biasa")
                        fig = go.Figure([go.Bar(name=lab, x=[short(n) + (" ⚠️" if n in SUSPECT else "") for n in CK.Model],
                                                y=CK[col], marker_color=c, hovertemplate="%{y:.2f}%")
                                         for lab, col, c in [("Hari biasa", "MAPE_hari_biasa", "#2a78d6"),
                                                             ("Hari libur", "MAPE_hari_libur", "#1baf7a"),
                                                             ("Awal lockdown", "MAPE_awal_lockdown", "#eb6834")]])
                        fig.update_layout(barmode="group", bargap=0.22)
                        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
                        show(style(fig, 340).update_yaxes(title="MAPE (%)"), "cek")
                        st.caption("⚠️ = dicurigai bocor: galat awal lockdown < ½ median model lain, tidak memburuk saat pola "
                                   "berubah, dan galat hari biasa < ½ median model lain.")
            with c2:
                with st.container(border=True):
                    hb = M.get("hybrid_base")
                    card_title("Hybrid vs peramal dasarnya", f"Peramal dasar: {short(hb) if hb else '–'} · MAPE uji per kondisi")
                    names = ([hb] if hb in set(LB.Model) else []) + list(LB[LB.Keluarga == "Hybrid"].Model)
                    if len(names) < 2:
                        st.caption("Model hybrid tidak ada pada hasil ini.")
                    else:
                        T = LB.set_index("Model").loc[names]
                        fig = go.Figure([go.Bar(name=k, x=[short(n) for n in T.index], y=T[c], marker_color=col,
                                                text=[f"{v:.2f}" for v in T[c]], textposition="outside", cliponaxis=False,
                                                hovertemplate="%{y:.2f}%")
                                         for k, c, col in [("Total uji", "TEST_MAPE", "#6250d6"),
                                                           ("Normal", "MAPE_NORMAL", "#2a78d6"),
                                                           ("COVID-19", "MAPE_COVID", "#eb6834")] if c in T])
                        fig.update_layout(barmode="group", bargap=0.25)
                        show(style(fig, 340).update_yaxes(title="MAPE (%)"), "hybcmp")

            with st.container(border=True):
                card_title("Tabel evaluasi lengkap", "Kolom VAL dari validasi (dasar pemilihan model); TEST dari periode uji penuh")
                tb = LB[LB.Keluarga.isin(fam_sel)].copy()
                tb.insert(0, "Peringkat", tb.VAL_MAE.rank().astype(int))
                tb["Model"] = tb.Model.map(short)
                tb = tb.sort_values("Peringkat")
                st.dataframe(tb, hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.2f") for c in tb.columns
                    if tb[c].dtype.kind == "f"})

# =============================================================================
# TAB 3 — AKTUAL VS PREDIKSI
# =============================================================================
with tabs[2]:
    if B is None:
        need_bundle()
    else:
        c1, c2 = st.columns([2, 2])
        msel = c1.multiselect("Model dibandingkan", MODELS, default=[sel_model], format_func=short, key="avp_m")
        days = sorted(TF.origin.dt.date.unique())
        if len(days) == 0:
            st.warning("Tidak ada data pada filter ini.")
        else:
            d_def = (days[max(0, len(days) - 14)], days[-1])
            dr = c2.select_slider("Zoom tanggal", options=days, value=d_def, format_func=lambda d: d.strftime("%d %b %y"))
            W = TF[(TF.origin.dt.date >= dr[0]) & (TF.origin.dt.date <= dr[1])]
            with st.container(border=True):
                card_title("Aktual vs prediksi per jam", f"{dr[0]:%d %b %Y} – {dr[1]:%d %b %Y}")
                act = W.drop_duplicates("datetime").sort_values("datetime")
                fig = go.Figure(go.Scatter(x=act.datetime, y=act.aktual, name="Aktual",
                                           line=dict(color=INK, width=2), hovertemplate="%{y:,.0f}"))
                for n in msel:
                    d = W[W.model == n].sort_values("datetime")
                    fig.add_trace(go.Scatter(x=d.datetime, y=d.prediksi, name=short(n), hovertemplate="%{y:,.0f}",
                                             line=dict(color=BLUE if n == sel_model else fam_color(n, FAM_OF), width=1.8)))
                show(style(fig, 360).update_yaxes(title="MWh"), "avp")

            G = TF[TF.model == sel_model]
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    card_title("Prediksi vs aktual", f"{short(sel_model)} · titik di garis = tepat")
                    smp = G.sample(min(3000, len(G)), random_state=1)
                    lo_, hi_ = float(np.nanmin(smp.aktual)), float(np.nanmax(smp.aktual))
                    fig = go.Figure([go.Scattergl(x=smp.aktual, y=smp.prediksi, mode="markers",
                                                  marker=dict(size=4, color=BLUE, opacity=0.35), name="jam",
                                                  hovertemplate="aktual %{x:,.0f}<br>prediksi %{y:,.0f}<extra></extra>"),
                                     go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", name="ideal",
                                                line=dict(color="#eb6834", dash="dash", width=1.5), hoverinfo="skip")])
                    show(style(fig, 310, legend=False, unified=False).update_xaxes(title="Aktual (MWh)")
                         .update_yaxes(title="Prediksi (MWh)"), "sc")
            with c2:
                with st.container(border=True):
                    card_title("Sebaran galat", "Aktual − prediksi · > 0 berarti ramalan terlalu rendah")
                    fig = go.Figure(go.Histogram(x=G.galat, nbinsx=60, marker=dict(color=BLUE, line=dict(color="#fff", width=1)),
                                                 hovertemplate="galat %{x}<br>%{y} jam<extra></extra>"))
                    fig.add_vline(x=0, line=dict(color="#eb6834", dash="dash"))
                    show(style(fig, 310, legend=False, unified=False).update_xaxes(title="MWh").update_yaxes(title="jumlah jam"), "hist")
            with c3:
                with st.container(border=True):
                    card_title("MAPE per hari dalam seminggu", f"{short(sel_model)}")
                    dw = G.assign(d=G.datetime.dt.dayofweek).groupby("d").ape.mean().reindex(range(7))
                    fig = go.Figure(go.Bar(x=HARI, y=dw.values, marker=dict(color=[BLUE if i < 5 else "#eb6834" for i in range(7)],
                                                                            line=dict(color="#fff", width=1.5)),
                                           text=[f"{v:.1f}%" for v in dw.values], textposition="outside", cliponaxis=False,
                                           hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
                    show(style(fig, 310, legend=False, unified=False).update_yaxes(title="MAPE (%)"), "dow")

            with st.container(border=True):
                card_title("Hari dengan galat terbesar", f"{short(sel_model)} · cek apakah bertepatan dengan hari libur/kejadian khusus")
                day = G.groupby("origin").agg(MAPE=("ape", "mean"), MAE=("galat", lambda e: e.abs().mean()),
                                              aktual_rata2=("aktual", "mean"), prediksi_rata2=("prediksi", "mean")).reset_index()
                day["tanggal"] = (day.origin + pd.Timedelta(days=1)).dt.date
                day["hari"] = [HARI[pd.Timestamp(d).dayofweek] for d in day.tanggal]
                hol = DATA.holiday.resample("D").max()
                day["libur"] = ["Ya" if hol.get(pd.Timestamp(d), 0) == 1 else "" for d in day.tanggal]
                worst = day.sort_values("MAPE", ascending=False).head(8)[["tanggal", "hari", "libur", "MAPE", "MAE",
                                                                         "aktual_rata2", "prediksi_rata2"]]
                cA, cB = st.columns([1.3, 1])
                with cA:
                    st.dataframe(worst, hide_index=True, column_config={
                        "MAPE": st.column_config.ProgressColumn("MAPE (%)", format="%.2f", min_value=0,
                                                                max_value=float(max(worst.MAPE.max(), 1))),
                        "MAE": st.column_config.NumberColumn(format="%.1f"),
                        "aktual_rata2": st.column_config.NumberColumn("aktual rata²", format="%.0f"),
                        "prediksi_rata2": st.column_config.NumberColumn("prediksi rata²", format="%.0f")})
                with cB:
                    fig = go.Figure(go.Scatter(x=day.origin + pd.Timedelta(days=1), y=day.MAPE, mode="lines",
                                               line=dict(color=BLUE, width=1.4), fill="tozeroy",
                                               fillcolor="rgba(31,111,209,0.10)", hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>"))
                    show(style(fig, 290, legend=False).update_yaxes(title="MAPE harian (%)"), "dailymape")

# =============================================================================
# TAB 4 — VARIABEL BERPENGARUH
# =============================================================================
with tabs[3]:
    if B is None:
        need_bundle()
    elif B["feat"] is None:
        st.warning("File pengaruh_fitur.csv tidak ada.")
    else:
        FT, GR = B["feat"], B["grp"]
        meth = M.get("imp_method", "importance")
        st.markdown(f'<div class="insight">🔍 Pengaruh dihitung pada model <b>{M.get("imp_model")}</b> di periode uji '
                    f'(metode: {meth}). <b>Riwayat beban</b> wajar dominan karena beban sangat berulang; untuk melihat faktor '
                    f'eksternal, aktifkan <b>"hanya variabel eksternal"</b>.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.25])
        with c1:
            with st.container(border=True):
                card_title("Kenaikan galat bila kelompok variabel dihilangkan", "Permutation importance (MWh) · makin panjang makin penting")
                G2 = GR.sort_values("kenaikan_mae_perm")
                fig = go.Figure(go.Bar(x=G2.kenaikan_mae_perm, y=G2.kelompok, orientation="h",
                                       marker=dict(color=[GRP_COL.get(k, BLUE) for k in G2.kelompok],
                                                   line=dict(color="#fff", width=1.5)),
                                       text=[f"+{v:.1f}" for v in G2.kenaikan_mae_perm], textposition="outside",
                                       cliponaxis=False, hovertemplate="%{y}: +%{x:.2f} MWh<extra></extra>"))
                fig.update_layout(margin=dict(l=175, r=45, t=10, b=35))
                show(style(fig, 360, legend=False, unified=False).update_xaxes(title="MWh"), "perm")
        with c2:
            with st.container(border=True):
                cc1, cc2 = st.columns([1, 1])
                only_ex = cc1.toggle("Hanya variabel eksternal", value=True)
                n_top = cc2.slider("Jumlah fitur", 5, min(30, len(FT)), 12)
                F2 = FT[FT.kelompok != "Riwayat beban (lag)"] if only_ex else FT
                F2 = F2.sort_values("nilai", ascending=False).head(n_top).iloc[::-1]
                card_title("Fitur paling berpengaruh", meth)
                fig = go.Figure(go.Bar(x=F2.nilai, y=F2.fitur, orientation="h",
                                       marker=dict(color=[GRP_COL.get(k, BLUE) for k in F2.kelompok],
                                                   line=dict(color="#fff", width=1.5)),
                                       customdata=F2.kelompok, hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.2f}<extra></extra>"))
                fig.update_layout(margin=dict(l=120, r=20, t=10, b=35))
                show(style(fig, max(290, 24 * len(F2)), legend=False, unified=False), "feat")
                st.markdown("".join(f'<span class="lg"><i style="background:{GRP_COL[k]}"></i>{k}</span>'
                                    for k in GRP_COL if k in set(F2.kelompok)), unsafe_allow_html=True)

        if B["hfeat"] is not None:
            with st.container(border=True):
                HF = B["hfeat"].head(15).iloc[::-1]
                card_title(f"Variabel yang mengoreksi foundation model (model hybrid · dasar {short(M.get('hybrid_base'))})",
                           "Apa yang belum diketahui foundation model dan dilengkapi oleh model buatan sendiri? (fm = ramalan foundation model)")
                fig = go.Figure(go.Bar(x=HF.nilai, y=HF.fitur, orientation="h",
                                       marker=dict(color=[GRP_COL.get(k, BLUE) for k in HF.kelompok], line=dict(color="#fff", width=1.5)),
                                       customdata=HF.kelompok, hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.2f}<extra></extra>"))
                fig.update_layout(margin=dict(l=130, r=20, t=10, b=35))
                show(style(fig, 360, legend=False, unified=False), "hfeat")

        c1, c2 = st.columns([1.25, 1])
        with c1:
            with st.container(border=True):
                SH = B["shap"]
                if SH is not None:
                    feats = [c[2:] for c in SH.columns if c.startswith("x_")]
                    fsel = st.selectbox("Efek variabel", feats, index=next((i for i, f in enumerate(feats) if f.startswith("T2M")), 0))
                    card_title(f"Efek {fsel} terhadap ramalan", "Nilai SHAP: + menaikkan, − menurunkan ramalan beban (MWh)")
                    fig = go.Figure(go.Scattergl(x=SH[f"x_{fsel}"], y=SH[f"s_{fsel}"], mode="markers",
                                                 marker=dict(size=5, color=SH[f"s_{fsel}"], colorscale="RdBu_r", opacity=0.6),
                                                 hovertemplate="nilai %{x:.2f}<br>efek %{y:+.1f} MWh<extra></extra>"))
                    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
                    show(style(fig, 330, legend=False, unified=False).update_xaxes(title=fsel).update_yaxes(title="SHAP (MWh)"), "shapdep")
                else:
                    city = st.selectbox("Kota", {"toc": "Tocumen (Kota Panama)", "san": "Santiago", "dav": "David"}.items(),
                                        format_func=lambda kv: kv[1], key="effcity")[0]
                    card_title("Efek suhu terhadap beban", "Rata-rata beban per kelompok suhu, dipisah siang (08–18) dan malam")
                    X = DATA.assign(tb=(DATA[f"T2M_{city}"] * 2).round() / 2,
                                    w=np.where(DATA.index.hour.isin(range(8, 19)), "Siang", "Malam"))
                    E = X.groupby(["w", "tb"]).nat_demand.agg(["mean", "count"]).reset_index()
                    E = E[E["count"] >= 30]
                    fig = go.Figure([go.Scatter(x=E[E.w == w].tb, y=E[E.w == w]["mean"], name=w, mode="lines+markers",
                                                line=dict(color=c, width=2.4), marker=dict(size=6), hovertemplate="%{y:,.0f} MWh")
                                     for w, c in [("Siang", "#eb6834"), ("Malam", "#2a78d6")]])
                    show(style(fig, 330).update_xaxes(title="Suhu 2 m (°C)").update_yaxes(title="Rata-rata beban (MWh)"), "effT")
        with c2:
            with st.container(border=True):
                card_title("Korelasi variabel eksogen dengan beban", "Pearson (linier) & Spearman (monoton)")
                if B["corr"] is not None:
                    C = B["corr"].sort_values("Pearson", key=abs)
                    fig = go.Figure([go.Bar(x=C.Pearson, y=C.variabel, orientation="h", name="Pearson", marker_color="#2a78d6"),
                                     go.Bar(x=C.Spearman, y=C.variabel, orientation="h", name="Spearman", marker_color="#eb6834")])
                    fig.update_layout(barmode="group", margin=dict(l=95, r=15, t=30, b=35))
                    show(style(fig, 380, unified=False).update_xaxes(range=[-1, 1], title="koefisien"), "corr")

# =============================================================================
# TAB 5 — EKSPLORASI DATA
# =============================================================================
with tabs[4]:
    if B is None:
        need_bundle()
    else:
        d0, d1 = DATA.index.min().date(), DATA.index.max().date()
        c1, c2 = st.columns([2, 1])
        rr = c1.date_input("Rentang tanggal data", (d0, d1), min_value=d0, max_value=d1, key="eda_rng")
        X = DATA.loc[str(rr[0]):str(rr[1])] if isinstance(rr, (tuple, list)) and len(rr) == 2 else DATA
        y = X.nat_demand
        kpi_row([("Rata-rata beban", f"{fmt(y.mean(), 0)} MWh", "per jam"),
                 ("Beban tertinggi", f"{fmt(y.max(), 0)} MWh", f"{y.idxmax():%d %b %Y %H:%M}"),
                 ("Beban terendah", f"{fmt(y.min(), 0)} MWh", f"{y.idxmin():%d %b %Y %H:%M}"),
                 ("Total energi", f"{fmt(y.sum() / 1e6, 2)} TWh", f"{len(y):,} jam".replace(",", "."))])
        with st.container(border=True):
            card_title("Tren beban harian", "Garis tipis = rata-rata harian · garis tebal = rata-rata bergerak 30 hari · area merah = pembatasan COVID-19")
            dly = y.resample("D").mean()
            fig = go.Figure([go.Scatter(x=dly.index, y=dly.values, name="harian", line=dict(color="#9cc0ec", width=1),
                                        hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra></extra>"),
                             go.Scatter(x=dly.index, y=dly.rolling(30, min_periods=7).mean(), name="MA 30 hari",
                                        line=dict(color=BLUE_D, width=2.4), hovertemplate="%{y:,.0f}<extra></extra>")])
            cs = pd.Timestamp(M.get("covid_start", "2020-03-15"))
            if dly.index.max() >= cs:
                fig.add_vrect(x0=max(cs, dly.index.min()), x1=dly.index.max(), fillcolor="#eb6834", opacity=0.10, line_width=0)
            for sp in M.get("split", [])[1:]:
                fig.add_vline(x=pd.Timestamp(sp["mulai"]), line=dict(color=MUTED, dash="dot"))
                fig.add_annotation(x=pd.Timestamp(sp["mulai"]), y=1, yref="paper", text=f"mulai {sp['bagian'].lower()}",
                                   showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, color=MUTED))
            show(style(fig, 330).update_yaxes(title="MWh"), "trend")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                card_title("Pola musiman tahunan", "Rata-rata beban per bulan, tiap garis = satu tahun")
                mo = X.assign(th=X.index.year, bl=X.index.month).groupby(["th", "bl"]).nat_demand.mean().unstack(0)
                cols = ["#c7dcf6", "#9cc0ec", "#6aa1e2", "#2a78d6", "#16509a", "#eb6834"]
                fig = go.Figure([go.Scatter(x=["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"],
                                            y=mo[yr].reindex(range(1, 13)).values, name=str(yr), mode="lines+markers",
                                            line=dict(color=cols[i % len(cols)], width=2.2 if yr == mo.columns.max() else 1.6),
                                            marker=dict(size=5), hovertemplate="%{y:,.0f} MWh")
                                 for i, yr in enumerate(mo.columns)])
                show(style(fig, 330).update_yaxes(title="MWh"), "season")
        with c2:
            with st.container(border=True):
                card_title("Profil beban: hari × jam", "Kapan beban paling tinggi dalam seminggu?")
                pv = X.assign(h=X.index.hour, d=X.index.dayofweek).pivot_table(index="d", columns="h", values="nat_demand")
                fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns], y=[HARI[i][:3] for i in pv.index],
                                           colorscale=SEQ, xgap=2, ygap=2, colorbar=dict(title="MWh", thickness=10),
                                           hovertemplate="%{y} jam %{x}:00<br>%{z:,.0f} MWh<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                show(style(fig, 330, legend=False, unified=False).update_xaxes(title="Jam"), "heat")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                card_title("Pola 24 jam: hari kerja vs akhir pekan vs libur")
                prof = X.assign(h=X.index.hour, tipe=np.where(X.holiday == 1, "Hari libur",
                                                              np.where(X.index.dayofweek >= 5, "Akhir pekan", "Hari kerja")))
                prof = prof.groupby(["tipe", "h"]).nat_demand.mean().unstack(0)
                fig = go.Figure([go.Scatter(x=prof.index, y=prof[c], name=c, line=dict(color=col, width=2.4),
                                            hovertemplate="%{y:,.0f} MWh") for c, col in
                                 [("Hari kerja", "#2a78d6"), ("Akhir pekan", "#eb6834"), ("Hari libur", "#1baf7a")] if c in prof])
                show(style(fig, 320).update_xaxes(title="Jam", dtick=2).update_yaxes(title="MWh"), "prof")
        with c2:
            with st.container(border=True):
                city = st.selectbox("Kota", {"toc": "Tocumen (Kota Panama)", "san": "Santiago", "dav": "David"}.items(),
                                    format_func=lambda kv: kv[1], key="edacity")[0]
                card_title("Suhu vs beban", "4.000 sampel acak · warna = jam")
                smp = X.sample(min(4000, len(X)), random_state=1)
                fig = go.Figure(go.Scattergl(x=smp[f"T2M_{city}"], y=smp.nat_demand, mode="markers",
                                             marker=dict(size=5, color=smp.index.hour, colorscale=SEQ, opacity=0.65,
                                                         colorbar=dict(title="Jam", thickness=10)),
                                             hovertemplate="%{x:.1f}°C<br>%{y:,.0f} MWh<extra></extra>"))
                show(style(fig, 290, legend=False, unified=False).update_xaxes(title="Suhu (°C)").update_yaxes(title="MWh"), "temp")

        with st.container(border=True):
            card_title("Pembagian data (70 : 10 : 20)", "Train untuk melatih, validasi untuk memilih model, test untuk evaluasi akhir")
            if M.get("split"):
                spd = pd.DataFrame(M["split"])
                fig = go.Figure([go.Bar(y=["Data"], x=[r.hari], name=f"{r.bagian} · {r.hari} hari ({r.persen:.0f}%)",
                                        orientation="h", marker_color=c, text=f"{r.bagian} {r.persen:.0f}%",
                                        textposition="inside", insidetextanchor="middle", textfont=dict(color="#fff", size=13),
                                        hovertemplate=f"{r.bagian}<br>{pd.Timestamp(r.mulai):%d %b %Y} – "
                                                      f"{pd.Timestamp(r.selesai):%d %b %Y}<br>{r.hari} hari<extra></extra>")
                                 for r, c in zip(spd.itertuples(), ["#2a78d6", "#1baf7a", "#eb6834"])])
                fig.update_layout(barmode="stack", margin=dict(l=10, r=10, t=30, b=10))
                fig.update_yaxes(visible=False); fig.update_xaxes(visible=False)
                show(style(fig, 140, unified=False), "split")

# =============================================================================
# TAB 6 — RAMALAN LANGSUNG
# =============================================================================
CHRONOS = {"Chronos-Bolt Tiny (9M, tercepat)": "amazon/chronos-bolt-tiny",
           "Chronos-Bolt Mini (21M)": "amazon/chronos-bolt-mini",
           "Chronos-Bolt Small (48M, disarankan)": "amazon/chronos-bolt-small",
           "Chronos-Bolt Base (205M, paling akurat)": "amazon/chronos-bolt-base"}


@st.cache_resource(show_spinner="Memuat foundation model (sekali saja)…")
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


with tabs[5]:
    st.markdown('<div class="insight">🚀 <b>Ramalan langsung</b>: upload data beban terbaru (CSV/XLSX berisi kolom waktu & beban). '
                'Foundation model <b>Chronos-Bolt</b> (Amazon) meramal secara zero-shot di CPU, tanpa pelatihan ulang. '
                'Aktifkan <b>mode uji</b> untuk mengukur akurasinya pada data Anda sendiri.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2.5], gap="medium")
    with left:
        with st.container(border=True):
            card_title("Pengaturan")
            opts = ["Upload file"] + (["Data historis bawaan"] if B is not None else [])
            how = st.radio("Sumber data", opts, horizontal=True)
            raw_df = None
            if how == "Upload file":
                up = st.file_uploader("File CSV / XLSX", type=["csv", "xlsx", "xls"], key="live_up")
                if up is not None:
                    try:
                        raw_df = read_any(up)
                    except Exception as e:
                        st.error(f"Gagal membaca file: {e}")
            else:
                raw_df = DATA.reset_index()
            tcol = ycol = None
            if raw_df is not None:
                cols = list(raw_df.columns)
                t_guess = next((c for c in cols if any(k in str(c).lower() for k in ("date", "time", "tanggal", "waktu"))), cols[0])
                num = [c for c in cols if pd.api.types.is_numeric_dtype(raw_df[c])]
                y_guess = "nat_demand" if "nat_demand" in cols else (num[0] if num else cols[-1])
                tcol = st.selectbox("Kolom waktu", cols, index=cols.index(t_guess))
                ycol = st.selectbox("Kolom beban", cols, index=cols.index(y_guess))
            mlabel = st.selectbox("Model", list(CHRONOS), index=2)
            h = int(st.number_input("Horizon (langkah)", 1, 168, 24, help="Chronos-Bolt optimal ≤ 64 langkah"))
            ctx_len = st.select_slider("Panjang konteks", [168, 336, 512, 1024, 2048], value=2048)
            backtest = st.toggle("Mode uji (sembunyikan data terakhir)", value=False)
            run = st.button("🚀 Jalankan ramalan", type="primary", disabled=raw_df is None)

    with right:
        if run and raw_df is not None:
            try:
                s, step, n_gap = prep_series(raw_df, tcol, ycol)
                if len(s) < 2 * h + 24:
                    raise ValueError("data terlalu pendek untuk horizon ini")
                actual, s_in = (s.iloc[-h:], s.iloc[:-h]) if backtest else (None, s)
                ctx = s_in.values[-ctx_len:].astype(np.float32)
                fidx = pd.date_range(s_in.index[-1] + step, periods=h, freq=step)
                naive = seasonal_naive(s_in, h, step)
                try:
                    with st.spinner(f"Meramal dengan {mlabel}…"):
                        med, lo, hi = chronos_forecast(CHRONOS[mlabel], ctx, h)
                    used = mlabel.split(" (")[0]
                except Exception as e:
                    st.warning(f"Chronos tidak tersedia ({type(e).__name__}). Memakai Seasonal Naive sebagai cadangan.")
                    sd = float(np.nanstd(s_in.diff(int(pd.Timedelta("1D") / step) or 1).values[-ctx_len:]))
                    med, lo, hi, used = naive, naive - 1.28 * sd, naive + 1.28 * sd, "Seasonal Naive"
                st.session_state["live"] = dict(s=s_in, fidx=fidx, med=med, lo=lo, hi=hi, used=used, actual=actual,
                                                naive=naive, n_gap=n_gap, step=step, ycol=ycol)
            except Exception as e:
                st.error(f"Data tidak bisa diproses: {e}")

        R = st.session_state.get("live")
        if R is None:
            with st.container(border=True):
                card_title("Belum ada ramalan", "Pilih data lalu klik Jalankan ramalan. Contoh format:")
                st.dataframe(pd.DataFrame({"datetime": ["2020-06-26 01:00", "2020-06-26 02:00", "…"],
                                           "nat_demand": [1012.5, 978.1, None]}), hide_index=True)
        else:
            pk_i = int(np.argmax(R["med"]))
            items = [("Model", R["used"], f"interval data {pd.Timedelta(R['step'])}"),
                     ("Puncak ramalan", fmt(R["med"].max(), 0), f"{R['fidx'][pk_i]:%d %b %Y %H:%M}"),
                     ("Rata-rata ramalan", fmt(R["med"].mean(), 0), f"{len(R['med'])} langkah ke depan")]
            if R["actual"] is not None:
                m1, m0 = metrics(R["actual"].values, R["med"]), metrics(R["actual"].values, R["naive"])
                better = m1["MAE"] < m0["MAE"]
                items += [("MAPE mode uji", f"{fmt(m1['MAPE'], 2)}%", f"MAE {fmt(m1['MAE'])}", "alt" if better else "warn"),
                          ("vs Seasonal Naive", f"{fmt(m0['MAPE'], 2)}%",
                           f"model {'lebih baik' if better else 'lebih buruk'} {fmt(abs(1 - m1['MAE'] / m0['MAE']) * 100)}%")]
            kpi_row(items)
            with st.container(border=True):
                card_title("Hasil ramalan", (f"{R['n_gap']} titik kosong diisi interpolasi · " if R["n_gap"] else "") +
                           "pita = kuantil 10%–90%")
                hist = R["s"].iloc[-min(len(R["s"]), 7 * 24):]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["lo"], fill="tonexty", fillcolor="rgba(31,111,209,0.16)",
                                         line=dict(width=0), name="Interval 80%", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historis", line=dict(color="#5b6778", width=1.6),
                                         hovertemplate="%{y:,.1f}"))
                if R["actual"] is not None:
                    fig.add_trace(go.Scatter(x=R["actual"].index, y=R["actual"].values, name="Aktual (disembunyikan)",
                                             line=dict(color="#1baf7a", width=2.2), hovertemplate="%{y:,.1f}"))
                    fig.add_trace(go.Scatter(x=R["fidx"], y=R["naive"], name="Seasonal Naive",
                                             line=dict(color="#9a9993", width=1.3, dash="dot"), hovertemplate="%{y:,.1f}"))
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["med"], name=f"Ramalan · {R['used']}", mode="lines+markers",
                                         marker=dict(size=5), line=dict(color=BLUE, width=3), hovertemplate="%{y:,.1f}"))
                fig.add_vline(x=R["s"].index[-1], line=dict(color=MUTED, dash="dash"))
                show(style(fig, 420).update_yaxes(title=str(R["ycol"])), "live")
                out = pd.DataFrame({"datetime": R["fidx"], "ramalan": R["med"], "q10": R["lo"], "q90": R["hi"]})
                if R["actual"] is not None:
                    out["aktual"] = R["actual"].values
                st.download_button("⬇️ Download hasil (CSV)", out.round(2).to_csv(index=False).encode(),
                                   "ramalan_langsung.csv", "text/csv")

st.markdown('<div class="foot">Evaluasi: pembagian kronologis 70% train · 10% validasi · 20% test dengan rolling-origin harian '
            '(meramal 24 jam setiap pukul 00:00) · model transfer learning dipakai zero-shot · Streamlit + Plotly</div>',
            unsafe_allow_html=True)
