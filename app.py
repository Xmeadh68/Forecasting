"""
Dashboard Peramalan Beban Listrik Nasional Panama
- Menampilkan hasil semua model yang dijalankan di Kaggle (klasik, ML, DL, transfer learning)
- Ramalan langsung dari data yang di-upload memakai foundation model Chronos-Bolt (CPU)

Jalankan lokal:  streamlit run app.py
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Peramalan Beban Listrik Panama", page_icon="⚡", layout="wide")

# -----------------------------------------------------------------------------
# Tema & gaya
# -----------------------------------------------------------------------------
CARD, GRID, INK, INK2, MUTED, ACC = "#1a1a19", "#2c2c2a", "#f0efec", "#c3c2b7", "#8b8a83", "#3987e5"
FAM = {"Baseline": "#8b8a83", "Klasik": "#3987e5", "Machine Learning": "#d95926",
       "Deep Learning": "#199e70", "Transfer Learning": "#c98500", "Ensemble": "#d55181"}
GRP_COL = {"Riwayat beban (lag)": "#3987e5", "Suhu": "#d95926", "Kelembapan": "#d95926",
           "Presipitasi": "#d95926", "Angin": "#d95926"}            # lainnya -> hijau (kalender/libur)
BLUES = [[0, "#104281"], [0.5, "#3987e5"], [1, "#cde2fb"]]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "hasil_kaggle")

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1400px}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 10% -10%,#1b2a44 0%,#0f1115 55%)}
[data-testid="stHeader"]{background:transparent}
h1,h2,h3{letter-spacing:-.3px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:6px 0 16px}
.kpi{background:#1a1a19;border:1px solid #2c2c2a;border-radius:14px;padding:14px 16px;position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:#3987e5}
.kl{color:#8b8a83;font-size:12px;text-transform:uppercase;letter-spacing:.6px}
.kv{font-size:23px;font-weight:700;margin:6px 0 3px;color:#f0efec;overflow-wrap:anywhere}
.kv.sm{font-size:17px}.ks{color:#c3c2b7;font-size:12px}
.insight{background:linear-gradient(135deg,#1c2b45,#1a1a19);border:1px solid #2a4f80;border-radius:14px;
padding:14px 18px;font-size:14px;line-height:1.6;color:#c3c2b7;margin-bottom:14px}.insight b{color:#f0efec}
.sub{color:#8b8a83;font-size:13px;margin:-6px 0 6px}
.lg{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:12px;color:#c3c2b7}
.lg i{width:10px;height:10px;border-radius:3px;display:inline-block}
[data-testid="stTabs"] button p{font-size:15px}
</style>""", unsafe_allow_html=True)


def style(fig, h=360, legend=True, unified=True):
    fig.update_layout(height=h, paper_bgcolor=CARD, plot_bgcolor=CARD, showlegend=legend,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK2, size=12),
                      margin=dict(l=55, r=20, t=30, b=45),
                      hovermode="x unified" if unified else "closest",
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      hoverlabel=dict(bgcolor="#26262a", font_color=INK, bordercolor=GRID))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def show(fig, key):
    cfg = {"displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:                      # versi Streamlit lama
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def kpi_row(items):
    html = "".join(f'<div class="kpi"><div class="kl">{a}</div><div class="kv{" sm" if len(str(b)) > 16 else ""}">{b}</div>'
                   f'<div class="ks">{c}</div></div>' for a, b, c in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def short(n):
    return "Ensemble (top-3)" if str(n).startswith("Ensemble") else n


def fmt(v, d=1):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan)
    return dict(MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                MAPE=np.mean(np.abs((y - p) / y)) * 100)


# -----------------------------------------------------------------------------
# Memuat hasil Kaggle (folder hasil_kaggle/ atau upload zip)
# -----------------------------------------------------------------------------
NEEDED = ["meta.json", "leaderboard_model.csv", "prediksi_model.csv.gz", "data_historis.csv.gz"]


@st.cache_data(show_spinner="Memuat hasil Kaggle…")
def load_bundle(files: dict):
    def csv(name, **kw):
        if name not in files:
            return None
        comp = "gzip" if name.endswith(".gz") else None
        return pd.read_csv(io.BytesIO(files[name]), compression=comp, **kw)
    B = dict(meta=json.loads(files["meta.json"]),
             leader=csv("leaderboard_model.csv"),
             pred=csv("prediksi_model.csv.gz", parse_dates=["origin", "datetime"]),
             data=csv("data_historis.csv.gz", parse_dates=["datetime"]),
             fut=csv("ramalan_24jam_kedepan.csv", parse_dates=["datetime"]),
             feat=csv("pengaruh_fitur.csv"), grp=csv("pengaruh_kelompok.csv"),
             corr=csv("korelasi_eksogen.csv"))
    B["data"] = B["data"].set_index("datetime").sort_index()
    return B


def files_from_dir(d):
    return {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))}


def files_from_zip(b):
    z = zipfile.ZipFile(io.BytesIO(b))
    return {os.path.basename(n): z.read(n) for n in z.namelist() if not n.endswith("/")}


with st.sidebar:
    st.markdown("## ⚡ Beban Listrik Panama")
    st.caption("Peramalan per jam · non-transfer learning vs transfer learning")
    src = st.radio("Sumber hasil Kaggle", ["Folder bawaan (hasil_kaggle/)", "Upload hasil_kaggle.zip"])
    files = None
    if src.startswith("Upload"):
        up = st.file_uploader("hasil_kaggle.zip dari tab Output Kaggle", type=["zip"])
        if up is not None:
            files = files_from_zip(up.getvalue())
    elif os.path.isdir(DEFAULT_DIR):
        files = files_from_dir(DEFAULT_DIR)

B = None
if files is not None:
    missing = [f for f in NEEDED if f not in files]
    if missing:
        st.sidebar.error("File berikut tidak ada di hasil: " + ", ".join(missing))
    else:
        B = load_bundle(files)

if B is not None:
    M, LB, PRED, DATA = B["meta"], B["leader"], B["pred"], B["data"]
    BEST = M["best"]
    with st.sidebar:
        st.divider()
        st.markdown(f"**Model terbaik:** {short(BEST)}")
        st.caption(f"Dibuat: {M.get('generated', '-')} · {len(LB)} model · uji {M['test_days']} hari")
        fams = [k for k in FAM if k in set(LB.Keluarga)]
        st.markdown("".join(f'<span class="lg"><i style="background:{FAM[k]}"></i>{k}</span><br>' for k in fams),
                    unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("# ⚡ Peramalan Beban Listrik Nasional Panama")
if B is not None:
    st.markdown(f'<div class="sub">Ramalan per jam {M["horizon"]} jam ke depan · data '
                f'{pd.Timestamp(M["data_start"]):%d %b %Y} – {pd.Timestamp(M["last_ts"]):%d %b %Y} · '
                f'{M["n_obs"]:,} observasi · evaluasi rolling-origin {M["test_days"]} hari uji</div>',
                unsafe_allow_html=True)

tabs = st.tabs(["📊 Ringkasan", "🏆 Perbandingan Model", "🔍 Variabel Berpengaruh",
                "📈 Eksplorasi Data", "🚀 Ramalan Langsung"])


def need_bundle():
    st.info("Hasil Kaggle belum dimuat. Letakkan folder **hasil_kaggle/** di samping app.py, "
            "atau upload **hasil_kaggle.zip** lewat sidebar. Tab **Ramalan Langsung** tetap bisa dipakai.")


# =============================================================================
# TAB 1 — RINGKASAN
# =============================================================================
with tabs[0]:
    if B is None:
        need_bundle()
    else:
        bt = LB.set_index("Model").loc[BEST]
        FUT = PRED[PRED.split == "future"]
        fb = FUT[FUT.model == BEST].sort_values("datetime")
        pk = fb.loc[fb.prediksi.idxmax()]
        kpi_row([("Model terbaik", short(BEST), bt.Keluarga),
                 ("MAPE test", f"{fmt(bt.TEST_MAPE, 2)}%", f"R² = {fmt(bt.TEST_R2, 3)}"),
                 ("MAE test", f"{fmt(bt.TEST_MAE)} MWh", f"RMSE {fmt(bt.TEST_RMSE)} MWh"),
                 ("Lebih baik dari naive", f"{fmt(bt['Skill_vs_Naive_%'])}%", "pengurangan MAE vs Seasonal Naive"),
                 ("Puncak 24 jam ke depan", f"{fmt(pk.prediksi, 0)} MWh", f"{pk.datetime:%d %b %Y, %H:%M}"),
                 ("Energi 24 jam ke depan", f"{fmt(fb.prediksi.sum() / 1000, 2)} GWh",
                  f"rata-rata {fmt(fb.prediksi.mean(), 0)} MWh/jam")])

        tl = LB[LB.Keluarga == "Transfer Learning"]
        ntl = LB[~LB.Keluarga.isin(["Transfer Learning", "Ensemble", "Baseline"])]
        txt = f"💡 Model terbaik (dipilih dari periode validasi) adalah <b>{short(BEST)}</b> dengan MAPE <b>{fmt(bt.TEST_MAPE, 2)}%</b>."
        if len(tl) and len(ntl):
            a, b = ntl.sort_values("TEST_MAPE").iloc[0], tl.sort_values("TEST_MAPE").iloc[0]
            txt += (f" Terbaik non-transfer learning: <b>{a.Model}</b> ({fmt(a.TEST_MAPE, 2)}%); "
                    f"terbaik transfer learning: <b>{b.Model}</b> ({fmt(b.TEST_MAPE, 2)}%).")
        if B["grp"] is not None:
            g = B["grp"].sort_values("kontribusi_pct", ascending=False)
            txt += f" Kelompok variabel paling berpengaruh: <b>{', '.join(g.kelompok.head(3))}</b>."
        st.markdown(f'<div class="insight">{txt}</div>', unsafe_allow_html=True)

        st.markdown("### Ramalan 24 jam ke depan")
        all_models = list(LB.Model)
        sel = st.multiselect("Tampilkan ramalan model", all_models, default=[BEST], format_func=short, key="fut_sel")
        last = pd.Timestamp(M["last_ts"])
        hist = DATA["nat_demand"].loc[last - pd.Timedelta(hours=71):last]
        fig = go.Figure()
        if B["fut"] is not None and BEST in sel:
            fu = B["fut"]
            fig.add_trace(go.Scatter(x=fu.datetime, y=fu.PI80_atas, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fu.datetime, y=fu.PI80_bawah, fill="tonexty", fillcolor="rgba(57,135,229,0.22)",
                                     line=dict(width=0), name="Interval 80% (model terbaik)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Aktual (72 jam terakhir)",
                                 line=dict(color=INK2, width=2), hovertemplate="%{y:,.0f} MWh"))
        for n in sel:
            d = FUT[FUT.model == n].sort_values("datetime")
            fam = LB.set_index("Model").loc[n, "Keluarga"]
            fig.add_trace(go.Scatter(x=d.datetime, y=d.prediksi, name=short(n), mode="lines+markers",
                                     marker=dict(size=5), hovertemplate="%{y:,.0f} MWh",
                                     line=dict(color=ACC if n == BEST else FAM[fam], width=2.6 if n == BEST else 1.6,
                                               dash=None if n == BEST else "dot")))
        fig.add_vline(x=last, line=dict(color=MUTED, dash="dot"))
        show(style(fig, 400).update_yaxes(title="MWh"), "fut")

        tbl = FUT[FUT.model.isin(sel)].pivot_table(index="datetime", columns="model", values="prediksi")
        tbl.columns = [short(c) for c in tbl.columns]
        c1, c2 = st.columns([3, 1])
        with c1:
            with st.expander("Tabel ramalan per jam"):
                st.dataframe(tbl.round(1), height=320)
        with c2:
            st.download_button("⬇️ Download ramalan (CSV)", tbl.to_csv().encode(), "ramalan_24jam.csv", "text/csv")

# =============================================================================
# TAB 2 — PERBANDINGAN MODEL
# =============================================================================
with tabs[1]:
    if B is None:
        need_bundle()
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        fam_sel = c1.multiselect("Keluarga model", list(FAM), default=[k for k in FAM if k in set(LB.Keluarga)])
        met = c2.selectbox("Metrik", ["MAPE", "MAE", "RMSE"])
        split = c3.radio("Periode", ["test", "val"], format_func=lambda s: "Uji" if s == "test" else "Validasi",
                         horizontal=True)
        P = PRED[(PRED.split == split) & (PRED.keluarga.isin(fam_sel))]
        rows = []
        for (n, k), g in P.groupby(["model", "keluarga"]):
            rows.append(dict(Model=n, Keluarga=k, **metrics(g.aktual, g.prediksi)))
        T = pd.DataFrame(rows)
        if T.empty:
            st.warning("Pilih minimal satu keluarga model.")
        else:
            T = T.sort_values(met)
            cA, cB = st.columns([7, 5])
            with cA:
                st.markdown(f"#### Peringkat model — {met} ({'uji' if split == 'test' else 'validasi'})")
                unit = "%" if met == "MAPE" else " MWh"
                fig = go.Figure(go.Bar(x=T[met], y=[short(n) for n in T.Model], orientation="h",
                                       marker=dict(color=[FAM[k] for k in T.Keluarga], line=dict(color=CARD, width=2)),
                                       text=[f"{v:.2f}{unit}" for v in T[met]], textposition="outside",
                                       textfont=dict(color=INK2), cliponaxis=False, customdata=T.Keluarga,
                                       hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.2f}<extra></extra>"))
                fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=170, r=60, t=10, b=40))
                show(style(fig, max(300, 38 * len(T)), legend=False, unified=False).update_xaxes(title=met), "lb")
            with cB:
                st.markdown("#### Galat menurut horizon")
                fig = go.Figure()
                for n in T.Model[:8]:
                    g = P[P.model == n].assign(e=lambda d: (d.aktual - d.prediksi).abs()).groupby("horizon").e.mean()
                    fam = T.set_index("Model").loc[n, "Keluarga"]
                    fig.add_trace(go.Scatter(x=g.index, y=g.values, name=short(n), mode="lines",
                                             line=dict(color=FAM[fam], width=2.6 if n == BEST else 1.4,
                                                       dash=None if n == BEST else "dot"),
                                             hovertemplate="%{y:.1f} MWh"))
                fig.update_layout(legend=dict(y=1.25, font=dict(size=10)))
                show(style(fig, 380).update_xaxes(title="Horizon (jam ke-)", dtick=2).update_yaxes(title="MAE (MWh)"), "hz")

            st.markdown("#### Tabel evaluasi lengkap (dari Kaggle)")
            show_lb = LB[LB.Keluarga.isin(fam_sel)].copy()
            show_lb["Model"] = show_lb.Model.map(short)
            st.dataframe(show_lb, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.2f") for c in show_lb.columns if show_lb[c].dtype != object})

            st.markdown("#### Aktual vs prediksi")
            origins = sorted(P.origin.unique())
            c1, c2 = st.columns([2, 2])
            msel = c1.multiselect("Model", list(T.Model), default=[m for m in [BEST] if m in set(T.Model)] or [T.Model.iloc[0]],
                                  format_func=short, key="avp")
            o0, o1 = c2.select_slider("Rentang origin (hari)", options=origins, value=(origins[0], origins[-1]),
                                      format_func=lambda x: pd.Timestamp(x).strftime("%d %b"))
            W = P[(P.origin >= o0) & (P.origin <= o1)]
            act = W.drop_duplicates("datetime").sort_values("datetime")
            fig = go.Figure(go.Scatter(x=act.datetime, y=act.aktual, name="Aktual", line=dict(color=INK2, width=1.8),
                                       hovertemplate="%{y:,.0f}"))
            for n in msel:
                d = W[W.model == n].sort_values("datetime")
                fam = T.set_index("Model").loc[n, "Keluarga"]
                fig.add_trace(go.Scatter(x=d.datetime, y=d.prediksi, name=short(n), hovertemplate="%{y:,.0f}",
                                         line=dict(color=ACC if n == BEST else FAM[fam], width=1.5)))
            show(style(fig, 380).update_yaxes(title="MWh"), "avp_fig")

            st.markdown("#### MAPE harian per origin")
            fig = go.Figure()
            for n in msel:
                d = W[W.model == n].groupby("origin").apply(lambda g: metrics(g.aktual, g.prediksi)["MAPE"])
                fam = T.set_index("Model").loc[n, "Keluarga"]
                fig.add_trace(go.Scatter(x=d.index, y=d.values, name=short(n), mode="lines+markers",
                                         line=dict(color=ACC if n == BEST else FAM[fam], width=1.8),
                                         hovertemplate="%{y:.2f}%"))
            show(style(fig, 300).update_yaxes(title="MAPE (%)"), "daily_mape")

# =============================================================================
# TAB 3 — VARIABEL BERPENGARUH
# =============================================================================
with tabs[2]:
    if B is None:
        need_bundle()
    elif B["feat"] is None:
        st.warning("File pengaruh_fitur.csv tidak ada di hasil Kaggle.")
    else:
        FT, GR = B["feat"], B["grp"].sort_values("kontribusi_pct")
        meth = M.get("imp_method", "importance")
        st.markdown(f'<div class="sub">Dihitung pada model <b>{M.get("imp_model")}</b> di periode uji · metode: {meth}</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Kontribusi per kelompok variabel")
            fig = go.Figure(go.Bar(x=GR.kontribusi_pct, y=GR.kelompok, orientation="h",
                                   marker=dict(color=ACC, line=dict(color=CARD, width=2)),
                                   text=[f"{v:.1f}%" for v in GR.kontribusi_pct], textposition="outside",
                                   textfont=dict(color=INK2), cliponaxis=False,
                                   hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.update_layout(margin=dict(l=170, r=50, t=10, b=40))
            show(style(fig, 340, legend=False, unified=False).update_xaxes(title="% dari total"), "grp")
        with c2:
            st.markdown("#### Kenaikan MAE bila kelompok diacak")
            G2 = GR.sort_values("kenaikan_mae_perm")
            fig = go.Figure(go.Bar(x=G2.kenaikan_mae_perm, y=G2.kelompok, orientation="h",
                                   marker=dict(color="#d95926", line=dict(color=CARD, width=2)),
                                   text=[f"+{v:.1f}" for v in G2.kenaikan_mae_perm], textposition="outside",
                                   textfont=dict(color=INK2), cliponaxis=False,
                                   hovertemplate="%{y}: +%{x:.2f} MWh<extra></extra>"))
            fig.update_layout(margin=dict(l=170, r=50, t=10, b=40))
            show(style(fig, 340, legend=False, unified=False).update_xaxes(title="MWh"), "perm")

        n_top = st.slider("Jumlah fitur ditampilkan", 5, min(40, len(FT)), 15)
        only_ex = st.toggle("Hanya variabel eksternal (tanpa riwayat beban/lag)", value=False)
        F2 = FT[FT.kelompok != "Riwayat beban (lag)"] if only_ex else FT
        F2 = F2.sort_values("nilai", ascending=False).head(n_top).iloc[::-1]
        st.markdown("".join(f'<span class="lg"><i style="background:{c}"></i>{k}</span>' for k, c in
                            [("Riwayat beban", "#3987e5"), ("Cuaca", "#d95926"), ("Kalender / libur", "#199e70")]),
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=F2.nilai, y=F2.fitur, orientation="h",
                               marker=dict(color=[GRP_COL.get(k, "#199e70") for k in F2.kelompok],
                                           line=dict(color=CARD, width=2)),
                               customdata=F2.kelompok, hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.2f}<extra></extra>"))
        fig.update_layout(margin=dict(l=130, r=20, t=10, b=40))
        show(style(fig, max(320, 26 * len(F2)), legend=False, unified=False).update_xaxes(title=meth), "feat")

        if B["corr"] is not None:
            st.markdown("#### Korelasi variabel eksogen dengan beban")
            C = B["corr"].sort_values("Pearson", key=abs)
            fig = go.Figure([go.Bar(x=C.Pearson, y=C.variabel, orientation="h", name="Pearson",
                                    marker=dict(color="#3987e5", line=dict(color=CARD, width=2))),
                             go.Bar(x=C.Spearman, y=C.variabel, orientation="h", name="Spearman",
                                    marker=dict(color="#d95926", line=dict(color=CARD, width=2)))])
            fig.update_layout(barmode="group", margin=dict(l=110, r=20, t=30, b=40))
            show(style(fig, 520, unified=False).update_xaxes(title="Koefisien korelasi", range=[-1, 1]), "corr")
        with st.expander("Cara membaca"):
            st.markdown("- **Kontribusi kelompok**: porsi total pengaruh fitur dalam satu kelompok.\n"
                        "- **Kenaikan MAE bila diacak**: seberapa memburuk ramalan kalau informasi kelompok itu dirusak "
                        "— makin besar, makin penting.\n- **Korelasi** hanya mengukur hubungan linier/monoton satu "
                        "variabel; model bisa menangkap pengaruh nonlinier dan interaksi.")

# =============================================================================
# TAB 4 — EKSPLORASI DATA
# =============================================================================
with tabs[3]:
    if B is None:
        need_bundle()
    else:
        d0, d1 = DATA.index.min().date(), DATA.index.max().date()
        rng = st.date_input("Rentang tanggal", (d0, d1), min_value=d0, max_value=d1)
        if isinstance(rng, tuple) and len(rng) == 2:
            X = DATA.loc[str(rng[0]):str(rng[1])]
        else:
            X = DATA
        y = X["nat_demand"]
        kpi_row([("Rata-rata beban", f"{fmt(y.mean(), 0)} MWh", "per jam"),
                 ("Beban puncak", f"{fmt(y.max(), 0)} MWh", f"{y.idxmax():%d %b %Y %H:%M}"),
                 ("Beban terendah", f"{fmt(y.min(), 0)} MWh", f"{y.idxmin():%d %b %Y %H:%M}"),
                 ("Jumlah jam", f"{len(y):,}".replace(",", "."), "observasi")])
        st.markdown("#### Tren beban harian")
        dly = y.resample("D").mean()
        fig = go.Figure(go.Scatter(x=dly.index, y=dly.values, line=dict(color=ACC, width=1.3),
                                   hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} MWh<extra></extra>"))
        if dly.index.max() >= pd.Timestamp("2020-03-15"):
            fig.add_vrect(x0=max(pd.Timestamp("2020-03-15"), dly.index.min()), x1=dly.index.max(),
                          fillcolor="#e66767", opacity=0.14, line_width=0)
        show(style(fig, 320, legend=False).update_yaxes(title="MWh"), "trend")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Profil beban: hari × jam")
            pv = X.assign(h=X.index.hour, d=X.index.dayofweek).pivot_table(index="d", columns="h", values="nat_demand")
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns],
                                       y=["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"][:len(pv)],
                                       colorscale=BLUES, xgap=2, ygap=2, colorbar=dict(title="MWh", thickness=10),
                                       hovertemplate="%{y} jam %{x}:00<br>%{z:,.0f} MWh<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 330, legend=False, unified=False).update_xaxes(title="Jam"), "heat")
        with c2:
            st.markdown("#### Suhu vs beban")
            city = st.selectbox("Kota", {"toc": "Tocumen (Kota Panama)", "san": "Santiago", "dav": "David"}.items(),
                                format_func=lambda kv: kv[1])[0]
            smp = X.sample(min(4000, len(X)), random_state=1)
            fig = go.Figure(go.Scattergl(x=smp[f"T2M_{city}"], y=smp["nat_demand"], mode="markers",
                                         marker=dict(size=5, color=smp.index.hour, colorscale=BLUES, opacity=0.6,
                                                     colorbar=dict(title="Jam", thickness=10)),
                                         hovertemplate="%{x:.1f}°C<br>%{y:,.0f} MWh<extra></extra>"))
            show(style(fig, 330, legend=False, unified=False).update_xaxes(title="Suhu 2 m (°C)").update_yaxes(title="MWh"), "temp")

        st.markdown("#### Pola 24 jam: hari kerja vs akhir pekan vs libur")
        prof = X.assign(h=X.index.hour, tipe=np.where(X.holiday == 1, "Hari libur",
                                                      np.where(X.index.dayofweek >= 5, "Akhir pekan", "Hari kerja")))
        prof = prof.groupby(["tipe", "h"]).nat_demand.mean().unstack(0)
        fig = go.Figure()
        for c, col in zip(["Hari kerja", "Akhir pekan", "Hari libur"], ["#3987e5", "#d95926", "#199e70"]):
            if c in prof:
                fig.add_trace(go.Scatter(x=prof.index, y=prof[c], name=c, line=dict(color=col, width=2),
                                         hovertemplate="%{y:,.0f} MWh"))
        show(style(fig, 320).update_xaxes(title="Jam", dtick=2).update_yaxes(title="MWh"), "prof")

# =============================================================================
# TAB 5 — RAMALAN LANGSUNG (upload + Chronos-Bolt)
# =============================================================================
CHRONOS = {"Chronos-Bolt Tiny (9M, tercepat)": "amazon/chronos-bolt-tiny",
           "Chronos-Bolt Mini (21M)": "amazon/chronos-bolt-mini",
           "Chronos-Bolt Small (48M, disarankan)": "amazon/chronos-bolt-small",
           "Chronos-Bolt Base (205M, paling akurat, lambat)": "amazon/chronos-bolt-base"}


@st.cache_resource(show_spinner="Memuat foundation model (sekali saja)…")
def load_chronos(name):
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return BaseChronosPipeline.from_pretrained(name, device_map="cpu")


def chronos_forecast(name, ctx, h):
    import torch
    pipe = load_chronos(name)
    q, _ = pipe.predict_quantiles(torch.tensor(ctx, dtype=torch.float32)[None, :], h,
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
        sheet = xl.sheet_names[0] if "data" not in xl.sheet_names else "data"
        return xl.parse(sheet)
    return pd.read_csv(up, sep=None, engine="python")


def prep_series(df, tcol, ycol):
    s = df[[tcol, ycol]].copy()
    s[tcol] = pd.to_datetime(s[tcol], errors="coerce")
    s[ycol] = pd.to_numeric(s[ycol], errors="coerce")
    s = s.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol, keep="last").set_index(tcol)[ycol]
    step = s.index.to_series().diff().median()
    full = pd.date_range(s.index[0], s.index[-1], freq=step)
    n_gap = len(full) - s.reindex(full).notna().sum()
    s = s.reindex(full).interpolate(limit_direction="both")
    return s, step, int(n_gap)


with tabs[4]:
    st.markdown("### 🚀 Ramalan langsung dengan foundation model")
    st.markdown('<div class="sub">Upload data beban terbaru (CSV/XLSX berisi kolom waktu & beban). '
                'Chronos-Bolt (Amazon) meramal zero-shot di CPU — tanpa pelatihan.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        opts = ["Upload file"] + (["Pakai data historis bawaan"] if B is not None else [])
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
        if raw_df is not None:
            cols = list(raw_df.columns)
            t_guess = next((c for c in cols if "date" in str(c).lower() or "time" in str(c).lower()
                            or "tanggal" in str(c).lower() or "waktu" in str(c).lower()), cols[0])
            num = [c for c in cols if pd.api.types.is_numeric_dtype(raw_df[c])]
            y_guess = "nat_demand" if "nat_demand" in cols else (num[0] if num else cols[-1])
            tcol = st.selectbox("Kolom waktu", cols, index=cols.index(t_guess))
            ycol = st.selectbox("Kolom target (beban)", cols, index=cols.index(y_guess))
        mlabel = st.selectbox("Model", list(CHRONOS), index=2)
        h = st.number_input("Horizon (langkah ke depan)", 1, 168, 24, help="Chronos-Bolt dilatih hingga 64 langkah; "
                                                                           "lebih dari itu akurasi menurun.")
        ctx_len = st.select_slider("Panjang konteks (langkah historis)", [168, 336, 512, 1024, 2048], value=2048)
        backtest = st.toggle("Mode uji (backtest): sembunyikan data terakhir lalu bandingkan", value=False)
        go_btn = st.button("🚀 Jalankan ramalan", type="primary", disabled=raw_df is None, width="stretch")

    with right:
        if go_btn and raw_df is not None:
            try:
                s, step, n_gap = prep_series(raw_df, tcol, ycol)
            except Exception as e:
                st.error(f"Data tidak bisa diproses: {e}")
                s = None
            if s is not None and len(s) < 2 * h + 24:
                st.error("Data terlalu pendek untuk horizon ini.")
                s = None
            if s is not None:
                actual = None
                if backtest:
                    actual, s_in = s.iloc[-h:], s.iloc[:-h]
                else:
                    s_in = s
                ctx = s_in.values[-ctx_len:].astype(np.float32)
                fidx = pd.date_range(s_in.index[-1] + step, periods=h, freq=step)
                try:
                    with st.spinner(f"Meramal dengan {mlabel}…"):
                        med, lo, hi = chronos_forecast(CHRONOS[mlabel], ctx, int(h))
                    used = mlabel
                except Exception as e:
                    st.warning(f"Chronos tidak tersedia ({type(e).__name__}: {str(e)[:120]}). "
                               "Memakai Seasonal Naive sebagai cadangan.")
                    med = seasonal_naive(s_in, int(h), step)
                    sd = float(np.nanstd(s_in.diff(int(pd.Timedelta("1D") / step) or 1).values[-ctx_len:]))
                    lo, hi, used = med - 1.28 * sd, med + 1.28 * sd, "Seasonal Naive (cadangan)"
                st.session_state["live"] = dict(s=s_in, fidx=fidx, med=med, lo=lo, hi=hi, used=used,
                                                actual=actual, naive=seasonal_naive(s_in, int(h), step),
                                                n_gap=n_gap, step=step, ycol=ycol)

        R = st.session_state.get("live")
        if R is None:
            st.info("Pilih data dan klik **Jalankan ramalan**. Contoh format:\n\n"
                    "| datetime | nat_demand |\n|---|---|\n| 2020-06-26 01:00 | 1012.5 |\n| 2020-06-26 02:00 | 978.1 |")
        else:
            if R["n_gap"]:
                st.caption(f"ℹ️ {R['n_gap']} titik waktu kosong diisi dengan interpolasi linier.")
            items = [("Model", R["used"].split(" (")[0], f"interval {pd.Timedelta(R['step'])}"),
                     ("Puncak ramalan", fmt(R["med"].max(), 0), f"{R['fidx'][int(np.argmax(R['med']))]:%d %b %Y %H:%M}"),
                     ("Rata-rata ramalan", fmt(R["med"].mean(), 0), f"{len(R['med'])} langkah ke depan")]
            if R["actual"] is not None:
                m1, m0 = metrics(R["actual"].values, R["med"]), metrics(R["actual"].values, R["naive"])
                items += [("MAPE (backtest)", f"{fmt(m1['MAPE'], 2)}%", f"MAE {fmt(m1['MAE'])}"),
                          ("MAPE Seasonal Naive", f"{fmt(m0['MAPE'], 2)}%",
                           f"model {'lebih baik' if m1['MAE'] < m0['MAE'] else 'lebih buruk'} "
                           f"{fmt(abs(1 - m1['MAE'] / m0['MAE']) * 100)}%")]
            kpi_row(items)
            hist = R["s"].iloc[-min(len(R["s"]), 7 * 24):]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=R["fidx"], y=R["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=R["fidx"], y=R["lo"], fill="tonexty", fillcolor="rgba(57,135,229,0.22)",
                                     line=dict(width=0), name="Interval 80% (q10–q90)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historis", line=dict(color=INK2, width=1.6),
                                     hovertemplate="%{y:,.1f}"))
            if R["actual"] is not None:
                fig.add_trace(go.Scatter(x=R["actual"].index, y=R["actual"].values, name="Aktual (disembunyikan)",
                                         line=dict(color="#199e70", width=2), hovertemplate="%{y:,.1f}"))
                fig.add_trace(go.Scatter(x=R["fidx"], y=R["naive"], name="Seasonal Naive",
                                         line=dict(color=MUTED, width=1.3, dash="dot"), hovertemplate="%{y:,.1f}"))
            fig.add_trace(go.Scatter(x=R["fidx"], y=R["med"], name=f"Ramalan · {R['used'].split(' (')[0]}",
                                     mode="lines+markers", marker=dict(size=5), line=dict(color=ACC, width=2.6),
                                     hovertemplate="%{y:,.1f}"))
            fig.add_vline(x=R["s"].index[-1], line=dict(color=MUTED, dash="dot"))
            show(style(fig, 440).update_yaxes(title=str(R["ycol"])), "live_fig")
            out = pd.DataFrame({"datetime": R["fidx"], "ramalan": R["med"], "batas_bawah_q10": R["lo"],
                                "batas_atas_q90": R["hi"]})
            if R["actual"] is not None:
                out["aktual"] = R["actual"].values
            c1, c2 = st.columns([3, 1])
            with c1:
                with st.expander("Tabel hasil ramalan"):
                    st.dataframe(out.round(2), hide_index=True, height=300)
            with c2:
                st.download_button("⬇️ Download hasil (CSV)", out.to_csv(index=False).encode(),
                                   "ramalan_langsung.csv", "text/csv")

st.markdown('<div class="sub" style="margin-top:24px">Model transfer learning dipakai zero-shot · cuaca '
            'diperlakukan sebagai prakiraan yang diketahui pada model non-TL · dibuat dengan Streamlit & Plotly</div>',
            unsafe_allow_html=True)
