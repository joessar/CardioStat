"""CardioStat Lab — Sistem Analisis Empiris Faktor Risiko Serangan Jantung Berbasis Statistika Terapan."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import stats_engine

importlib.reload(stats_engine)
from stats_engine import (
    CLINICAL_CUTOFFS,
    classify_columns,
    export_results_to_excel,
    load_dataset,
    run_assumption_tests,
    run_chi_square_association,
    run_correlation,
    run_multi_group_comparison,
    run_multiple_regression,
    run_profile_and_risk,
    run_two_group_comparison,
)

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "heart_attack_prediction_dataset.csv"

ANALYSIS_MODULES = [
    ("profil", "1. Deskriptif & Risiko (PAN/PAK)"),
    ("asumsi", "2. Uji Asumsi (Shapiro-Wilk & Levene)"),
    ("komparasi_2", "3. Uji Beda 2 Kelompok (t-test / Mann-Whitney)"),
    ("komparasi_multi", "4. Uji Beda >2 Kelompok (ANOVA / Kruskal-Wallis)"),
    ("asosiasi", "5. Uji Asosiasi (Chi-Square & Cramér's V)"),
    ("korelasi_regresi", "6. Korelasi & Regresi Berganda"),
]

MODULE_INFO: dict[str, dict[str, str]] = {
    "profil": {
        "title": "Statistik Deskriptif & Kategorisasi Risiko Klinis (PAN / PAK)",
        "use": "Merangkum karakteristik parameter numerik pasien (Mean, Median, SD, Min, Max, IQR, Skewness) serta mengelompokkan pasien ke dalam 3 tingkat risiko (Rendah, Sedang, Tinggi).",
        "criteria": "Metode PAK menggunakan ambang batas klinis tetap (AHA/WHO/NCEP). Jika kolom belum memiliki ambang klinis tetap, sistem otomatis memakai PAN (Mean ± 1 SD). Periksa Skewness dan grafik Histogram/Boxplot/Violin untuk melihat bentuk sebaran data.",
        "ref": "Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics; Azwar, S. (2012/2021). Penyusunan Skala Psikologi.",
    },
    "asumsi": {
        "title": "Uji Prasyarat / Asumsi — Normalitas (Shapiro–Wilk) & Homogenitas (Levene)",
        "use": "Memeriksa apakah sebaran data mendekati distribusi normal dan apakah varians antar kelompok pasien homogen sebelum memilih uji parametrik atau nonparametrik.",
        "criteria": "Shapiro–Wilk: p >= 0.05 menandakan data konsisten dengan distribusi normal (otomatis mengambil maks. 5.000 sampel acak jika N > 5.000), selalu kombinasikan dengan Q-Q Plot. Levene's Test: p >= 0.05 mendukung homogenitas varians; jika tidak homogen, gunakan Welch t-test atau uji nonparametrik.",
        "ref": "Shapiro & Wilk (1965); Levene (1960); Ghasemi & Zahediasl (2012); Field (2018).",
    },
    "komparasi_2": {
        "title": "Komparasi 2 Kelompok Independen — Independent t-test & Mann–Whitney U",
        "use": "Membandingkan parameter klinis numerik antara tepat dua kelompok pasien independen (misalnya Berisiko vs Tidak Berisiko, atau Perokok vs Non-Perokok).",
        "criteria": "Gunakan Independent t-test (Student jika varians homogen, Welch jika tidak homogen) untuk data normal, atau Mann–Whitney U untuk data non-normal/ordinal. Signifikan jika p < 0.05. Laporkan besaran efek Cohen's d (<0.2 sangat kecil, 0.2–0.5 kecil, 0.5–0.8 sedang, >=0.8 besar).",
        "ref": "Field, A. (2018); Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences; Welch (1947).",
    },
    "komparasi_multi": {
        "title": "Komparasi >2 Kelompok Independen — One-Way ANOVA, Kruskal–Wallis & Tukey HSD",
        "use": "Membandingkan parameter klinis numerik antar tiga atau lebih kelompok pasien independen (misalnya pola Diet: Healthy/Average/Unhealthy, atau antar Benua).",
        "criteria": "Gunakan One-Way ANOVA (parametrik) atau Kruskal–Wallis (nonparametrik). Jika uji omnibus signifikan (p < 0.05), periksa uji lanjut Post-Hoc Tukey HSD untuk mengetahui pasangan kelompok yang berbeda signifikan serta laporkan Eta-squared (η²).",
        "ref": "Field, A. (2018); Cohen, J. (1988).",
    },
    "asosiasi": {
        "title": "Uji Asosiasi Faktor Risiko Kategorik — Chi-Square & Cramér's V",
        "use": "Menguji apakah terdapat hubungan yang signifikan antara dua variabel kategorik (misalnya Riwayat Merokok, Diabetes, atau Obesitas terhadap Risiko Serangan Jantung).",
        "criteria": "Signifikan jika p < 0.05. Periksa jumlah sel dengan frekuensi harapan (expected count) < 5 agar hasil Chi-Square tetap andal. Kekuatan hubungan diukur dengan Cramér's V (<0.1 lemah, 0.1–0.3 sedang, >=0.3 kuat).",
        "ref": "Agresti, A. (2019). An Introduction to Categorical Data Analysis.",
    },
    "korelasi_regresi": {
        "title": "Korelasi (Pearson / Spearman) & Regresi Linear Berganda (OLS)",
        "use": "Mengukur arah dan kekuatan hubungan antar variabel numerik (Pearson untuk linear, Spearman untuk nonparametrik/monoton) serta memodelkan pengaruh simultan beberapa prediktor terhadap satu variabel target.",
        "criteria": "Korelasi: periksa koefisien (r / ρ) dan p < 0.05. Regresi Berganda: evaluasi R², Adjusted R², uji simultan F (p < 0.05), signifikansi parsial tiap koefisien B, serta VIF (< 5–10 menandakan bebas multikolinearitas serius) dan plot residual.",
        "ref": "Schober et al. (2018); Fox, J. (2016); Field, A. (2018).",
    },
}


st.set_page_config(
    page_title="CardioStat Lab — Analisis Statistika Kardiovaskular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 55%, #b91c1c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(127, 29, 29, 0.18);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.96rem;
        color: #fee2e2;
    }
    .stage-badge {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .clinical-box {
        background-color: #f8fafc;
        border-left: 5px solid #991b1b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0 1rem 0;
        font-size: 0.96rem;
        line-height: 1.6;
        color: #1e293b;
    }
    div.stButton > button[kind="primary"] {
        background-color: #991b1b;
        border-color: #991b1b;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #7f1d1d;
        border-color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "df": None,
        "dataset_name": None,
        "bp_parsed": False,
        "active_module": None,
        "analysis_result": None,
        "analysis_module_ran": None,
        "table_page": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>CardioStat Lab</h1>
            <p>Sistem Analisis Empiris Faktor Risiko Serangan Jantung Berbasis Statistika Terapan (Non-Machine Learning)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_1_upload() -> None:
    st.markdown('<span class="stage-badge">Tahap 1 — Panel Unggah Data</span>', unsafe_allow_html=True)
    col_up, col_btn = st.columns([3, 1.4], vertical_alignment="bottom")

    with col_up:
        uploaded_file = st.file_uploader(
            "Unggah Dataset Pasien (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            help="Kolom 'Blood Pressure' berformat 'Sistolik/Diastolik' akan otomatis dipecah menjadi Systolic_BP dan Diastolic_BP.",
        )

    with col_btn:
        load_default = st.button(
            "Muat Dataset Pasien Jantung",
            type="primary",
            use_container_width=True,
            help="Memuat dataset bawaan heart_attack_prediction_dataset.csv (8.763 pasien).",
        )

    if uploaded_file is not None:
        if st.session_state.dataset_name != f"upload:{uploaded_file.name}:{uploaded_file.size}":
            try:
                df, bp_parsed = load_dataset(uploaded_file, filename=uploaded_file.name)
                st.session_state.df = df
                st.session_state.dataset_name = f"upload:{uploaded_file.name}:{uploaded_file.size}"
                st.session_state.bp_parsed = bp_parsed
                st.session_state.active_module = None
                st.session_state.analysis_result = None
                st.session_state.table_page = 1
                st.success(f"Dataset **{uploaded_file.name}** berhasil dimuat ({len(df):,} baris, {df.shape[1]} kolom).")
            except Exception as exc:
                st.error(f"Gagal membaca file: {exc}")

    if load_default:
        if not DEFAULT_DATASET_PATH.exists():
            st.error(f"Dataset bawaan tidak ditemukan di: `{DEFAULT_DATASET_PATH}`")
        else:
            try:
                df, bp_parsed = load_dataset(DEFAULT_DATASET_PATH)
                st.session_state.df = df
                st.session_state.dataset_name = "default:heart_attack_prediction_dataset.csv"
                st.session_state.bp_parsed = bp_parsed
                st.session_state.active_module = None
                st.session_state.analysis_result = None
                st.session_state.table_page = 1
                st.success(
                    f"Dataset bawaan **heart_attack_prediction_dataset.csv** berhasil dimuat "
                    f"({len(df):,} pasien, {df.shape[1]} variabel)."
                )
            except Exception as exc:
                st.error(f"Gagal memuat dataset bawaan: {exc}")


def render_stage_2_preview(df: pd.DataFrame) -> None:
    st.divider()
    st.markdown('<span class="stage-badge">Tahap 2 — Pratinjau Tabel Dataset</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Pasien (Baris)", f"{len(df):,}")
    c2.metric("Jumlah Variabel (Kolom)", f"{df.shape[1]}")

    if "Heart Attack Risk" in df.columns:
        risk_rate = pd.to_numeric(df["Heart Attack Risk"], errors="coerce").mean() * 100
        c3.metric("Prevalensi Risiko (1)", f"{risk_rate:.2f}%")
    else:
        c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    c4.metric(
        "Auto-Parse Blood Pressure",
        "Aktif (Systolic & Diastolic)" if st.session_state.bp_parsed else "Tidak Ditemukan",
    )

    # Search & pagination controls
    f1, f2, f3 = st.columns([2.5, 1, 1], vertical_alignment="bottom")
    with f1:
        search_query = st.text_input(
            "Cari data pada tabel",
            placeholder="Ketik ID pasien, negara, benua, pola diet, dll...",
        )
    with f2:
        page_size = st.selectbox("Baris per halaman", [10, 25, 50, 100], index=0)

    filtered_df = df
    if search_query.strip():
        q = search_query.strip().lower()
        mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
        filtered_df = df[mask]

    total_rows = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    with f3:
        page_num = st.number_input(
            f"Halaman (1 - {total_pages})",
            min_value=1,
            max_value=total_pages,
            value=min(st.session_state.table_page, total_pages),
            step=1,
        )
        st.session_state.table_page = int(page_num)

    start_idx = (st.session_state.table_page - 1) * page_size
    end_idx = start_idx + page_size
    st.caption(f"Menampilkan baris {start_idx + 1:,} - {min(end_idx, total_rows):,} dari {total_rows:,} hasil.")
    st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)


def render_stage_3_navbar() -> None:
    st.divider()
    st.markdown('<span class="stage-badge">Tahap 3 — Bilah Pilihan Analisis Horizontal</span>', unsafe_allow_html=True)
    cols = st.columns(len(ANALYSIS_MODULES))
    for idx, (mod_key, mod_label) in enumerate(ANALYSIS_MODULES):
        is_active = st.session_state.active_module == mod_key
        if cols[idx].button(
            mod_label,
            key=f"nav_{mod_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_module = mod_key
            st.session_state.analysis_result = None
            st.session_state.analysis_module_ran = None
            st.rerun()


def render_stage_4_and_5(df: pd.DataFrame, col_meta: dict[str, list[str]]) -> None:
    mod = st.session_state.active_module
    if not mod:
        st.info("Pilih salah satu jenis analisis pada bilah navigasi di atas untuk mengonfigurasi variabel.")
        return

    st.divider()
    st.markdown('<span class="stage-badge">Tahap 4 — Konfigurasi Variabel Dinamis</span>', unsafe_allow_html=True)

    info = MODULE_INFO[mod]
    st.subheader(info["title"])
    with st.expander("Panduan Uji, Kriteria Keputusan & Referensi", expanded=True):
        st.markdown(
            f"- **Kegunaan:** {info['use']}\n"
            f"- **Kriteria Keputusan:** {info['criteria']}\n"
            f"- **Referensi:** *{info['ref']}*"
        )

    num_cols = col_meta["numeric"]
    cat_cols = col_meta["categorical"]
    two_cols = col_meta["two_group"]
    multi_cols = col_meta["multi_group"]

    with st.container(border=True):
        if mod == "profil":
            c1, c2, c3 = st.columns([1.3, 1.5, 0.8])
            sel_num = c1.selectbox("Pilih Variabel Numerik", num_cols, index=num_cols.index("Cholesterol") if "Cholesterol" in num_cols else 0)
            sel_method = c2.selectbox(
                "Metode Kategorisasi Risiko",
                [
                    "PAK (Acuan Klinis Medis)",
                    "PAK (Skor Ideal Teoretis — Mi & SDi)",
                    "PAN (Mean ± SD Empiris)",
                ],
            )
            sel_k = c3.selectbox("Jumlah Kategori", [2, 3, 4, 5], index=1)

            ideal_min_val = None
            ideal_max_val = None
            if "Skor Ideal Teoretis" in sel_method:
                col_series = pd.to_numeric(df[sel_num], errors="coerce").dropna()
                def_min = float(col_series.min()) if not col_series.empty else 0.0
                def_max = float(col_series.max()) if not col_series.empty else 100.0
                m1, m2 = st.columns(2)
                ideal_min_val = m1.number_input("Skor Minimum Teoretis", value=def_min, step=1.0)
                ideal_max_val = m2.number_input("Skor Maksimum Teoretis", value=def_max, step=1.0)

            if st.button("Jalankan Analisis", type="primary"):
                try:
                    res = run_profile_and_risk(
                        df,
                        sel_num,
                        method=sel_method,
                        k_categories=int(sel_k),
                        ideal_min=ideal_min_val,
                        ideal_max=ideal_max_val,
                    )
                    res["col"] = sel_num
                    st.session_state.analysis_result = res
                    st.session_state.analysis_module_ran = mod
                except Exception as exc:
                    st.error(str(exc))

        elif mod == "asumsi":
            c1, c2 = st.columns(2)
            sel_num = c1.selectbox("Pilih Variabel Numerik", num_cols, index=num_cols.index("Cholesterol") if "Cholesterol" in num_cols else 0)
            group_options = ["(Tanpa Pengelompokan)"] + cat_cols
            sel_grp = c2.selectbox("Variabel Kelompok (untuk Levene's Test)", group_options, index=group_options.index("Heart Attack Risk") if "Heart Attack Risk" in group_options else 0)
            if st.button("Jalankan Analisis", type="primary"):
                try:
                    grp = None if sel_grp == "(Tanpa Pengelompokan)" else sel_grp
                    res = run_assumption_tests(df, sel_num, group_col=grp)
                    res["col"] = sel_num
                    res["group_col"] = grp
                    st.session_state.analysis_result = res
                    st.session_state.analysis_module_ran = mod
                except Exception as exc:
                    st.error(str(exc))

        elif mod == "komparasi_2":
            if not two_cols:
                st.warning("Tidak ditemukan kolom kategorik dengan tepat 2 kelompok pada dataset ini.")
                return
            c1, c2 = st.columns(2)
            sel_num = c1.selectbox("Variabel Numerik (Skor / Biomarker)", num_cols, index=num_cols.index("Cholesterol") if "Cholesterol" in num_cols else 0)
            sel_grp = c2.selectbox("Variabel Kelompok (Tepat 2 Kategori)", two_cols, index=two_cols.index("Heart Attack Risk") if "Heart Attack Risk" in two_cols else 0)
            if st.button("Jalankan Analisis", type="primary"):
                try:
                    res = run_two_group_comparison(df, sel_num, sel_grp)
                    res["col"] = sel_num
                    res["group_col"] = sel_grp
                    st.session_state.analysis_result = res
                    st.session_state.analysis_module_ran = mod
                except Exception as exc:
                    st.error(str(exc))

        elif mod == "komparasi_multi":
            if not multi_cols:
                st.warning("Tidak ditemukan kolom kategorik dengan minimal 3 kelompok pada dataset ini.")
                return
            c1, c2 = st.columns(2)
            sel_num = c1.selectbox("Variabel Numerik (Skor / Biomarker)", num_cols, index=num_cols.index("Cholesterol") if "Cholesterol" in num_cols else 0)
            sel_grp = c2.selectbox("Variabel Kelompok (>= 3 Kategori)", multi_cols, index=multi_cols.index("Diet") if "Diet" in multi_cols else 0)
            if st.button("Jalankan Analisis", type="primary"):
                try:
                    res = run_multi_group_comparison(df, sel_num, sel_grp)
                    res["col"] = sel_num
                    res["group_col"] = sel_grp
                    st.session_state.analysis_result = res
                    st.session_state.analysis_module_ran = mod
                except Exception as exc:
                    st.error(str(exc))

        elif mod == "asosiasi":
            if len(cat_cols) < 2:
                st.warning("Minimal dibutuhkan 2 variabel kategorik untuk uji Chi-Square.")
                return
            c1, c2 = st.columns(2)
            sel_r = c1.selectbox("Variabel Faktor Risiko (Baris)", cat_cols, index=cat_cols.index("Smoking") if "Smoking" in cat_cols else 0)
            sel_c = c2.selectbox("Variabel Target / Pasangan (Kolom)", cat_cols, index=cat_cols.index("Heart Attack Risk") if "Heart Attack Risk" in cat_cols else min(1, len(cat_cols) - 1))
            if st.button("Jalankan Analisis", type="primary"):
                try:
                    res = run_chi_square_association(df, sel_r, sel_c)
                    res["row_col"] = sel_r
                    res["col_col"] = sel_c
                    st.session_state.analysis_result = res
                    st.session_state.analysis_module_ran = mod
                except Exception as exc:
                    st.error(str(exc))

        elif mod == "korelasi_regresi":
            sub_mode = st.radio("Jenis Analisis", ["Korelasi (Pearson / Spearman)", "Regresi Linear Berganda"], horizontal=True)
            if sub_mode.startswith("Korelasi"):
                c1, c2 = st.columns([2.5, 1])
                default_corr = [c for c in ["Age", "Cholesterol", "Systolic_BP", "Diastolic_BP", "BMI", "Triglycerides"] if c in num_cols]
                if len(default_corr) < 2:
                    default_corr = num_cols[:4]
                sel_cols = c1.multiselect("Pilih Variabel Numerik (Minimal 2)", num_cols, default=default_corr)
                corr_method = c2.selectbox("Metode Korelasi", ["Pearson", "Spearman", "Kendall"])
                if st.button("Jalankan Analisis", type="primary"):
                    try:
                        res = run_correlation(df, sel_cols, method=corr_method)
                        res["sub_mode"] = "korelasi"
                        res["method"] = corr_method
                        st.session_state.analysis_result = res
                        st.session_state.analysis_module_ran = mod
                    except Exception as exc:
                        st.error(str(exc))
            else:
                c1, c2 = st.columns([1, 2.2])
                target_opts = num_cols + [c for c in two_cols if c not in num_cols]
                sel_target = c1.selectbox("Variabel Target (Y)", target_opts, index=target_opts.index("Systolic_BP") if "Systolic_BP" in target_opts else 0)
                pred_opts = [c for c in num_cols + two_cols if c != sel_target]
                default_preds = [c for c in ["Age", "Cholesterol", "BMI", "Exercise Hours Per Week", "Stress Level", "Sleep Hours Per Day"] if c in pred_opts]
                if not default_preds:
                    default_preds = pred_opts[:3]
                sel_preds = c2.multiselect("Variabel Prediktor (X)", pred_opts, default=default_preds)
                if st.button("Jalankan Analisis", type="primary"):
                    try:
                        res = run_multiple_regression(df, sel_target, sel_preds)
                        res["sub_mode"] = "regresi"
                        res["target"] = sel_target
                        st.session_state.analysis_result = res
                        st.session_state.analysis_module_ran = mod
                    except Exception as exc:
                        st.error(str(exc))

    # Stage 5: Render results if available for active module
    res = st.session_state.analysis_result
    if res is not None and st.session_state.analysis_module_ran == mod:
        render_stage_5_results(mod, res)


def render_stage_5_results(mod: str, res: dict[str, Any]) -> None:
    st.divider()
    st.markdown('<span class="stage-badge">Tahap 5 — Hasil Uji, Visualisasi & Penjelasan Klinis</span>', unsafe_allow_html=True)

    st.markdown(f'<div class="clinical-box"><b>Interpretasi Klinis Otomatis:</b><br>{res["narrative"]}</div>', unsafe_allow_html=True)

    export_sheets: dict[str, pd.DataFrame] = {
        "Narasi_Klinis": pd.DataFrame([{"Modul": mod, "Interpretasi Klinis": res["narrative"].replace("**", "").replace("*", "")}])
    }

    if mod == "profil":
        st.caption(res["cutoff_note"])
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("Statistik Deskriptif")
            st.dataframe(res["stats_df"], use_container_width=True, hide_index=True)
        with t2:
            st.subheader("Distribusi Kategori Risiko")
            st.dataframe(res["risk_df"], use_container_width=True, hide_index=True)
            st.subheader("Tabel Distribusi Frekuensi")
            st.dataframe(res["freq_df"], use_container_width=True, hide_index=True)

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)
        plot_df = pd.DataFrame({res["col"]: res["series"], "Kategori": res["categories"]})
        with g1:
            fig_hist = px.histogram(plot_df, x=res["col"], color="Kategori", nbins=30, title=f"Histogram {res['col']}")
            st.plotly_chart(fig_hist, use_container_width=True)
        with g2:
            risk_plot_df = res["risk_df"].copy()
            risk_plot_df["Label"] = risk_plot_df.apply(
                lambda r: f"{int(r['Frekuensi'])} ({r['Persentase (%)']:.2f}%)", axis=1
            )
            fig_cat_bar = px.bar(
                risk_plot_df,
                x="Kategori Risiko",
                y="Frekuensi",
                color="Kategori Risiko",
                text="Label",
                title=f"Distribusi {len(risk_plot_df)} Kategori — {res['col']}",
            )
            fig_cat_bar.update_traces(textposition="outside", showlegend=False)
            fig_cat_bar.update_yaxes(range=[0, float(risk_plot_df["Frekuensi"].max()) * 1.15])
            st.plotly_chart(fig_cat_bar, use_container_width=True)
        with g3:
            fig_box = px.box(plot_df, y=res["col"], title=f"Boxplot {res['col']}", color_discrete_sequence=["#991b1b"])
            st.plotly_chart(fig_box, use_container_width=True)
        with g4:
            fig_vio = px.violin(plot_df, y=res["col"], box=True, title=f"Violin Plot {res['col']}", color_discrete_sequence=["#b91c1c"])
            st.plotly_chart(fig_vio, use_container_width=True)

        with st.expander("Detail Kategorisasi per Pasien (500 Baris Pertama)"):
            st.dataframe(res["detail_df"].head(500), use_container_width=True, hide_index=True)

        export_sheets["Statistik_Deskriptif"] = res["stats_df"]
        export_sheets["Kategori_Risiko"] = res["risk_df"]
        export_sheets["Distribusi_Frekuensi"] = res["freq_df"]

    elif mod == "asumsi":
        st.subheader("Hasil Uji Normalitas & Homogenitas")
        st.dataframe(res["summary_df"], use_container_width=True, hide_index=True)

        qq_df = res["qq_df"]
        fig_qq = px.scatter(qq_df, x="Theoretical Quantiles", y="Sample Quantiles", title=f"Normal Q-Q Plot ({res['col']})", opacity=0.6)
        slope, intercept = np.polyfit(qq_df["Theoretical Quantiles"], qq_df["Sample Quantiles"], 1)
        line_x = np.array([qq_df["Theoretical Quantiles"].min(), qq_df["Theoretical Quantiles"].max()])
        fig_qq.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines", name="Garis Normal", line=dict(color="#991b1b", width=2)))

        if res.get("group_col"):
            g_left, g_right = st.columns(2)
            with g_left:
                st.plotly_chart(fig_qq, use_container_width=True)
            with g_right:
                fig_hom = px.box(
                    res["clean_df"],
                    x=res["group_col"],
                    y=res["col"],
                    color=res["group_col"],
                    title=f"Sebaran {res['col']} per Kelompok ({res['group_col']})",
                )
                st.plotly_chart(fig_hom, use_container_width=True)
        else:
            st.plotly_chart(fig_qq, use_container_width=True)

        export_sheets["Uji_Asumsi"] = res["summary_df"]

    elif mod == "komparasi_2":
        c1, c2 = st.columns([1.1, 1.3])
        with c1:
            st.subheader("Ringkasan per Kelompok")
            st.dataframe(res["group_df"], use_container_width=True, hide_index=True)
            st.subheader("Hasil Uji Hipotesis")
            st.dataframe(res["test_df"], use_container_width=True, hide_index=True)
        with c2:
            fig_box = px.box(
                res["clean_df"],
                x=res["group_col"],
                y=res["col"],
                color=res["group_col"],
                title=f"Perbandingan {res['col']} menurut {res['group_col']}",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        export_sheets["Statistik_Kelompok"] = res["group_df"]
        export_sheets["Hasil_Uji_2_Kelompok"] = res["test_df"]

    elif mod == "komparasi_multi":
        c1, c2 = st.columns([1.1, 1.3])
        with c1:
            st.subheader("Ringkasan per Kelompok")
            st.dataframe(res["group_df"], use_container_width=True, hide_index=True)
            st.subheader("Hasil Uji ANOVA & Kruskal-Wallis")
            st.dataframe(res["test_df"], use_container_width=True, hide_index=True)
        with c2:
            fig_box = px.box(
                res["clean_df"],
                x=res["group_col"],
                y=res["col"],
                color=res["group_col"],
                title=f"Perbandingan {res['col']} antar {res['group_col']}",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Uji Lanjut Post-Hoc Tukey HSD")
        st.dataframe(res["tukey_df"], use_container_width=True, hide_index=True)

        export_sheets["Statistik_Kelompok"] = res["group_df"]
        export_sheets["Hasil_Uji_Omnibus"] = res["test_df"]
        export_sheets["Post_Hoc_Tukey_HSD"] = res["tukey_df"]

    elif mod == "asosiasi":
        c1, c2 = st.columns([1.1, 1.3])
        with c1:
            st.subheader("Hasil Uji Chi-Square & Cramér's V")
            st.dataframe(res["test_df"], use_container_width=True, hide_index=True)
            st.subheader("Tabel Kontingensi (Frekuensi)")
            st.dataframe(res["crosstab_df"], use_container_width=True, hide_index=True)
        with c2:
            ct_long = res["crosstab"].reset_index().melt(id_vars=res["row_col"], var_name=res["col_col"], value_name="Jumlah Pasien")
            fig_bar = px.bar(
                ct_long,
                x=res["row_col"],
                y="Jumlah Pasien",
                color=res["col_col"],
                barmode="group",
                title=f"Grafik Batang Berkelompok: {res['row_col']} vs {res['col_col']}",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        export_sheets["Hasil_Chi_Square"] = res["test_df"]
        export_sheets["Tabel_Kontingensi"] = res["crosstab_df"]

    elif mod == "korelasi_regresi":
        if res["sub_mode"] == "korelasi":
            c1, c2 = st.columns([1.1, 1.2])
            with c1:
                st.subheader(f"Tabel Pasangan Korelasi ({res['method']})")
                st.dataframe(res["pairs_df"], use_container_width=True, hide_index=True)
            with c2:
                fig_heat = px.imshow(
                    res["corr_matrix"],
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=f"Heatmap Korelasi {res['method']}",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            export_sheets["Matriks_Korelasi"] = res["corr_matrix"].reset_index()
            export_sheets["Pasangan_Korelasi"] = res["pairs_df"]
        else:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.subheader("Ringkasan Model Regresi")
                st.dataframe(res["summary_df"], use_container_width=True, hide_index=True)
            with c2:
                fig_res = px.scatter(
                    x=res["fitted"],
                    y=res["residuals"],
                    labels={"x": f"Nilai Prediksi ({res['target']})", "y": "Residual"},
                    title="Plot Residual vs Nilai Prediksi",
                    opacity=0.5,
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="#991b1b")
                st.plotly_chart(fig_res, use_container_width=True)

            st.subheader("Koefisien Regresi Linear Berganda, Std. Beta & VIF")
            st.dataframe(res["coef_df"], use_container_width=True, hide_index=True)

            st.subheader("Statistik Residual")
            st.dataframe(res["resid_stats_df"], use_container_width=True, hide_index=True)

            export_sheets["Ringkasan_Regresi"] = res["summary_df"]
            export_sheets["Koefisien_Regresi"] = res["coef_df"]
            export_sheets["Statistik_Residual"] = res["resid_stats_df"]

    excel_bytes = export_results_to_excel(export_sheets)
    st.download_button(
        label="Unduh Laporan Hasil Analisis (.xlsx)",
        data=excel_bytes,
        file_name=f"CardioStat_Lab_{mod}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def render_dashboard(df: pd.DataFrame) -> None:
    """Render interactive clinical executive dashboard with global filters, KPIs, and Plotly charts."""
    st.markdown('<span class="stage-badge">Dashboard Ringkasan Klinis</span>', unsafe_allow_html=True)

    filtered = df.copy()
    with st.container(border=True):
        st.markdown("**Filter Interaktif Populasi**")
        f_cols = st.columns(4)

        if "Continent" in filtered.columns:
            continents = sorted(filtered["Continent"].dropna().astype(str).unique())
            sel_cont = f_cols[0].multiselect("Benua (Continent)", continents, default=continents)
            if sel_cont:
                filtered = filtered[filtered["Continent"].astype(str).isin(sel_cont)]

        if "Sex" in filtered.columns:
            sexes = sorted(filtered["Sex"].dropna().astype(str).unique())
            sel_sex = f_cols[1].multiselect("Jenis Kelamin (Sex)", sexes, default=sexes)
            if sel_sex:
                filtered = filtered[filtered["Sex"].astype(str).isin(sel_sex)]

        if "Diet" in filtered.columns:
            diets = sorted(filtered["Diet"].dropna().astype(str).unique())
            sel_diet = f_cols[2].multiselect("Pola Diet (Diet)", diets, default=diets)
            if sel_diet:
                filtered = filtered[filtered["Diet"].astype(str).isin(sel_diet)]

        if "Age" in filtered.columns:
            age_num = pd.to_numeric(filtered["Age"], errors="coerce").dropna()
            if not age_num.empty and float(age_num.min()) < float(age_num.max()):
                min_a, max_a = int(age_num.min()), int(age_num.max())
                sel_age = f_cols[3].slider("Rentang Usia (Age)", min_a, max_a, (min_a, max_a))
                filtered = filtered[pd.to_numeric(filtered["Age"], errors="coerce").between(sel_age[0], sel_age[1])]

    if filtered.empty:
        st.warning("Tidak ada data pasien yang cocok dengan kombinasi filter saat ini.")
        return

    st.caption(f"Menampilkan **{len(filtered):,}** dari **{len(df):,}** pasien.")

    # Row 1: KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Pasien", f"{len(filtered):,}")

    if "Heart Attack Risk" in filtered.columns:
        risk_series = pd.to_numeric(filtered["Heart Attack Risk"], errors="coerce").dropna()
        risk_count = int((risk_series == 1).sum())
        risk_pct = float(risk_series.mean() * 100) if not risk_series.empty else 0.0
        k2.metric("Berisiko Serangan Jantung", f"{risk_pct:.2f}%", f"{risk_count:,} pasien", delta_color="off")
    else:
        k2.metric("Jumlah Variabel", f"{filtered.shape[1]}")

    if "Cholesterol" in filtered.columns:
        chol_mean = pd.to_numeric(filtered["Cholesterol"], errors="coerce").mean()
        k3.metric("Rata-rata Kolesterol", f"{chol_mean:.1f} mg/dL")
    else:
        k3.metric("Baris Lengkap", f"{int(filtered.dropna().shape[0]):,}")

    if "Systolic_BP" in filtered.columns and "Diastolic_BP" in filtered.columns:
        sys_m = pd.to_numeric(filtered["Systolic_BP"], errors="coerce").mean()
        dia_m = pd.to_numeric(filtered["Diastolic_BP"], errors="coerce").mean()
        k4.metric("Rata-rata Tekanan Darah", f"{sys_m:.0f}/{dia_m:.0f} mmHg")
    elif "Age" in filtered.columns:
        k4.metric("Rata-rata Usia", f"{pd.to_numeric(filtered['Age'], errors='coerce').mean():.1f} thn")

    comorb_parts = []
    for col_name, label in [("Smoking", "Merokok"), ("Diabetes", "Diabetes"), ("Obesity", "Obesitas")]:
        if col_name in filtered.columns:
            pct = pd.to_numeric(filtered[col_name], errors="coerce").mean() * 100
            comorb_parts.append(f"{label} {pct:.0f}%")
    if comorb_parts:
        k5.metric("Faktor Risiko Utama", comorb_parts[0], " | ".join(comorb_parts[1:]), delta_color="off")
    else:
        k5.metric("Missing Values", f"{int(filtered.isna().sum().sum()):,}")

    st.divider()

    # Row 2: Profil Risiko & Biomarker
    r1_left, r1_right = st.columns([1, 1.35])
    with r1_left:
        if "Heart Attack Risk" in filtered.columns:
            risk_counts = (
                filtered["Heart Attack Risk"]
                .map({0: "Tidak Berisiko (0)", 1: "Berisiko (1)"})
                .fillna(filtered["Heart Attack Risk"].astype(str))
                .value_counts()
                .reset_index()
            )
            risk_counts.columns = ["Status Risiko", "Jumlah"]
            fig_donut = px.pie(
                risk_counts,
                names="Status Risiko",
                values="Jumlah",
                hole=0.52,
                color="Status Risiko",
                color_discrete_map={"Tidak Berisiko (0)": "#2563eb", "Berisiko (1)": "#991b1b"},
                title="Proporsi Risiko Serangan Jantung",
            )
            fig_donut.update_traces(textinfo="percent+value")
            st.plotly_chart(fig_donut, use_container_width=True)

    with r1_right:
        cut_rows = []
        for param in ["Cholesterol", "Systolic_BP", "BMI"]:
            if param in filtered.columns and param in CLINICAL_CUTOFFS:
                cfg = CLINICAL_CUTOFFS[param]
                s = pd.to_numeric(filtered[param], errors="coerce").dropna()
                cats = pd.cut(s, bins=cfg["bins"], labels=["Rendah / Normal", "Sedang / Borderline", "Tinggi"])
                vc = cats.value_counts(sort=False)
                for cat_name, cnt in vc.items():
                    cut_rows.append(
                        {
                            "Parameter": param,
                            "Kategori Klinis": str(cat_name),
                            "Persentase (%)": round(float(cnt / len(s) * 100), 2) if len(s) > 0 else 0.0,
                        }
                    )
        if cut_rows:
            cut_df = pd.DataFrame(cut_rows)
            fig_cut = px.bar(
                cut_df,
                x="Parameter",
                y="Persentase (%)",
                color="Kategori Klinis",
                barmode="stack",
                text_auto=".1f",
                color_discrete_map={
                    "Rendah / Normal": "#16a34a",
                    "Sedang / Borderline": "#f59e0b",
                    "Tinggi": "#991b1b",
                },
                title="Profil Kategori Risiko Klinis Biomarker (%)",
            )
            st.plotly_chart(fig_cut, use_container_width=True)

    # Row 3: Komparasi Kelompok & Wilayah
    r2_left, r2_right = st.columns(2)
    with r2_left:
        if "Heart Attack Risk" in filtered.columns:
            dim_opts = [c for c in ["Continent", "Diet", "Sex"] if c in filtered.columns]
            if dim_opts:
                sel_dim = st.selectbox("Bandingkan Prevalensi Risiko (%) menurut:", dim_opts, key="dash_dim")
                grp_risk = (
                    filtered.assign(_risk=pd.to_numeric(filtered["Heart Attack Risk"], errors="coerce"))
                    .groupby(sel_dim, dropna=True)["_risk"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                grp_risk["Prevalensi (%)"] = np.round(grp_risk["mean"] * 100, 2)
                fig_grp = px.bar(
                    grp_risk.sort_values("Prevalensi (%)", ascending=False),
                    x=sel_dim,
                    y="Prevalensi (%)",
                    color="Prevalensi (%)",
                    text="Prevalensi (%)",
                    color_continuous_scale="Reds",
                    title=f"Tingkat Risiko Serangan Jantung menurut {sel_dim}",
                )
                fig_grp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_grp.update_yaxes(range=[0, max(5.0, float(grp_risk["Prevalensi (%)"].max()) * 1.2)])
                st.plotly_chart(fig_grp, use_container_width=True)

    with r2_right:
        bio_opts = [c for c in ["Cholesterol", "Systolic_BP", "Diastolic_BP", "BMI", "Triglycerides", "Age"] if c in filtered.columns]
        if bio_opts and "Heart Attack Risk" in filtered.columns:
            sel_bio = st.selectbox("Bandingkan Sebaran Biomarker:", bio_opts, key="dash_bio")
            box_df = filtered[[sel_bio, "Heart Attack Risk"]].dropna().copy()
            box_df["Status Risiko"] = box_df["Heart Attack Risk"].map({0: "Tidak Berisiko (0)", 1: "Berisiko (1)"}).fillna(
                box_df["Heart Attack Risk"].astype(str)
            )
            fig_bio = px.box(
                box_df,
                x="Status Risiko",
                y=sel_bio,
                color="Status Risiko",
                color_discrete_map={"Tidak Berisiko (0)": "#2563eb", "Berisiko (1)": "#991b1b"},
                title=f"Sebaran {sel_bio} berdasarkan Status Risiko",
            )
            st.plotly_chart(fig_bio, use_container_width=True)

    # Row 4: Faktor Gaya Hidup & Heatmap Korelasi
    r3_left, r3_right = st.columns(2)
    with r3_left:
        if "Heart Attack Risk" in filtered.columns:
            factor_cols = [c for c in ["Smoking", "Diabetes", "Obesity", "Family History", "Alcohol Consumption"] if c in filtered.columns]
            factor_rows = []
            for fc in factor_cols:
                tmp = filtered[[fc, "Heart Attack Risk"]].dropna()
                for val in sorted(tmp[fc].unique()):
                    sub = tmp[tmp[fc] == val]
                    pct = float(pd.to_numeric(sub["Heart Attack Risk"], errors="coerce").mean() * 100)
                    factor_rows.append(
                        {
                            "Faktor Risiko": fc,
                            "Kondisi": "Ya (1)" if val == 1 else ("Tidak (0)" if val == 0 else str(val)),
                            "Prevalensi Risiko (%)": round(pct, 2),
                        }
                    )
            if factor_rows:
                fdf = pd.DataFrame(factor_rows)
                fig_fac = px.bar(
                    fdf,
                    y="Faktor Risiko",
                    x="Prevalensi Risiko (%)",
                    color="Kondisi",
                    barmode="group",
                    orientation="h",
                    text_auto=".1f",
                    color_discrete_map={"Ya (1)": "#991b1b", "Tidak (0)": "#64748b"},
                    title="Prevalensi Serangan Jantung per Faktor Gaya Hidup & Riwayat",
                )
                st.plotly_chart(fig_fac, use_container_width=True)

    with r3_right:
        heat_cols = [
            c
            for c in ["Age", "Cholesterol", "Systolic_BP", "Diastolic_BP", "BMI", "Triglycerides", "Stress Level", "Sleep Hours Per Day"]
            if c in filtered.columns
        ]
        if len(heat_cols) >= 2:
            corr_mat = filtered[heat_cols].apply(pd.to_numeric, errors="coerce").corr()
            fig_heat = px.imshow(
                corr_mat,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Heatmap Korelasi Parameter Klinis & Gaya Hidup",
            )
            st.plotly_chart(fig_heat, use_container_width=True)


def main() -> None:
    init_session_state()
    render_header()
    render_stage_1_upload()

    df = st.session_state.df
    if df is None:
        st.info("Silakan unggah dataset pasien (.csv/.xlsx) atau klik **Muat Dataset Pasien Jantung** untuk memulai.")
        return

    tab_dash, tab_lab = st.tabs(["Dashboard Ringkasan Klinis", "Laboratorium Uji Statistik"])
    with tab_dash:
        render_dashboard(df)
    with tab_lab:
        render_stage_2_preview(df)
        render_stage_3_navbar()
        col_meta = classify_columns(df)
        render_stage_4_and_5(df, col_meta)


if __name__ == "__main__":
    main()
