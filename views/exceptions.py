from collections import Counter, defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from business_logic import generate_single_sheet_excel
from config import DATE_COL, TICKET_ID_COL, EXCEPTION_COL


@st.cache_data
def _prepare_exc_dataframe(df_source: pd.DataFrame, drop_duplicates: bool) -> pd.DataFrame:
    """Clean, sort and optionally deduplicate the source dataframe."""
    df = df_source.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df[TICKET_ID_COL] = df[TICKET_ID_COL].astype(str)
    df = df.sort_values(by=[DATE_COL]).reset_index(drop=True)
    if drop_duplicates:
        df = df.drop_duplicates(subset=[TICKET_ID_COL], keep="last")
    return df


@st.cache_data
def _build_pivot(df_period: pd.DataFrame):
    """Build pivot_dt (Timestamp columns) and pivot (DD-MM string columns)."""
    pivot_dt = df_period.pivot_table(
        index=EXCEPTION_COL,
        columns=DATE_COL,
        aggfunc="size",
        fill_value=0,
    )
    pivot_dt["Total"] = pivot_dt.sum(axis=1)

    pivot = pivot_dt.copy()
    new_columns = []
    for col in pivot.columns:
        if col == "Total":
            new_columns.append(col)
        else:
            try:
                new_columns.append(pd.to_datetime(col).strftime("%d-%m"))
            except Exception:
                new_columns.append(str(col))
    pivot.columns = new_columns
    return pivot_dt, pivot


@st.cache_data
def _daily_series_for_exception(
    df_period: pd.DataFrame,
    exception: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Compute the daily occurrence count for a single exception."""
    df_sel = df_period[df_period[EXCEPTION_COL] == exception].copy()
    df_sel[DATE_COL] = pd.to_datetime(df_sel[DATE_COL], errors="coerce").dt.normalize()
    all_dates = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
    series = df_sel.groupby(DATE_COL).size().reindex(all_dates, fill_value=0)
    result = series.reset_index()
    result.columns = [DATE_COL, "Occurrences"]
    return result


def render_exceptions(df_source: pd.DataFrame):
    """
    Onglet Exceptions : analyse des exceptions sur une période choisie,
    pivot journalier, Top 10 et évolution par exception individuelle.
    """
    df_exc = df_source.copy()

    # ── Vérification des colonnes ─────────────────────────────────────────────
    missing_cols = [
        col for col in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL]
        if col not in df_source.columns
    ]
    if missing_cols:
        st.error(
            "Les colonnes suivantes sont manquantes pour l'analyse des exceptions : "
            + ", ".join(missing_cols)
        )
        return

    # ── Métriques globales (sur la source brute, avant dédoublonnage) ─────────
    df_raw = df_source.copy()
    df_raw[TICKET_ID_COL] = df_raw[TICKET_ID_COL].astype(str)
    nb_refs_distinctes = int(df_raw[TICKET_ID_COL].nunique())
    nb_lignes_doublons_supprimees = int(
        df_raw.duplicated(subset=[TICKET_ID_COL], keep="last").sum()
    )
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Nombre de références distinctes", nb_refs_distinctes)
    with m2:
        st.metric("Lignes doublons supprimées", nb_lignes_doublons_supprimees)

    df_exc = _prepare_exc_dataframe(df_source, drop_duplicates=st.toggle("Drop Duplicates"))

    st.markdown(
        "Sélectionnez une période pour analyser les exceptions distinctes "
        "et leur nombre total d'occurrences (après dédoublonnage par référence)."
    )

    # ── Sélection de la période ───────────────────────────────────────────────
    min_date = df_exc[DATE_COL].min()
    max_date = df_exc[DATE_COL].max()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Date de début",
            value=min_date.date() if hasattr(min_date, "date") else None,
            min_value=min_date.date() if hasattr(min_date, "date") else None,
            max_value=max_date.date() if hasattr(max_date, "date") else None,
            key="exc_start_date",
        )
    with col_d2:
        end_date = st.date_input(
            "Date de fin",
            value=max_date.date() if hasattr(max_date, "date") else None,
            min_value=min_date.date() if hasattr(min_date, "date") else None,
            max_value=max_date.date() if hasattr(max_date, "date") else None,
            key="exc_end_date",
        )

    try:
        start_ts = pd.to_datetime(start_date).normalize()
        end_ts = pd.to_datetime(end_date).normalize()
    except Exception:
        st.error("Les dates sélectionnées pour l'analyse des exceptions sont invalides.")
        return

    if start_ts > end_ts:
        st.error("La date de début doit être inférieure ou égale à la date de fin.")
        return

    # ── Filtrage ──────────────────────────────────────────────────────────────
    df_period = df_exc[
        (df_exc[DATE_COL] >= start_ts) & (df_exc[DATE_COL] <= end_ts)
    ].copy()

    if df_period.empty:
        st.info(
            "Aucune donnée d'exception pour la période sélectionnée. "
            "Veuillez choisir une autre plage de dates."
        )
        return

    # ── Pivot par exception × date ────────────────────────────────────────────
    pivot_dt, pivot = _build_pivot(df_period)

    st.subheader("Exceptions distinctes sur la période sélectionnée")
    exceptions_stats = (
        pivot.sort_values("Total", ascending=False).reset_index()
    )
    st.dataframe(exceptions_stats)

    # ── Sélection et export des exceptions choisies ───────────────────────────
    if "selected_exceptions" not in st.session_state:
        st.session_state.selected_exceptions = []

    selected = st.multiselect(
        "Sélectionnez les exceptions à ajouter au tableau",
        options=exceptions_stats[EXCEPTION_COL].tolist(),
        default=st.session_state.selected_exceptions,
    )
    st.session_state.selected_exceptions = selected

    selected_exceptions_df = (
        exceptions_stats[exceptions_stats[EXCEPTION_COL].isin(selected)].copy()
        if selected
        else exceptions_stats.iloc[0:0].copy()
    )

    st.subheader("Exceptions sélectionnées")
    st.dataframe(selected_exceptions_df)

    if not selected_exceptions_df.empty:
        excel_selected_bytes = generate_single_sheet_excel(
            selected_exceptions_df,
            sheet_name="Exceptions Sélectionnées",
        )
        st.download_button(
            label="Télécharger les exceptions sélectionnées (Excel)",
            data=excel_selected_bytes,
            file_name=f"exceptions_selectionnees_{start_ts.date()}_{end_ts.date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_selected_exceptions",
        )

    # ── Top 10 exceptions – évolution journalière ─────────────────────────────
    # pivot_dt has real Timestamp columns → Plotly renders dates correctly
    top_10 = pivot_dt.nlargest(10, "Total")
    max_len = 60

    line_data = top_10.drop(columns="Total").T

    # Noms courts et uniques pour la légende
    short_names = [
        exc if len(exc) <= max_len else exc[:max_len] + "..."
        for exc in line_data.columns
    ]
    counts = Counter(short_names)
    seen: dict = defaultdict(int)
    unique_names = []
    for name in short_names:
        if counts[name] > 1:
            seen[name] += 1
            unique_names.append(f"{name} ({seen[name]})")
        else:
            unique_names.append(name)

    exception_mapping = dict(zip(line_data.columns, unique_names))
    line_data_short = line_data.rename(columns=exception_mapping)

    selected_short = []
    for i, short in enumerate(unique_names):
        if st.checkbox(short, value=True, key=f"exc_tab_cb_{i}"):
            selected_short.append(short)

    if not selected_short:
        st.info("Aucune exception sélectionnée pour le Top 10.")
    else:
        mapping_inv = {v: k for k, v in exception_mapping.items()}
        selected_full = [mapping_inv[s] for s in selected_short if s in mapping_inv]

        try:
            total_selected_exc = int(top_10.loc[selected_full]["Total"].sum())
        except Exception:
            total_selected_exc = 0

        line_data_plot = line_data_short[selected_short].copy()

        fig = go.Figure()
        for col in line_data_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(line_data_plot.index),
                    y=line_data_plot[col].values.tolist(),
                    mode="lines",
                    name=col,
                    hovertemplate=(
                        "Exception=%{fullData.name}<br>"
                        "Date=%{x}<br>"
                        "Occurrences=%{y}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis_title="Date",
            xaxis_type="date",
            yaxis_title="Number of Occurrences",
            height=500,
        )

        st.subheader("Évolution journalière des Top 10 exceptions")
        st.plotly_chart(fig, width="stretch")
        st.metric("Total des occurrences (Top10 sélectionnées)", total_selected_exc)

    # ── Évolution d'une exception individuelle ────────────────────────────────
    st.subheader("Évolution quotidienne d'une exception sur la période sélectionnée")

    exceptions_list = (
        exceptions_stats[EXCEPTION_COL].dropna().astype(str).tolist()
    )
    if not exceptions_list:
        return

    selected_exception = st.selectbox(
        "Choisissez une exception",
        options=exceptions_list,
        key="exc_selected_exception",
    )

    df_exc_selected = _daily_series_for_exception(df_period, selected_exception, start_ts, end_ts)
    daily_counts = df_exc_selected

    if daily_counts["Occurrences"].sum() == 0:
        st.info("Aucune occurrence de cette exception sur la période sélectionnée.")
        return

    fig_single = go.Figure()
    fig_single.add_trace(
        go.Scatter(
            x=daily_counts[DATE_COL],
            y=daily_counts["Occurrences"],
            mode="lines",
            name=selected_exception,
            hovertemplate=(
                "Exception=%{fullData.name}<br>"
                "Date=%{x}<br>"
                "Occurrences=%{y}<extra></extra>"
            ),
        )
    )
    fig_single.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Number of Occurrences",
        legend=dict(font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    st.plotly_chart(fig_single, width="stretch")

    # ── Export Excel toutes exceptions ────────────────────────────────────────
    excel_exc_bytes = generate_single_sheet_excel(exceptions_stats, sheet_name="Exceptions")
    st.download_button(
        label="Télécharger les exceptions (Excel)",
        data=excel_exc_bytes,
        file_name=f"exceptions_{start_ts.date()}_{end_ts.date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_report_1",
    )
