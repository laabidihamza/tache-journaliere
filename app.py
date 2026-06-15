# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import io
import gc
from datetime import datetime, timedelta
from typing import Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Constants & configuration
# =========================

DATE_COL = "Date"
TICKET_ID_COL = "Référence du ticket"
EXCEPTION_COL = "Exception"

st.set_page_config(
    page_title="Analyse quotidienne des tickets",
    layout="wide",
)


# =========================
# Utility functions
# =========================

def parse_date_input(date_input) -> pd.Timestamp:
    return pd.to_datetime(date_input).normalize()


def compute_j_minus_1(j_date: pd.Timestamp) -> pd.Timestamp:
    """
    Business rules for j-1:
    - If j = 16/12/2025 → j-1 = 13/12/2025
    - If j = 01/02/2026 → j-1 = 29/01/2026
    - Otherwise: j-1 = j - 1 day
    """
    if j_date == pd.Timestamp("2025-12-16"):
        return pd.Timestamp("2025-12-13")
    if j_date == pd.Timestamp("2026-02-01"):
        return pd.Timestamp("2026-01-29")
    return j_date - pd.Timedelta(days=1)

@st.cache_data(show_spinner=False)
def load_data_from_excel(uploaded_file) -> pd.DataFrame:
    """Load and cache data from Excel with advanced memory optimization."""
    try:
        # 1. Read only the necessary columns to save massive amounts of RAM
        required_cols = [DATE_COL, TICKET_ID_COL, EXCEPTION_COL]
        
        # 2. Use the 'calamine' engine which is much lighter on memory than openpyxl
        df = pd.read_excel(
            uploaded_file,
            engine="calamine",
            usecols=lambda x: x in required_cols,
            dtype={TICKET_ID_COL: str, EXCEPTION_COL: str}
        )
        
        if DATE_COL not in df.columns:
            raise ValueError(f"La colonne obligatoire '{DATE_COL}' est manquante.")
        
        # 3. Convert date column to datetime efficiently
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        
        # 4. Remove rows with missing dates IMMEDIATELY to free memory
        # df = df.dropna(subset=[DATE_COL])
        
        # 5. Normalize dates (remove time component)
        df[DATE_COL] = df[DATE_COL].dt.normalize()
        
        # 6. Drop complete duplicate rows early to reduce memory footprint
        # df = df.drop_duplicates().reset_index(drop=True)
        
        # 7. Sort by date for better performance
        df = df.reset_index(drop=True)
        
        # 8. Force Python to release unreferenced memory blocks immediately
        gc.collect()
        
        return df
    except MemoryError as e:
        raise ValueError(
            "Le fichier est trop volumineux pour être traité. "
            "Erreur mémoire : " + str(e)
        )
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier Excel : {str(e)}")

def validate_columns(df: pd.DataFrame) -> Optional[str]:
    missing = [c for c in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL] if c not in df.columns]
    if missing:
        return "Les colonnes suivantes sont manquantes : " + ", ".join(missing)
    return None


# =========================
# Business logic
# =========================

def compute_ticket_sets(
    df: pd.DataFrame, date_j1: pd.Timestamp, date_j: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickets_j1 = df[df[DATE_COL] == date_j1].copy()
    tickets_j = df[df[DATE_COL] == date_j].copy()
    return tickets_j1, tickets_j


def compute_synthesis(
    tickets_j1: pd.DataFrame,
    tickets_j: pd.DataFrame,
    date_j1: pd.Timestamp,
    date_j: pd.Timestamp,
) -> pd.DataFrame:
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    set_j = set(tickets_j[TICKET_ID_COL].astype(str))
    treated_ids = set_j1 - set_j
    new_ids = set_j - set_j1
    return pd.DataFrame({
        "Date": [date_j.normalize()],
        "Nombre des cas traités à la date j": [len(treated_ids)],
        "Nombre des nouveaux cas à la date j": [len(new_ids)],
        "Nombre des tickets à la date j": [len(set_j)],
    })


@st.cache_data(show_spinner=False)
def compute_synthesis_all_dates(df: pd.DataFrame, shift_metrics_by_one_day: bool = True) -> pd.DataFrame:
    """Precomputed cached synthesis for all dates."""
    if df.empty:
        return pd.DataFrame(columns=[
            "Date",
            "Nombre des cas traités à la date j",
            "Nombre des nouveaux cas à la date j",
            "Nombre des tickets à la date j",
        ])

    data = df.copy()
    # Dates are already normalized from load_data_from_excel
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce").dt.normalize()
    data = data.dropna(subset=[DATE_COL])
    data[TICKET_ID_COL] = data[TICKET_ID_COL].astype(str)

    groups = (
        data.groupby(DATE_COL)[TICKET_ID_COL]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    rows = []
    for date_j in sorted(groups.keys()):
        set_j = groups.get(date_j, set())
        date_j1 = pd.to_datetime(compute_j_minus_1(date_j)).normalize()
        set_j1 = groups.get(date_j1, set())
        treated_ids = set_j1 - set_j
        new_ids = set_j - set_j1
        rows.append({
            "Date": date_j,
            "Nombre des cas traités à la date j": len(treated_ids),
            "Nombre des nouveaux cas à la date j": len(new_ids),
            "Nombre des tickets à la date j": len(set_j),
        })

    result = pd.DataFrame(rows).sort_values(by="Date").reset_index(drop=True)

    if shift_metrics_by_one_day and len(result) > 1:
        result["Nombre des cas traités à la date j"] = result["Nombre des cas traités à la date j"].shift(-1)
        result["Nombre des nouveaux cas à la date j"] = result["Nombre des nouveaux cas à la date j"].shift(-1)
        result = result.iloc[:-1].reset_index(drop=True)

    return result


def compute_new_tickets(tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame) -> pd.DataFrame:
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    tickets_j = tickets_j.copy()
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)
    new_tickets = tickets_j[~tickets_j[TICKET_ID_COL].isin(set_j1)].copy()

    if new_tickets.empty:
        return new_tickets

    if EXCEPTION_COL in new_tickets.columns:
        exc_counts = new_tickets[EXCEPTION_COL].value_counts().rename("Exception_Count").to_frame()
        new_tickets = new_tickets.merge(exc_counts, left_on=EXCEPTION_COL, right_index=True, how="left")
        new_tickets = new_tickets.sort_values(by=["Exception_Count", DATE_COL], ascending=[False, True])
        new_tickets = new_tickets.drop(columns=["Exception_Count"])
    else:
        new_tickets = new_tickets.sort_values(by=DATE_COL)

    return new_tickets.reset_index(drop=True)


def compute_treated_tickets(tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame) -> pd.DataFrame:
    tickets_j1 = tickets_j1.copy()
    tickets_j = tickets_j.copy()
    tickets_j1[TICKET_ID_COL] = tickets_j1[TICKET_ID_COL].astype(str)
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)
    set_j = set(tickets_j[TICKET_ID_COL])
    treated = tickets_j1[~tickets_j1[TICKET_ID_COL].isin(set_j)].copy()
    return treated.reset_index(drop=True)


def generate_excel_bytes(synthese_df, nouveaux_df, traites_df) -> bytes:
    """Generate Excel with multiple sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        synthese_df.to_excel(writer, sheet_name="Synthèse", index=False)
        nouveaux_df.to_excel(writer, sheet_name="Nouveaux tickets", index=False)
        traites_df.to_excel(writer, sheet_name="Tickets traités", index=False)
    output.seek(0)
    return output.getvalue()


def generate_single_sheet_excel(df: pd.DataFrame, sheet_name: str = "Données") -> bytes:
    """Generate single sheet Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def prepare_dashboard_data(df: pd.DataFrame):
    """Precompute dashboard aggregations."""
    df_daily = (
        df.groupby(DATE_COL)[TICKET_ID_COL]
        .nunique()
        .reset_index()
        .rename(columns={TICKET_ID_COL: "Nombre de tickets"})
    )

    # Efficient duplicate handling using drop_duplicates
    # Keep DATE_COL for later daily evolution analysis
    df_pie = (
        df[[TICKET_ID_COL, EXCEPTION_COL, DATE_COL]]
        .drop_duplicates(subset=[TICKET_ID_COL], keep="last")
    )

    exception_counts = (
        df_pie[EXCEPTION_COL]
        .value_counts()
        .reset_index(name="Nombre")
    )
    if EXCEPTION_COL not in exception_counts.columns:
        exception_counts.columns = [EXCEPTION_COL, "Nombre"]

    return df_daily, exception_counts, df_pie


# =========================
# Dashboard / Visualisations
# =========================

def build_dashboard(df: pd.DataFrame):
    if df.empty:
        st.info("Aucune donnée disponible pour le dashboard.")
        return

    # Use cached precomputed data
    df_daily, exception_counts, df_pie = prepare_dashboard_data(df)

    st.subheader("Évolution du nombre total de tickets par date")
    fig_daily = px.line(df_daily, x=DATE_COL, y="Nombre de tickets", markers=True)
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("Nombre de tickets par date")
    fig_bar = px.bar(df_daily, x=DATE_COL, y="Nombre de tickets")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Top 10 des exceptions (camembert)")
    if EXCEPTION_COL not in df.columns:
        st.info(f"La colonne '{EXCEPTION_COL}' n'existe pas dans les données.")
        return

    if exception_counts.empty:
        st.info("Aucune exception à afficher.")
        return

    top_exceptions = exception_counts.head(10).copy()
    max_len = 60
    top_exceptions["Exception_courte"] = top_exceptions[EXCEPTION_COL].astype(str).str.slice(0, max_len)
    mask = top_exceptions[EXCEPTION_COL].str.len() > max_len
    top_exceptions.loc[mask, "Exception_courte"] = top_exceptions.loc[mask, "Exception_courte"] + "..."

    fig_exceptions = px.pie(top_exceptions, names="Exception_courte", values="Nombre")
    fig_exceptions.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=10)))
    st.plotly_chart(fig_exceptions, use_container_width=True)

    st.markdown("### Détail des Top 10 exceptions")
    st.dataframe(
        top_exceptions[[EXCEPTION_COL, "Nombre"]].rename(
            columns={EXCEPTION_COL: "Exception", "Nombre": "Nombre d'occurrences"}
        ),
        use_container_width=True,
    )

    # Évolution journalière des Top 10 exceptions
    st.subheader("Évolution journalière des Top 10 exceptions")

    top_10_names = top_exceptions[EXCEPTION_COL].tolist()
    df_top_exc_time = df_pie[df_pie[EXCEPTION_COL].isin(top_10_names)].copy()
    df_top_exc_time[DATE_COL] = pd.to_datetime(df_top_exc_time[DATE_COL]).dt.normalize()
    df_top_exc_time["Exception_courte"] = df_top_exc_time[EXCEPTION_COL].astype(str).str.slice(0, max_len)
    mask2 = df_top_exc_time[EXCEPTION_COL].str.len() > max_len
    df_top_exc_time.loc[mask2, "Exception_courte"] = df_top_exc_time.loc[mask2, "Exception_courte"] + "..."

    daily_exception_counts = (
        df_top_exc_time.groupby([DATE_COL, "Exception_courte"]).size().reset_index(name="Nombre d'occurrences")
    )
    available_exceptions = sorted(daily_exception_counts["Exception_courte"].unique().tolist())

    st.markdown("**Sélectionnez les exceptions à afficher :**")
    # OPTIMIZATION: Replace checkboxes with multiselect
    selected_exceptions = st.multiselect(
        "Exceptions",
        options=available_exceptions,
        default=available_exceptions,
        label_visibility="collapsed",
        key="dashboard_exc_select",
    )

    if not selected_exceptions:
        st.info("Aucune exception sélectionnée.")
        return

    total_occ = int(
        daily_exception_counts[daily_exception_counts["Exception_courte"].isin(selected_exceptions)][
            "Nombre d'occurrences"
        ].sum()
    )
    df_plot = daily_exception_counts[daily_exception_counts["Exception_courte"].isin(selected_exceptions)].copy()

    fig_exc_trend = px.line(
        df_plot,
        x=DATE_COL,
        y="Nombre d'occurrences",
        color="Exception_courte",
        markers=True,
        log_y=True,
    )
    fig_exc_trend.update_layout(
        legend_title_text="Exception",
        legend=dict(font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_exc_trend, use_container_width=True)
    st.metric("Occurrences totales sélectionnées", total_occ)


# =========================
# Main Streamlit interface
# =========================

def main():
    st.title("Analyse quotidienne des tickets")
    st.markdown("Cette application permet d'analyser quotidiennement les tickets à partir d'un fichier Excel.")

    tab_analyse, tab_dashboard, tab_exceptions = st.tabs(
        ["Analyse & Résultats", "Dashboard", "Exceptions"]
    )

    # Session state initialisation
    for key in ["df_source", "synthese_df", "nouveaux_df", "traites_df", "excel_bytes"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "selected_exceptions" not in st.session_state:
        st.session_state.selected_exceptions = []

    # ==============================
    # Tab 1: Analyse & Résultats
    # ==============================
    with tab_analyse:
        st.header("Analyse & Résultats")

        uploaded_file = st.file_uploader("Uploader un fichier Excel", type=["xlsx", "xls"])

        today = datetime.today().date()
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_input_1 = st.date_input("Date 1 (j-1)", value=today - timedelta(days=1), key="date_j1")
        with col_date2:
            date_input_2 = st.date_input("Date 2 (j)", value=today, key="date_j")

        run_analysis = st.button("Lancer l'analyse", type="primary")

        df = None
        if uploaded_file is not None:
            st.subheader("Aperçu du fichier Excel")
            try:
                df = load_data_from_excel(uploaded_file)
                if df.empty:
                    st.error("Le fichier ne contient aucune donnée après nettoyage.")
                    df = None
                else:
                    st.dataframe(df.head(50), use_container_width=True)
                    st.session_state.df_source = df
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

        if run_analysis:
            try:
                date_j = parse_date_input(date_input_2)
                computed_j1 = compute_j_minus_1(date_j)
                date_j1 = parse_date_input(date_input_1)
            except Exception:
                st.error("Les dates fournies sont invalides.")
                st.stop()

            if df is None:
                st.error("Aucun fichier valide n'a été chargé.")
                st.stop()

            error_cols = validate_columns(df)
            if error_cols:
                st.error(error_cols)
                st.stop()

            if date_j1 != computed_j1:
                st.warning(
                    f"Attention : pour j = {date_j.date()}, la règle métier attend j-1 = {computed_j1.date()}. "
                    f"Vous avez saisi j-1 = {date_j1.date()}."
                )

            tickets_j1, tickets_j = compute_ticket_sets(df, date_j1, date_j)

            if tickets_j.empty and tickets_j1.empty:
                st.error(
                    "Aucun ticket trouvé pour les dates sélectionnées. "
                    "Vérifiez que les dates existent dans la colonne 'Date'."
                )
                st.stop()

            synthese_j_df = compute_synthesis(tickets_j1, tickets_j, date_j1, date_j)
            synthese_all_df = compute_synthesis_all_dates(df)
            nouveaux_df = compute_new_tickets(tickets_j1, tickets_j)
            traites_df = compute_treated_tickets(tickets_j1, tickets_j)

            st.session_state.df_source = df
            st.session_state.synthese_df = synthese_all_df
            st.session_state.nouveaux_df = nouveaux_df
            st.session_state.traites_df = traites_df

            # Summary metrics
            st.subheader("Synthèse")
            if not synthese_j_df.empty:
                row = synthese_j_df.iloc[0]
                try:
                    date_str = row["Date"].strftime("%Y-%m-%d")
                except Exception:
                    date_str = str(row["Date"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Date j", date_str)
                with col2:
                    st.metric("Cas traités", int(row["Nombre des cas traités à la date j"]))
                with col3:
                    st.metric("Nouveaux cas", int(row["Nombre des nouveaux cas à la date j"]))
                with col4:
                    st.metric("Total tickets", int(row["Nombre des tickets à la date j"]))

            st.subheader("Synthèse complète (toutes dates)")
            st.dataframe(synthese_all_df, use_container_width=True)

            st.subheader("Nouveaux tickets (présents à j, absents à j-1)")
            if nouveaux_df.empty:
                st.info("Aucun nouveau ticket pour la période sélectionnée.")
            else:
                st.dataframe(nouveaux_df, use_container_width=True)

            st.subheader("Tickets traités (présents à j-1, absents à j)")
            if traites_df.empty:
                st.info("Aucun ticket traité pour la période sélectionnée.")
            else:
                st.dataframe(traites_df, use_container_width=True)

            # Sort before export
            if EXCEPTION_COL in nouveaux_df.columns:
                nouveaux_df = nouveaux_df.sort_values(by=EXCEPTION_COL, ascending=True)
            if EXCEPTION_COL in traites_df.columns:
                traites_df = traites_df.sort_values(by=EXCEPTION_COL, ascending=True)

            excel_bytes = generate_excel_bytes(synthese_all_df, nouveaux_df, traites_df)
            st.session_state.excel_bytes = excel_bytes

            st.download_button(
                label="Télécharger le fichier Excel de résultats",
                data=excel_bytes,
                file_name=f"analyse_tickets_{date_j.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ==============================
    # Tab 2: Dashboard
    # ==============================
    with tab_dashboard:
        st.header("Dashboard")
        if st.session_state.df_source is None:
            st.info("Veuillez d'abord charger un fichier dans l'onglet 'Analyse & Résultats'.")
        else:
            build_dashboard(st.session_state.df_source)

    # ==============================
    # Tab 3: Exceptions
    # ==============================
    with tab_exceptions:
        st.header("Exceptions")
        if st.session_state.df_source is None:
            st.info("Veuillez d'abord charger un fichier dans l'onglet 'Analyse & Résultats'.")
        else:
            df_exc = st.session_state.df_source.copy()

            # Validate required columns
            missing_cols = [c for c in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL] if c not in df_exc.columns]
            if missing_cols:
                st.error("Colonnes manquantes : " + ", ".join(missing_cols))
                st.stop()

            # Dates already normalized from load_data_from_excel, just ensure consistency
            df_exc[DATE_COL] = pd.to_datetime(df_exc[DATE_COL], errors="coerce")
            df_exc = df_exc.dropna(subset=[DATE_COL]).copy()
            df_exc[TICKET_ID_COL] = df_exc[TICKET_ID_COL].astype(str)

            # Stats
            nb_refs = int(df_exc[TICKET_ID_COL].nunique())
            nb_dupes = int(df_exc.duplicated(subset=[TICKET_ID_COL], keep="last").sum())

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Nombre de références distinctes", nb_refs)
            with m2:
                st.metric("Lignes doublons supprimées", nb_dupes)

            df_exc = df_exc.sort_values(by=[DATE_COL]).reset_index(drop=True)

            drop_dupes = st.toggle("Drop Duplicates")
            if drop_dupes:
                df_exc = df_exc.drop_duplicates(subset=[TICKET_ID_COL], keep="last")

            st.markdown(
                "Sélectionnez une période pour analyser les exceptions distinctes "
                "et leur nombre total d'occurrences."
            )

            min_date = df_exc[DATE_COL].min()
            max_date = df_exc[DATE_COL].max()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input(
                    "Date de début",
                    value=min_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="exc_start_date",
                )
            with col_d2:
                end_date = st.date_input(
                    "Date de fin",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="exc_end_date",
                )

            try:
                start_ts = pd.to_datetime(start_date).normalize()
                end_ts = pd.to_datetime(end_date).normalize()
            except Exception:
                st.error("Les dates sélectionnées sont invalides.")
                st.stop()

            if start_ts > end_ts:
                st.error("La date de début doit être inférieure ou égale à la date de fin.")
            else:
                mask_period = (df_exc[DATE_COL] >= start_ts) & (df_exc[DATE_COL] <= end_ts)
                df_period = df_exc[mask_period].copy()

                if df_period.empty:
                    st.info("Aucune donnée d'exception pour la période sélectionnée.")
                else:
                    # Pivot table: exception × date
                    pivot = df_period.pivot_table(
                        index=EXCEPTION_COL,
                        columns=DATE_COL,
                        aggfunc="size",
                        fill_value=0,
                    )
                    pivot["Total"] = pivot.sum(axis=1)

                    # Format date columns to DD-MM
                    new_cols = []
                    for col in pivot.columns:
                        if col == "Total":
                            new_cols.append(col)
                        else:
                            try:
                                new_cols.append(pd.to_datetime(col).strftime("%d-%m"))
                            except Exception:
                                new_cols.append(str(col))
                    pivot.columns = new_cols

                    exceptions_stats = (
                        pivot.sort_values("Total", ascending=False)
                        .reset_index()
                        .rename(columns={EXCEPTION_COL: EXCEPTION_COL})
                    )

                    st.subheader("Exceptions distinctes sur la période sélectionnée")
                    st.dataframe(exceptions_stats, use_container_width=True)

                    # Multiselect for custom table
                    selected = st.multiselect(
                        "Sélectionnez les exceptions à ajouter au tableau",
                        options=exceptions_stats[EXCEPTION_COL].tolist(),
                        default=st.session_state.selected_exceptions,
                        key="exc_multiselect",
                    )
                    st.session_state.selected_exceptions = selected

                    selected_exceptions_df = (
                        exceptions_stats[exceptions_stats[EXCEPTION_COL].isin(selected)].copy()
                        if selected else exceptions_stats.iloc[0:0].copy()
                    )

                    st.subheader("Exceptions sélectionnées")
                    st.dataframe(selected_exceptions_df, use_container_width=True)

                    if not selected_exceptions_df.empty:
                        st.download_button(
                            label="Télécharger les exceptions sélectionnées (Excel)",
                            data=generate_single_sheet_excel(selected_exceptions_df, "Exceptions Sélectionnées"),
                            file_name=f"exceptions_selectionnees_{start_ts.date()}_{end_ts.date()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_selected_exceptions",
                        )

                    # Top 10 exceptions chart
                    top_10 = pivot.nlargest(10, "Total")
                    max_len = 60
                    line_data = top_10.drop(columns="Total").T

                    short_names = []
                    for exc in line_data.columns:
                        s = exc if len(exc) <= max_len else exc[:max_len] + "..."
                        short_names.append(s)

                    counts = Counter(short_names)
                    seen = defaultdict(int)
                    unique_names = []
                    for name in short_names:
                        if counts[name] > 1:
                            seen[name] += 1
                            unique_names.append(f"{name} ({seen[name]})")
                        else:
                            unique_names.append(name)

                    exception_mapping = dict(zip(line_data.columns, unique_names))
                    line_data_short = line_data.rename(columns=exception_mapping)

                    st.markdown("**Sélectionnez les exceptions Top 10 à afficher :**")
                    # OPTIMIZATION: Replace checkboxes with multiselect
                    selected_short = st.multiselect(
                        "Exceptions",
                        options=unique_names,
                        default=unique_names,
                        label_visibility="collapsed",
                        key="exc_top10_multiselect",
                    )

                    if not selected_short:
                        st.info("Aucune exception sélectionnée pour le Top 10.")
                    else:
                        mapping_inv = {v: k for k, v in exception_mapping.items()}
                        selected_full = [mapping_inv[s] for s in selected_short if s in mapping_inv]

                        try:
                            total_selected_exc = int(top_10.loc[selected_full]["Total"].sum()) if selected_full else 0
                        except Exception:
                            total_selected_exc = 0

                        line_data_plot = line_data_short[selected_short].copy()

                        fig = go.Figure()
                        for col in line_data_plot.columns:
                            fig.add_trace(go.Scatter(
                                x=list(line_data_plot.index),
                                y=line_data_plot[col].values.tolist(),
                                mode="lines",
                                name=col,
                                hovertemplate=(
                                    "Exception=%{fullData.name}<br>"
                                    "Date=%{x}<br>"
                                    "Occurrences=%{y}<extra></extra>"
                                ),
                            ))

                        fig.update_layout(
                            margin=dict(l=0, r=0, t=50, b=0),
                            xaxis_title="Date",
                            yaxis_title="Number of Occurrences",
                            legend=dict(font=dict(size=10)),
                            height=500,
                        )

                        st.subheader("Évolution journalière des Top 10 exceptions")
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("Total des occurrences (Top10 sélectionnées)", total_selected_exc)

                    # Single exception daily evolution
                    st.subheader("Évolution quotidienne d'une exception sur la période sélectionnée")
                    exceptions_list = (
                        exceptions_stats[EXCEPTION_COL].dropna().astype(str).tolist()
                    )
                    if exceptions_list:
                        selected_exception = st.selectbox(
                            "Choisissez une exception",
                            options=exceptions_list,
                            key="exc_selected_exception",
                        )

                        df_exc_selected = df_period[df_period[EXCEPTION_COL] == selected_exception].copy()
                        df_exc_selected[DATE_COL] = pd.to_datetime(df_exc_selected[DATE_COL], errors="coerce").dt.normalize()

                        all_dates_range = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
                        daily_counts_series = (
                            df_exc_selected.groupby(DATE_COL).size().reindex(all_dates_range, fill_value=0)
                        )
                        daily_counts = daily_counts_series.reset_index()
                        daily_counts.columns = [DATE_COL, "Occurrences"]

                        if daily_counts["Occurrences"].sum() == 0:
                            st.info("Aucune occurrence de cette exception sur la période sélectionnée.")
                        else:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=daily_counts[DATE_COL],
                                y=daily_counts["Occurrences"],
                                mode="lines",
                                name=selected_exception,
                                hovertemplate=(
                                    "Exception=%{fullData.name}<br>"
                                    "Date=%{x}<br>"
                                    "Occurrences=%{y}<extra></extra>"
                                ),
                            ))
                            fig2.update_layout(
                                template="plotly_dark",
                                xaxis_title="Date",
                                yaxis_title="Number of Occurrences",
                                legend=dict(font=dict(size=10)),
                                margin=dict(l=0, r=0, t=40, b=0),
                                height=500,
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                    # Download full exceptions report
                    st.download_button(
                        label="Télécharger les exceptions (Excel)",
                        data=generate_single_sheet_excel(exceptions_stats, sheet_name="Exceptions"),
                        file_name=f"exceptions_{start_ts.date()}_{end_ts.date()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_report_1",
                    )


if __name__ == "__main__":
    main()
