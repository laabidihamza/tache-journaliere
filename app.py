from datetime import datetime, timedelta

import streamlit as st

from business_logic import (
    compute_new_tickets,
    compute_synthesis,
    compute_synthesis_all_dates,
    compute_ticket_sets,
    compute_treated_tickets,
    generate_excel_bytes,
)
from config import EXCEPTION_COL
from data_loader import (
    compute_j_minus_1,
    load_data_from_excel,
    parse_date_input,
    validate_columns,
)
from views.dashboard import render_dashboard
from views.exceptions import render_exceptions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analyse quotidienne des tickets", layout="wide")


def main():
    st.title("Analyse quotidienne des tickets")
    st.markdown(
        "Cette application permet d'analyser quotidiennement les tickets "
        "à partir d'un fichier Excel."
    )

    tab_analyse, tab_dashboard, tab_exceptions = st.tabs(
        ["Analyse & Résultats", "Dashboard", "Exceptions"]
    )

    # ── Session state initialisation ──────────────────────────────────────────
    for key in ("df_source", "synthese_df", "nouveaux_df", "traites_df", "excel_bytes"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 1 – Analyse & Résultats
    # ══════════════════════════════════════════════════════════════════════════
    with tab_analyse:
        st.header("Analyse & Résultats")

        uploaded_file = st.file_uploader("Uploader un fichier Excel", type=["xlsx", "xls"])

        today = datetime.today().date()
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_input_1 = st.date_input("Date 1 (j-1)", value=today - timedelta(days=1), key="date_j1")
        with col_date2:
            date_input_2 = st.date_input("Date 2 (j)", value=today, key="date_j")

        run_analysis = st.button("Lancer l'analyse")

        df = None
        if uploaded_file is not None:
            st.subheader("Aperçu du fichier Excel")
            try:
                df = load_data_from_excel(uploaded_file)
                if df.empty:
                    st.error("Le fichier ne contient aucune donnée après nettoyage.")
                    df = None
                else:
                    st.dataframe(df.head(50))
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
                return

            if df is None:
                st.error("Aucun fichier valide n'a été chargé.")
                return

            error_cols = validate_columns(df)
            if error_cols:
                st.error(error_cols)
                return

            if date_j1 != computed_j1:
                st.warning(
                    f"Attention : selon les règles métiers, pour j = {date_j.date()}, "
                    f"la date j-1 attendue est {computed_j1.date()}. "
                    f"Vous avez saisi j-1 = {date_j1.date()}."
                )

            tickets_j1, tickets_j = compute_ticket_sets(df, date_j1, date_j)

            if tickets_j.empty and tickets_j1.empty:
                st.error(
                    "Aucun ticket trouvé pour les dates sélectionnées. "
                    "Vérifiez que les dates existent dans la colonne 'Date'."
                )
                return

            synthese_j_df = compute_synthesis(tickets_j1, tickets_j, date_j1, date_j)
            synthese_all_df = compute_synthesis_all_dates(df)
            nouveaux_df = compute_new_tickets(tickets_j1, tickets_j)
            traites_df = compute_treated_tickets(tickets_j1, tickets_j)

            st.session_state.df_source = df
            st.session_state.synthese_df = synthese_all_df
            st.session_state.nouveaux_df = nouveaux_df
            st.session_state.traites_df = traites_df

            # Métriques de synthèse
            st.subheader("Synthèse")
            if not synthese_j_df.empty:
                synth_row = synthese_j_df.iloc[0]
                try:
                    date_j_str = synth_row["Date"].strftime("%Y-%m-%d")
                except Exception:
                    date_j_str = str(synth_row["Date"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Date j", date_j_str)
                with col2:
                    st.metric("Cas traités à la date j", int(synth_row["Nombre des cas traités à la date j"]))
                with col3:
                    st.metric("Nouveaux cas à la date j", int(synth_row["Nombre des nouveaux cas à la date j"]))
                with col4:
                    st.metric("Nombre total de tickets à la date j", int(synth_row["Nombre des tickets à la date j"]))

            st.subheader("Nouveaux tickets (présents à j, absents à j-1)")
            if nouveaux_df.empty:
                st.info("Aucun nouveau ticket pour la période sélectionnée.")
            else:
                st.dataframe(nouveaux_df)

            st.subheader("Tickets traités (présents à j-1, absents à j)")
            if traites_df.empty:
                st.info("Aucun ticket traité pour la période sélectionnée.")
            else:
                st.dataframe(traites_df)

            # Tri avant export
            if EXCEPTION_COL in nouveaux_df.columns:
                nouveaux_df = nouveaux_df.sort_values(by=EXCEPTION_COL)
            if EXCEPTION_COL in traites_df.columns:
                traites_df = traites_df.sort_values(by=EXCEPTION_COL)

            excel_bytes = generate_excel_bytes(synthese_all_df, nouveaux_df, traites_df)
            st.session_state.excel_bytes = excel_bytes

            st.download_button(
                label="Télécharger le fichier Excel de résultats",
                data=excel_bytes,
                file_name=f"analyse_tickets_{date_j.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 2 – Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with tab_dashboard:
        st.header("Dashboard")
        if st.session_state.df_source is None:
            st.info(
                "Veuillez d'abord charger un fichier et lancer une analyse dans l'onglet "
                "'Analyse & Résultats'."
            )
        else:
            render_dashboard(st.session_state.df_source)

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 3 – Exceptions
    # ══════════════════════════════════════════════════════════════════════════
    with tab_exceptions:
        st.header("Exceptions")
        if st.session_state.df_source is None:
            st.info(
                "Veuillez d'abord charger un fichier et lancer une analyse dans l'onglet "
                "'Analyse & Résultats' pour initialiser les données."
            )
        else:
            render_exceptions(st.session_state.df_source)


if __name__ == "__main__":
    main()
