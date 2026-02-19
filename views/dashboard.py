import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATE_COL, TICKET_ID_COL, EXCEPTION_COL


@st.cache_data
def _dashboard_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(DATE_COL)[TICKET_ID_COL]
        .nunique()
        .reset_index()
        .rename(columns={TICKET_ID_COL: "Nombre de tickets"})
    )


@st.cache_data
def _dashboard_top_exceptions(df: pd.DataFrame, top_n: int = 10):
    """Deduplicate by ticket ref, compute top-N exceptions and daily trend data."""
    df_pie = df.copy()
    df_pie[TICKET_ID_COL] = df_pie[TICKET_ID_COL].astype(str)
    df_pie = df_pie.drop_duplicates(subset=[TICKET_ID_COL], keep="last")

    exception_counts = (
        df_pie[EXCEPTION_COL]
        .value_counts()
        .reset_index(name="Nombre")
        .rename(columns={"index": EXCEPTION_COL})
    )
    top_exceptions = exception_counts.head(top_n).copy()

    max_len = 60
    top_exceptions["Exception_courte"] = (
        top_exceptions[EXCEPTION_COL].astype(str).str.slice(0, max_len)
    )
    mask = top_exceptions[EXCEPTION_COL].str.len() > max_len
    top_exceptions.loc[mask, "Exception_courte"] = (
        top_exceptions.loc[mask, "Exception_courte"] + "..."
    )

    # Daily trend for top exceptions
    top_names = top_exceptions[EXCEPTION_COL].tolist()
    df_trend = df_pie[df_pie[EXCEPTION_COL].isin(top_names)].copy()
    df_trend[DATE_COL] = pd.to_datetime(df_trend[DATE_COL]).dt.normalize()
    df_trend["Exception_courte"] = df_trend[EXCEPTION_COL].astype(str).str.slice(0, max_len)
    mask2 = df_trend[EXCEPTION_COL].str.len() > max_len
    df_trend.loc[mask2, "Exception_courte"] = df_trend.loc[mask2, "Exception_courte"] + "..."

    daily_exception_counts = (
        df_trend.groupby([DATE_COL, "Exception_courte"])
        .size()
        .reset_index(name="Nombre d'occurrences")
    )
    return top_exceptions, exception_counts, daily_exception_counts


def render_dashboard(df: pd.DataFrame):
    """
    Construit les graphiques principaux pour le dashboard :
    - Évolution du nombre total de tickets par date
    - Nombre de tickets par date
    - Répartition des exceptions (Top 10)
    - Évolution journalière des Top 10 exceptions
    """
    if df.empty:
        st.info("Aucune donnée disponible pour le dashboard.")
        return

    # ── Tickets par date ───────────────────────────────────────────────────────
    df_daily = _dashboard_daily_counts(df)

    st.subheader("Évolution du nombre total de tickets par date")
    st.plotly_chart(
        px.line(df_daily, x=DATE_COL, y="Nombre de tickets", markers=True),
        width="stretch",
    )

    st.subheader("Nombre de tickets par date")
    st.plotly_chart(
        px.bar(df_daily, x=DATE_COL, y="Nombre de tickets"),
        width="stretch",
    )

    # ── Top 10 exceptions (camembert) ─────────────────────────────────────────
    st.subheader("Top 10 des exceptions (camembert)")

    if EXCEPTION_COL not in df.columns:
        st.info(f"La colonne '{EXCEPTION_COL}' n'existe pas dans les données.")
        return

    top_exceptions, exception_counts, daily_exception_counts = _dashboard_top_exceptions(df)

    if exception_counts.empty:
        st.info("Aucune exception à afficher.")
        return

    fig_exceptions = px.pie(top_exceptions, names="Exception_courte", values="Nombre")
    fig_exceptions.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_exceptions, width="stretch")

    st.markdown("### Détail des Top 10 exceptions")
    st.dataframe(
        top_exceptions[[EXCEPTION_COL, "Nombre"]].rename(
            columns={EXCEPTION_COL: "Exception", "Nombre": "Nombre d'occurrences"}
        ),
        width="stretch",
    )

    # ── Évolution journalière des Top 10 exceptions ───────────────────────────
    st.subheader("Évolution journalière des Top 10 exceptions")

    available_exceptions = sorted(daily_exception_counts["Exception_courte"].unique().tolist())

    selected_exceptions = []
    for i, exc in enumerate(available_exceptions):
        if st.checkbox(exc, value=True, key=f"dash_exc_cb_{i}"):
            selected_exceptions.append(exc)

    if not selected_exceptions:
        st.info("Aucune exception sélectionnée. Sélectionnez au moins une exception pour afficher le graphique.")
        return

    df_plot = daily_exception_counts[
        daily_exception_counts["Exception_courte"].isin(selected_exceptions)
    ].copy()

    total_occurrences = int(df_plot["Nombre d'occurrences"].sum())

    fig_exc_trend = px.line(
        df_plot,
        x=DATE_COL,
        y="Nombre d'occurrences",
        color="Exception_courte",
        markers=True,
    )
    fig_exc_trend.update_layout(
        legend_title_text="Exception",
        legend=dict(font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_exc_trend.update_yaxes(type="log")
    st.plotly_chart(fig_exc_trend, width="stretch")
    st.metric("Occurrences totales sélectionnées", total_occurrences)
