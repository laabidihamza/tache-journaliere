import io
from typing import Tuple

import pandas as pd
import streamlit as st

from config import DATE_COL, TICKET_ID_COL, EXCEPTION_COL
from data_loader import compute_j_minus_1


def compute_ticket_sets(
    df: pd.DataFrame, date_j1: pd.Timestamp, date_j: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne deux DataFrames :
    - tickets_j1 : tickets présents à la date j-1
    - tickets_j  : tickets présents à la date j
    """
    tickets_j1 = df[df[DATE_COL] == date_j1].copy()
    tickets_j = df[df[DATE_COL] == date_j].copy()
    return tickets_j1, tickets_j


def compute_synthesis(
    tickets_j1: pd.DataFrame,
    tickets_j: pd.DataFrame,
    date_j1: pd.Timestamp,
    date_j: pd.Timestamp,
) -> pd.DataFrame:
    """
    Calcule la synthèse pour la date j.

    Colonnes :
    - Date
    - Nombre des cas traités à la date j : présents à j-1 et absents à j
    - Nombre des nouveaux cas à la date j : présents à j et absents à j-1
    - Nombre des tickets à la date j
    """
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    set_j = set(tickets_j[TICKET_ID_COL].astype(str))

    treated_ids = set_j1 - set_j
    new_ids = set_j - set_j1

    synthese_data = {
        "Date": [date_j.normalize()],
        "Nombre des cas traités à la date j": [len(treated_ids)],
        "Nombre des nouveaux cas à la date j": [len(new_ids)],
        "Nombre des tickets à la date j": [len(set_j)],
    }
    return pd.DataFrame(synthese_data)


@st.cache_data(show_spinner="Calcul de la synthèse…")
def compute_synthesis_all_dates(
    df: pd.DataFrame, shift_metrics_by_one_day: bool = True
) -> pd.DataFrame:
    """
    Calcule la synthèse pour toutes les dates présentes dans le fichier d'entrée.

    Si shift_metrics_by_one_day=True (par défaut), les métriques « cas traités » et
    « nouveaux cas » sont décalées d'une ligne vers le bas pour refléter que les données
    reçues le jour j reflètent l'état de j-1.
    """
    empty_cols = [
        "Date",
        "Nombre des cas traités à la date j",
        "Nombre des nouveaux cas à la date j",
        "Nombre des tickets à la date j",
    ]

    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    data = df.copy()
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
        date_j1_norm = pd.to_datetime(compute_j_minus_1(date_j)).normalize()
        set_j1 = groups.get(date_j1_norm, set())

        treated_ids = set_j1 - set_j
        new_ids = set_j - set_j1

        rows.append(
            {
                "Date": date_j,
                "Nombre des cas traités à la date j": len(treated_ids),
                "Nombre des nouveaux cas à la date j": len(new_ids),
                "Nombre des tickets à la date j": len(set_j),
            }
        )

    synthese_all_df = pd.DataFrame(rows).sort_values(by="Date").reset_index(drop=True)

    if shift_metrics_by_one_day and len(synthese_all_df) > 1:
        synthese_all_df["Nombre des cas traités à la date j"] = (
            synthese_all_df["Nombre des cas traités à la date j"].shift(-1)
        )
        synthese_all_df["Nombre des nouveaux cas à la date j"] = (
            synthese_all_df["Nombre des nouveaux cas à la date j"].shift(-1)
        )
        synthese_all_df = synthese_all_df.iloc[:-1].reset_index(drop=True)

    return synthese_all_df


@st.cache_data
def compute_new_tickets(
    tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame
) -> pd.DataFrame:
    """
    Retourne les nouveaux tickets (présents à j, absents à j-1),
    triés par fréquence de l'exception (décroissante) puis par date.
    """
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    tickets_j = tickets_j.copy()
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)

    new_tickets = tickets_j[~tickets_j[TICKET_ID_COL].isin(set_j1)].copy()

    if new_tickets.empty:
        return new_tickets

    if EXCEPTION_COL in new_tickets.columns:
        exception_counts = (
            new_tickets[EXCEPTION_COL]
            .value_counts()
            .rename("Nombre d'occurrences")
            .to_frame()
        )
        new_tickets = new_tickets.merge(
            exception_counts,
            left_on=EXCEPTION_COL,
            right_index=True,
            how="left",
        )
        new_tickets = new_tickets.sort_values(
            by=["Nombre d'occurrences", DATE_COL], ascending=[False, True]
        )
    else:
        new_tickets = new_tickets.sort_values(by=DATE_COL)

    return new_tickets.reset_index(drop=True)


@st.cache_data
def compute_treated_tickets(
    tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame
) -> pd.DataFrame:
    """
    Retourne les tickets traités (présents à j-1, absents à j).
    """
    tickets_j1 = tickets_j1.copy()
    tickets_j = tickets_j.copy()

    tickets_j1[TICKET_ID_COL] = tickets_j1[TICKET_ID_COL].astype(str)
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)

    set_j = set(tickets_j[TICKET_ID_COL])
    treated_tickets = tickets_j1[~tickets_j1[TICKET_ID_COL].isin(set_j)].copy()

    if not treated_tickets.empty and EXCEPTION_COL in treated_tickets.columns:
        exception_counts = (
            treated_tickets[EXCEPTION_COL]
            .value_counts()
            .rename("Nombre d'occurrences")
            .to_frame()
        )
        treated_tickets = treated_tickets.merge(
            exception_counts,
            left_on=EXCEPTION_COL,
            right_index=True,
            how="left",
        )
        treated_tickets = treated_tickets.sort_values(
            by=["Nombre d'occurrences", DATE_COL], ascending=[False, True]
        )

    return treated_tickets.reset_index(drop=True)


# =========================
# Export Excel
# =========================

@st.cache_data
def generate_excel_bytes(
    synthese_df: pd.DataFrame,
    nouveaux_df: pd.DataFrame,
    traites_df: pd.DataFrame,
) -> bytes:
    """
    Génère un fichier Excel en mémoire (bytes) avec 3 feuilles :
    1) Synthèse
    2) Nouveaux tickets
    3) Tickets traités
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        synthese_df.to_excel(writer, sheet_name="Synthèse", index=False)
        nouveaux_df.to_excel(writer, sheet_name="Nouveaux tickets", index=False)
        traites_df.to_excel(writer, sheet_name="Tickets traités", index=False)
    output.seek(0)
    return output.getvalue()


@st.cache_data
def generate_single_sheet_excel(df: pd.DataFrame, sheet_name: str = "Données") -> bytes:
    """
    Génère un fichier Excel en mémoire (bytes) avec une seule feuille.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()
