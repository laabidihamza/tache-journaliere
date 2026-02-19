from typing import Optional

import pandas as pd
import streamlit as st

from config import DATE_COL, TICKET_ID_COL, EXCEPTION_COL


def parse_date_input(date_input) -> pd.Timestamp:
    """
    Convertit un input Streamlit (datetime.date/datetime) en pd.Timestamp normalisé (sans heure).
    """
    return pd.to_datetime(date_input).normalize()


def compute_j_minus_1(j_date: pd.Timestamp) -> pd.Timestamp:
    """
    Calcule la date j-1 en tenant compte des cas particuliers imposés.

    Règles spéciales :
    - Si j = 16/12/2025 alors j-1 = 13/12/2025
    - Si j = 01/02/2026 alors j-1 = 29/01/2026
    Sinon : j-1 = j - 1 jour
    """
    if j_date == pd.Timestamp("2025-12-16"):
        return pd.Timestamp("2025-12-13")
    if j_date == pd.Timestamp("2026-02-01"):
        return pd.Timestamp("2026-01-29")

    return j_date - pd.Timedelta(days=1)


@st.cache_data(show_spinner="Chargement du fichier Excel…")
def load_data_from_excel(uploaded_file) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel uploadé dans Streamlit.
    Retourne un DataFrame pandas.
    """
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    if DATE_COL not in df.columns:
        raise ValueError(f"La colonne obligatoire '{DATE_COL}' est manquante.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame) -> Optional[str]:
    """
    Vérifie la présence des colonnes essentielles.
    Retourne un message d'erreur si nécessaire, sinon None.
    """
    missing_cols = [
        col for col in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL]
        if col not in df.columns
    ]

    if missing_cols:
        return (
            "Les colonnes suivantes sont manquantes dans le fichier : "
            + ", ".join(missing_cols)
        )
    return None
