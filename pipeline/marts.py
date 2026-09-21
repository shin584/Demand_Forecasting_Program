"""Mart-construction pipeline: turns raw extracted visit-level rows into
Mart 1 (Visit-Probability), Mart 2 (Customer Drug Profile) and Mart 3
(Acute Drug Statistics).

See CONTEXT.md and docs/adr/ for the design decisions this pipeline encodes.
This module currently implements only the pipeline scaffolding and Mart 1's
Y label; later tickets fill in the remaining columns of all three marts.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

CUSTOMER_ID_COL = "고객ID"
VISIT_DATE_COL = "내방일"
NEXT_DAY_VISIT_COL = "내일_방문"

MART1_COLUMNS = [CUSTOMER_ID_COL, NEXT_DAY_VISIT_COL]
MART2_COLUMNS = ["고객ID", "약품ID", "최근소모량"]
MART3_COLUMNS = ["약품ID", "계절", "요일", "소모량"]


class MartResult(NamedTuple):
    mart1: pd.DataFrame
    mart2: pd.DataFrame
    mart3: pd.DataFrame


def build_marts(raw_visits: pd.DataFrame, as_of_date) -> MartResult:
    """Build Mart 1/2/3 from raw visit-level rows, as of a given snapshot date.

    Pure and deterministic: reads only `as_of_date` (never wall-clock "today")
    and never mutates `raw_visits`, so the same inputs always produce the same
    outputs and the same entry point can be reused for training, backtesting,
    and live inference (see docs/adr/0002-point-in-time-correctness.md).
    """
    as_of_date = pd.Timestamp(as_of_date)
    mart1 = _build_mart1(raw_visits, as_of_date)
    mart2 = pd.DataFrame(columns=MART2_COLUMNS)
    mart3 = pd.DataFrame(columns=MART3_COLUMNS)
    return MartResult(mart1=mart1, mart2=mart2, mart3=mart3)


def _build_mart1(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    # Includes every customer in raw_visits, unfiltered by Chronic-Patient /
    # Revisit Match status — that inclusion filter is a later ticket's job
    # (see issue #1); today's output isn't yet the final Mart 1 population.
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    target_date = as_of_date + pd.Timedelta(days=1)

    customers = (
        raw_visits[CUSTOMER_ID_COL].drop_duplicates().sort_values().reset_index(drop=True)
    )

    # 내일_방문 is the literal "did the customer actually show up on
    # as_of_date + 1" boolean — independent of Revisit Match, which is a
    # separate behavioral definition used only for chronic/acute routing.
    # A visit dated exactly as_of_date + 1 day is necessarily that customer's
    # *next* visit after as_of_date, since no calendar day falls between the
    # two — this equivalence would not hold if target_date were more than one
    # day out.
    visited_next_day = set(raw_visits.loc[visit_dates == target_date, CUSTOMER_ID_COL])

    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: customers,
            NEXT_DAY_VISIT_COL: customers.isin(visited_next_day),
        }
    )
