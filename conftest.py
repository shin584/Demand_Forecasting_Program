"""Shared synthetic-data test harness for the mart-construction pipeline.

Reusable by every ticket under issue #1: build a `raw_visits`-shaped
DataFrame (matching Extract_By_CSV.py's output columns) from a handful of
hand-constructed visit rows, without having to fill in all ~27 columns by
hand for every test.
"""

import pandas as pd

from pipeline.pharmacy_calendar import PharmacyCalendar

RAW_VISIT_COLUMNS = [
    "조제판매ID",
    "고객ID",
    "내방일",
    "처방조제일수",
    "다음내방일",
    "처방전발행기관ID",
    "가족ID",
    "성별",
    "생년월일",
    "가족총내방",
    "가족총매출",
    "보험구분",
    "발행기관기호",
    "병명코드",
    "분류기호",
    "약품ID",
    "투약일수",
    "일회투약량",
    "일일투여회수",
    "소모량",
    "약품명",
    "속명",
    "제약사",
    "중증암등록대상자",
    "산전산모대상자",
    "희귀난치대상자",
    "차상위대상자",
]

_DATE_COLUMNS = ["내방일", "다음내방일"]

_DEFAULTS = {
    "조제판매ID": 1,
    "고객ID": 1,
    "내방일": "2024-01-01",
    "처방조제일수": 30,
    "다음내방일": "2024-01-31",
    "처방전발행기관ID": 1,
    "가족ID": 1,
    "성별": "여",
    "생년월일": "1970-01-01",
    "가족총내방": 1,
    "가족총매출": 10000,
    "보험구분": "건강보험",
    "발행기관기호": "10000000",
    "병명코드": None,
    "분류기호": None,
    "약품ID": 1,
    "투약일수": 30,
    "일회투약량": 1.0,
    "일일투여회수": 2,
    "소모량": 60.0,
    "약품명": "테스트약",
    "속명": "테스트약",
    "제약사": "테스트제약",
    "중증암등록대상자": None,
    "산전산모대상자": None,
    "희귀난치대상자": None,
    "차상위대상자": None,
}


def make_visit_row(**overrides) -> dict:
    """One synthetic raw_visits row, with sensible defaults for every column.

    Pass only the fields a given test actually cares about, e.g.
    `make_visit_row(고객ID=1, 내방일="2024-03-01")`.
    """
    row = dict(_DEFAULTS)
    row.update(overrides)
    return row


def make_raw_visits(rows: list[dict]) -> pd.DataFrame:
    """Assemble make_visit_row(...) dicts into a raw_visits-shaped DataFrame."""
    df = pd.DataFrame(rows, columns=RAW_VISIT_COLUMNS)
    for date_col in _DATE_COLUMNS:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


# Two dedicated drug IDs, each given more occurrences than any real test
# fixture is expected to use for its own drugs.
NOISE_DRUG_IDS = (90001, 90002)


def make_high_frequency_filler_visits(occurrences: int = 3) -> list[dict]:
    """Visit rows guaranteeing NOISE_DRUG_IDS are the dataset's top-2
    highest-frequency drugs, each on its own single-visit customer.

    Prepend this to a raw_visits fixture (`make_raw_visits(make_high_frequency_filler_visits() + [...])`)
    so a test's own drugs are never accidentally swept into the Revisit
    Match top-2 exclusion (see ADR-0001) just because too few distinct
    drugs exist in the fixture - a real drug only needs to appear in fewer
    than `occurrences` visits to be safe from exclusion.
    """
    rows = []
    visit_id = 900000
    customer_id = 900000
    for drug_id in NOISE_DRUG_IDS:
        for _ in range(occurrences):
            rows.append(make_visit_row(조제판매ID=visit_id, 고객ID=customer_id, 약품ID=drug_id))
            visit_id += 1
            customer_id += 1
    return rows


def make_independent_chronic_match_visits(
    customer_id: int, drug_id: int, visit_id_start: int
) -> list[dict]:
    """Two visit rows forming an already-resolved Revisit Match, dated well
    before any as_of_date a test built around this fixture would exercise --
    establishes `customer_id` as Chronic (see ADR-0001) without depending on,
    or disturbing the timing of, whatever other visit that test anchors on or
    needs to still be in the future relative to as_of_date.

    Needed because Chronic/Acute classification only considers visits with
    내방일 <= as_of_date (see issue #11 and
    docs/adr/0002-point-in-time-correctness.md): a test that also wants to
    exercise some other as-of-date-sensitive behavior (which visit anchors,
    a Y-label lookup, a family visit count, ...) needs its customer's Chronic
    status established independently of that behavior's own visit dates.

    Give each call its own `drug_id` and `visit_id_start` (unused elsewhere
    in that test's fixture) so this pair doesn't collide with other visits'
    조제판매ID values or get swept into the Revisit Match top-2 frequency
    exclusion.
    """
    return [
        make_visit_row(
            조제판매ID=visit_id_start,
            고객ID=customer_id,
            내방일="2023-11-01",
            다음내방일="2023-12-01",
            약품ID=drug_id,
        ),
        make_visit_row(
            조제판매ID=visit_id_start + 1, 고객ID=customer_id, 내방일="2023-11-15", 약품ID=drug_id
        ),
    ]


def closed_on(*dates: str) -> PharmacyCalendar:
    """A pharmacy calendar closed on exactly `dates` and open every other
    day, Sundays included -- so a test's closures are only the ones it
    names."""
    return PharmacyCalendar(closed_days=pd.DatetimeIndex(dates), observed_through=None)
