import pandas as pd

from conftest import make_raw_visits, make_visit_row
from pipeline.pharmacy_calendar import PharmacyCalendar


def _is_closed(calendar: PharmacyCalendar, *dates: str) -> list[bool]:
    return calendar.is_closed(pd.Series(pd.to_datetime(list(dates)))).tolist()


def _visits_on(*dates: str) -> pd.DataFrame:
    return make_raw_visits(
        [make_visit_row(조제판매ID=i, 내방일=date) for i, date in enumerate(dates)]
    )


def test_a_day_inside_the_extract_with_no_dispensing_is_closed():
    # 2024-01-01 Mon .. 2024-01-04 Thu, nothing dispensed on Wednesday.
    calendar = PharmacyCalendar.from_visits(_visits_on("2024-01-01", "2024-01-02", "2024-01-04"))

    assert _is_closed(calendar, "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04") == [
        False,
        False,
        True,
        False,
    ]


def test_a_sunday_inside_the_extract_with_dispensing_is_open():
    # 2024-01-07 is a Sunday the pharmacy happened to open.
    calendar = PharmacyCalendar.from_visits(_visits_on("2024-01-06", "2024-01-07", "2024-01-08"))

    assert _is_closed(calendar, "2024-01-07") == [False]


def test_past_the_extract_only_sundays_are_closed_by_default():
    # The extract ends Wednesday 2024-01-03; Thursday to the next Monday follow.
    calendar = PharmacyCalendar.from_visits(_visits_on("2024-01-01", "2024-01-03"))

    assert _is_closed(
        calendar, "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"
    ) == [False, False, False, True, False]


def test_listed_closures_past_the_extract_are_closed():
    calendar = PharmacyCalendar.from_visits(
        _visits_on("2024-01-01", "2024-01-03"), closure_dates=["2024-01-05"]
    )

    assert _is_closed(calendar, "2024-01-04", "2024-01-05") == [False, True]


def test_a_listed_closure_inside_the_extract_is_closed_even_with_dispensing():
    calendar = PharmacyCalendar.from_visits(
        _visits_on("2024-01-01", "2024-01-02", "2024-01-03"), closure_dates=["2024-01-02"]
    )

    assert _is_closed(calendar, "2024-01-02") == [True]


def test_no_calendar_means_no_closed_days():
    assert _is_closed(PharmacyCalendar.always_open(), "2024-01-07", "2030-12-25") == [False, False]


def test_result_keeps_the_input_index():
    calendar = PharmacyCalendar.from_visits(_visits_on("2024-01-01", "2024-01-03"))
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]), index=[10, 20])

    assert calendar.is_closed(dates).index.tolist() == [10, 20]


def test_is_closed_on_answers_for_a_single_day():
    calendar = PharmacyCalendar.from_visits(_visits_on("2024-01-01", "2024-01-02", "2024-01-04"))

    assert calendar.is_closed_on("2024-01-03") is True
    assert calendar.is_closed_on(pd.Timestamp("2024-01-04")) is False
