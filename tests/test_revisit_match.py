from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import revisit_match


def test_true_positive_shared_non_top2_drug_within_window():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            # Shares 약품ID=1 (not top-2) and falls within +/-30 days of 다음내방일.
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == True  # noqa: E712
    assert matches.loc[2] == False  # noqa: E712 (no later visit to match against)


def test_false_positive_guard_top2_drug_overlap_does_not_count():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                다음내방일="2024-01-31",
                약품ID=90001,
            ),
            # Only shared drug is 90001, one of the dataset's top-2 - must NOT match.
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=90001),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == False  # noqa: E712


def test_no_match_when_later_visit_outside_window():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            # 60 days after 다음내방일 - outside the +/-30 day window.
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-03-31", 약품ID=1),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == False  # noqa: E712


def test_no_match_when_no_shared_drug():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=2),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == False  # noqa: E712


def test_earlier_or_same_day_visits_are_not_candidates():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # An earlier visit by the same customer must not count as the
            # "later visit" for this later one, even though it shares a drug
            # and falls inside the window measured from this visit's own
            # 다음내방일.
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 다음내방일="2024-01-05", 약품ID=1
            ),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[2] == False  # noqa: E712


def test_matches_across_multiple_drug_line_rows_of_the_same_visit():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Visit 1 dispenses two drugs (two rows sharing 조제판매ID=1).
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=2
            ),
            # Later visit shares only 약품ID=2 from visit 1.
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=2),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == True  # noqa: E712


def test_match_found_past_an_in_window_visit_that_shares_no_drug():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            # Inside visit 1's window, but shares no drug with it...
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-20", 약품ID=2),
            # ...so the match has to come from this later in-window visit.
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-02-10", 약품ID=1),
        ]
    )

    matches = revisit_match(raw_visits)

    assert matches.loc[1] == True  # noqa: E712
