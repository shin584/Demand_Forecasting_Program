from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import sample_mart1_weights


def test_severity_flag_gets_severity_weight_even_when_not_chronic():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Customer 1: only one visit ever, so no Revisit Match is
            # possible - not chronic. Still gets the severity weight.
            make_visit_row(조제판매ID=1, 고객ID=1, 중증암등록대상자="Y"),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 3.0


def test_severity_weight_takes_priority_over_chronic_weight():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Customer 1: severe (산전산모대상자) AND chronic (genuine
            # Revisit Match) - severity must win.
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                다음내방일="2024-01-31",
                약품ID=1,
                산전산모대상자="Y",
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 3.0


def test_any_of_the_three_severity_flags_triggers_the_severity_weight():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(조제판매ID=1, 고객ID=1, 중증암등록대상자="Y"),
            make_visit_row(조제판매ID=2, 고객ID=2, 산전산모대상자="Y"),
            make_visit_row(조제판매ID=3, 고객ID=3, 희귀난치대상자="Y"),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 3.0
    assert weights.loc[2] == 3.0
    assert weights.loc[3] == 3.0


def test_chronic_but_not_severe_gets_chronic_weight():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 2.0


def test_neither_severe_nor_chronic_gets_baseline_weight():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Only one visit ever - no Revisit Match possible - and no
            # severity flags set.
            make_visit_row(조제판매ID=1, 고객ID=1),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 1.0


def test_charhoowi_daesang_has_no_effect_alone():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(조제판매ID=1, 고객ID=1, 차상위대상자="Y"),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 1.0


def test_charhoowi_daesang_has_no_effect_alongside_chronic_status():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                다음내방일="2024-01-31",
                약품ID=1,
                차상위대상자="Y",
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )

    weights = sample_mart1_weights(raw_visits)

    assert weights.loc[1] == 2.0


def test_weight_tiers_are_overridable_parameters():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Distinct drug (998) so this visit doesn't add to 약품ID=1's
            # frequency count below.
            make_visit_row(조제판매ID=1, 고객ID=1, 희귀난치대상자="Y", 약품ID=998),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=3, 고객ID=2, 내방일="2024-02-15", 약품ID=1),
            # Distinct drug so this single visit doesn't inflate 약품ID=1's
            # frequency into a tie with the top-2 filler drugs.
            make_visit_row(조제판매ID=4, 고객ID=3, 약품ID=999),
        ]
    )

    weights = sample_mart1_weights(
        raw_visits, severity_weight=5.0, chronic_weight=2.5, baseline_weight=1.5
    )

    assert weights.loc[1] == 5.0
    assert weights.loc[2] == 2.5
    assert weights.loc[3] == 1.5


def test_reuses_precomputed_chronic_customer_ids():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(조제판매ID=1, 고객ID=1),
        ]
    )

    # Force customer 1 to be treated as chronic via the precomputed set,
    # even though the raw data alone wouldn't classify it that way.
    weights = sample_mart1_weights(raw_visits, chronic_customer_ids={1})

    assert weights.loc[1] == 2.0
