"""Runs the Track 1 + Track 2 backtest end-to-end against the real pharmacy
extract and writes the headline WAPE plus a per-drug breakdown to a report
file (see issue #18 and `pipeline.backtest.run_backtest`).

Trains Track 1's model once on `split_mart1_training_set`'s train/val split
-- sampled train rows, early-stopped against, Platt-calibrated on and
cutoff-tuned on full daily validation snapshots (matching production's own
training path, see `pipeline.model.train_track1_model`) -- reports the test
window's Σp against actual Chronic next-day visits, then walks the trained
model forward across the split's test window at the cutoff it carries (the
tuned, F1-maximising one, or the default `CHRONIC_VISIT_PROB_CUTOFF` if none
could be tuned). Closed days come from the extract itself
(`PharmacyCalendar.from_visits`: a day with no dispensing is closed), and
the report splits Track 1's figures by whether the target day was closed,
the first open day after a closure, or any other open day (issue #39), and
Track 2's by closed vs open target day (issue #42). A
full run walks one calendar day at a time across the
whole test window (6 months by default), rebuilding Mart 1/2/3 as of each
day -- a few seconds per day on the v0.3 extract; pass a narrower
`--test-dates-limit` while iterating.

Usage: python scripts/run_backtest.py [--test-dates-limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.backtest import (
    ACTUAL_CHRONIC_VISITS_COL,
    ACTUAL_COL,
    ERROR_COL,
    PREDICTED_COL,
    SUM_VISIT_PROB_COL,
    TRACK1_ACTUAL_COL,
    TRACK1_PREDICTED_COL,
    TRACK2_ACTUAL_COL,
    TRACK2_PREDICTED_COL,
    VISIT_LIST_SIZE_COL,
    run_backtest,
)
from pipeline.marts import (
    CLOSED_TOMORROW_COL,
    DRUG_ID_COL,
    SNAPSHOT_DATE_COL,
    build_mart1_training_set,
    mart1_split_windows,
    split_mart1_training_set,
)
from pipeline.model import CHRONIC_VISIT_PROB_CUTOFF, train_track1_model
from pipeline.pharmacy_calendar import PharmacyCalendar

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / "dataset" / "pharmacy_raw_data_v0.3.csv"
REPORT_PATH = REPO_ROOT / "reports" / "backtest_summary.txt"

# Raw extract date columns needing parsing -- 생년월일 mixes "%Y-%m-%d" and
# "%Y-%m-%d %H:%M:%S.%f" formatted values in the real v0.3 extract, so
# format="mixed" is required here (an unqualified `pd.to_datetime` errors on
# the inconsistency).
_DATE_COLUMNS = ["내방일", "다음내방일", "생년월일"]


def _load_raw_visits(path: Path) -> pd.DataFrame:
    raw_visits = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for column in _DATE_COLUMNS:
        raw_visits[column] = pd.to_datetime(raw_visits[column], format="mixed")
    return raw_visits


def _wape_breakdown(daily: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Same WAPE formula as `pipeline.backtest._wape`
    (`sum(abs error) / sum(actual)`), grouped by `group_col` instead of taken
    once over the whole window -- see issue #18's Implementation Decisions
    ("WAPE formula ... computed both per as-of-date and once overall")."""
    breakdown = daily.groupby(group_col)[[PREDICTED_COL, ACTUAL_COL, ERROR_COL]].sum()
    breakdown["WAPE"] = breakdown[ERROR_COL] / breakdown[ACTUAL_COL].replace(0, pd.NA)
    return breakdown


def _track_totals(daily: pd.DataFrame, predicted_col: str, actual_col: str) -> str:
    return f"predicted {daily[predicted_col].sum():.2f}, actual {daily[actual_col].sum():.2f}"


def _visit_probability_table(per_date: pd.DataFrame) -> pd.DataFrame:
    """`result.per_date` indexed by as-of-date, with a "Total" row summing
    every column over the window (see issue #25)."""
    table = per_date.set_index(SNAPSHOT_DATE_COL)
    table.index = table.index.strftime("%Y-%m-%d")
    dtypes = table.dtypes
    table.loc["Total"] = table.sum()
    return table.astype(dtypes)


def _target_day_kind_table(result, calendar: PharmacyCalendar) -> pd.DataFrame:
    """Track 1's figures per kind of target day (see issue #39): closed, the
    first open day after a closure (the as-of date itself was closed), and
    every other open day -- Σp, actual Chronic next-day visits, mean Visit
    List size, and Track 1's predicted/actual volume and WAPE."""
    per_date = result.per_date.set_index(SNAPSHOT_DATE_COL)
    kind = pd.Series("open", index=per_date.index)
    kind[calendar.is_closed(per_date.index.to_series()).to_numpy()] = "open, after a closure"
    kind[per_date[CLOSED_TOMORROW_COL].to_numpy()] = "closed"

    daily = result.daily.assign(
        _error=(result.daily[TRACK1_PREDICTED_COL] - result.daily[TRACK1_ACTUAL_COL]).abs()
    )
    track1 = daily.groupby(SNAPSHOT_DATE_COL)[
        [TRACK1_PREDICTED_COL, TRACK1_ACTUAL_COL, "_error"]
    ].sum()
    table = (
        per_date.join(track1)
        .assign(kind=kind)
        .groupby("kind")
        .agg(
            days=(SUM_VISIT_PROB_COL, "size"),
            sum_p=(SUM_VISIT_PROB_COL, "sum"),
            actual_visits=(ACTUAL_CHRONIC_VISITS_COL, "sum"),
            mean_visit_list=(VISIT_LIST_SIZE_COL, "mean"),
            track1_predicted=(TRACK1_PREDICTED_COL, "sum"),
            track1_actual=(TRACK1_ACTUAL_COL, "sum"),
            _error=("_error", "sum"),
        )
    )
    table["track1_WAPE"] = table["_error"] / table["track1_actual"].replace(0, pd.NA)
    return table.drop(columns="_error")


def _track2_closed_open_summary(summary) -> str:
    """Track 2 split by target day (see issue #42): a closed day has no
    actual dispensing inside the extract, so only its predicted volume."""
    return (
        f"Track 2 on closed target days: predicted {summary.track2_closed_predicted:.2f}\n"
        f"Track 2 on open target days: predicted {summary.track2_open_predicted:.2f}, "
        f"actual {summary.track2_open_actual:.2f}, WAPE {summary.track2_open_wape:.4f}\n"
    )


def _training_summary(trained) -> str:
    tuning = trained.cutoff_tuning
    if tuning.cutoff is None:
        cutoff_line = (
            "Chronic cutoff: none tuned (no validation Next-Day Visits); "
            f"falling back to the provisional {CHRONIC_VISIT_PROB_CUTOFF}"
        )
    else:
        cutoff_line = (
            f"Chronic cutoff: {tuning.cutoff:.4f} (max F1) -- validation precision "
            f"{tuning.precision:.4f}, recall {tuning.recall:.4f}, F1 {tuning.f1:.4f}"
        )
    metrics = ", ".join(f"{name} {value:.4f}" for name, value in trained.metrics.items())
    check = trained.sum_p_check
    return (
        f"Validation: {metrics}\n"
        f"{cutoff_line}\n"
        f"Test window: Σp {check.sum_p:.1f} vs {check.actual_visits} actual Chronic "
        f"next-day visits (ratio {check.ratio:.3f})\n"
    )


def _write_report(
    path: Path, result, training_summary: str, calendar: PharmacyCalendar
) -> None:
    summary = result.summary
    wape_by_date = _wape_breakdown(result.daily, SNAPSHOT_DATE_COL)
    per_drug = _wape_breakdown(result.daily, DRUG_ID_COL).sort_values(ERROR_COL, ascending=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(training_summary + "\n")
        f.write(f"WAPE: {summary.wape:.4f}\n")
        f.write(f"  Track 1 WAPE: {summary.track1_wape:.4f} ")
        f.write(f"({_track_totals(result.daily, TRACK1_PREDICTED_COL, TRACK1_ACTUAL_COL)})\n")
        f.write(f"  Track 2 WAPE: {summary.track2_wape:.4f} ")
        f.write(f"({_track_totals(result.daily, TRACK2_PREDICTED_COL, TRACK2_ACTUAL_COL)})\n")
        f.write(f"Total predicted: {summary.total_predicted:.2f}\n")
        f.write(f"Total actual: {summary.total_actual:.2f}\n")
        f.write(f"Days: {len(result.per_date)}\n\n")
        f.write(_track2_closed_open_summary(summary))
        f.write("\n")
        f.write("Track 1 by target day:\n")
        f.write(_target_day_kind_table(result, calendar).to_string())
        f.write("\n\n")
        f.write(
            "Track 1 visit probabilities per as-of-date (Σp vs actual Chronic next-day "
            "visits; Visit List size vs scored Chronic population):\n"
        )
        f.write(_visit_probability_table(result.per_date).to_string())
        f.write("\n\nPer-as-of-date WAPE:\n")
        f.write(wape_by_date.to_string())
        f.write("\n\nPer-drug breakdown (sorted by absolute error, descending):\n")
        f.write(per_drug.to_string())
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dates-limit",
        type=int,
        default=None,
        help="Only walk the first N dates of the test window (for a faster, partial run).",
    )
    args = parser.parse_args()

    raw_visits = _load_raw_visits(RAW_DATA_PATH)
    calendar = PharmacyCalendar.from_visits(raw_visits)

    training_set = build_mart1_training_set(raw_visits, pharmacy_calendar=calendar)
    started = time.perf_counter()
    split = split_mart1_training_set(training_set, raw_visits, pharmacy_calendar=calendar)
    print(
        f"Built val/test daily snapshots in {time.perf_counter() - started:.1f}s "
        f"({len(split.val):,} val rows, {len(split.test):,} test rows)"
    )
    trained = train_track1_model(split.train, split.val, split.test)
    training_summary = _training_summary(trained)
    print(training_summary, end="")

    test_dates = None
    if args.test_dates_limit is not None:
        windows = mart1_split_windows(training_set)
        full_range = pd.date_range(windows.test_start, windows.end, freq="D")
        test_dates = list(full_range[: args.test_dates_limit])

    result = run_backtest(
        raw_visits, trained.model, test_dates=test_dates, pharmacy_calendar=calendar
    )
    _write_report(REPORT_PATH, result, training_summary, calendar)
    print(_target_day_kind_table(result, calendar).to_string())
    print(_track2_closed_open_summary(result.summary), end="")

    print(
        f"WAPE: {result.summary.wape:.4f} (Track 1 {result.summary.track1_wape:.4f}, "
        f"Track 2 {result.summary.track2_wape:.4f})"
    )
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
