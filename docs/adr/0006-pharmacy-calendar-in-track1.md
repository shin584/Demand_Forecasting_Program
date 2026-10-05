---
status: accepted
---

# Track 1 reads a pharmacy calendar: closed-day features, and p = 0 on a closed target day

Track 1's X-features had no notion of whether the target day is open. In the full v0.3 backtest (issue #33: 185 as-of dates, 2025-07-15 → 2026-01-15, cutoff 0.108), 31 target days had zero Chronic visits: 26 Sundays plus holidays. On those days Σp averaged ≈32 and the Visit List ≈59 customers. Closed days made up ≈1,000 of the window's 6,737 Σp, so the window's 1.04 Σp ratio was partly closed-day over-prediction cancelling open-day under-prediction (≈0.89×). Issue #39.

**Decisions**:

- **The calendar.** `PharmacyCalendar.from_visits(raw_visits, closure_dates)`. Inside the extract, a day with no dispensing at all is closed. Past the extract's last 내방일, a day is closed if it's a Sunday or on the pharmacist's closure list (`closure_dates`, which also overrides the extract). Closures are planned in advance, so the backtest reading a past target day's closure from the extract isn't leakage.
- **Features.** Mart 1 gains 내일_휴무 (target day closed) and 오늘_휴무 (snapshot day closed). The second lets the model learn that the first open day after a closure picks up its visits. Both are computed per row, by the same `_mart1_x_features` path, for `build_marts`, the Mart 1 Training Set and the daily snapshots.
- **Hard zero.** On a closed target day every probability is 0 (`pipeline.model.visit_probabilities`): an empty Visit List and no Track 1 demand, rare-drug allocation included. The model alone gets close but never to exactly 0, and nobody can visit a closed pharmacy.
- **Calibration on open days.** The Platt calibrator is fit on validation rows whose target day is open. Closed rows are zeroed anyway, and fitting on them would drag every open day below its real rate. Cutoff tuning, the validation metrics and the test Σp check use the zeroed probabilities, closed days included, exactly as inference sees them.
- **No calendar, no closures.** Every entry point takes an optional `pharmacy_calendar`. Without one, no day is closed and behaviour is unchanged. The inference rule is only valid on a full extract, so it's an explicit input, never derived inside `build_marts`.

**Relation to [[0002-point-in-time-correctness]]**: _Contradicts ADR-0002's rule that every X-feature is recomputed as of its row's own snapshot date, but deliberately._ 내일_휴무 reads the calendar as it stands: a closure is a schedule fact the pharmacist knows ahead of time, not an observation that accumulates. The cost is honest to state. Inside the extract, "closed" means "nothing was dispensed that day", which is also the outcome being predicted. A training row with 내일_휴무 = True therefore has Y = 0 by construction, and in the backtest every closed day's Σp is 0 by construction. Production only knows Sundays and the pharmacist's list, so a holiday closure missing from that list stays unzeroed. The backtest therefore flatters production on non-Sunday closures, by exactly as much as the pharmacist's list is incomplete. An open day on which nobody happened to come in would also read as closed, which on a full extract is vanishingly rare.

**Considered options**:

- *A Korean public-holiday list (e.g. the `holidays` package) plus Sundays*: rejected. Which holidays the pharmacy closes for has changed over time. Every non-Sunday closure in v0.3 is a statutory holiday, but alternative and temporary holidays are sometimes open (2023-05-29, 2025-01-27, 2025-03-03, 2025-05-06) and sometimes closed (2023-01-24, 2024-02-12, 2024-05-06). By late 2025 the pharmacy opens on most holidays (2025-10-03, 10-07–09, 12-25). A list would mislabel training rows. It stays useful as the pharmacist's input for future days.
- *Pharmacist-supplied list only*: rejected for history, for the same reason. Kept for future days.
- *Hard zero only, no features*: rejected. It removes the closed-day over-forecast but leaves the day after a closure under-predicted.
- *Features only*: rejected. Closed days would keep a small non-zero Σp and a non-empty Visit List.
- *Reallocating a closed day's probability onto the next open day*: rejected. 오늘_휴무 lets the model learn the pile-up from data instead of a hand-written rule.

**Consequences**: Saved models from before this change lack the two features and must be retrained. Track 2 doesn't read the calendar: a closed Sunday already forecasts 0 through its weekday bucket, but holidays don't. Zeroing Track 2 (and the order) on closed target days is a separate decision. A low-volume holiday opening (2–8 visits, e.g. 2025-12-25) counts as open and is scored like any other day.

**Result (full v0.3 backtest, 185 as-of dates, 2025-07-15 → 2026-01-15)**: the model was retrained with the two features. The calendar inferred 27 closed target days in the window.

| | before | after |
|---|---|---|
| Combined WAPE | 1.115 | 0.968 |
| Track 1 WAPE | 1.191 | 1.030 |
| Track 1 predicted (actual 614,332) | 679,019 | 645,677 |
| Test-window Σp (actual 6,478) | 6,737 (1.04×) | 6,915 (1.07×) |
| Tuned cutoff | 0.108 | 0.160 |
| Validation precision / recall / F1 | 0.17 / 0.35 / 0.23 | 0.25 / 0.28 / 0.27 |
| Validation AUC | 0.846 | 0.874 |

The AUC and precision gains are partly mechanical: validation now includes closed days, which have p = 0 and Y = 0.

By target day, after:

| target day | days | Σp | actual visits | ratio | Visit List / day | Track 1 WAPE |
|---|---|---|---|---|---|---|
| closed | 27 | 0 | 0 | — | 0 | — (no actual) |
| first open day after a closure | 27 | 1,153 | 1,412 | 0.82× | 51 | 0.83 |
| other open | 131 | 5,762 | 5,066 | 1.14× | 51 | 1.09 |

Closed-day over-forecast is gone, and the Visit List shrinks from ~71 to ~51 a day. 오늘_휴무 only partly captures the pile-up: the first open day after a closure is still under-predicted (0.82×), while ordinary open days are now over-predicted (1.14×). Calibration fits one sigmoid across both, so a residual remains for a later issue.

**Update (issue #42): Track 2 on closed target days.** The separate decision above is now made. On a closed target day Track 2's statistical value is 0 for every drug, so with Track 1 already 0 the final order quantity is 0. The rare-drug stock floor (희귀약_최소재고) is kept, since it is a stock level, not daily flow. Closed-day Acute demand is not rolled forward onto the next open day: walk-in Acute patients often go to another pharmacy when this one is closed, and moving all of it forward would worsen open-day over-forecasting. Mart 3's calendar-day denominators still include closed days; excluding them is deferred, to be decided from the open-day Track 2 WAPE below.

Full v0.3 backtest, same 185 as-of dates and same retrained Track 1 model as above (cutoff 0.160, 27 closed target days):

| | before | after |
|---|---|---|
| Combined WAPE | 0.968 | 0.966 |
| Track 1 WAPE | 1.030 | 1.030 |
| Track 2 WAPE | 1.229 | 1.210 |
| Track 2 predicted (actual 86,764) | 72,407 (0.83×) | 70,759 (0.82×) |
| Track 2 predicted on closed target days | 1,648 | 0 |
| Track 2 on open target days: predicted / actual / WAPE | 70,759 / 86,764 / 1.210 | 70,759 / 86,764 / 1.210 |

The whole gain is the removed closed-day forecast (about 61 units per closed day, from holidays and young drugs' backed-off Sundays); open days are unchanged, as intended. Track 2 still under-forecasts open days (0.82×), so the ~2% per-weekday dilution from closed days in Mart 3's denominators is small next to Track 2's remaining error, and is left as deferred.
