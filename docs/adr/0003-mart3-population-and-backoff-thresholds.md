---
status: accepted
---

# Mart 3 excludes rare drugs entirely; bucket and season backoff thresholds are separate

Track 2 Sparse-Bucket Backoff's threshold was left as "N, TBD empirically," with a single shared value reused for both bucket-level (drug×season×weekday) and season-level sufficiency. Real distribution analysis of the v0.3 extract (168k rows, 1,075 drugs) showed two problems with leaving it that way: reusing one threshold makes season-sufficiency almost automatic — a season pools up to 7 weekdays of bucket data, so it clears any bucket-sized bar with room to spare, making the season fallback tier do little independent work — and separately, 61.7% of drugs in the catalog have fewer than 5 unique patients, the same population CONTEXT.md's rare-drug cutoff already flags for 100%-allocation stockout handling rather than a modeled forecast.

**Decision**: `N_bucket` and `N_season` are separate, independently-set thresholds (placeholders `N_bucket=5`, `N_season=15`, chosen from the real bucket/season count distributions — median bucket count 4, median season count 8 — not yet backtested, since no Track 2 evaluation harness exists in the repo). Drugs below the rare-drug patient cutoff (<5 patients, as-of-date trailing 12 months) get no Mart 3 row at all: they're routed to the rare-drug 100%-allocation rule instead of a backed-off statistical estimate, reusing that cutoff as Mart 3's own population filter rather than maintaining a second, independently-tunable "rare" threshold.

**Considered options**: a single shared bucket/season threshold (rejected — makes the season tier vestigial); a separate Mart-3-specific rarity floor distinct from the order-quantity rule's cutoff (rejected — two overlapping "rare" definitions add complexity with no validation data yet to justify tuning them apart); building a real backtest harness now to tune all of this empirically (rejected as out of scope until Track 1 exists and there's a combined forecast to validate against — deferred, not abandoned).

**Consequences**: Mart 3's population is now narrower than "all Acute-Patient visits" — a majority of the catalog, by drug count, never receives a Track 2 statistical estimate at all. The rare-drug patient count must be computed as-of each snapshot date (trailing 12 months ending at `as_of_date`), matching the Point-in-Time Correctness discipline ([[0002-point-in-time-correctness]]), not as a live/full-history count. Not yet implemented in `pipeline/marts.py` — design only.

**Update (implementing #36)**: The first full backtest (v0.3, 185-day test window) put Track 2 at WAPE 3.24, forecasting ~3× the Acute consumption actually dispensed. Two biases, pulling in opposite directions:

1. **Wrong estimand.** Each bucket held the mean 소모량 *per dispensing event*, and Track 2 forecast that for every non-rare drug, every day. Days the drug wasn't dispensed were missing from the average, so a drug dispensed once a week was still forecast at a full dispensing on all seven days. Replaying Track 2 alone over the test window: 3.4× actual volume.
2. **Retroactive population.** Mart 3's history kept only customers who were Acute *as of the snapshot*, so every visit a now-Chronic customer made while still Acute was dropped. Those customers make most of Track 2's actual demand (69% of Acute consumption in the test window comes from customers who later turn Chronic). With the estimand fixed but this left alone, Track 2 fell to ~0.3× actual volume.

**Decision**: a bucket holds **expected daily consumption** — total 소모량 over the bucket's calendar days, counting days with no dispensing as zeros. A drug's calendar starts at its first Acute dispensing, so a newly-introduced drug isn't diluted by days before it existed. `N_bucket` and `N_season` now count **calendar days** in the drug's history, not dispensings: a bucket with enough days but no dispensing (the closed Sunday) is a genuine 0 rather than something to back off from. The placeholders stay 5 and 15, untuned. Mart 3's population — both the grid and the rare-drug patient count — classifies **each visit as of its own 내방일**: a visit on or before the customer's Chronic-since Date is Acute. That includes the visit on the Chronic-since Date itself, which Track 1 didn't score the day before and which the backtest attributes to Track 2's actual.

**Considered options** (replaying Track 2 alone over the test window, rebuilding Mart 3 weekly; WAPE here is per drug per day):

| Estimand / population / thresholds | Volume vs actual | WAPE |
|---|---|---|
| Per-dispensing mean, snapshot population (before) | 3.43× | 3.62 |
| Daily, snapshot population, day thresholds | 0.27× | 1.04 |
| Daily, per-visit population, day thresholds from extract start | 0.84× | 1.27 |
| **Daily, per-visit population, day thresholds from drug's first dispensing** | **0.94×** | **1.31** |
| Daily, per-visit population, dispensing-count thresholds | 1.02–1.14× | 1.44–1.50 |

The 0.27× option has the lowest WAPE, but only because WAPE rewards under-forecasting intermittent demand: forecasting 0 everywhere scores 1.0. Volume matching decided the choice, since Track 2's output feeds order quantities.

**Consequences**: A drug first dispensed only days ago has every bucket backed off to its overall daily average over those few days. On its first day that average is a whole dispensing, so the old overshoot comes back for brand-new drugs until enough days accumulate (the rare-drug filter keeps most such drugs out of Mart 3 anyway). Track 2 now forecasts a fraction of a dispensing per drug per day, so its per-day error is dominated by whether the drug is dispensed at all — expect per-drug daily WAPE near or above 1 even when volume is right. Rare Acute drugs (~16% of Acute consumption in the test window, measured under the old population) still get no Track 2 forecast at all. The 100%-allocation rule this ADR routes them to exists only on Track 1's side ([[0004-track1-rare-drug-population-and-allocation]]).
