---
status: accepted
---

# Mart 3 excludes rare drugs entirely; bucket and season backoff thresholds are separate

Track 2 Sparse-Bucket Backoff's threshold was left as "N, TBD empirically," with a single shared value reused for both bucket-level (drug×season×weekday) and season-level sufficiency. Real distribution analysis of the v0.3 extract (168k rows, 1,075 drugs) showed two problems with leaving it that way: reusing one threshold makes season-sufficiency almost automatic — a season pools up to 7 weekdays of bucket data, so it clears any bucket-sized bar with room to spare, making the season fallback tier do little independent work — and separately, 61.7% of drugs in the catalog have fewer than 5 unique patients, the same population CONTEXT.md's rare-drug cutoff already flags for 100%-allocation stockout handling rather than a modeled forecast.

**Decision**: `N_bucket` and `N_season` are separate, independently-set thresholds (placeholders `N_bucket=5`, `N_season=15`, chosen from the real bucket/season count distributions — median bucket count 4, median season count 8 — not yet backtested, since no Track 2 evaluation harness exists in the repo). Drugs below the rare-drug patient cutoff (<5 patients, as-of-date trailing 12 months) get no Mart 3 row at all: they're routed to the rare-drug 100%-allocation rule instead of a backed-off statistical estimate, reusing that cutoff as Mart 3's own population filter rather than maintaining a second, independently-tunable "rare" threshold.

**Considered options**: a single shared bucket/season threshold (rejected — makes the season tier vestigial); a separate Mart-3-specific rarity floor distinct from the order-quantity rule's cutoff (rejected — two overlapping "rare" definitions add complexity with no validation data yet to justify tuning them apart); building a real backtest harness now to tune all of this empirically (rejected as out of scope until Track 1 exists and there's a combined forecast to validate against — deferred, not abandoned).

**Consequences**: Mart 3's population is now narrower than "all Acute-Patient visits" — a majority of the catalog, by drug count, never receives a Track 2 statistical estimate at all. The rare-drug patient count must be computed as-of each snapshot date (trailing 12 months ending at `as_of_date`), matching the Point-in-Time Correctness discipline ([[0002-point-in-time-correctness]]), not as a live/full-history count. Not yet implemented in `pipeline/marts.py` — design only.
