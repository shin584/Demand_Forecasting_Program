---
status: accepted
---

# Family-loyalty and Mart 2 features must be recomputed as-of each row's visit date

`Extract_By_CSV.py` joins `tbl가족총매출` (가족총내방/가족총매출) as a single live snapshot, attached identically to every historical row for a family regardless of that row's own 내방일. We confirmed this empirically: for families with visits spanning >180 days, the value is constant across every row (n=1018 distinct-value-count-of-1 vs 0 elsewhere) — it's the current cumulative total as of extraction date (2026-01-16), not a historical time series.

Used as-is, this bakes look-ahead bias directly into Mart 1's training set: a 2023-dated training row would see 2025-level family loyalty as if already known in 2023. The 2yr/6mo/6mo temporal split exists specifically to prevent this kind of leakage, but a static snapshot join defeats it regardless of how the split is drawn. The same risk applies to Mart 2 (latest per-customer×drug consumption) whenever it's used to backtest Track 1+2's combined demand forecast on the validation/test period, rather than only at live production inference.

**Decision**: recompute 가족총내방/가족총매출 (and Mart 2, when used for backtesting) as of each row's own 내방일 — cumulative up to but excluding that visit — for both training and backtesting. The live current snapshot is legitimate only at actual production inference time, where "now" genuinely is now. This is a real trade-off (as-of recomputation is materially more expensive to build than a static join) accepted because the alternative produces a model that validates well in a leaky backtest and then underperforms in production — a train/serve skew that would otherwise surface late and be hard to diagnose.

**Update (implementing #4)**: 가족총내방 is recomputed as designed (`family_totals_as_of` in `pipeline/marts.py`, a count of distinct visits). 가족총매출 could not be: `raw_visits` (Extract_By_CSV.py's output) carries no per-visit monetary amount anywhere — only 소모량, a drug consumption quantity in units that aren't comparable across drugs, whereas `tbl가족총매출.총매출` is a genuine `money`-typed column with no per-visit equivalent extracted. Recomputing 가족총매출 point-in-time needs a raw-extract change first (a per-visit revenue/amount column) — tracked in #10. Mart 2's as-of recomputation was implemented as designed, since it only depends on 내방일 and 소모량, both already present per visit.
