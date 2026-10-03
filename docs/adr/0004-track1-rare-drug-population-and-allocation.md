---
status: accepted
---

# Track 1's rare-drug exception counts the Chronic population and sums full allocations, not Mart 3's Acute count or expected-value multiplication

CONTEXT.md's rare-drug cutoff was described as "one shared definition" used by both Mart 3 (excluding thin drugs from Track 2's statistical backoff, per [[0003-mart3-population-and-backoff-thresholds]]) and Track 1's own 100%-allocation exception for low-volume drugs (Research-Log.md's "재고 부족 상황 회피 방안"). In practice the only implementation of that cutoff, `_rare_drug_ids` in `pipeline/marts.py`, counts distinct patients over `_mart3_eligible_visits` — the **Acute** population, by Mart 3's own design. Track 1's exception is a different question entirely: whether a drug has too few **Chronic** patients (the population Track 1's expected-value calculation, probability × Mart 2 latest consumption, actually runs over) to trust the law of large numbers. Reusing `_rare_drug_ids` as-is for Track 1 would silently answer the wrong question.

Separately, neither Plan.md nor Research-Log.md specifies how the 100%-allocation exception combines across multiple qualifying patients on the same rare drug, once triggered.

**Decision**: introduce a Chronic-population counterpart to `_rare_drug_ids` — same `rare_drug_patient_threshold` value and as-of-date/trailing-12-month methodology, but counting distinct Chronic patients on that drug instead of Acute ones. The two counts are independent: the same drug can be "rare" under one and not the other, since Track 1 and Track 2 never share a patient population. For a drug that trips the Chronic-population cutoff, Track 1's per-drug expected demand is the **sum, over that drug's Chronic patients, of each patient's full latest Mart 2 consumption where their predicted visit probability meets the Chronic cutoff** — patients below the cutoff contribute 0, not a scaled-down amount, and no probability-weighted multiplication is applied to any patient on that drug.

**Considered options**:
- Reuse `_rare_drug_ids` (Acute-scoped) directly for Track 1 — rejected, it counts the wrong population and would misclassify rarity for the population Track 1 actually serves.
- Maintain a single blended patient count across both populations — rejected, it would conflate two disjoint patient sets (a drug's Chronic and Acute patients never overlap, since Chronic/Acute is a per-patient classification) into a number that answers neither track's question correctly.
- Expected-value multiplication (probability × consumption) even for rare drugs — rejected; this is exactly the underestimation risk the exception exists to avoid, since the law-of-large-numbers assumption behind multiplication doesn't hold for a handful of patients.
- Average or max across qualifying patients instead of summing — rejected; averaging would understate the drug needed when multiple patients qualify (each needs their own dose, not a shared average), and max ignores every qualifying patient but one.

**Consequences**: the rare-drug cutoff remains one shared *threshold value and methodology* (as CONTEXT.md's Decision Thresholds describes) but is now explicitly two independent *counts* over two disjoint populations — future readers must not assume a single boolean "is this drug rare" applies to both tracks. The final combined per-drug output table is a union of Track 1's and Track 2's drug sets, zero-filled on whichever side is absent, since there is no independently-extracted full drug catalog in this repo to enumerate a canonical list from instead. Not yet implemented in `pipeline/marts.py` or any Track 1 module — design only, to be built as part of the Track 1 + Track 2 model-building/inference phase.
