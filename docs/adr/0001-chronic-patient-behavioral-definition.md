---
status: accepted
---

# Chronic patient status is defined behaviorally (Revisit Match), not by diagnosis code

The plan originally leaned on diagnosis code (병명코드) and/or prescription duration to decide whether a patient is "chronic," feeding Mart 1's inclusion filter, the 만성질환여부 feature, and sample-weight tier 2. That's unreliable: ~30-34% of prescriptions have no diagnosis code, and Research-Log.md proposed patching that gap with a Random Forest pseudo-labeling model.

Instead, chronic status is now defined by **Revisit Match**: for a given visit, does a *later* visit by the same customer share ≥1 drug — excluding the top-2 highest-frequency drugs in the catalog, to avoid false matches on near-universal OTC drugs — within ±30 days of that visit's recorded expected next-visit date (criterion "H", time window "T2")? This replaces the diagnosis-code path everywhere it was used and retires the pseudo-labeling effort entirely, since it depends on drug data (0% missing) rather than diagnosis code (30-34% missing).

We evaluated 24 candidate definitions (8 match criteria × 3 time windows) against 3 years of actual visit data (168k rows, v0.3 extract) before picking H+T2 (74.4% resulting rate). Diagnosis-code-based criteria (E, G) were rejected specifically because their lower rates (~28-34%) reflect missing-data coverage, not genuine non-recurrence. The 74.4% rate itself was sanity-checked against a random-pair baseline (6.6% coincidental overlap between different customers) and a same-customer/no-timing baseline (81.6%) — both confirm this pharmacy's customer base is genuinely repeat-heavy rather than the number being a matching artifact.
