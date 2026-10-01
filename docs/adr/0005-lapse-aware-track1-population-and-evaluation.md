---
status: accepted
---

# Track 1 learns lapse from post-cycle negatives, excludes Lapsed customers, scores only the Current Regimen, and is evaluated on full daily snapshots

The first full backtest (issue #24, v0.3 extract, 182 days) reported WAPE ≈130: Track 1 predicted ~80× the actual Chronic demand. Two causes:

1. **The model never saw a customer stop coming back.** Positives exist for every real visit (dated the day before it, including returns after long absences), but Negative Sampling only placed Y=0 rows inside a patient-cycle (≤96% of 처방조제일수), and a customer's last visit anchored no cycle. Every long-gap training row was therefore a positive, so the model learned "overdue ⇒ visits tomorrow". Chronic status is permanent once earned, so on 2025-09-08 all 2,637 Chronic customers were scored — including 1,102 unseen for 180+ days — at a mean probability of 0.83 against 35 actual visits. Validation AUC (0.94) hid this because validation rows were sampled the same way.
2. **Track 1 multiplied each probability by every drug the customer had ever received** (all of Mart 2), not what they currently take.

**Decisions**:

- **Post-cycle negatives.** Negative Sampling continues past each cycle's expected end at ~1.25×, 1.5×, 2×, 3×, 4× 처방조제일수, plus one sample just before the Lapse Horizon, each kept only if it precedes the next visit and lies within the horizon. This includes cycles anchored by a customer's last known visit. The Next-Day label needs no censoring handling: "no visit on d+1" is known for every d before the extract's end.
- **Lapsed Chronic Patient.** A Chronic customer with no visit in the last 180 days (the provisional Lapse Horizon) stays Chronic but is outside Track 1's inference population. Their return is forecast by neither track — an accepted gap (~1.8% of Chronic visits), since Mart 3 still excludes every Chronic customer.
- **Current Regimen.** Track 1 demand uses only the drugs on the customer's Anchoring Visit. Mart 2 itself is unchanged.
- **Chronic-since Date.** Chronic membership as of *d* is "the later visit of the customer's first Revisit-Matched pair is ≤ *d*". That's one date per customer, so every consumer — `build_marts`, the Mart 1 Training Set, and the validation/test snapshots — shares the same as-of-correct population. This supersedes the full-history shortcut the Mart 1 Training Set took under [[0002-point-in-time-correctness]] ("Update (wiring 학습_가중치/negative sampling into Mart 1)").
- **Evaluation on the inference distribution.** In the Temporal Split, only train stays sampled; validation and test are full daily snapshots (every non-Lapsed Chronic customer, every day).
- **Calibration and cutoff.** The training base rate is an artifact of sampling, while the real daily Chronic visit rate is ~1.3%. Raw scores are therefore Platt-calibrated on the validation daily snapshots, and the test window checks that Σp is in the same order of magnitude as actual visits. The Chronic cutoff (Visit List and rare-drug allocation) is re-tuned on calibrated validation probabilities to reach ≥70% recall with a ≥20% precision floor. These targets are provisional. Rejected alternatives: an analytic prior-shift correction, which is invalid because negatives aren't sampled uniformly at random; isotonic calibration, whose ties blur the cutoff; and a top-K Visit List, which ignores how many visits a given day actually expects.

**Considered options**:

- *Lapse cap only, training unchanged* — rejected: the model still believes "overdue ⇒ visits tomorrow" inside the cap (31–90-day-gap customers scored 0.93–0.99).
- *Post-cycle negatives only, no cap* — rejected: scoring customers unseen for years is noise, and a cap bounds how far post-cycle sampling must reach.
- *Lapse horizon relative to the customer's own cycle* — rejected in favour of a fixed 180 days, which is simpler to explain; the model now handles finer-grained lapse itself.
- *Daily, fixed-step, or absolute-day post-cycle sampling* — rejected: daily explodes the training set; cycle multiples keep per-customer density bounded and match the existing relative-proportion rule.
- *Current Regimen alternatives*, measured on 38,409 consecutive Chronic visit pairs. Scores are next-visit drug-set Jaccard and predicted ÷ actual consumption:
  - latest visit: 0.73 / 0.99× — **chosen**
  - latest chronic-pattern visit: 0.73 / 1.01×, identical on 97% of pairs, so not worth the extra rule
  - drugs with supply still active: 0.65 / 1.11×, collapsing to ~0 for returning customers
  - full history: 0.51 / 2.34×
- *Routing Lapsed customers into Mart 3* — rejected: it would pull chronic-regimen history into the acute seasonal statistics, for ~1.8% of visits.
- *Keep sampled validation/test and gate on the backtest only* — rejected: early stopping on a distribution the model never meets at inference is how this bug stayed hidden.

**Consequences**: The ~1:5 positive:negative ratio no longer holds and is not a target. Before replacing `build_marts`'s per-date Revisit Match with the Chronic-since Date, verify on the extract that the two give identical Chronic populations. The Lapse Horizon joins the provisional Decision Thresholds.

**Update (implementing #26)**: Verified on the v0.3 extract before switching `build_marts` over. The Chronic-since population and the per-date Revisit Match population are identical on all 182 dates of the backtest window (2025-09-07 → 2026-03-07). The Chronic-since Date takes the top-2 high-frequency drug exclusion from the full extract, not from each date's as-of-filtered visits. In this window that makes no difference, because the as-of top-2 set is already the full extract's {917, 2561}. Before 2024-11 the as-of top-2 set was {2561, 6202}, and the two definitions differ by 0–5 customers per snapshot. This affects only early training-history snapshots, and it was accepted rather than reproducing a per-date top-2 set. Details are in issue #26.
