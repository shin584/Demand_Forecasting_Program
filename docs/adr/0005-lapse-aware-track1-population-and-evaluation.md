---
status: accepted
---

# Track 1 learns lapse from post-cycle negatives, excludes Lapsed customers, scores only the Current Regimen, and is evaluated on full daily snapshots

The first full backtest (issue #24, v0.3 extract, 182 days) reported WAPE ≈130: Track 1 predicted ~80× the actual Chronic demand. Two causes:

1. **The model never saw a customer stop coming back.** Positives exist for every real visit (dated the day before it, including returns after long absences), but Negative Sampling only placed Y=0 rows inside a patient-cycle (≤96% of 처방조제일수), and a customer's last visit anchored no cycle. Every long-gap training row was therefore a positive, so the model learned "overdue ⇒ visits tomorrow". Chronic status is permanent once earned, so on 2025-09-08 all 2,637 Chronic customers were scored — including 1,102 unseen for 180+ days — at a mean probability of 0.83 against 35 actual visits. Validation AUC (0.94) hid this because validation rows were sampled the same way.
2. **Track 1 multiplied each probability by every drug the customer had ever received** (all of Mart 2), not what they currently take.

**Decisions**:

- **Post-cycle negatives.** Negative Sampling continues past each cycle's expected end at ~1.25×, 1.5×, 2×, 3×, 4× 처방조제일수, plus one sample just before the Lapse Horizon, each kept only if it precedes the next visit and lies within the horizon. This includes cycles anchored by a customer's last known visit. The Next-Day label needs no censoring handling: "no visit on d+1" is known for every d before the extract's end.
- **Lapsed Chronic Patient.** A Chronic customer with no visit in the last 180 days (the provisional Lapse Horizon) stays Chronic but is outside Track 1's inference population. Their return is forecast by neither track — an accepted gap (~1.8% of Chronic visits), since Mart 3 still excludes every Chronic customer's visits after their Chronic-since Date (see [[0003-mart3-population-and-backoff-thresholds]], "Update (implementing #36)").
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

**Update (implementing #35)**: Post-cycle negatives dated at fixed multiples of 처방조제일수 let the model recognise them. 마지막방문_경과일 + 남은_약_일수 recovers the cycle length, so "is 경과일 exactly on the sampling grid?" separated the classes. 99.4% of training negatives sat on the grid, against 35.4% of positives, and the model learned "off the grid ⇒ visits tomorrow" (inference mean p 0.877 off the grid). Negatives are now **jittered within windows, weighted by window width**:

- Each anchoring visit's sample points (the in-cycle and post-cycle multiples within the Lapse Horizon, plus the day before it) are sorted. Each point gets a window running between the midpoints with its neighbours. The first window starts on day 1 and the last ends on the horizon's last day, so the windows tile the horizon. Edges are rounded to whole days, and windows that round to nothing are dropped, which is how short cycles collapse.
- One negative is drawn on a uniformly random day of each window, from a seeded RNG, so the training set is reproducible. The next-visit and extract-end bounds are applied after the draw rather than by shrinking the window. That keeps each window's expected contribution equal to its number of in-bounds days.
- Each negative's 학습_가중치 is multiplied by its window's width, normalised to mean 1 over the training set's negatives. Positives keep their tier weight. Every in-bounds day then counts the same in expectation, as under daily sampling, so window width isn't a pattern the model can learn by 경과일. Platt calibration only has to correct the overall base rate.
- Rejected: uniform random days across each anchor's whole range, which gives up the window structure.

On v0.3, with the model retrained and scored on 2025-09-08, 33.9% of training negatives fall on the old grid (35.4% of positives). Raw mean p per 마지막방문_경과일 bucket is now 0.194 / 0.177 / 0.126 / 0.102 for 0–30 / 31–60 / 61–90 / 91–180 days, down from 0.641 / 0.866 / 0.935 / 0.959. Σp is 252.5 against 33 actual visits (was 1,192). Sampled-validation AUC falls from 0.937 to 0.725, because the grid leak had been inflating it. Details are in issue #35.

**Update (implementing #32)**: The ~1.3% daily rate above was measured over every Chronic customer. Over Track 1's actual population, the non-Lapsed Chronic customers, the v0.3 validation window's rate is 2.4%. The Platt calibrator is fit unweighted on the validation snapshots and saved with the classifier (`CalibratedTrack1Model`), so every consumer of a visit probability gets calibrated values. On v0.3 the calibrated validation mean matches the base rate exactly, and the test window's Σp is 6,737 against 6,478 actual visits (ratio 1.04). No cutoff met both of the tuning targets set above (validation AUC 0.85). Reaching 70% recall cost precision 9% (cutoff 0.034, ~270 customers a day on the Visit List), and a 20% precision floor capped recall at 23%.

The **cutoff is now tuned by maximising F1** instead of against those targets. Each training run picks the cutoff that maximises F1 against next-day Chronic visits on the calibrated validation snapshots, breaking ties towards the higher cutoff. This supersedes the "≥70% recall with a ≥20% precision floor" rule in *Calibration and cutoff* above.

- On v0.3 the chosen cutoff is 0.108, with validation precision 17%, recall 35% and F1 0.23 (test F1 0.21), and ~71 customers a day on the Visit List. The previous 0.3 gave F1 0.12 with 7% recall.
- The tuned cutoff is saved with the model, alongside its calibrator (`CalibratedTrack1Model.chronic_visit_prob_cutoff`), and inference and the backtest default to it. `CHRONIC_VISIT_PROB_CUTOFF` (0.11) is only the fallback for a model that carries none.
- Rejected:
  - *a fixed cutoff in the 0.10–0.12 band*: equivalent today, but it goes stale when retraining shifts the calibrated probabilities.
  - *F2*, which weights recall: cutoff 0.073, 48% recall, but ~126 customers a day for the pharmacist to review.

F1 weighs a missed visit the same as a wrong listing. For rare-drug allocation a miss means a stockout, so revisit F-beta if stockouts of rare drugs show up. Details are in issue #32.
