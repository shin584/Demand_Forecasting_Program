# Demand Forecast System

A two-track pharmacy demand-forecasting system: chronic-patient demand is predicted deterministically (visit probability × last prescription quantity), acute-patient demand is predicted statistically (seasonal/day-of-week averages). Output is a per-drug order-quantity table feeding automated reordering.

## Labels & Classification

**Next-Day Visit (내일_방문, Y)**:
The literal boolean label used to train Track 1: did this customer walk into the pharmacy on the calendar day immediately following the prediction date? This is a separate concept from chronic-patient classification below — it is not derived from revisit-window matching.
_Avoid_: using this interchangeably with "revisit" or "재방문"

**Revisit Match**:
For a given visit, the next later visit by the same customer whose drug-set or diagnosis-set overlaps the current visit's, found within a day-window tolerance around the visit's expected next-visit date (다음내방일). Parameterized by a match criterion (candidates A–H: common-drug count, drug Jaccard similarity, diagnosis-code intersection, or combinations) and a time-tolerance window (candidates T1–T3). Used to behaviorally define chronic-patient status — not used to produce the Y label.
_Avoid_: conflating with 내일_방문/Y

**Chronic Patient (만성질환자)**:
A **customer-level** label: a customer with at least one visit that is a Revisit Match under criterion **H+T2** — some later visit by the same customer shares ≥1 drug with it (excluding the top-2 highest-frequency drugs in the catalog) and falls within ±30 days of that visit's expected next-visit date. Once earned, it is never revoked (see Lapsed Chronic Patient for the long-absence case). Every visit by a Chronic customer — including one-off acute-looking visits — belongs to the Chronic population (Track 1), not Mart 3. Status is evaluated as of a date via the customer's Chronic-since Date. This replaces the old diagnosis-code/duration-based definition everywhere it appeared (Mart 1 inclusion filter, the 만성질환여부 feature, and sample-weight tier 2) and retires the diagnosis-code pseudo-labeling effort entirely. See [[0001-chronic-patient-behavioral-definition]].
_Avoid_: defining chronic status purely from 병명코드 (diagnosis code) or from a duration threshold alone; treating diagnosis-code-based classification as still authoritative anywhere in the pipeline

**Acute Patient (급성질환자)**:
Complement of Chronic Patient, also customer-level: a customer none of whose visits (as of the date in question) is a Revisit Match. All of an Acute customer's visits feed Mart 3.
_Avoid_: classifying individual visits as Chronic or Acute — Revisit Match is a per-visit test, but the Chronic/Acute label it yields belongs to the customer

**Chronic-since Date**:
For a Chronic customer, the date of the *later* visit in their first Revisit-Matched pair — the earliest date on which that match could actually have been observed. A customer is Chronic as of date *d* exactly when their Chronic-since Date ≤ *d*. This one per-customer date gives the as-of-correct Chronic population for any snapshot (inference, backtest, validation/test snapshots, and every Mart 1 Training Set row) without re-running Revisit Match per date. See [[0005-lapse-aware-track1-population-and-evaluation]].

**Lapsed Chronic Patient**:
A Chronic Patient with no visit in the last *L* days before the snapshot date (the Lapse Horizon — provisional, 180 days; see Decision Thresholds). Still Chronic, but outside Track 1's inference population: not scored, not on the Visit List, contributing no Track 1 demand. The horizon is a hard population boundary on top of, not instead of, the model itself learning that a long absence lowers visit probability (see Negative Sampling).
A Lapsed customer who returns is forecast by neither track — Track 1 excludes them, and Mart 3 excludes every Chronic customer. This is an accepted, known gap (~1.8% of Chronic visits in the v0.3 extract come after a 180+ day absence), left to the Safety Stock buffer.
_Avoid_: treating "Lapsed" as a reclassification to Acute or feeding Lapsed customers' visits into Mart 3; relying on the horizon alone to suppress overdue customers' probabilities

## Data Marts

**Mart 1 (Visit-Probability Mart)**:
Chronic-patient training set for Track 1's ML model. X = patient/visit profile features (including MPR adherence score and per-patient no-show rate, both derived purely from `tbl매출` and added alongside the originally-planned feature set, plus 차상위대상자 alongside 보험구분 as copay-sensitivity signals), Y = Next-Day Visit label, sample-weighted by severity/chronic tier (see Severe/특례 Weight Tier). Inclusion filter and negative sampling both key off Chronic Patient / Revisit Match status, not diagnosis code. 주요_진료과 (primary department) is dropped entirely — the underlying data was never extractable (see Test-Log) and isn't proxied by another column. 주요_약품속명 (primary active ingredient) is defined as the 속명 of whichever drug in the visit has the longest 투약일수 — the same drug that drives the 장기투약_일수 feature, so both features name the same "primary drug."
_Avoid_: 처방전발행기관ID as a stand-in for 주요_진료과 — it's an opaque institution ID with no department-level meaning

**Anchoring Visit**:
For a Mart 1 row, the customer's most recent visit on or before that row's snapshot date. Visit-level features (마지막방문_경과일, 남은_약_일수, 내일이_예약일, 장기투약_일수, 주요_약품속명, 보험구분, 차상위대상자) are read from this one visit only: a value missing on the Anchoring Visit stays missing, never back-filled from an earlier visit. A row with no visit on or before its snapshot date has no Anchoring Visit, and all these features are missing.
_Avoid_: "last known value" — e.g. a 차상위대상자 'Y' on an earlier visit does not make a later blank visit 'Y'

**Negative Sampling**:
For building Mart 1's Y=0 examples, the early/mid/late-window sampling scheme (1 sample early, 1 mid, 2–3 late per patient-cycle) covers the inside of each cycle. It is extended **past each cycle's expected end** — continuing until the next visit, or, after a customer's last known visit, up to the Lapse Horizon — so the training set contains customers who are overdue and customers who have stopped coming back. Without this, every long-gap training row was a positive and the model learned "overdue ⇒ visits tomorrow" (issue #24). The old ~1:5 positive:negative ratio no longer holds once post-cycle negatives exist; neither it nor the "1:3" figure from older planning notes is a target.
_Avoid_: citing "1:3" or "1:5" as the target ratio; confining Y=0 samples to inside a patient-cycle

**Mart 1 Training Set**:
The full historical assembly used to actually train Track 1's model, as distinct from Mart 1 itself (`build_marts`'s single as-of-date snapshot, used for daily inference/backtesting). Built by `build_mart1_training_set`: one Next-Day Visit positive row per real visit, plus `sample_mart1_negatives`'s Y=0 rows, spanning many historical snapshot dates per customer — each row carries its own snapshot date (기준일자) precisely so it can later be temporally split (see Temporal Split below). Chronic membership is as-of-correct per row: a row belongs only if its customer's Chronic-since Date ≤ that row's 기준일자, and not if the customer is Lapsed as of it — the same population rule inference uses. This supersedes the earlier full-history shortcut (see [[0005-lapse-aware-track1-population-and-evaluation]], superseding the relevant update in [[0002-point-in-time-correctness]]).
Severity tier (학습_가중치) membership is still derived once from the full visit history: it only weights rows, it is never a model input, so it can't leak into predictions.
_Avoid_: deriving Chronic membership for training rows from the full, unfiltered visit history

**Temporal Split**:
`split_mart1_training_set` divides the Mart 1 Training Set into train/validation/test by 기준일자: test is the most recent 6 months, validation is the 6 months immediately before that, and train is whatever history remains earlier than both. Both eval-window boundaries are computed from the training set's own max 기준일자, not a hardcoded calendar date, so they land on the same two most-recent windows regardless of how much history is available — train is the one window that grows or shrinks with the data, never the other two. A boundary date belongs to the newer of the two windows it separates. Only the train window is drawn from the sampled Mart 1 Training Set; validation and test are **full daily snapshots** — every non-lapsed Chronic customer on every day of the window, exactly the population and label distribution Track 1 sees at inference — so early stopping and evaluation metrics can't be flattered by the sampling scheme (validation AUC 0.94 on sampled rows hid a ~60× over-prediction of daily visits; see issue #24).
_Avoid_: shrinking or padding val/test to hit a fixed train:val:test proportion — only train's size is allowed to vary; evaluating on sampled val/test rows

**Current Regimen**:
The drug set on a Chronic customer's Anchoring Visit. Track 1 demand multiplies a customer's visit probability only by Mart 2 amounts for drugs in their Current Regimen — never by every drug they've ever been dispensed. Chosen against the v0.3 extract: the latest visit's drug set matches the next visit's (Jaccard 0.73, predicted ÷ actual consumption 0.99×), while full history over-predicts ~2.3× (Jaccard 0.51); "latest chronic-pattern visit" performed identically, and "drugs whose supply is still active" drops returning customers' drugs entirely.
_Avoid_: using the whole of Mart 2 as a customer's expected basket

**Mart 2 (Customer Drug Profile Mart)**:
Per customer×drug, stores the **latest single-visit consumption amount** (not an average across visits) — deliberately avoiding "the averaging pitfall" where a changed dosage gets smoothed away. Same no-back-fill rule as Anchoring Visit: if that latest visit's 소모량 is missing, the value is missing (that customer×drug contributes nothing to Track 1 demand), never an earlier visit's amount.
_Avoid_: "average consumption" for this mart's value column

**Mart 3 (Acute Drug Statistics Mart)**:
Statistical basis for Track 2. Aggregates consumption by **season** (4 buckets), not calendar month — chosen despite month giving finer granularity, because season keeps per-bucket sample sizes viable given the available data span. Population is Acute-Patient visits further restricted to non-rare drugs: a drug whose as-of-date trailing-12-month patient count is below the rare-drug cutoff (see Decision Thresholds) gets no Mart 3 row at all and is routed to the rare-drug 100%-allocation rule instead — Track 2's statistical backoff never runs on a drug too thin to estimate a distribution from.
_Avoid_: assuming every drug in the Acute population gets a Mart 3 row — rare drugs are excluded from the mart entirely, not backed off within it

**Track 2 Sparse-Bucket Backoff**:
Most drugs don't have enough history to fill every season×weekday bucket (median drug: ~72% of its 28 buckets have zero observations) — this is the typical case, not an edge case. Track 2's statistical estimate uses **hierarchical backoff per drug**: season×weekday average if that bucket has ≥`N_bucket` observations, else that drug's season-only average if the season has ≥`N_season` observations, else that drug's overall average. `N_bucket` and `N_season` are separate thresholds, not one shared value — reusing a single threshold for both made the season tier nearly vestigial, since a season pools up to 7 weekdays of data and so clears any bucket-sized bar almost automatically. Current placeholders (`N_bucket=5`, `N_season=15`), chosen against the real v0.3 extract's bucket/season count distributions, not yet backtested — no Track 2 evaluation harness exists yet in the repo, so these remain provisional like the rest of Decision Thresholds until one does. See [[0003-mart3-population-and-backoff-thresholds]].
_Avoid_: treating `N_bucket` and `N_season` as the same value; assuming these numbers were empirically tuned against forecast accuracy — they weren't, there's nothing yet to tune against

## Business Rules

**Visit-Probability Calibration**:
Track 1's raw model scores are mapped to calibrated probabilities by a sigmoid (Platt) fit on the validation window's full daily snapshots, so that on a typical day the sum of predicted probabilities matches the real number of Chronic next-day visits (~1.3% of Chronic customers in the v0.3 extract). This is needed because the training set's positive rate is an artifact of Negative Sampling, not the real visit rate, and Track 1's expected-value demand assumes calibrated probabilities. The test window is never used for fitting; it checks that Σp stays in the same order of magnitude as actual visits. The Chronic cutoff (see Decision Thresholds) is applied to calibrated probabilities.
_Avoid_: a closed-form prior-shift correction by the sampling ratio — negatives are placed at deliberate cycle positions, not sampled uniformly at random, so it doesn't hold; isotonic calibration, whose many tied probabilities around the cutoff blur the Visit List

**Safety Stock**:
Final order quantity = (Track 1 expected demand + Track 2 statistical demand) × 1.2 buffer. This is buffer-only — no additional round-up to box/minimum-order-unit is applied on top.

**Decision Thresholds (provisional)**:
Chronic visit-probability cutoff (0.3 until re-tuned — see Visit-Probability Calibration: it is to be chosen on the validation snapshots as the value reaching ≥70% recall of next-day Chronic visitors while keeping precision ≥20%, both targets themselves provisional), rare-drug special-handling cutoff (<5 patients, counted over a trailing 12-month window), the Lapse Horizon (180 days since last visit), the safety-stock buffer (×1.2), and the sample-weight scalars (severe/특례 tier starts at 3.0 within a 3.0–5.0 band; chronic tier at 2.0) are all placeholder values, not fixed business requirements from the pharmacist. There is no separate "general-patient" cutoff: Track 1 is Chronic-only end to end (training population, inference population, visit list) per [[0001-chronic-patient-behavioral-definition]] — Plan.md's original two-bucket dynamic threshold (chronic 0.3 / general 0.7) predates that behavioral redefinition and the general-patient branch is dead. Everyone outside the Chronic population is covered by Track 2's population-level statistics, not scored individually by Track 1. They're expected to be tuned empirically against validation-set precision/recall once a model exists. The rare-drug cutoff does double duty: it also gates Mart 3 population inclusion (see Track 2 Sparse-Bucket Backoff) and Track 1's rare-drug allocation rule (see Track 1 Rare-Drug Allocation) — one shared threshold *value* and as-of-date methodology, not two independently-tunable numbers. It is applied to two different populations, though: Mart 3 counts distinct patients over the **Acute** population, while Track 1's rule counts distinct patients over the **Chronic** population — the same drug can be "rare" under one count and not the other, since the two tracks never share a patient population. See [[0004-track1-rare-drug-population-and-allocation]]. Its trailing-12-month patient count is as-of-date correct (recomputed per snapshot, excluding future visits), matching the Point-in-Time Correctness discipline below, not a live/full-history count.

**Track 1 Rare-Drug Allocation**:
For a drug below the rare-drug patient cutoff (counted over the Chronic population — see Decision Thresholds), Track 1's per-drug expected demand is the **sum, over that drug's Chronic patients whose Current Regimen includes it, of each patient's full latest Mart 2 consumption where their predicted visit probability meets the Chronic cutoff** — not a probability-weighted expected value. A patient below the cutoff contributes 0, not a scaled-down amount. This replaces the ordinary expected-value multiplication (probability × Mart 2 consumption) specifically to avoid underestimating stock for drugs too thin to rely on the law of large numbers. See [[0004-track1-rare-drug-population-and-allocation]].
_Avoid_: applying probability-weighted multiplication to rare drugs; averaging or taking the max across qualifying patients instead of summing

**Severe/특례 Weight Tier**:
The sample-weight tier-1 trigger is: 중증암등록대상자 OR 산전산모대상자 OR **희귀난치대상자** (any true → weight 3.0-5.0). 희귀난치대상자 is used as the umbrella severity flag in place of the narrower 산정특례중증난치등록대상자 named in early planning notes — it subsumes that and the other narrow 산정특례 sub-registrations (burn, TB, dementia, etc.) without needing to OR in each one individually. **차상위대상자** (near-poverty-line copay-assistance eligibility) is explicitly excluded from this weight — it's a socioeconomic/copay signal, not a severity signal, and instead belongs as an X-feature alongside 보험구분.
_Avoid_: including 차상위대상자 in the severity weight; citing 산정특례중증난치등록대상자 as the active tier-1 column

**Negative Sampling Windows**:
The early/mid/late day-offsets (~1-5일차 / ~15일차 / ~27-29일차) in the negative-sampling scheme are **relative proportions of each patient's own 처방조제일수 cycle length** (~15% / ~50% / ~90-95%), not fixed absolute day counts — read literally as absolute days they'd be meaningless for the shorter end of the observed duration range (7 to 60+ days). Every in-cycle sample must fall at least two days before the anchoring visit's next visit: one on the day before it is really a Next-Day Visit positive, and one on or after it describes a cycle that's already over — so Y=0 is the actual label, never just assumed. No Mart 1 Training Set row is dated on or after the extract's last 내방일, since its label would fall outside the data. Past the cycle's expected end, the same relative rule continues: post-cycle samples at ~1.25×, 1.5×, 2×, 3×, and 4× 처방조제일수 after the visit, plus one final sample just before the Lapse Horizon — each kept only if it falls at least two days before the customer's next visit and within the horizon.

**Point-in-Time Correctness**:
Any feature sourced from a live cumulative/aggregate table (e.g. 가족총내방/가족총매출 from `tbl가족총매출`, and Mart 2's "latest visit consumption" when used for backtesting) must be recomputed **as of each row's own 내방일** (cumulative up to but excluding that visit) for training and backtesting — never joined as the current live snapshot, which silently leaks future information into past training/eval rows. The current live snapshot is only valid at actual production inference time. This also covers Chronic/Acute (Revisit Match) classification inside `build_marts`: a customer counts as Chronic only once their Chronic-since Date ≤ that call's `as_of_date`, so a visit after the snapshot date can't retroactively make an earlier snapshot's customer "Chronic." The Chronic-since Date makes this same as-of rule cheap enough to apply everywhere — inference, backtest, validation/test snapshots, and the Mart 1 Training Set alike. See [[0002-point-in-time-correctness]] and [[0005-lapse-aware-track1-population-and-evaluation]].
_Avoid_: joining `tbl가족총매출` (or any similar live-aggregate table) directly without an as-of-date recomputation step; computing Chronic/Acute classification inside `build_marts` from the full, unfiltered `raw_visits` table

## Data Handling

**Deliverable Scope**:
This project's deliverables are a validated per-drug order-quantity table/CSV and the Visit List below. Integration with a live automated ordering system is explicitly out of scope for this phase — no such system exists in this repo today.

**Visit List**:
A per-run, customer-level output: every Chronic patient whose predicted next-day visit probability meets the Chronic cutoff, for pharmacist review ahead of the order-quantity table. It's a byproduct of the same Track 1 inference pass that produces the order-quantity table's chronic-demand component, not a separately-modeled output.

**PII Policy**:
Extractions select minimal columns only (excluding name, SSN, and other non-essential identifying fields from `tbl고객` and related tables). Raw extracts must never be committed to any shared or public location — local/offline use only.
