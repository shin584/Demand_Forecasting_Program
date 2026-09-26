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
A patient classified via Revisit Match, using criterion **H+T2**: a later visit exists with ≥1 common drug (excluding the top-2 highest-frequency drugs in the catalog) within ±30 days of the current visit's expected next-visit date. This replaces the old diagnosis-code/duration-based definition everywhere it appeared (Mart 1 inclusion filter, the 만성질환여부 feature, and sample-weight tier 2) and retires the diagnosis-code pseudo-labeling effort entirely. See [[0001-chronic-patient-behavioral-definition]].
_Avoid_: defining chronic status purely from 병명코드 (diagnosis code) or from a duration threshold alone; treating diagnosis-code-based classification as still authoritative anywhere in the pipeline

**Acute Patient (급성질환자)**:
Complement of Chronic Patient — visits that don't match the Revisit Match pattern.

## Data Marts

**Mart 1 (Visit-Probability Mart)**:
Chronic-patient training set for Track 1's ML model. X = patient/visit profile features (including MPR adherence score and per-patient no-show rate, both derived purely from `tbl매출` and added alongside the originally-planned feature set, plus 차상위대상자 alongside 보험구분 as copay-sensitivity signals), Y = Next-Day Visit label, sample-weighted by severity/chronic tier (see Severe/특례 Weight Tier). Inclusion filter and negative sampling both key off Chronic Patient / Revisit Match status, not diagnosis code. 주요_진료과 (primary department) is dropped entirely — the underlying data was never extractable (see Test-Log) and isn't proxied by another column. 주요_약품속명 (primary active ingredient) is defined as the 속명 of whichever drug in the visit has the longest 투약일수 — the same drug that drives the 장기투약_일수 feature, so both features name the same "primary drug."
_Avoid_: 처방전발행기관ID as a stand-in for 주요_진료과 — it's an opaque institution ID with no department-level meaning

**Negative Sampling**:
For building Mart 1's Y=0 examples, the early/mid/late-window sampling scheme (1 sample early, 1 mid, 2–3 late per patient-cycle) is the authoritative spec; its resulting ~1:5 positive:negative ratio is the accepted outcome, not the "1:3" figure that appears in older planning notes.
_Avoid_: citing "1:3" as the target ratio

**Mart 1 Training Set**:
The full historical assembly used to actually train Track 1's model, as distinct from Mart 1 itself (`build_marts`'s single as-of-date snapshot, used for daily inference/backtesting). Built by `build_mart1_training_set`: one Next-Day Visit positive row per real visit, plus `sample_mart1_negatives`'s Y=0 rows, spanning many historical snapshot dates per customer — each row carries its own snapshot date (기준일자) precisely so it can later be temporally split (see Temporal Split below). Chronic/severity population membership is derived once from the full, unfiltered visit history (matching `sample_mart1_weights`/`sample_mart1_negatives`'s own live/full-history-by-default behavior), not re-derived per row's snapshot date — only the per-row X-features are as-of-correct. See [[0002-point-in-time-correctness]] ("Update (wiring 학습_가중치/negative sampling into Mart 1)").
_Avoid_: assuming this shares Mart 1's per-row leakage-safety property for Chronic/severity status — that's a population-level decision here, not a per-row one

**Temporal Split**:
`split_mart1_training_set` divides the Mart 1 Training Set into train/validation/test by 기준일자: test is the most recent 6 months, validation is the 6 months immediately before that, and train is whatever history remains earlier than both. Both eval-window boundaries are computed from the training set's own max 기준일자, not a hardcoded calendar date, so they land on the same two most-recent windows regardless of how much history is available — train is the one window that grows or shrinks with the data, never the other two. A boundary date belongs to the newer of the two windows it separates.
_Avoid_: shrinking or padding val/test to hit a fixed train:val:test proportion — only train's size is allowed to vary

**Mart 2 (Customer Drug Profile Mart)**:
Per customer×drug, stores the **latest single-visit consumption amount** (not an average across visits) — deliberately avoiding "the averaging pitfall" where a changed dosage gets smoothed away.
_Avoid_: "average consumption" for this mart's value column

**Mart 3 (Acute Drug Statistics Mart)**:
Statistical basis for Track 2. Aggregates consumption by **season** (4 buckets), not calendar month — chosen despite month giving finer granularity, because season keeps per-bucket sample sizes viable given the available data span. Population is Acute-Patient visits further restricted to non-rare drugs: a drug whose as-of-date trailing-12-month patient count is below the rare-drug cutoff (see Decision Thresholds) gets no Mart 3 row at all and is routed to the rare-drug 100%-allocation rule instead — Track 2's statistical backoff never runs on a drug too thin to estimate a distribution from.
_Avoid_: assuming every drug in the Acute population gets a Mart 3 row — rare drugs are excluded from the mart entirely, not backed off within it

**Track 2 Sparse-Bucket Backoff**:
Most drugs don't have enough history to fill every season×weekday bucket (median drug: ~72% of its 28 buckets have zero observations) — this is the typical case, not an edge case. Track 2's statistical estimate uses **hierarchical backoff per drug**: season×weekday average if that bucket has ≥`N_bucket` observations, else that drug's season-only average if the season has ≥`N_season` observations, else that drug's overall average. `N_bucket` and `N_season` are separate thresholds, not one shared value — reusing a single threshold for both made the season tier nearly vestigial, since a season pools up to 7 weekdays of data and so clears any bucket-sized bar almost automatically. Current placeholders (`N_bucket=5`, `N_season=15`), chosen against the real v0.3 extract's bucket/season count distributions, not yet backtested — no Track 2 evaluation harness exists yet in the repo, so these remain provisional like the rest of Decision Thresholds until one does. See [[0003-mart3-population-and-backoff-thresholds]].
_Avoid_: treating `N_bucket` and `N_season` as the same value; assuming these numbers were empirically tuned against forecast accuracy — they weren't, there's nothing yet to tune against

## Business Rules

**Safety Stock**:
Final order quantity = (Track 1 expected demand + Track 2 statistical demand) × 1.2 buffer. This is buffer-only — no additional round-up to box/minimum-order-unit is applied on top.

**Decision Thresholds (provisional)**:
Chronic visit-probability cutoff (0.3), rare-drug special-handling cutoff (<5 patients, counted over a trailing 12-month window), the safety-stock buffer (×1.2), and the sample-weight scalars (severe/특례 tier starts at 3.0 within a 3.0–5.0 band; chronic tier at 2.0) are all placeholder values, not fixed business requirements from the pharmacist. There is no separate "general-patient" cutoff: Track 1 is Chronic-only end to end (training population, inference population, visit list) per [[0001-chronic-patient-behavioral-definition]] — Plan.md's original two-bucket dynamic threshold (chronic 0.3 / general 0.7) predates that behavioral redefinition and the general-patient branch is dead. Everyone outside the Chronic population is covered by Track 2's population-level statistics, not scored individually by Track 1. They're expected to be tuned empirically against validation-set precision/recall once a model exists. The rare-drug cutoff does double duty: it also gates Mart 3 population inclusion (see Track 2 Sparse-Bucket Backoff) and Track 1's rare-drug allocation rule (see Track 1 Rare-Drug Allocation) — one shared threshold *value* and as-of-date methodology, not two independently-tunable numbers. It is applied to two different populations, though: Mart 3 counts distinct patients over the **Acute** population, while Track 1's rule counts distinct patients over the **Chronic** population — the same drug can be "rare" under one count and not the other, since the two tracks never share a patient population. See [[0004-track1-rare-drug-population-and-allocation]]. Its trailing-12-month patient count is as-of-date correct (recomputed per snapshot, excluding future visits), matching the Point-in-Time Correctness discipline below, not a live/full-history count.

**Track 1 Rare-Drug Allocation**:
For a drug below the rare-drug patient cutoff (counted over the Chronic population — see Decision Thresholds), Track 1's per-drug expected demand is the **sum, over that drug's Chronic patients, of each patient's full latest Mart 2 consumption where their predicted visit probability meets the Chronic cutoff** — not a probability-weighted expected value. A patient below the cutoff contributes 0, not a scaled-down amount. This replaces the ordinary expected-value multiplication (probability × Mart 2 consumption) specifically to avoid underestimating stock for drugs too thin to rely on the law of large numbers. See [[0004-track1-rare-drug-population-and-allocation]].
_Avoid_: applying probability-weighted multiplication to rare drugs; averaging or taking the max across qualifying patients instead of summing

**Severe/특례 Weight Tier**:
The sample-weight tier-1 trigger is: 중증암등록대상자 OR 산전산모대상자 OR **희귀난치대상자** (any true → weight 3.0-5.0). 희귀난치대상자 is used as the umbrella severity flag in place of the narrower 산정특례중증난치등록대상자 named in early planning notes — it subsumes that and the other narrow 산정특례 sub-registrations (burn, TB, dementia, etc.) without needing to OR in each one individually. **차상위대상자** (near-poverty-line copay-assistance eligibility) is explicitly excluded from this weight — it's a socioeconomic/copay signal, not a severity signal, and instead belongs as an X-feature alongside 보험구분.
_Avoid_: including 차상위대상자 in the severity weight; citing 산정특례중증난치등록대상자 as the active tier-1 column

**Negative Sampling Windows**:
The early/mid/late day-offsets (~1-5일차 / ~15일차 / ~27-29일차) in the negative-sampling scheme are **relative proportions of each patient's own 처방조제일수 cycle length** (~15% / ~50% / ~90-95%), not fixed absolute day counts — read literally as absolute days they'd be meaningless for the shorter end of the observed duration range (7 to 60+ days).

**Point-in-Time Correctness**:
Any feature sourced from a live cumulative/aggregate table (e.g. 가족총내방/가족총매출 from `tbl가족총매출`, and Mart 2's "latest visit consumption" when used for backtesting) must be recomputed **as of each row's own 내방일** (cumulative up to but excluding that visit) for training and backtesting — never joined as the current live snapshot, which silently leaks future information into past training/eval rows. The current live snapshot is only valid at actual production inference time. This also covers Chronic/Acute (Revisit Match) classification inside `build_marts`: it's recomputed from only the visits at or before that call's `as_of_date`, so a visit after the snapshot date can't retroactively make an earlier snapshot's customer "Chronic." See [[0002-point-in-time-correctness]].
_Avoid_: joining `tbl가족총매출` (or any similar live-aggregate table) directly without an as-of-date recomputation step; computing Chronic/Acute classification inside `build_marts` from the full, unfiltered `raw_visits` table

## Data Handling

**Deliverable Scope**:
This project's deliverables are a validated per-drug order-quantity table/CSV and the Visit List below. Integration with a live automated ordering system is explicitly out of scope for this phase — no such system exists in this repo today.

**Visit List**:
A per-run, customer-level output: every Chronic patient whose predicted next-day visit probability meets the Chronic cutoff, for pharmacist review ahead of the order-quantity table. It's a byproduct of the same Track 1 inference pass that produces the order-quantity table's chronic-demand component, not a separately-modeled output.

**PII Policy**:
Extractions select minimal columns only (excluding name, SSN, and other non-essential identifying fields from `tbl고객` and related tables). Raw extracts must never be committed to any shared or public location — local/offline use only.
