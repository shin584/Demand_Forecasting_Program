Q1 - What is "revisit," and is it one definition or two?: Plan.md describes Y (내일_방문) as a simple literal-next-calendar-day boolean. But preexam_rawdata2.py implements something very different: a multi-day-window match (up to 30 days past the expected exhaustion date) confirmed by drug/disease-set overlap. Test-Log.md:179 then ties the A–H sensitivity analysis (revisit rate 28%–75% depending on criterion) explicitly to "만성질환 분류 로직" (chronic classification), not to the Y label. So: is 내일_방문 (Y) strictly "did they walk in tomorrow" — a separate, simpler thing from "is this patient chronic," which uses the window-matched A–H-style definition? Or did the project intend these to be the same signal?



➡️ Keep them separate: Y stays the literal next-day boolean (matches the described production inference process), and chronic/acute classification uses the window-matched behavioral definition. Which exact A–H criterion to adopt is a follow-up question once this split is confirmed.



---



❓ Q2 - Mart 2 value: latest visit or average?: Plan.md:69 states Mart 2 should store the latest single-visit consumption per customer×drug, explicitly to avoid "평균의 함정" (the averaging pitfall). But Research-Log.md's own worked example table (lines 98–102) labels that same column "1회 내방 시 평균 소모량" (average). These give different numbers whenever a patient's dose has changed over time. Which is the real design?



➡️ Latest — it matches Plan.md's explicitly stated rationale; the Research-Log table header looks like a copy-paste imprecision.



---



❓ Q3 - Track 2 time bucket: season or month?: Mart 3's schema (Plan.md:73) uses "계절" (season, 4 buckets) as a column, but the worked formula in Research-Log.md:133,138 computes statistics over "동월" (same calendar month, 12 buckets). With only ~2.4 years of actual data (see Q4), month buckets give roughly 2 same-weekday-same-month observations per drug — quite sparse. Which granularity is intended?



➡️ Proceed on a seasonal basis.   



---



❓ Q4 - Train/val/test split doesn't fit the actual data span: Plan.md's 2yr/6mo/6mo split assumes 3 years of data, but Test-Log.md:115 shows the validated extract only spans 2023-08-28 to 2026-01-16 — about 2.37 years. How should the split be resized: (a) shrink all three proportionally, (b) keep val/test at 6 months each (protect the eval signal) and shrink train to ~1.4yr, or (c) go back to MSSQL and check whether more than 3 years of raw history actually exists before deciding?



➡️ I re-extract the data. Please examine '/dataset/pharmacy_raw_data_v0.3.csv'.



---



❓ Q5 - Are external time-series factors in scope now?: Research-Log.md:78-83 brainstorms weather, holiday calendar, and flu-epidemic index as inputs, but none of them appear in Plan.md's actual Mart 1/2/3 schemas — they never made it from brainstorm to spec. Build them into this version, or defer?



➡️ Defer to a later version — day-of-week + month already covers most seasonal signal for Track 2, and sourcing external data is a separate data-engineering effort from the modeling work at hand.



---



❓ Q6 - Is there an existing automated ordering system to integrate with?: Plan.md's last step says output "feeds into 자동 발주 시스템" (automated ordering system). I looked through the repo and only found rotating MSSQL .bak backups under D/<요일>/ — no ordering-system code or integration spec. Does one already exist elsewhere that this project must target, or is producing a clean per-drug order-quantity table the actual deliverable for now?



➡️ Treat integration as out of scope for now; deliverable is a validated order-quantity table/CSV, with real-system integration as a later, separate project phase.



---



❓ Q7 - Are the hardcoded thresholds fixed requirements or placeholders?: The plan hardcodes several constants: chronic-patient visit-probability cutoff 0.3, general-patient cutoff 0.7, rare-drug "handle specially" cutoff of <5 patients, and safety-stock buffer +20%. Did these come from the pharmacist as actual business requirements, or are they your own provisional guesses to be tuned once there's a trained model and a validation set to test against?



➡️ Provisional — tune them empirically against validation-set precision/recall and, ideally, an estimated stockout-cost tradeoff, rather than locking them in now.



---



❓ Q8 - Safety stock: buffer, round-up, or both?: Research-Log.md:153 lists "×1.2 buffer" and "round up to box/min-order-unit" joined by "또는" (or), but never states whether they combine. Should the final quantity be: (a) buffer only, (b) box round-up only, (c) buffer applied first, then round up to box size, or (d) round up only, skipping the percentage buffer entirely?



➡️ (a) buffer only



---



❓ Q9 - Any PII/compliance constraint beyond "select minimal columns"?: tbl고객 has 100+ columns including encrypted SSNs and disease-registration flags (cancer, rare disease, etc.) per Test-Log.md:39,46. Extract_By_CSV.py already excludes name/SSN columns in practice, which is good — but is there a stated or implied rule (data must stay on one offline machine, no PII-adjacent columns in anything committed to version control, a retention limit, etc.), or is "pick minimal columns as needed" the entire policy?



➡️ keep it that way, never commit raw extracts to any shared/public location.