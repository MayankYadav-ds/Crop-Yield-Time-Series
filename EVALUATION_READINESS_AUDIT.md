# Final evaluation readiness audit

## Verdict

The expanded submission is ready for a transparent demonstration of the implemented and locally verified scope. It is not a certification of full institutional compliance, a successful hosted Colab run, a successful Prophet fit, or 100% forecasting accuracy. The evidence distinguishes successful models from excluded models.

The original working submission is unchanged and its pre-change ZIP backup was checked file by file. New deliverables are in FINAL_EVALUATION_SUBMISSION. Total paid-service expenditure: ₹0; all experiments use local CPU.

## Requested audit actions

| Action | Verified result | Qualification |
|---|---|---|
| Sequential notebook execution | PASS: 39 cells, 19 code cells completed in a real Jupyter kernel; zero error outputs; 29 genuine PNG outputs | Prophet’s caught optimizer failure is a recorded model exclusion, not a successful fit |
| Statistics, figures and results agree | PASS: raw counts checked; all 36 model/protocol metric rows independently recalculated; report and presentation tables checked; 9 native chart data sets checked | Chart values use 12 significant digits for Excel portability; CSVs retain full precision |
| All three milestones | Functional evidence mapped in MILESTONE_COMPLIANCE.md | Exact faculty-PDF compliance is UNVERIFIED because official requirement documents were unavailable |
| Google Colab | Installation-command construction and upload adapter cases tested locally; CPU mode and portable paths verified | Browser was signed out of Google; hosted Colab/Linux installation was not executed |
| Expanded XGBoost tuning | PASS: 12 completed seeded chronological candidates; selected using 2010–2011 validation RMSE | The previously winning setting remains best; no additional improvement is invented |
| LSTM / GRU / patch Transformer | PASS: all three genuinely trained, checkpointed and evaluated on common district targets | Small architectures, three-year input; patch model is not an exact PatchTST reproduction |
| Prophet | Implemented for longer crop aggregates; native optimization attempted | FAILED/EXCLUDED locally; no Prophet metrics or successful training claim |
| Exclusions | Explicit failure logs, applicability diagnostics and MODEL_EXCLUSIONS.md | No policy bypass, invented fit or unsupported seasonality |
| Regenerated artifacts | Seven-page editable report/PDF, sixteen-slide editable presentation/PDF, executed notebook | Old working files preserved; the new presentation supersedes obsolete slide claims |
| Teacher preparation | Exactly 50 short Hinglish answers and an 11-minute demonstration script | Contribution claims require actual team records |

## Experimental facts

- Confirmed input: crop_production.csv, 246,091 rows and seven columns; 3,730 missing production entries; no exact or seasonal business-key duplicates.
- Raw labels: 1997–2015; modeled matched panel: 2005–2014. Raw 2015 reporting has only 562 records.
- Annual panel: 210,131 accepted groups. Accepted raw rows 241,302; invalid/missing-component rows 3,997; Whole Year overlap rows 792. No automatic outlier removal or production imputation.
- Four crops, 1,112 complete histories, 11,120 modeled observations and 20 modeled state labels.
- Outer validation targets: 2010 and 2011, at origins 2009 and 2010. Inner neural stopping: train through 2008, monitor 2009; patience 8 and maximum 60 epochs.
- Final pre-test fit through 2011. Expanding one-step and fixed-origin 2011 tests both target 2012–2014, with 3,336 common targets per model/protocol.
- Twelve district-level models enter the final leaderboard: eight baseline/classical methods, selected XGBoost, LSTM, GRU and PatchTransformerTiny.
- Neural parameter counts/selected epochs: LSTM 1,233/49; GRU 929/52; patch Transformer 657/31.
- Forecasts are for 2015–2017 beyond the modeled cutoff 2014. They are not verified observations and include a partly reported raw year. No calibrated prediction intervals are available.

## Results and selection

Validation selects XGBoost_Tuned with RMSE 0.5294001588. Its selected configuration remains 250 estimators, depth 3, learning rate 0.03, subsample 0.8, column sampling 0.9 and L2 penalty 10.

| Model | One-step test MAE | RMSE | MAPE % |
|---|---:|---:|---:|
| XGBoost selected on validation | 0.3369 | 0.5617 | 24.77 |
| LSTM | 0.3265 | 0.5457 | 23.56 |
| GRU | 0.3321 | 0.5511 | 25.16 |
| PatchTransformerTiny | 0.3253 | 0.5496 | 22.67 |
| Naive | 0.3298 | 0.5680 | 20.66 |

LSTM has the lowest retrospective one-step RMSE, the patch model the lowest MAE, and Naive the lowest MAPE. Test ranking does not change validation selection. Selected XGBoost fixed-origin RMSE is 0.6283, compared with Naive 0.6191 and LSTM 0.5992. Scores use a numerical production/area scale; physical units are unverified.

## Accuracy and reproducibility checks

Every exported metric was independently recalculated from saved predictions for 36 model/protocol rows. Actual targets match the annual model panel. All four validation/test/future prediction files match the first real-kernel run with maximum numeric difference exactly zero. Feature future-perturbation checks, chronological training-target checks, neural parameter-change checks, checkpoint inference replay and forecast-row alignment passed. The original input SHA-256 remains unchanged.

The final notebook kernel completed in 152.2 seconds; the main experiment portion reports 140.9 seconds. Timings vary by machine and are not guaranteed for free Colab. Training/inference logs distinguish global fits from baseline loop overhead. The extra EDA exports do not change predictions.

## Visual and application verification

The research PDF contains seven A4 pages, a 167-word abstract, eight numbered figures, four tables and ten cited references. Word exported the final editable DOCX; bundled PDFium rendered the pages. Every page was visually inspected. The packaged render_docx.py path was previously diagnosed as unavailable because LibreOffice is absent, so Word/PDFium was used without changing installed security settings.

The presentation has sixteen slides, nine native editable charts and seven native tables. The first-party finalizer passed package, geometry, font, chart-title, data/workbook and import checks. The final deck opened in PowerPoint; all sixteen native slide renders were inspected and a PDF preview was exported. Report and presentation metric cells match the same CSVs at their displayed precision.

## Remaining limitations

1. Official milestone/report PDFs were not supplied or found; the available user brief and archived implementation are not substitutes for faculty specifications. No full marks or unconditional compliance claim is made.
2. Hosted Colab was not signed in. Local tests cover upload payload/schema handling and installer command construction, not an actual Google-hosted upload or Linux install.
3. Prophet’s native optimizer failed; ARIMA imports are blocked locally. Neither receives a fabricated score. GARCH/VAR and seasonal variants are excluded with explicit scope/assumption reasons.
4. Complete-history cohort selection depends on test-era availability, creating survivor selection. The test years were already examined in a previous study; this extension is retrospective, not independent prospective validation.
5. Units, exact dataset lineage/license and district boundary stability remain unverified. Short histories, shared yearly shocks, missingness and area weights limit interpretation.
6. No weather/soil/management variables, calibrated intervals or proven causal effects are available. No significance claim is made for small model-score gaps.
7. Historical milestone presentations are not used as proof of the updated results. The regenerated Final_Presentation.pptx is the current evidence-aligned presentation. Individual contribution evidence was not invented.
