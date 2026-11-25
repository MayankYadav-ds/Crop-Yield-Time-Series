from pathlib import Path
import os
import json,pandas as pd,numpy as np
from docx import Document
from docx.shared import Inches,Pt,Cm,RGBColor
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
base=Path.cwd();R=Path(os.environ.get('APY_OUTPUT','results'));fig=R/'figures';O=Path(os.environ.get('APY_SUBMISSION','submission'));O.mkdir(exist_ok=True)
s=json.loads((R/'dataset_summary.json').read_text());pr=json.loads((R/'preprocessing_summary.json').read_text());design=json.loads((R/'experimental_design.json').read_text());info=json.loads((R/'best_model_summary.json').read_text());status=json.loads((R/'model_training_status.json').read_text());xsel=json.loads((R/'xgboost_selection.json').read_text());refs=json.loads(Path(__file__).with_name('references.json').read_text(encoding='utf-8'));tl=pd.read_csv(R/'test_leaderboard.csv');fl=pd.read_csv(R/'fixed_origin_leaderboard.csv');vl=pd.read_csv(R/'validation_leaderboard.csv');station_df=pd.read_csv(R/'stationarity_results.csv');met=pd.read_csv(R/'selected_series.csv');logs=pd.read_csv(R/'training_and_inference.csv');best=info['best_by_validation_RMSE'];b=tl.set_index('model').loc[best];n=tl.set_index('model').loc['Naive'];l=tl.set_index('model').loc['LSTM'];bf=fl.set_index('model').loc[best]
# Preserve the established editable report layout.
exec(Path(__file__).with_name('report_layout_helpers.py').read_text(encoding='utf-8'),globals())
p('BENNETT UNIVERSITY',True,17,WD_ALIGN_PARAGRAPH.CENTER);p('CSET 369 Time Series Analysis',False,11,WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(4):p('')
q=d.add_paragraph('Agricultural Crop Yield Forecasting Using Classical and Advanced Time Series Models','Title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
for run in q.runs:run.font.size=Pt(23)
p('Final evaluation edition',False,13,WD_ALIGN_PARAGRAPH.CENTER);p('Course Coordinator  Anurag Choubey',False,11,WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3):p('')
for txt in ['Mayank Yadav — S24CSEU1494']:p(txt,False,12,WD_ALIGN_PARAGRAPH.CENTER)
p('18 November 2026',False,11,WD_ALIGN_PARAGRAPH.CENTER)
sec=d.add_section(WD_SECTION.NEW_PAGE);cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
q=d.sections[0].footer.paragraphs[0];q.clear();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');q._p.append(fld)
abstract=f'''Agricultural forecasting requires consistent annual targets, chronological evaluation and transparent treatment of incomplete reporting. This study analyzes 246,091 crop-production records covering 1997–2015 and constructs annual production-to-area ratios after conservative quality checks. A matched 2005–2014 panel contains 1,112 district–crop histories for Rice, Maize, Wheat and Rapeseed &Mustard. Eight local baseline or classical estimators are compared with a globally tuned XGBoost model, LSTM, GRU and a lightweight patch-based Transformer. Twelve XGBoost candidates use expanding validation targets in 2010–2011; neural epoch selection uses an earlier 2009 stopping year. The validation-selected XGBoost achieves one-step test MAE {b.MAE:.4f}, RMSE {b.RMSE:.4f} and MAPE {b.MAPE:.2f}% over 2012–2014. LSTM has the lowest retrospective test RMSE, {l.RMSE:.4f}, while Naive has the lowest MAPE. Fixed-origin three-step results are reported separately. Prophet was implemented for longer crop-level aggregates but its native optimizer failed locally, so no Prophet accuracy is asserted. Source units, district boundaries and cohort selection limit interpretation. This retrospective extension provides reproducible evidence for model comparison rather than guaranteed operational accuracy or an independent prospective validation.'''
p('Abstract',True);p(abstract);p('Keywords: agricultural forecasting, annual panel, XGBoost, LSTM, GRU, patch Transformer, chronological validation.',size=9)
h('I. INTRODUCTION')
p('Agricultural planners need forecasts whose target definition and information set remain clear. A district-level yield forecast can support investigation and production planning, but reported accuracy is misleading when aggregation combines overlapping seasonal records or when future values enter model features. Short annual histories also make persistence a demanding benchmark for more complex methods.')
p('The objective is to compare classical and advanced methods on common targets while preserving an auditable progression from raw records to predictions. The experiment uses crop_production.csv, with seven fields and 246,091 observations. It extends the existing ten-year study with a larger chronological XGBoost search and genuinely trained small neural models. The contribution is a reproducible application and evidence audit; no novel architecture or universal forecasting superiority is claimed.')
p('The extension keeps model selection separate from retrospective test ranking. Its test years were already examined in an earlier project version. Therefore, the present results are retrospective benchmark evidence, not an independently blinded or prospective holdout. The report states this limitation and does not substitute a test winner for the validation-selected method.')
h('II. LITERATURE REVIEW')
p('Statistical forecasting. Hyndman and Athanasopoulos [2] explain baseline methods and out-of-sample evaluation, while Box et al. [3] establish autoregressive modeling. Their relevance is the need to define a forecast origin and compare complex models against simple alternatives. This experiment includes persistence, historical mean, moving average, drift and an explicitly labeled OLS AR(1).')
p('Exponential smoothing. Hyndman et al. [4] provide a state-space framework for smoothing. This study uses custom conditional least-squares SES, Holt and damped Holt recursions. They share the underlying level/trend ideas but are not claimed to reproduce full likelihood-based ETS estimation or a specific library implementation.')
p('Agricultural machine learning. Van Klompenburg et al. [5] review crop-yield prediction and frequently used environmental inputs. Temperature, rainfall, soil and management variables are absent here. The present comparison concerns historical predictability; its observed correlations and yield-ratio declines cannot establish agricultural causes.')
p('Boosting. Chen and Guestrin [6] describe regularized tree boosting in XGBoost. A global regressor shares lag–target relationships across districts and crops. Chronological candidate evaluation is required because nonlinear capacity does not guarantee better generalization on annual histories.')
p('Sequence models. LSTM [7] and the recurrent encoder–decoder of Cho et al. [8] motivate gated recurrent models. Nie et al. [10] motivate patching and channel-independent temporal attention. The current patch-based model is a small inspired implementation, not an exact PatchTST reproduction. Prophet [9] motivates a separate trend model with unjustified seasonal components disabled; its local optimizer failure is an execution finding, not evidence that the method is intrinsically unsuitable.')
h('III. DATASET AND PREPROCESSING')
sub('A. Provenance and verified characteristics')
p('The empirical input is the supplied crop_production.csv. Its digest is recorded in the evidence package, and the original input is preserved. The official APY query portal [1] provides contextual agricultural statistics. Exact record-level lineage, collection methodology and redistribution license for this local file remain unverified; the report does not assert direct download from that portal. The earlier 16-column dataset is not combined with this experiment.')
table(['Property','Verified observation'],[['Raw rows / fields','246,091 / 7'],['Crop-year labels','1997–2015'],['State labels','33'],['District names / state–district pairs','646 / 652'],['Crop names','124'],['Missing production','3,730'],['Exact / business-key duplicates','0 / 0'],['Annual processed groups',f"{pr['annual_panel_rows']:,}"]], 'TABLE I. INPUT AND DATA QUALITY')
p('The columns are State_Name, District_Name, Crop_Year, Season, Crop, Area and Production. Unit columns and stable geographic identifiers are absent. Accordingly, the target is called a production/area ratio; tonnes/hectare is not asserted as independently verified. State plus district labels identify series but do not guarantee unchanged geographic boundaries.')
sub('B. Conservative annual aggregation')
p('Categorical whitespace is standardized. Missing production is not imputed. Nonfinite measures, nonpositive area and negative production invalidate a component. An annual state–district–crop group is excluded when any selected component is invalid, a seasonal business key is duplicated, or Whole Year overlaps another season. The original input has no Total category. No outliers are automatically removed.')
p('The accepted annual target is:');p('Y(G) = Σ(r∈G) Production(r) / Σ(r∈G) Area(r).       (1)',size=9,align=WD_ALIGN_PARAGRAPH.CENTER)
p('Equation (1) is an area-weighted aggregation, not an average of individual ratios. Accepted annual groups account for 241,302 raw rows; invalid or missing-component groups account for 3,997 rows; overlapping Whole Year groups account for 792 rows. These decisions sum to the raw 246,091 records. Nonpositive-area and negative-production counts are both zero. Recorded-versus-calculated yield discrepancy testing is inapplicable because the file has no recorded yield column.')
sub('C. Temporal coverage and modeling cohort')
image('reporting.png','Fig. 1. Raw reporting volume by crop year. The final 2015 label contains only 562 records; the modeled cutoff is 2014.')
p('The four modeled crops are Rice, Maize, Wheat and Rapeseed &Mustard. Their complete 2005–2014 cohort contains 1,112 histories across 20 state labels and 11,120 annual targets: 319 Rice, 278 Maize, 267 Wheat and 248 Rapeseed &Mustard histories. The complete-case rule excludes 1,283 incomplete or unavailable histories within this window. This retrospective requirement uses test-era availability and introduces survivor selection. It is not presented as prospective sample construction.')
p('Annual district–crop, state–crop and national-by-crop exports preserve commodity separation. Combining heterogeneous production measures across all crops would need unit verification. Reporting counts quantify observed volume, not completeness against a known expected universe. Internal missing-year counts and all group decisions are exported.')
h('IV. RESEARCH METHODOLOGY')
image('final_workflow.png','Fig. 2. Analysis workflow. Validation choices freeze before the current extension’s test evaluation; the test period was previously examined in the archived study.')
sub('A. Time series EDA and stationarity')
image('yield_trends.png','Fig. 3. Area-weighted production/area ratios in the matched crop cohort. Changing area weights can influence aggregate trends.')
p('Rice and Maize weighted ratios increase by 26.23% and 37.11% from 2005 to 2014. Wheat and Rapeseed &Mustard change by −11.21% and −5.08%, respectively. Wheat declines 29.60% from 2013 to 2014. These are observed numerical ratios; they do not identify climate, disease or fertilizer effects. Exports include area/production trends, missingness heatmaps, state comparisons, distributions, IQR flags, descriptive correlations, rolling statistics and year-over-year changes. A centered rolling smooth provides a descriptive trend/residual view, without imposing annual seasonality.')
lev=station_df[station_df['transform'].eq('level')];dif=station_df[station_df['transform'].eq('difference')]
p(f"ADF and KPSS use the 769 selected identities with complete 1997–2011 histories, before final test years. ADF rejects the unit-root null at 5% for {100*lev.adf_p.lt(.05).mean():.1f}% of levels and {100*dif.adf_p.lt(.05).mean():.1f}% of differences; KPSS rejects its level-stationarity null for {100*lev.kpss_p.lt(.05).mean():.1f}% and {100*dif.kpss_p.lt(.05).mean():.1f}%. ADF uses a constant and AIC selection among at most one lag; KPSS uses a constant and automatic lag selection. ACF/PACF plots are representative pre-test diagnostics, not pooled cross-district autocorrelations.")
p('The tests are descriptive and not adjusted for multiple comparisons. Limited sample length reduces power, and tabulated KPSS p-values may be bounds rather than exact values. Diagnostic warnings are exported. Stationarity evidence does not establish that differenced forecasts will win out of sample.')
sub('B. Baselines and classical estimators')
p('Naive repeats the latest observation; Mean averages the history; MA3 averages its latest three values; Drift extrapolates the average first-to-last change. AR1_OLS estimates an intercept and one lag by unconstrained least squares. Its multi-step forecast recursively applies the fitted coefficients, which can be unstable on short or nearly constant histories.')
p('SES_LS updates level as ℓ(t)=αy(t)+(1−α)ℓ(t−1). Holt_LS adds trend; DampedHolt_LS damps it by φ. Conditional one-step SSE is minimized with L-BFGS-B. Initial level is the first observation; initial trend is the first difference, except SES uses zero. α and β lie in [0.01,0.99] and damping in [0.8,0.98]. Fixed-origin forecasts use one fitted state, without refitting to synthetic predictions. The LS labels distinguish these models from full ETS. Statsmodels ARIMA imports remain blocked by local application-control policy; no ARIMA accuracy is reported.')
sub('C. Expanded XGBoost tuning')
p('A global XGBoost regressor uses yield lags 1–3, their trailing mean, their population standard deviation and year minus 2005. Current production, current area and future targets are excluded. No geographic or crop encoding is added, preserving the existing feature definition. Twelve candidates comprise the two earlier settings and ten seeded combinations of estimators, depth, learning rate, row/column sampling and regularization. Selection uses pooled validation RMSE only.')
image('xgboost_tuning.png','Fig. 4. Actual expanding-validation RMSE across twelve XGBoost candidates. Expanding the search did not produce a better validation setting than the earlier selected candidate.')
p('The winning candidate remains trial 1: 250 estimators, depth 3, learning rate 0.03, subsample 0.8, column sampling 0.9 and L2 penalty 10. The additional candidates vary estimator count among 100/200/350, depth 2/3/4, learning rate 0.03/0.05/0.08, row and column sampling, L1 penalty and minimum child weight. Every candidate configuration and score is exported. No extra improvement is fabricated merely because the search budget increased.')
sub('D. Small recurrent and patch-based models')
p('The global LSTM and GRU use one recurrent layer with 16 hidden units, a final-state linear head and dropout 0.1. PatchTransformerTiny uses a three-year input, two overlapping length-two patches, an eight-dimensional patch embedding, learned positional information, two attention heads, one pre-normalized encoder layer, a 16-unit feed-forward block, dropout 0.1 and a flattened linear forecasting head. It is a lightweight patch-based model inspired by [10], not an exact PatchTST reproduction.')
rows=[]
for kind in ['LSTM','GRU','PatchTransformerTiny']:
 a=status[kind];rows.append([kind.replace('PatchTransformerTiny','Patch TF'),a['parameter_count'],a['selected_epochs']])
table(['Neural model','Parameters','Selected epochs'],rows,'TABLE II. ACTUAL NEURAL CONFIGURATIONS')
p('All neural fits use CPU, float32 tensors, Adam learning rate 0.003, weight decay 10⁻⁴, batch size 256 and gradient-norm clipping at 1. Training-row shuffling occurs only inside the chronological training split. A scalar mean and standard deviation are fitted separately at each training origin and applied to both inputs and targets. No test observations enter normalization.')
p('An inner fit through 2008 monitors 2009 MSE, with patience 8 and at most 60 epochs. The selected epoch counts are then frozen. Outer validation refits through 2009 and 2010 predict 2010 and 2011, respectively. Test refits use those same epoch counts. Parameter updates are checked, checkpoints include normalization values, and reloaded inference is verified against the original predictions.')
image('neural_losses_panel.png','Fig. 5. Genuine neural training and inner-stopping losses. The validation curves concern 2009; they are not test losses or outer 2010–2011 selection scores.')
sub('E. Chronological evaluation and metrics')
p('The common modeled window is 2005–2014. Outer validation targets are 2010 and 2011 with origins 2009 and 2010. Selection freezes on validation RMSE. Final pre-test fitting uses 2005–2011. An expanding one-step test refits at origins 2011, 2012 and 2013; a separate fixed-origin test uses only information through 2011 for all three 2012–2014 horizons. Each model has 2,224 validation targets and 3,336 targets per test protocol. Supervised global training starts in 2008 because three historical values are required.')
p('For N aligned targets, e(i)=y(i)−ŷ(i):');p('MAE = N⁻¹ Σ |e(i)|.                                               (2)',size=9,align=WD_ALIGN_PARAGRAPH.CENTER);p('RMSE = sqrt[N⁻¹ Σ e(i)²].                                     (3)',size=9,align=WD_ALIGN_PARAGRAPH.CENTER);p('MAPE = 100 |S|⁻¹ Σ(i∈S) |e(i)/y(i)|.                   (4)',size=9,align=WD_ALIGN_PARAGRAPH.CENTER)
p('S excludes |y|≤10⁻⁸; one of the 3,336 test targets is excluded from MAPE. MAE and RMSE retain all targets. Scores weight district/year observations equally. Additional exports include sMAPE, MASE and signed mean error. MASE uses training first differences from 2005–2009 for validation and 2005–2011 for testing; undefined zero scales are omitted. Models enter the leaderboard only when all common targets have finite predictions. Seeds, versions, timings, fit statuses and checkpoints are recorded.')
h('V. RESULTS AND DISCUSSION')
sub('A. Validation selection and retrospective ranking')
# Compact native Word table, one column wide.
p('TABLE III. EXPANDING ONE-STEP TEST, 2012–2014',True,8,WD_ALIGN_PARAGRAPH.CENTER)
t=d.add_table(rows=1,cols=5);t.autofit=False;widths=[.91,.48,.65,.66,.65]
for c,v in zip(t.rows[0].cells,['Model','Type','MAE','RMSE','MAPE %']):c.text=v
labels={'PatchTransformerTiny':'Patch TF','XGBoost_Tuned':'XGB tuned'}
for _,a in tl.iterrows():
 cat='DL' if a.model in ['LSTM','GRU','PatchTransformerTiny'] else ('ML' if a.model=='XGBoost_Tuned' else ('Base' if a.model in ['Naive','Mean','MA3','Drift'] else 'Stat'))
 for c,v in zip(t.add_row().cells,[labels.get(a.model,a.model),cat,f'{a.MAE:.4f}',f'{a.RMSE:.4f}',f'{a.MAPE:.2f}']):c.text=v
for col,w in zip(t.columns,widths):col.width=Inches(w)
for ri,row in enumerate(t.rows):
 row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
 for c,w in zip(row.cells,widths):
  c.width=Inches(w);prp=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
  for edge in ['top','left','bottom','right']:
   a=OxmlElement('w:'+edge);a.set(qn('w:val'),'single');a.set(qn('w:sz'),'4');a.set(qn('w:color'),'D9D9D9');borders.append(a)
  prp.append(borders)
  if ri==0:sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EAEAEA');prp.append(sh)
  for q in c.paragraphs:
   q.paragraph_format.space_after=Pt(3);q.paragraph_format.space_before=Pt(3);q.paragraph_format.keep_with_next=ri<len(t.rows)-1
   for rr in q.runs:rr.font.size=Pt(7.5);rr.bold=ri==0
p('Errors use the numerical production/area scale; physical units are unverified. DL denotes neural models, ML boosting, Stat local estimators and Base baselines. Patch TF denotes PatchTransformerTiny.',size=8)
p(f"XGBoost_Tuned remains selected by validation RMSE ({vl.set_index('model').loc[best,'RMSE']:.4f}), ahead of LSTM ({vl.set_index('model').loc['LSTM','RMSE']:.4f}). Its one-step test RMSE is {b.RMSE:.4f}, a {b.RMSE_improvement_vs_naive_pct:.2f}% reduction relative to Naive’s {n.RMSE:.4f}. This is a metric-specific empirical difference, not a statistically established or operational benefit.")
p(f"LSTM has the lowest retrospective one-step test RMSE ({l.RMSE:.4f}); PatchTransformerTiny has the lowest MAE ({tl.set_index('model').loc['PatchTransformerTiny','MAE']:.4f}); Naive has the lowest MAPE ({n.MAPE:.2f}%). These rankings do not replace the validation selection. The full search and neural comparisons show that one method does not dominate every metric.")
image('competitive_rmse.png','Fig. 6. One-step RMSE for the global models and Naive. Table III contains every evaluated model, including unstable AR1_OLS.')
p('AR1_OLS has one-step RMSE 5.3499 and fixed-origin RMSE 14,099.2581. Unconstrained coefficients on short histories can create explosive forecasts. Across all one-step models, 29 predictions are negative and remain in the metrics; they are not clipped or hidden. There are no negative selected-model future forecasts. Classical caught-fit exception and optimizer non-success counts are zero; this does not imply that every requested library fit succeeded.')
sub('B. Forecast horizon and model reliability')
rows=[]
for name in ['XGBoost_Tuned','LSTM','Naive']:
 a=fl.set_index('model').loc[name];rows.append([labels.get(name,name),f'{a.MAE:.4f}',f'{a.RMSE:.4f}',f'{a.MAPE:.2f}'])
# Helper supports arbitrary column counts; explicit widths are adjusted below for this table.
p('TABLE IV. SELECTED MODEL, LSTM AND NAIVE FIXED-ORIGIN TEST',True,8,WD_ALIGN_PARAGRAPH.CENTER)
t2=d.add_table(rows=1,cols=4);t2.autofit=False
for c,v in zip(t2.rows[0].cells,['Model','MAE','RMSE','MAPE %']):c.text=v
for row in rows:
 for c,v in zip(t2.add_row().cells,row):c.text=v
for col,w in zip(t2.columns,[1.1,.75,.75,.75]):col.width=Inches(w)
for ri,row in enumerate(t2.rows):
 for c,w in zip(row.cells,[1.1,.75,.75,.75]):
  c.width=Inches(w)
  for q in c.paragraphs:
   q.paragraph_format.space_after=Pt(3);q.paragraph_format.keep_with_next=ri<len(t2.rows)-1
   for rr in q.runs:rr.font.size=Pt(8);rr.bold=ri==0
p('For the selected XGBoost, fixed-origin RMSE 0.6283 is worse than Naive’s 0.6191. LSTM’s retrospective fixed-origin RMSE is 0.5992. The one-step protocol receives newly observed years at subsequent origins; the fixed-origin protocol does not. Accuracy cannot be transferred between these horizons without a matched backtest.')
image('horizon_errors.png','Fig. 7. Actual fixed-origin MAE by horizon. Curves use the same matched district/crop targets; no later observations enter the fixed-origin model.')
sub('C. Execution, applicability and future forecasts')
p(f"The expanded local run took {info['runtime_seconds']:.1f} seconds. Logged fit and inference times and neural parameter counts are exported separately; baseline loop times include evaluation overhead. The random seed is 42 and neural computation uses two CPU threads. The work uses free local tools and no paid services.")
p('Prophet was implemented on four crop-level annual aggregates using 769 longer complete histories over 1997–2014. Daily, weekly and yearly seasonality were disabled, two changepoints were allowed and three prior scales were specified for chronological tuning. Its first native optimization failed locally. Consequently, no successful Prophet fit, accuracy, tuning winner or prediction interval is asserted, and the different aggregate target population is never mixed into the district leaderboard. Failure logs and runnable code are retained.')
p('Seasonal ARIMA and seasonal Holt–Winters are excluded because no repeating annual-period structure is established. ARCH/Ljung–Box diagnostics on one longer history use only 14 pre-test residuals and remain exploratory. GARCH and VAR are outside the validated short-history, univariate design; the report does not claim that these methods are impossible or that the whole panel lacks ARCH effects.')
image('expanded_future.png','Fig. 8. Selected-model cohort mean after refitting through 2014. Forecast years 2015–2017 are model outputs; 2015 is partially present in the raw file.')
p('The validation-selected XGBoost is refitted through the modeled cutoff and recursively forecasts each selected series. CSV exports include every horizon. The plotted mean is unweighted across series, not a national agricultural yield. Calibrated prediction intervals and interval coverage are unavailable; point forecasts must not be presented as verified future observations.')
h('VI. LIMITATIONS AND CHALLENGES')
p('The individual histories are short and the validation/test periods contain few independent years. Shared shocks reduce temporal replication despite many districts. Complete-history selection excludes changing or incomplete regions and depends on test-era availability. The test years were examined previously, so the extension is retrospective and cannot support independent prospective generalization claims.')
p('Units, exact source lineage/license and stable district boundaries remain unresolved. Production missingness and seasonal overlap exclusions may be nonrandom. Aggregate ratios can change with area weights and reporting coverage. The study concerns four crops and cannot establish performance for all 124 crop labels. No causal agricultural explanation follows without environmental and management variables.')
p('The neural architectures are deliberately small and their three-year context limits long-range dependence learning. They are not extensively tuned. Early stopping uses a single inner year; expanded XGBoost tuning still has a bounded 12-candidate budget. No significance test establishes that the small score gaps persist elsewhere. Prophet and ARIMA execution limitations are explicit, and probabilistic calibration remains future work.')
p('The final notebook executes in a real local Jupyter kernel. The Colab browser is signed out, so hosted execution cannot be certified. Dependency installation and upload adapter behavior are tested locally and the code uses CPU. Official faculty milestone PDFs were unavailable; exact institutional compliance and individual contribution evidence therefore remain unverified.')
h('VII. CONCLUSION AND FUTURE WORK')
p(f"The study completed raw-data auditing, annual aggregation, time series EDA, pre-test diagnostics, 12-candidate XGBoost tuning and genuine LSTM, GRU and patch-based Transformer training on a common 1,112-series cohort. Validation selects XGBoost_Tuned; its one-step test MAE is {b.MAE:.4f}, RMSE {b.RMSE:.4f} and MAPE {b.MAPE:.2f}%. LSTM has the lowest retrospective RMSE, while metric and horizon rankings differ. These distinctions are retained rather than selecting a winner from the test set.")
p('For farmers, insurers and planners, the outputs can serve as investigated baselines after domain validation, not guaranteed recommendations. Future work should verify provenance and units, obtain stable geographic crosswalks, add weather/soil/satellite variables, evaluate genuinely new years, and calibrate multi-horizon uncertainty. More observations and prospective evaluation would support stronger claims than increasing model complexity alone.')
h('REFERENCES')
for i,(ref,url) in enumerate(refs,1):q=p(f'[{i}] {ref} {url}',size=8);q.paragraph_format.keep_together=True
# Deterministic removal of theme font/border residue.
for a in list(d.styles.element.iter(qn('w:pBdr'))):a.getparent().remove(a)
for q in list(d.paragraphs)+[q for t in d.tables for row in t.rows for c in row.cells for q in c.paragraphs]:
 for a in list(q._p.iter(qn('w:pBdr'))):a.getparent().remove(a)
 for rr in q.runs:
  rr.font.name='Times New Roman';rf=rr._element.get_or_add_rPr().find(qn('w:rFonts'))
  if rf is not None:
   for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
    if qn('w:'+a) in rf.attrib:del rf.attrib[qn('w:'+a)]
  if q.style.name=='Caption':rr.bold=False
for style in d.styles:
 if style.type==1:
  style.font.name='Times New Roman';rf=style.element.get_or_add_rPr().find(qn('w:rFonts'))
  if rf is not None:
   for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
    if qn('w:'+a) in rf.attrib:del rf.attrib[qn('w:'+a)]
for tab in d.tables:
 for ri,row in enumerate(tab.rows):
  for cell in row.cells:
   prp=cell._tc.get_or_add_tcPr()
   if prp.find(qn('w:tcBorders')) is None:
    borders=OxmlElement('w:tcBorders')
    for edge in ['top','left','bottom','right']:
     el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    prp.append(borders)
   if ri==0 and prp.find(qn('w:shd')) is None:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EAEAEA');prp.append(sh)
d.save(O/'Final_Time_Series_Research_Report.docx');(O/'report_abstract.txt').write_text(abstract,encoding='utf-8');(O/'report_metadata.json').write_text(json.dumps({'abstract_words':len(abstract.split()),'references':len(refs),'numbered_figures':8,'tables':4},indent=2));print('Report created; abstract words:',len(abstract.split()))

