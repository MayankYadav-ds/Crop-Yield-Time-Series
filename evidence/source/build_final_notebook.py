from pathlib import Path
p=Path('work/final_audit/extended_experiment.py');s=p.read_text(encoding='utf-8').replace('m.fit(frame);yrs=','m.fit(frame,seed=42);yrs=');p.write_text(s,encoding='utf-8')
import nbformat as nbf,json,ast
root=Path('work/final_audit');nb=nbf.v4.new_notebook();cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):ast.parse(s);cells.append(nbf.v4.new_code_cell(s))
md('''# Agricultural crop yield forecasting — final evaluation edition
**Bennett University · CSET 369**  
Mayank Yadav — S24CSEU1494

This edition preserves the original working experiment’s seven-column input and 2005–2014 matched cohort, expands XGBoost tuning to 12 chronological trials, and adds genuinely trained LSTM, GRU and a tiny patch-based Transformer. Prophet has a separate longer-history aggregate experiment and an explicit failure status if its native optimizer cannot run. AR1_OLS and custom least-squares smoothing are labeled accurately; no ARIMA result is invented.

Use **crop_production.csv** (246,091 rows), not the older 16-column input. Scores are calculated, not prefilled. The raw file includes partial 2015 reporting, while the modeled cutoff is 2014. Forecasts cover 2015–2017 and are not observations.

**Scientific limits:** retrospective balanced-cohort selection; seven pre-test annual observations; only two outer validation years and three test years; units, source/license and district boundaries unverified. The test period was examined in the previous study, so this extension is not an independent prospective holdout. The official milestone PDFs were not available for exact certification.''')
md('## 1 Free CPU environment and dependency check\nColab normally supplies PyTorch. Missing PyTorch uses the official CPU wheel index. The local audit records exact installed versions; Colab resource availability and versions can differ. No paid runtime or service is requested.')
setup='''import os,sys,subprocess,importlib.util,importlib.metadata as package_metadata
from pathlib import Path
COLAB_MODE=bool(importlib.util.find_spec('google.colab')) if importlib.util.find_spec('google') else False
if COLAB_MODE:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy>=1.26','pandas>=2.2','scipy>=1.13','statsmodels>=0.14','scikit-learn>=1.5','xgboost>=2.1','matplotlib>=3.9','prophet>=1.1'])
    try:
        current_torch=tuple(int(x) for x in package_metadata.version('torch').split('+')[0].split('.')[:2])
    except package_metadata.PackageNotFoundError:
        current_torch=(0,0)
    if current_torch<(2,4):
        subprocess.check_call([sys.executable,'-m','pip','install','-q','torch>=2.4','--index-url','https://download.pytorch.org/whl/cpu'])
print('CPU-only experiment; no paid services. Colab mode:',COLAB_MODE)
'''
code(setup)
md('## 2 Upload and validate the confirmed dataset\nOnly the returned CSV payload is accepted; the original source file is preserved. A different CSV with the same schema can change every result and requires a new report.')
upload='''EXPECTED_COLUMNS={'State_Name','District_Name','Crop_Year','Season','Crop','Area','Production'}
def configure_dataset(csv_path=None,upload_callback=None):
    if upload_callback is not None:
        uploaded=upload_callback()
        names=[name for name in uploaded if name.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError('Upload exactly one crop_production.csv file.')
        path=Path(names[0]).name
        Path(path).write_bytes(uploaded[names[0]])
    else:
        path=str(csv_path or os.environ.get('APY_CSV','crop_production.csv'))
    if not Path(path).is_file():raise FileNotFoundError('Set APY_CSV or upload crop_production.csv.')
    import csv
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:header=next(csv.reader(handle))
    if set(header)!=EXPECTED_COLUMNS:raise ValueError(f'Expected seven-column crop_production.csv; received {header}')
    os.environ['APY_CSV']=str(Path(path).resolve())
    return Path(path)
if COLAB_MODE:
    from google.colab import files
    data_path=configure_dataset(upload_callback=files.upload)
    os.environ['APY_OUTPUT']='/content/crop_yield_final_results'
else:
    data_path=configure_dataset()
    os.environ.setdefault('APY_OUTPUT','results')
print('Dataset:',data_path.name)
'''
code(upload)
core=(root/'core.py').read_text(encoding='utf-8').replace("'work/final_audit/results'","'results'")
# Split only at top-level boundaries; every code cell parses independently.
marks=[('# Representative EDA','## 4 Time series EDA'),('# Training-only diagnostics','## 5 Pre-test stationarity and dependence'),('# Models:','## 6 Baselines and classical implementations')]
start=0;head='## 3 Data quality and conservative annual panel'
for mark,nexthead in marks:
 i=core.index(mark);md(head);code(core[start:i]);start=i;head=nexthead
md(head+'\nThe classical estimators use past observations only. Fixed-origin forecasts retain the same fitted states or coefficients for all horizons. Unconstrained AR(1) can be unstable and produce extreme or negative forecasts; those results remain visible.');code(core[start:])
md('## 7 Small neural architectures and training protocol\nAll inputs are three historical yields. Each fit normalizes on its training observations only. Inner fitting ends in 2008 and selects an epoch using 2009 with patience 8 and a 60-epoch ceiling. The selected epoch count stays fixed for outer 2010–2011 validation and subsequent test refits. Training-row shuffling occurs only within the already chronological training split. The patch Transformer uses length-two overlapping patches, positional embeddings and two attention heads; it is not an exact PatchTST reproduction.')
code((root/'neural_models.py').read_text(encoding='utf-8'))
main=(root/'extended_experiment.py').read_text(encoding='utf-8');main=main[main.index('import torch,copy,logging,traceback'):]
marks=[('# Inner 2009','## 9 Neural inner stopping and chronological outer validation'),('# No further tuning','## 10 Frozen selection and final tests'),('# Refit only','## 11 Selected-model refit and forecasts'),('# Prophet uses','## 12 Separate annual Prophet applicability experiment'),('# Audits, errors','## 13 Fit status, disaggregated errors and actual comparison plots'),('# ARCH applicability','## 14 Volatility and multivariate applicability'),('# Integrity tests','## 15 Integrity tests and execution evidence')]
start=0;head='## 8 Expanded XGBoost chronological tuning\nThe two previous candidates and ten seeded configurations vary depth, estimators, learning rate, subsampling and regularization. Only validation RMSE chooses the candidate; test results never change it.'
for mark,nexthead in marks:
 i=main.index(mark);md(head);code(main[start:i]);start=i;head=nexthead
md(head);code(main[start:])
md('## 16 Additional EDA and diagnostic exports\nState and national series remain separated by crop. Numerical correlations and outlier flags are descriptive. No outliers are removed merely because they exceed an IQR threshold. Centered rolling trend views are never forecasting inputs.')
extra=(root/'additional_eda.py').read_text(encoding='utf-8');extra=extra[:extra.index("if __name__=='__main__'")];code(extra+'\nadditional_eda(ROOT)\n')
md('## 17 Results to show during teacher evaluation\nThe validation-selected model and retrospective test winner answer different questions. A successful notebook execution can still contain an explicitly excluded model; read the status table and logs.')
code('''from IPython.display import display
print('Validation selection:',best)
display(vl[['model','MAE','RMSE','MAPE']])
print('Expanding one-step test:');display(tl[['model','MAE','RMSE','MAPE','RMSE_improvement_vs_naive_pct']])
print('Fixed-origin three-step test:');display(fl[['model','MAE','RMSE','MAPE']])
print('Model execution statuses:');display(pd.DataFrame(model_status).T)
print('Future forecast sample:');display(future.head(12))
''')
md('## 18 Quick live demonstration check\nThis cell refits only the selected XGBoost configuration at origin 2011 and checks one 2012 prediction against the saved one-step result. It does not tune using that target or change the selected model.')
code("""demo_X,demo_y=xy(2011)
live_check_model=XGBRegressor(**chosen_xgb,objective='reg:squarederror',random_state=42,n_jobs=2).fit(demo_X,demo_y)
demo_sid=wide.index[0]
demo_feature=np.array([feat(wide.loc[demo_sid,:2011].values,2012)])
live_value=float(live_check_model.predict(demo_feature)[0])
stored_value=float(test[(test.model=='XGBoost_Tuned')&(test.sid==demo_sid)&(test.year==2012)].prediction.iloc[0])
np.testing.assert_allclose(live_value,stored_value,rtol=1e-7,atol=1e-7)
print('PASS: one-origin refit reproduces saved prediction.')
print('Series:',demo_sid,'Target: 2012','Prediction:',live_value)
""")
md('## 19 Dataset identity and download\nSave this notebook with outputs after a fresh Colab run. A runtime or environment difference may change numerical results; regenerate the report if results change.')
code('''import hashlib,shutil
identity={'name':Path(os.environ['APY_CSV']).name,'bytes':Path(os.environ['APY_CSV']).stat().st_size,'sha256':hashlib.sha256(Path(os.environ['APY_CSV']).read_bytes()).hexdigest()}
js('input_identity.json',identity)
archive=shutil.make_archive(str(ROOT.resolve().parent/'crop_yield_final_evidence'),'zip',ROOT)
print('Completed. Evidence ZIP:',Path(archive).name)
print('Input SHA-256:',identity['sha256'])
if COLAB_MODE:files.download(archive)
''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'Complete_Time_Series_Project_EXECUTED.ipynb'}}
nbf.write(nb,root/'final_unexecuted.ipynb');(root/'colab_setup.py').write_text(setup,encoding='utf-8');(root/'upload_adapter.py').write_text(upload,encoding='utf-8');print('Notebook prepared:',len(cells),'cells,',sum(c.cell_type=='code' for c in cells),'code cells')
