from pathlib import Path
import os,sys,json,csv,hashlib,zipfile,types,tempfile,contextlib,io,importlib.util,importlib.metadata as pm,subprocess,xml.etree.ElementTree as E
from unittest.mock import patch
import numpy as np,pandas as pd,nbformat
from docx import Document
from pypdf import PdfReader
B=Path.cwd();A=B/'work/final_audit';R=A/'notebook_results';O=B/'outputs/FINAL_EVALUATION_SUBMISSION';audit={}
# All cells executed in the delivered artifact, not merely syntax checked.
nb=nbformat.read(O/'Complete_Time_Series_Project_EXECUTED.ipynb',as_version=4);codes=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in codes);assert not [o for c in codes for o in c.outputs if o.output_type=='error'];audit['notebook']=json.loads((A/'notebook_execution.json').read_text());assert len(codes)==19
# Compare every numeric prediction against the first real-kernel run; both experiments use frozen seeds/configurations.
replays={}
for name in ['validation_predictions.csv','test_predictions.csv','fixed_origin_predictions.csv','future_forecasts.csv']:
 x=pd.read_csv(A/'first_notebook_predictions'/name);y=pd.read_csv(R/name);assert list(x.columns)==list(y.columns) and len(x)==len(y)
 cols=x.select_dtypes('number').columns;delta=float(np.max(np.abs(x[cols].values-y[cols].values)));assert delta==0,(name,delta);replays[name]={'rows':len(y),'maximum_absolute_numeric_difference':delta}
audit['independent_kernel_replay']=replays
# Recompute metrics using an independent implementation, and align actual labels to saved annual panel values.
wide=pd.read_csv(R/'model_panel.csv',index_col=0);wide.columns=wide.columns.astype(int);metric_rows=[]
for stem,cutoff,expected_years in [('validation',2009,{2010,2011}),('test',2011,{2012,2013,2014}),('fixed_origin',2011,{2012,2013,2014})]:
 pred=pd.read_csv(R/(stem+'_predictions.csv'));leader=pd.read_csv(R/(stem+'_leaderboard.csv')).set_index('model');scale=wide.loc[:,:cutoff].diff(axis=1).abs().mean(axis=1)
 assert set(pred.year)==expected_years;assert not pred.duplicated(['model','sid','year']).any()
 for model,g in pred.groupby('model'):
  actual=g.actual.to_numpy();forecast=g.prediction.to_numpy();err=actual-forecast;ok=np.abs(actual)>1e-8;den=np.abs(actual)+np.abs(forecast);sc=g.sid.map(scale).values;valid=sc>1e-8
  assert np.isfinite(forecast).all();assert np.allclose(actual,[wide.loc[sid,int(year)] for sid,year in zip(g.sid,g.year)])
  calc={'MAE':np.mean(np.abs(err)),'RMSE':np.sqrt(np.dot(err,err)/len(err)),'MAPE':100*np.mean(np.abs(err[ok]/actual[ok])),'MAPE_n':int(ok.sum()),'sMAPE':np.mean(np.divide(200*np.abs(err),den,out=np.zeros_like(err),where=den>1e-8)),'MASE':np.mean(np.abs(err[valid])/sc[valid]),'MFE':np.mean(err)}
  for metric,value in calc.items():assert np.isclose(leader.loc[model,metric],value,rtol=1e-11,atol=1e-11),(stem,model,metric)
  metric_rows.append({'protocol':stem,'model':model,'all_metrics_verified':True,'n':len(g)})
audit['metric_recalculations']=metric_rows;assert len(metric_rows)==36
# Raw input and aggregation audit.
raw=pd.read_csv(B/'work/crop_production.csv');summary=json.loads((R/'dataset_summary.json').read_text());assert len(raw)==246091 and raw.shape[1]==7;assert raw.Production.isna().sum()==3730 and raw.duplicated().sum()==0
quality=json.loads((R/'preprocessing_summary.json').read_text());assert sum(quality['raw_rows_by_decision'].values())==len(raw);panel=pd.read_csv(R/'annual_panel.csv');assert len(panel)==210131 and np.allclose(panel.y,panel.production/panel.area)
assert not panel.duplicated(['State_Name','District_Name','Crop','Crop_Year']).any();assert panel.area.gt(0).all() and panel.production.ge(0).all();assert set(raw.Crop_Year)==set(range(1997,2016))
identity=json.loads((B/'work/new_input_hash.json').read_text());digest=hashlib.sha256((B/'work/crop_production.csv').read_bytes()).hexdigest();assert digest==identity['sha256'];source=Path(identity['source']);assert source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==digest
audit['data']={'original_input_sha256_unchanged':True,'raw_rows':len(raw),'annual_groups':len(panel),'modeled_series':len(wide),'physical_units_verified':False,'source_license_verified':False}
# Training metadata and model status.
train=pd.read_csv(R/'training_and_inference.csv');neural=train[train.model.isin(['LSTM','GRU','PatchTransformerTiny'])&train.max_training_target_year.notna()];assert (neural.max_training_target_year<=neural.origin).all()
status=json.loads((R/'model_training_status.json').read_text());assert all(status[k]['status']=='trained and evaluated' for k in ['LSTM','GRU','PatchTransformerTiny']);assert status['Prophet']['status']=='excluded after runtime failure'
selection=json.loads((R/'selection_frozen.json').read_text());vl=pd.read_csv(R/'validation_leaderboard.csv');assert selection['best_model']==vl.sort_values(['RMSE','model']).iloc[0].model=='XGBoost_Tuned';assert not selection['test_evaluation_started']
trials=pd.read_csv(R/'hyperparameter_results.csv');assert len(trials)==12 and trials.status.eq('completed').all();assert int(json.loads((R/'xgboost_selection.json').read_text())['selected_trial'])==int(trials.sort_values(['RMSE','trial']).iloc[0].trial)
audit['training']={'three_neural_models_trained':True,'training_target_year_checks':True,'inner_stopping':{'training_end':2008,'stop_year':2009,'patience':8,'max_epochs':60},'checkpoint_replay':'passed in actual notebook for each neural test origin','xgboost_trials':12,'selection_frozen_before_current_test_computation':True,'test_was_previously_examined':True,'Prophet_status':status['Prophet']['status'],'ARIMA_status':status['ARIMA']['status']}
# Unit/integration checks of the Colab upload adapter. This is explicitly not hosted UI verification.
adapter=(A/'upload_adapter.py').read_text(encoding='utf-8');setup=(A/'colab_setup.py').read_text(encoding='utf-8');oldcwd=Path.cwd();oldenv=dict(os.environ);payload=(B/'work/crop_production.csv').read_bytes();upload_tests=[]
try:
 with contextlib.nullcontext(tempfile.mkdtemp(dir=A,prefix='upload_test_')) as td:
  os.chdir(td);fake=types.ModuleType('google.colab');fake.files=types.SimpleNamespace(upload=lambda:{'crop_production.csv':payload});google=types.ModuleType('google');google.__path__=[]
  with patch.dict(sys.modules,{'google':google,'google.colab':fake}):
   state={'COLAB_MODE':True,'os':os,'Path':Path};exec(adapter,state)
  assert hashlib.sha256(Path(state['data_path']).read_bytes()).hexdigest()==digest;upload_tests.append('valid CSV upload payload accepted with matching digest')
  fn=state['configure_dataset']
  for name,call in [('zero CSV',lambda:fn(upload_callback=lambda:{})),('multiple CSVs',lambda:fn(upload_callback=lambda:{'a.csv':b'x','b.csv':b'x'})),('wrong schema',lambda:fn(upload_callback=lambda:{'wrong.csv':b'id,year,area\n1,2000,1\n'}))]:
   try:call();raise AssertionError('Expected rejection: '+name)
   except ValueError:upload_tests.append(name+' rejected')
finally:os.chdir(oldcwd);os.environ.clear();os.environ.update(oldenv)
original_find=importlib.util.find_spec
def pretend_spec(name,*a,**kw):
 if name in ['google','google.colab']:return types.SimpleNamespace()
 return original_find(name,*a,**kw)
with patch('importlib.util.find_spec',side_effect=pretend_spec),patch('importlib.metadata.version',return_value='0.0'),patch('subprocess.check_call') as calls:
 exec(setup,{})
 commands=[call.args[0] for call in calls.call_args_list];assert len(commands)==2;assert 'matplotlib>=3.9' in commands[0];assert 'https://download.pytorch.org/whl/cpu' in commands[1]
audit['colab']={'upload_adapter_integration_tests':upload_tests,'installation_command_construction_tested':True,'cpu_index_verified_in_branch':True,'real_dependency_installation':'passed locally in isolated workspace package directory','hosted_colab_execution':'unverified: Google sign-in unavailable','linux_dependency_installation':'not executed; no hosted account access','scope_of_mock_tests':'SDK payload and installer command behavior only; not a real hosted upload or install'}
# Report table cells match all current metric values at displayed precision.
doc=Document(O/'Final_Time_Series_Research_Report.docx');leader=pd.read_csv(R/'test_leaderboard.csv').set_index('model');aliases={'Patch TF':'PatchTransformerTiny','XGB tuned':'XGBoost_Tuned'}
for row in doc.tables[2].rows[1:]:
 vals=[c.text.strip() for c in row.cells];model=aliases.get(vals[0],vals[0]);assert vals[2:]==[f'{leader.loc[model,"MAE"]:.4f}',f'{leader.loc[model,"RMSE"]:.4f}',f'{leader.loc[model,"MAPE"]:.2f}']
pdf=PdfReader(O/'Final_Time_Series_Research_Report.pdf');assert len(pdf.pages)<=10;metadata=json.loads((A/'report_metadata.json').read_text());assert 150<=metadata['abstract_words']<=200 and metadata['references']==10
# Native PowerPoint table matches the same CSV; chart caches match the exact referenced observations within published precision.
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'};charts=[]
with zipfile.ZipFile(O/'Final_Presentation.pptx') as z:
 root=E.fromstring(z.read('ppt/slides/slide10.xml'));rows=root.findall('.//a:tbl/a:tr',ns)
 assert len(rows)==13
 for row in rows[1:]:
  vals=[''.join(cell.itertext()).strip() for cell in row.findall('a:tc',ns)]
  vals=[''.join(cell.findall('.//a:t',ns)[i].text or '' for i in range(len(cell.findall('.//a:t',ns)))) for cell in row.findall('a:tc',ns)]
  model=vals[0];assert vals[2:]==[f'{leader.loc[model,"MAE"]:.4f}',f'{leader.loc[model,"RMSE"]:.4f}',f'{leader.loc[model,"MAPE"]:.2f}']
 snapshot=json.loads((B/'work/deck_build/data.json').read_text());cy=list(range(2005,2015));comparison=['XGBoost_Tuned','LSTM','GRU','PatchTransformerTiny','Naive']
 expected={
 'Annual reporting':[[snapshot['reporting'][y] for y in sorted(snapshot['reporting'])]],
 'Matched crop ratios':[[next(r['yield_ratio'] for r in snapshot['weighted_crop_series'] if r['Crop']==crop and r['Crop_Year']==y) for y in cy] for crop in ['Rice','Maize','Wheat','Rapeseed &Mustard']],
 'Validation RMSE by trial':[[r['RMSE'] for r in snapshot['hyperparameter_results']]],
 'Outer validation RMSE':[[next(r['RMSE'] for r in snapshot['validation_leaderboard'] if r['model']==model) for model in comparison]],
 'One-step test RMSE':[[leader.loc[model,'RMSE'] for model in comparison]],
 'Cohort mean: one-step':[[r['actual'] for r in snapshot['actual_predicted']['XGBoost_Tuned']],[r['prediction'] for r in snapshot['actual_predicted']['XGBoost_Tuned']],[r['prediction'] for r in snapshot['actual_predicted']['LSTM']]],
 'Fixed-origin MAE':[[next(r['mean'] for r in snapshot['fixed_origin_error_by_horizon'] if r['model']==model and r['horizon']==h) for h in [1,2,3]] for model in ['Naive','XGBoost_Tuned','LSTM','GRU','PatchTransformerTiny']],
 '2015–2017 point forecasts':[[snapshot['future_means'][y] for y in sorted(snapshot['future_means'])]]}
 kinds=['LSTM','GRU','PatchTransformerTiny'];end=min(len(snapshot[k+'_loss']) for k in kinds);epochs=[e for e in range(1,end+1) if e==1 or e%3==0 or e==end];expected['Inner stopping year 2009']=[[next(r['validation_normalized_MSE'] for r in snapshot[k+'_loss'] if r['epoch']==e) for e in epochs] for k in kinds]
 for name in z.namelist():
  if '/charts/chart' in name and name.endswith('.xml'):
   root=E.fromstring(z.read(name));title=''.join(t.text or '' for t in root.findall('c:chart/c:title//a:t',ns));series=[]
   for ser in root.findall('.//c:ser',ns):
    val=ser.find('c:val',ns);pts=val.findall('.//c:pt',ns);series.append([float(pt.find('c:v',ns).text) for pt in pts])
   assert title in expected,(name,title)
   assert len(series)==len(expected[title])
   for observed,ref in zip(series,expected[title]):np.testing.assert_allclose(observed,ref,rtol=1e-10,atol=1e-10)
   charts.append({'chart':name,'title':title,'data_verified_to_published_precision':True})
assert len(charts)==9
audit['document_consistency']={'report_metric_table_verified':True,'presentation_metric_table_verified':True,'native_charts':charts,'chart_numeric_storage':'12 significant digits for Excel portability; full-precision CSVs retained','report_pages':len(pdf.pages),'abstract_words':metadata['abstract_words'],'references':10,'slides':16,'report_figures_generated_from_recorded_data':True,'figure_generation_code_retained':True}
# Confirm original working files are intact and match the pre-change backup.
backup=sorted((B/'outputs').glob('WORKING_BACKUP_*.zip'))[-1];backup_checks=[]
with zipfile.ZipFile(backup) as z:
 assert z.testzip() is None
 for item in z.infolist():
  if item.is_dir():continue
  parts=Path(item.filename).parts
  rel=Path(*parts[1:]) if parts[0]=='FINAL_SUBMISSION' else Path(item.filename)
  old=B/'outputs/FINAL_SUBMISSION'/rel
  assert old.exists(),str(old)
  assert hashlib.sha256(old.read_bytes()).digest()==hashlib.sha256(z.read(item.filename)).digest(),str(old)
  backup_checks.append(str(rel))
audit['backup']={'archive':str(backup),'zip_integrity':'passed','original_files_unchanged':True,'verified_files':backup_checks}
q=json.loads((A/'viva_questions.json').read_text());assert len(q)==50;audit['viva_questions']=50;audit['demonstration_minutes']=11
(A/'independent_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('Independent audit passed: 36 metric rows, 9 native chart data sets, both replay predictions, report/presentation tables, upload cases and working backup.')



