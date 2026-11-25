"""Retrospective expanded study; untouched backup retains the prior experiment."""
from pathlib import Path
exec(Path(__file__).with_name('core.py').read_text(encoding='utf-8'),globals())
from neural_models import train_model,predict_batch,forecast_inputs,verify_checkpoint
import torch,copy,logging,traceback,importlib.metadata as pkg
(RootModels:=ROOT/'models').mkdir(exist_ok=True)
model_status={};training_records=[];test_cache={};inner_epochs={};trial_records=[]
# Expanded XGBoost search: two previous settings plus ten seeded candidates.
rng=np.random.default_rng(42)
CANDIDATES=list(PARAMS)
while len(CANDIDATES)<12:
 p={'n_estimators':int(rng.choice([100,200,350])),'max_depth':int(rng.choice([2,3,4])),'learning_rate':float(rng.choice([.03,.05,.08])),'subsample':float(rng.choice([.7,.85,1.])),'colsample_bytree':float(rng.choice([.7,.9,1.])),'reg_lambda':float(rng.choice([1.,5.,10.])),'reg_alpha':float(rng.choice([0.,.1,1.])),'min_child_weight':int(rng.choice([1,5,10]))}
 if p not in CANDIDATES:CANDIDATES.append(p)
def xgb_eval(params,years,name,mode):
 records=[];models={}
 for origin in ([2011] if mode=='fixed_origin' else [y-1 for y in years]):
  X,Y=xy(origin);tic=time.perf_counter();m=XGBRegressor(**params,objective='reg:squarederror',random_state=42,n_jobs=2).fit(X,Y)
  training_records.append({'model':name,'origin':origin,'mode':mode,'train_seconds':time.perf_counter()-tic,'training_rows':len(Y)})
  models[origin]=m;hist=wide.loc[:,:origin].values.copy();yrs=years if mode=='fixed_origin' else [origin+1]
  for year in yrs:
   Xp=np.array([feat(h,year) for h in hist]);tinf=time.perf_counter();pr=m.predict(Xp)
   training_records.append({'model':name,'origin':origin,'mode':mode,'inference_seconds':time.perf_counter()-tinf,'inference_rows':len(pr)})
   for sid,v in zip(wide.index,pr):records.append({'model':name,'sid':sid,'year':year,'horizon':year-origin,'mode':mode,'actual':wide.loc[sid,year],'prediction':float(v)})
   hist=np.column_stack([hist,pr])
 return pd.DataFrame(records),models
scales=wide.loc[:,:2009].diff(axis=1).abs().mean(axis=1)
val=local_predictions([2010,2011],'validation_one_step')
trial_predictions={}
for i,params in enumerate(CANDIDATES):
 try:
  pr,_=xgb_eval(params,[2010,2011],f'XGB_trial_{i:02d}','validation_one_step');metric=score(pr,len(wide)*2).iloc[0].to_dict();trial_records.append({'trial':i,'status':'completed',**params,**metric});trial_predictions[i]=pr
 except Exception as e:trial_records.append({'trial':i,'status':'failed',**params,'error':str(e)})
 print('XGBoost trial',i,'done',flush=True)
trials=pd.DataFrame(trial_records);trials.to_csv(ROOT/'hyperparameter_results.csv',index=False)
valid_trials=trials[trials.status.eq('completed')];chosen_trial=int(valid_trials.sort_values(['RMSE','trial']).iloc[0].trial);chosen_xgb=CANDIDATES[chosen_trial]
xval=trial_predictions[chosen_trial].copy();xval['model']='XGBoost_Tuned';val=pd.concat([val,xval],ignore_index=True)
js('xgboost_selection.json',{'trial_count':len(CANDIDATES),'selected_trial':chosen_trial,'parameters':chosen_xgb,'criterion':'expanding validation RMSE, 2010 and 2011','seed':42})
# Inner 2009 stopping year is earlier than both outer validation targets.
for kind in ['LSTM','GRU','PatchTransformerTiny']:
 try:
  m,pack,trace,rec=train_model(wide,kind,2008,epochs=60,inner_stop=True,checkpoint=RootModels/(kind+'_inner.pt'));inner_epochs[kind]=pack['epochs'];training_records.append({'model':kind,'mode':'inner_stopping',**rec})
  pd.DataFrame(trace).to_csv(ROOT/(kind+'_loss.csv'),index=False)
  records=[]
  for year in [2010,2011]:
   mm,pp,tt,rr=train_model(wide,kind,year-1,epochs=inner_epochs[kind],checkpoint=RootModels/f'{kind}_validation_{year-1}.pt');training_records.append({'model':kind,'mode':'validation_one_step',**rr})
   pred=predict_batch(mm,pp,forecast_inputs(wide,year))
   for sid,v in zip(wide.index,pred):records.append({'model':kind,'sid':sid,'year':year,'horizon':1,'mode':'validation_one_step','actual':wide.loc[sid,year],'prediction':float(v)})
  val=pd.concat([val,pd.DataFrame(records)],ignore_index=True);model_status[kind]={'status':'validation completed','selected_epochs':inner_epochs[kind],'parameter_count':rec['parameter_count'],'lookback':3,'normalization':'fit separately on each training origin','inner_train_end':2008,'inner_stop_year':2009,'max_epochs':60,'patience':8,'dropout':.1,'learning_rate':.003,'batch_size':256}
 except Exception as e:model_status[kind]={'status':'excluded after failure','error':str(e),'traceback':traceback.format_exc()}
 print('Neural validation',kind,model_status[kind]['status'],flush=True)
vl=score(val,len(wide)*2);best=vl.sort_values(['RMSE','model']).iloc[0].model
js('selection_frozen.json',{'best_model':best,'criterion':'outer expanding validation RMSE','test_evaluation_started':False,'test_limitation':'retrospective extension; these test years were examined in the prior archived study; no independent prospective holdout claimed'})
val.to_csv(ROOT/'validation_predictions.csv',index=False);vl.to_csv(ROOT/'validation_leaderboard.csv',index=False)
print('FROZEN OVERALL WINNER',best,flush=True)
# No further tuning takes place after this point.
scales=wide.loc[:,:2011].diff(axis=1).abs().mean(axis=1)
test=local_predictions([2012,2013,2014],'test_one_step');fixed=local_predictions([2012,2013,2014],'fixed_origin')
xp_test,xmodels=xgb_eval(chosen_xgb,[2012,2013,2014],'XGBoost_Tuned','test_one_step');xp_fixed,_=xgb_eval(chosen_xgb,[2012,2013,2014],'XGBoost_Tuned','fixed_origin');test=pd.concat([test,xp_test],ignore_index=True);fixed=pd.concat([fixed,xp_fixed],ignore_index=True)
for kind in inner_epochs:
 try:
  records=[];models={}
  for year in [2012,2013,2014]:
   m,pack,trace,rec=train_model(wide,kind,year-1,epochs=inner_epochs[kind],checkpoint=RootModels/f'{kind}_test_origin_{year-1}.pt');training_records.append({'model':kind,'mode':'test_one_step',**rec});models[year-1]=(m,pack)
   Xp=forecast_inputs(wide,year);tic=time.perf_counter();pr=predict_batch(m,pack,Xp);training_records.append({'model':kind,'origin':year-1,'mode':'test_one_step','inference_seconds':time.perf_counter()-tic,'inference_rows':len(pr)})
   replay=verify_checkpoint(RootModels/f'{kind}_test_origin_{year-1}.pt',Xp[:5]);np.testing.assert_allclose(replay,pr[:5],atol=1e-7,rtol=1e-7)
   for sid,v in zip(wide.index,pr):records.append({'model':kind,'sid':sid,'year':year,'horizon':1,'mode':'test_one_step','actual':wide.loc[sid,year],'prediction':float(v)})
  test=pd.concat([test,pd.DataFrame(records)],ignore_index=True)
  m,pack=models[2011];hist=wide.loc[:,:2011].values.copy();records=[]
  for year in [2012,2013,2014]:
   pr=predict_batch(m,pack,hist[:,-3:])
   for sid,v in zip(wide.index,pr):records.append({'model':kind,'sid':sid,'year':year,'horizon':year-2011,'mode':'fixed_origin','actual':wide.loc[sid,year],'prediction':float(v)})
   hist=np.column_stack([hist,pr])
  fixed=pd.concat([fixed,pd.DataFrame(records)],ignore_index=True);model_status[kind]['status']='trained and evaluated';test_cache[kind]=models
 except Exception as e:model_status[kind].update(status='evaluation failed',error=str(e),traceback=traceback.format_exc())
for pred,stem in [(test,'test'),(fixed,'fixed_origin')]:
 lead=score(pred,len(wide)*3);lead['RMSE_improvement_vs_naive_pct']=100*(1-lead.RMSE/lead.set_index('model').loc['Naive','RMSE']);pred.to_csv(ROOT/(stem+'_predictions.csv'),index=False);lead.to_csv(ROOT/(stem+'_leaderboard.csv'),index=False)
 print(stem,'\n',lead.to_string(index=False),flush=True)
tl=pd.read_csv(ROOT/'test_leaderboard.csv');fl=pd.read_csv(ROOT/'fixed_origin_leaderboard.csv')
assert best in set(tl.model),'Validation-selected model failed testing; report selection failure rather than silently replacing it'
# Refit only the validation-selected district-level model; keep classical and neural multistep forecasts origin-fixed.
future=[]
if best=='XGBoost_Tuned':
 X,Y=xy(2014);final=XGBRegressor(**chosen_xgb,objective='reg:squarederror',random_state=42,n_jobs=2).fit(X,Y);final.save_model(str(RootModels/'selected_xgboost.json'))
 pd.Series(final.feature_importances_,index=FEATURES).to_csv(ROOT/'feature_importance.csv');hist=wide.values.copy()
 for yr in [2015,2016,2017]:
  pr=final.predict(np.array([feat(h,yr) for h in hist]));hist=np.column_stack([hist,pr])
  for sid,v in zip(wide.index,pr):future.append({'sid':sid,'year':yr,'horizon':yr-2014,'forecast_yield_ratio':float(v),'model':best,'modeled_cutoff':2014})
elif best in inner_epochs:
 final,pack,trace,rec=train_model(wide,best,2014,epochs=inner_epochs[best],checkpoint=RootModels/'selected_neural.pt');training_records.append({'model':best,'mode':'future_refit',**rec});hist=wide.values.copy()
 for yr in [2015,2016,2017]:
  pr=predict_batch(final,pack,hist[:,-3:]);hist=np.column_stack([hist,pr])
  for sid,v in zip(wide.index,pr):future.append({'sid':sid,'year':yr,'horizon':yr-2014,'forecast_yield_ratio':float(v),'model':best,'modeled_cutoff':2014})
else:
 for sid,r in wide.iterrows():
  fc,pa,res=fit(best,r.values)
  for yr,v in zip([2015,2016,2017],fc(3)):future.append({'sid':sid,'year':yr,'horizon':yr-2014,'forecast_yield_ratio':float(v),'model':best,'modeled_cutoff':2014})
future=pd.DataFrame(future);future.to_csv(ROOT/'future_forecasts.csv',index=False)
# Prophet uses longer, stable-cohort annual crop aggregates: a different target population.
try:
 from prophet import Prophet
 logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
 eligible_long=panel[panel.sid.isin(long.index)&panel.Crop_Year.between(1997,2014)]
 aggregate=eligible_long.groupby(['Crop','Crop_Year'])[['area','production']].sum();aggregate['y']=aggregate.production/aggregate.area;aggregate.to_csv(ROOT/'prophet_crop_series.csv')
 def prophet_predictions(prior,years,mode):
  records=[]
  for crop in CROPS:
   s=aggregate.loc[crop].y
   for origin in ([2011] if mode=='fixed_origin' else [y-1 for y in years]):
    tr=s.loc[:origin];frame=pd.DataFrame({'ds':pd.to_datetime(tr.index.astype(str)+'-01-01'),'y':tr.values});m=Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,n_changepoints=2,changepoint_prior_scale=prior,uncertainty_samples=0)
    m.fit(frame,seed=42);yrs=years if mode=='fixed_origin' else [origin+1];pr=m.predict(pd.DataFrame({'ds':pd.to_datetime([str(y)+'-01-01' for y in yrs])})).yhat.values
    for y,v in zip(yrs,pr):records.append({'model':'Prophet','crop':crop,'year':y,'horizon':y-origin,'mode':mode,'actual':s[y],'prediction':float(v)})
  return pd.DataFrame(records)
 pv=[];prior_predictions={}
 for prior in [.01,.05,.1]:
  pp=prophet_predictions(prior,[2010,2011],'validation_one_step');rmse=np.sqrt(np.mean((pp.actual-pp.prediction)**2));pv.append({'changepoint_prior_scale':prior,'validation_RMSE':rmse});prior_predictions[prior]=pp
 prior=min(pv,key=lambda x:x['validation_RMSE'])['changepoint_prior_scale'];pd.DataFrame(pv).to_csv(ROOT/'prophet_tuning.csv',index=False)
 prop_all=[]
 for mode in ['test_one_step','fixed_origin']:
  prop=prophet_predictions(prior,[2012,2013,2014],mode);prop_all.append(prop)
  for crop in CROPS:
   s=aggregate.loc[crop].y
   for year in [2012,2013,2014]:
    origin=2011 if mode=='fixed_origin' else year-1;h=s.loc[:origin].values
    for name in ['Naive','Mean','Drift']:
     fc,_,_=fit(name,h);prop_all.append(pd.DataFrame([{'model':name,'crop':crop,'year':year,'horizon':year-origin,'mode':mode,'actual':s[year],'prediction':float(fc(year-origin)[-1])}]))
 pp=pd.concat(prop_all,ignore_index=True);pp.to_csv(ROOT/'prophet_predictions.csv',index=False);rows=[]
 for (mode,model),g in pp.groupby(['mode','model']):
  a=g.actual.to_numpy();e=a-g.prediction.to_numpy();ok=np.abs(a)>1e-8;rows.append({'mode':mode,'model':model,'n':len(g),'MAE':np.mean(abs(e)),'RMSE':np.sqrt(np.mean(e**2)),'MAPE':100*np.mean(abs(e[ok]/a[ok]))})
 pd.DataFrame(rows).to_csv(ROOT/'prophet_separate_leaderboard.csv',index=False);model_status['Prophet']={'status':'trained and evaluated separately','target':'four crop aggregates from 769 stable histories, 1997–2014','daily_weekly_yearly_seasonality':False,'selected_changepoint_prior_scale':prior,'not_in_district_leaderboard':True,'uncertainty_intervals':'not calibrated; not exported'}
except Exception as e:model_status['Prophet']={'status':'excluded after runtime failure','error':str(e),'traceback':traceback.format_exc()}
# Audits, errors and real plots.
training=pd.DataFrame(training_records);training.to_csv(ROOT/'training_and_inference.csv',index=False);pd.DataFrame(fitlog).to_csv(ROOT/'fitted_parameters.csv',index=False);pd.DataFrame(failures,columns=['model','sid','origin','mode','error']).to_csv(ROOT/'failed_fits.csv',index=False);pd.DataFrame(timings).to_csv(ROOT/'baseline_runtime.csv',index=False)
for pred,stem in [(test,'one_step'),(fixed,'fixed_origin')]:
 e=pred.merge(metadata.reset_index(),on='sid');e['absolute_error']=abs(e.actual-e.prediction)
 for col in ['Crop','State_Name','year','horizon']:e.groupby(['model',col]).absolute_error.agg(['mean','count']).to_csv(ROOT/f'{stem}_error_by_{col}.csv')
for kind in inner_epochs:
 losses=pd.read_csv(ROOT/(kind+'_loss.csv'));fig,ax=plt.subplots(figsize=(3.5,2.2));ax.plot(losses.epoch,losses.train_normalized_MSE,label='Training through 2008');ax.plot(losses.epoch,losses.validation_normalized_MSE,label='Inner stop year 2009');ax.set(xlabel='Epoch',ylabel='Normalized MSE',title=kind+' actual training loss');ax.legend(fontsize=6);save(kind+'_loss')
fig,ax=plt.subplots(figsize=(3.5,3));tl.set_index('model').RMSE.sort_values().plot.barh(ax=ax);ax.set(xlabel='RMSE (production / area)',title='Expanded one-step test RMSE');save('expanded_comparison')
selected=test[test.model.eq(best)].copy();selected['residual']=selected.actual-selected.prediction
fig,ax=plt.subplots(figsize=(3.5,2.2));selected.groupby('year')[['actual','prediction']].mean().plot(ax=ax,marker='o');ax.set(xlabel='Crop year',ylabel='Mean production / area',title='Validation-selected model: one-step test');save('expanded_actual_predicted')
fig,ax=plt.subplots(figsize=(3.5,2.2));wide.mean(axis=0).plot(ax=ax,label='Observed cohort mean');f=future.groupby('year').forecast_yield_ratio.mean();ax.plot([2014]+list(f.index),[wide[2014].mean()]+list(f),'--o',label='Forecast');ax.set(xlabel='Crop year',ylabel='Mean production / area',title='Forecast after modeled cutoff 2014');ax.legend(fontsize=6);save('expanded_future')
fig,ax=plt.subplots(figsize=(3.5,2.2));trials[trials.status.eq('completed')].plot.scatter(x='trial',y='RMSE',ax=ax);ax.set(xlabel='Candidate index',ylabel='Validation RMSE',title='Chronological XGBoost tuning');save('xgboost_tuning')
# ARCH applicability evidence: representative longer history remains very short for volatility estimation.
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
rep=long.index[0];_,_,res=fit('AR1_OLS',long.loc[rep].values)
try:
 lb=acorr_ljungbox(res,lags=[1,2],return_df=True);arch=het_arch(res,nlags=1);js('applicability_diagnostics.json',{'representative_sid':rep,'residual_n':len(res),'ARCH_LM_p':arch[1],'Ljung_Box':lb.to_dict(),'GARCH_VAR_exclusion':'14 annual representative residuals and no validated stationary multivariate design; diagnostic is exploratory and not panel-wide evidence'})
except Exception as e:js('applicability_diagnostics.json',{'error':str(e)})
# Integrity tests on actual objects and independently reloaded predictions.
assert len(wide)==1112 and wide.shape[1]==10 and not panel.duplicated(key).any();assert np.allclose(panel.y,panel.production/panel.area)
assert set(val.year)=={2010,2011} and set(test.year)=={2012,2013,2014};assert len(future)==3*len(wide)
r=wide.iloc[0];saved=feat(r.loc[:2009],2010);changed=r.copy();changed.loc[2010:]=999999;assert np.allclose(saved,feat(changed.loc[:2009],2010))
read=pd.read_csv(ROOT/'test_predictions.csv');computed=score(read,len(wide)*3).set_index('model');assert np.allclose(computed.loc[tl.model,'RMSE'],tl.RMSE)
js('best_model_summary.json',{'best_by_validation_RMSE':best,'validation':vl[vl.model.eq(best)].to_dict('records'),'test_one_step':tl[tl.model.eq(best)].to_dict('records'),'fixed_origin_three_step':fl[fl.model.eq(best)].to_dict('records'),'test_lowest_RMSE':tl.iloc[0].model,'test_lowest_MAE':tl.sort_values('MAE').iloc[0].model,'test_lowest_MAPE':tl.sort_values('MAPE').iloc[0].model,'evaluated_models':list(tl.model),'failed_fits':len(failures),'optimizer_non_success':int((pd.DataFrame(fitlog).get('optimizer_success',pd.Series(dtype=bool))==False).sum()),'negative_test_predictions':int(test.prediction.lt(0).sum()),'negative_future_predictions':int(future.forecast_yield_ratio.lt(0).sum()),'runtime_seconds':time.perf_counter()-T0})
model_status['XGBoost_Tuned']={'status':'trained and evaluated','trial_count':12,'parameters':chosen_xgb};model_status['ARIMA']={'status':'excluded: environment block','error':'Windows application-control policy blocks statsmodels state-space _inversions extension; no policy changes attempted'};model_status['SARIMA_seasonal_HW']={'status':'excluded: no justified annual periodicity'};model_status['ARCH_GARCH_VAR']={'status':'not fit; short annual histories and inadequate validated multivariate/volatility design'}
js('model_training_status.json',model_status);js('runtime_environment.json',{'python':platform.python_version(),'platform':platform.platform(),'packages':{n:pkg.version(n) for n in ['numpy','pandas','scipy','statsmodels','scikit-learn','xgboost','matplotlib','torch','prophet']},'seed':42,'device':'CPU','torch_threads':2});js('hyperparameters.json',{'XGBoost_candidates':CANDIDATES,'selected_xgboost':chosen_xgb,'neural_models':{k:v for k,v in model_status.items() if k in inner_epochs},'selection_metric':'outer validation RMSE','test_previously_seen':True})
(ROOT/'test_results.txt').write_text('PASS weighted aggregation identity; unique annual keys; complete panel; chronological target windows; future-target feature invariance; forecast alignment; saved-prediction RMSE recalculation; neural parameter updates; neural checkpoint inference replay.\nProphet is evaluated separately on different aggregate targets. Colab hosted execution not certified by this local run.\n')
print('FINAL EXTENSION FINISHED',round(time.perf_counter()-T0,2),'seconds',flush=True)
