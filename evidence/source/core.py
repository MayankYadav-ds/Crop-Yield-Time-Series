from pathlib import Path
import os
import time,json,hashlib,warnings,platform,importlib.metadata as meta
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from xgboost import XGBRegressor
ROOT=Path(os.environ.get('APY_OUTPUT','work/final_audit/results'));ROOT.mkdir(exist_ok=True);FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
np.random.seed(42); T0=time.perf_counter()
def js(name,o):(ROOT/name).write_text(json.dumps(o,indent=2,default=str),encoding='utf-8')
def save(name):
 plt.tight_layout(); path=FIG/(name+'.png'); plt.savefig(path,dpi=300,bbox_inches='tight'); plt.close()
 try:
  from IPython.display import display, Image
  display(Image(filename=str(path)))
 except ImportError:
  print('Saved figure:',path)

plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
raw=pd.read_csv(os.environ.get('APY_CSV','raw/crop_production.csv'));d=raw.copy()
for c in ['State_Name','District_Name','Season','Crop']:d[c]=d[c].str.strip().str.replace(r'\s+',' ',regex=True)
key=['State_Name','District_Name','Crop','Crop_Year'];bk=key+['Season']
num=['Area','Production'];d['valid']=np.isfinite(d.Area)&np.isfinite(d.Production)&d.Area.gt(0)&d.Production.ge(0)
d['duplicate']=d.duplicated(bk,keep=False)
summary={'rows':len(raw),'columns':len(raw.columns),'states':d.State_Name.nunique(),'district_name_labels':d.District_Name.nunique(),'state_district_pairs':d[['State_Name','District_Name']].drop_duplicates().shape[0],'crops_raw':raw.Crop.nunique(),'crops_standardized':d.Crop.nunique(),'start_year':int(d.Crop_Year.min()),'end_year':int(d.Crop_Year.max()),'missing':raw.isna().sum().to_dict(),'exact_duplicates':int(raw.duplicated().sum()),'business_duplicate_rows':int(d.duplicate.sum()),'nonpositive_area':int(d.Area.le(0).sum()),'negative_production':int(d.Production.lt(0).sum()),'season_counts':d.Season.value_counts().to_dict(),'raw_rows_by_year':d.Crop_Year.value_counts().sort_index().to_dict(),'units':'not encoded; production/area ratio reported without asserting verified tonnes/hectare'}
js('dataset_summary.json',summary); print('RAW',summary,flush=True)
rows=[];aud=[]
for k,g in d.groupby(key):
 reason='included'
 if g.duplicate.any():reason='duplicate_business_key'
 elif g.Season.eq('Whole Year').any() and len(g)>1:reason='whole_year_overlap'
 elif not g.valid.all():reason='invalid_or_missing_component'
 aud.append(dict(zip(key,k),decision=reason,n_raw=len(g)))
 if reason=='included':
  a=g.Area.sum();p=g.Production.sum();rows.append(dict(zip(key,k),area=a,production=p,y=p/a))
panel=pd.DataFrame(rows).sort_values(key);panel['sid']=panel.State_Name+'|'+panel.District_Name+'|'+panel.Crop
panel.to_csv(ROOT/'annual_panel.csv',index=False);audit=pd.DataFrame(aud);audit.to_csv(ROOT/'aggregation_audit.csv',index=False)
quality={'annual_panel_rows':len(panel),'annual_groups_by_decision':audit.decision.value_counts().to_dict(),'raw_rows_by_decision':audit.groupby('decision').n_raw.sum().to_dict(),'no_production_imputation':True,'outliers_removed':0}
js('preprocessing_summary.json',quality)
cov=panel.groupby('sid').Crop_Year.agg(['min','max','count']);cov['missing_internal_years']=cov['max']-cov['min']+1-cov['count'];cov.to_csv(ROOT/'continuity_report.csv')
CROPS=['Rice','Maize','Wheat','Rapeseed &Mustard']
subset=panel[panel.Crop.isin(CROPS)]
w=subset.pivot(index='sid',columns='Crop_Year',values='y').reindex(columns=list(range(2005,2015)))
wide=w.dropna(); ids=wide.index
metadata=panel[['sid','State_Name','District_Name','Crop']].drop_duplicates('sid').set_index('sid').loc[ids]
metadata.to_csv(ROOT/'selected_series.csv');wide.to_csv(ROOT/'model_panel.csv')
assert len(wide)>0
print('SELECTED',wide.shape,metadata.Crop.value_counts().to_dict(),flush=True)
js('experimental_design.json',{'crops':CROPS,'n_series':len(wide),'n_panel_observations':int(wide.size),'excluded_incomplete_series':len(w)-len(wide),'train_final':[2005,2011],'validation_targets':[2010,2011],'validation_origins':[2009,2010],'one_step_test_targets':[2012,2013,2014],'fixed_origin_test':{'origin':2011,'targets':[2012,2013,2014]},'raw_2015_excluded':'562 records; incomplete coverage relative to other years','modeled_cutoff':2014,'future_targets':[2015,2016,2017],'survivor_bias':'eligibility requires complete test-era observations','seed':42,'series_selection':'four named crops matching earlier experiment scope','units':'production units / area units; physical labels absent'})


# Representative EDA uses matched districts; all non-modeling EDA can use full panel.
sel=panel[panel.sid.isin(ids)&panel.Crop_Year.between(2005,2014)]
pooled=sel.groupby(['Crop','Crop_Year'])[['area','production']].sum();pooled['yield_ratio']=pooled.production/pooled.area;pooled.to_csv(ROOT/'weighted_crop_series.csv')
for crop in CROPS:
 annual=pooled.loc[crop];fig,ax=plt.subplots(3,1,figsize=(6.6,7),sharex=True)
 for axis,col,lab in zip(ax,['yield_ratio','production','area'],['Production / area','Production units','Area units']):annual[col].plot(ax=axis,marker='o');axis.set_ylabel(lab)
 ax[0].set_title(crop+' annual trends in the balanced panel');ax[-1].set_xlabel('Crop start year');save('trends_'+crop.split()[0])
fig,ax=plt.subplots(figsize=(3.3,2.1));s=d.Crop_Year.value_counts().sort_index();ax.plot(s.index,s.values,marker='o',ms=2);ax.set(xlabel='Crop start year',ylabel='Raw records',title='Annual reporting volume');save('reporting')
fig,ax=plt.subplots(figsize=(3.3,2.1));d.isna().groupby(d.Crop_Year).mean().Production.plot.bar(ax=ax);ax.set(ylabel='Missing production proportion',xlabel='Crop start year',title='Missing production by year');save('missingness')
fig,ax=plt.subplots(figsize=(3.3,2.2))
for crop in CROPS:pooled.loc[crop].yield_ratio.plot(ax=ax,marker='o',ms=2,label=crop)
ax.set(xlabel='Crop start year',ylabel='Production / area',title='Weighted yield ratios in the matched panel');ax.legend(fontsize=6);save('yield_trends')
fig,ax=plt.subplots(figsize=(3.3,2.2));cov['missing_internal_years'].plot.hist(ax=ax,bins=20);ax.set(xlabel='Missing internal years',ylabel='Series',title='Continuity across full available histories');save('coverage')


# Training-only diagnostics on longest complete 1997-2011 histories for the selected identities.
long=panel[panel.sid.isin(ids)].pivot(index='sid',columns='Crop_Year',values='y').reindex(columns=list(range(1997,2012))).dropna()
station=[]; diagnostic_warnings=[]
for sid,r in long.iterrows():
 for transform,v in [('level',r.values),('difference',np.diff(r.values))]:
  try:
   with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    a=adfuller(v,maxlag=1,autolag='AIC');k=kpss(v,regression='c',nlags='auto')
   diagnostic_warnings.extend({'sid':sid,'transform':transform,'category':type(w.message).__name__,'message':str(w.message)} for w in captured)
   station.append({'sid':sid,'transform':transform,'n':len(v),'adf_stat':a[0],'adf_p':a[1],'kpss_stat':k[0],'kpss_p':k[1]})
  except Exception as ex:station.append({'sid':sid,'transform':transform,'error':str(ex)})
station=pd.DataFrame(station);station.to_csv(ROOT/'stationarity_results.csv',index=False)
print('STATIONARITY',len(long),flush=True)
pd.DataFrame(diagnostic_warnings,columns=['sid','transform','category','message']).to_csv(ROOT/'diagnostic_warnings.csv',index=False)
print('Diagnostic warnings captured and exported:',len(diagnostic_warnings))
if len(long):
 representative=long.iloc[0]
 fig,axes=plt.subplots(1,2,figsize=(7,2.8))
 plot_acf(representative.values,lags=5,ax=axes[0]);plot_pacf(representative.values,lags=5,method='ywm',ax=axes[1])
 fig.suptitle('Representative training-era history, 1997–2011: '+str(long.index[0]),fontsize=8)
 save('training_acf_pacf')



# Models: local baseline methods, a transparent OLS AR(1), and custom least-squares smoothing.
NAMES=['Naive','Mean','MA3','Drift','AR1_OLS','SES_LS','Holt_LS','DampedHolt_LS']
fitlog=[]
def smooth(h,alpha,beta=0,phi=1,kind='SES'):
 l=float(h[0]);b=0 if kind=='SES' else float(h[1]-h[0]);sse=0;res=[]
 for v in h[1:]:
  pr=l+phi*b;e=float(v-pr);res.append(e);sse+=e*e
  ln=alpha*v+(1-alpha)*pr
  b=0 if kind=='SES' else beta*(ln-l)+(1-beta)*phi*b
  l=ln
 return sse,l,b,np.array(res)
def fit(name,h):
 h=np.asarray(h,dtype=float);n=len(h)
 if name=='Naive':return lambda H:np.repeat(h[-1],H),{},None
 if name=='Mean':return lambda H:np.repeat(h.mean(),H),{},None
 if name=='MA3':return lambda H:np.repeat(h[-3:].mean(),H),{'window':3},None
 if name=='Drift':return lambda H:h[-1]+np.arange(1,H+1)*(h[-1]-h[0])/(n-1),{},None
 if name=='AR1_OLS':
  X=np.column_stack([np.ones(n-1),h[:-1]]);c,phi=np.linalg.lstsq(X,h[1:],rcond=None)[0]
  def fc(H):
   ans=[];v=h[-1]
   for _ in range(H):v=c+phi*v;ans.append(v)
   return np.array(ans)
  return fc,{'constant':c,'phi':phi},h[1:]-X@np.array([c,phi])
 kind='SES' if name=='SES_LS' else 'Holt';damped=name=='DampedHolt_LS'
 x0=[.5] if kind=='SES' else ([.5,.2,.9] if damped else [.5,.2])
 bounds=[(.01,.99)] if kind=='SES' else ([(.01,.99),(.01,.99),(.8,.98)] if damped else [(.01,.99),(.01,.99)])
 def unpack(p):return p[0],(0 if kind=='SES' else p[1]),(p[2] if damped else 1)
 opt=minimize(lambda p:smooth(h,*unpack(p),kind=kind)[0],x0,bounds=bounds,method='L-BFGS-B',options={'maxiter':150})
 a,b,phi=unpack(opt.x);_,level,trend,res=smooth(h,a,b,phi,kind=kind)
 def fc(H):return np.array([level+trend*sum(phi**j for j in range(1,i+1)) for i in range(1,H+1)])
 return fc,{'alpha':a,'beta':b,'phi':phi,'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message)},res
failures=[];timings=[]
def local_predictions(years,mode):
 records=[]
 for name in NAMES:
  tic=time.perf_counter()
  for sid,r in wide.iterrows():
   origins=[2011] if mode=='fixed_origin' else [yr-1 for yr in years]
   for origin in origins:
    try:
     fc,params,res=fit(name,r.loc[:origin].values)
     fitlog.append({'model':name,'sid':sid,'origin':origin,'mode':mode,**params})
     yrs=years if mode=='fixed_origin' else [origin+1];pred=fc(len(yrs))
     if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
     for yr,v in zip(yrs,pred):records.append({'model':name,'sid':sid,'year':yr,'horizon':yr-origin,'mode':mode,'actual':r[yr],'prediction':float(v)})
    except Exception as ex:failures.append({'model':name,'sid':sid,'origin':origin,'mode':mode,'error':str(ex)})
  timings.append({'model':name,'mode':mode,'targets':str(years),'seconds':time.perf_counter()-tic})
  print(mode,name,'done',round(time.perf_counter()-tic,2),flush=True)
 return pd.DataFrame(records)
FEATURES=['lag1','lag2','lag3','mean3','std3','time']
def feat(h,yr):h=np.asarray(h);return [h[-1],h[-2],h[-3],h[-3:].mean(),h[-3:].std(),yr-2005]
def xy(end):
 X=[];Y=[]
 for _,r in wide.iterrows():
  for yr in range(2008,end+1):X.append(feat(r.loc[:yr-1],yr));Y.append(r[yr])
 return np.array(X),np.array(Y)
PARAMS=[dict(n_estimators=150,max_depth=2,learning_rate=.05,subsample=.8,colsample_bytree=.9,reg_lambda=5),dict(n_estimators=250,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.9,reg_lambda=10)]
def make_xgb(params):return XGBRegressor(**params,objective='reg:squarederror',random_state=42,n_jobs=2)
def xp(params,years,name,mode):
 records=[];tic=time.perf_counter()
 origins=[2011] if mode=='fixed_origin' else [yr-1 for yr in years]
 for origin in origins:
  X,Y=xy(origin);m=make_xgb(params).fit(X,Y)
  for sid,r in wide.iterrows():
   history=r.loc[:origin].tolist();yrs=years if mode=='fixed_origin' else [origin+1]
   for yr in yrs:
    v=float(m.predict(np.array([feat(history,yr)]))[0]);records.append({'model':name,'sid':sid,'year':yr,'horizon':yr-origin,'mode':mode,'actual':r[yr],'prediction':v});history.append(v)
 timings.append({'model':name,'mode':mode,'targets':str(years),'seconds':time.perf_counter()-tic})
 return pd.DataFrame(records)
scales=wide.loc[:,:2009].diff(axis=1).abs().mean(axis=1) # validation scaling uses only initial training
def score(pred,expected):
 ans=[]
 for name,g in pred.groupby('model'):
  if len(g)!=expected or g.duplicated(['sid','year']).any() or not np.isfinite(g.prediction).all():continue
  a=g.actual.to_numpy();p=g.prediction.to_numpy();e=a-p;valid=np.abs(a)>1e-8;den=np.abs(a)+np.abs(p);sc=g.sid.map(scales).to_numpy();ms=sc>1e-8
  ans.append({'model':name,'n':len(g),'MAE':abs(e).mean(),'RMSE':np.sqrt((e**2).mean()),'MAPE':100*np.mean(abs(e[valid]/a[valid])) if valid.any() else None,'MAPE_n':int(valid.sum()),'sMAPE':np.mean(np.divide(200*abs(e),den,out=np.zeros_like(e),where=den>1e-8)),'MASE':np.mean(abs(e[ms])/sc[ms]) if ms.any() else None,'MFE':e.mean()})
 return pd.DataFrame(ans).sort_values('RMSE')
