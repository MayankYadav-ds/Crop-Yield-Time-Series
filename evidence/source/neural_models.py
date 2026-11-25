"""Small CPU neural models with training-only normalization and nested stopping."""
import time,copy,random,json
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
torch.backends.mkldnn.enabled=False
class TinySequence(nn.Module):
 def __init__(self,kind,hidden=16,dropout=.1):
  super().__init__();self.kind=kind
  if kind=='LSTM':self.rnn=nn.LSTM(1,hidden,batch_first=True)
  elif kind=='GRU':self.rnn=nn.GRU(1,hidden,batch_first=True)
  else:
   self.patch=nn.Linear(2,8);self.position=nn.Parameter(torch.randn(1,2,8)*.02)
   self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(8,2,16,dropout,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
  self.dropout=nn.Dropout(dropout);self.head=nn.Linear(16 if kind=='PatchTransformerTiny' else hidden,1)
 def forward(self,x):
  if self.kind=='PatchTransformerTiny':
   patches=x.squeeze(-1).unfold(1,2,1);h=self.encoder(self.patch(patches)+self.position).flatten(1)
  else:h=self.rnn(x)[0][:,-1,:]
  return self.head(self.dropout(h)).squeeze(-1)
def seed_all(seed=42):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def sequences(wide,end):
 X=[];Y=[];ident=[]
 for sid,r in wide.iterrows():
  for yr in range(2008,end+1):X.append(r.loc[yr-3:yr-1].values);Y.append(r[yr]);ident.append((sid,yr))
 return np.asarray(X,dtype=np.float32),np.asarray(Y,dtype=np.float32),ident
def forecast_inputs(wide,year):return np.stack([r.loc[year-3:year-1].values for _,r in wide.iterrows()]).astype(np.float32)
def train_model(wide,kind,end,epochs=60,inner_stop=False,checkpoint=None):
 seed_all(42);mu=float(wide.loc[:,:end].values.mean());sd=float(wide.loc[:,:end].values.std());sd=max(sd,1e-6)
 X,Y,ident=sequences(wide,end);xx=torch.tensor(((X-mu)/sd)[...,None]);yy=torch.tensor((Y-mu)/sd)
 m=TinySequence(kind);optimizer=torch.optim.Adam(m.parameters(),lr=.003,weight_decay=1e-4);lossfn=nn.MSELoss();trace=[];best=float('inf');beststate=None;bestepoch=0;stale=0;tic=time.perf_counter();initial=copy.deepcopy(m.state_dict())
 if inner_stop:
  assert end==2008
  vx=torch.tensor(((forecast_inputs(wide,2009)-mu)/sd)[...,None]);vy=torch.tensor((wide[2009].values.astype(np.float32)-mu)/sd)
 for ep in range(1,epochs+1):
  m.train();order=torch.randperm(len(xx));total=0
  for start in range(0,len(xx),256):
   idx=order[start:start+256];optimizer.zero_grad();pred=m(xx[idx]);loss=lossfn(pred,yy[idx]);assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);optimizer.step();total+=float(loss.detach())*len(idx)
  m.eval()
  with torch.no_grad():valid=float(lossfn(m(vx),vy)) if inner_stop else None
  trace.append({'epoch':ep,'train_normalized_MSE':total/len(xx),'validation_normalized_MSE':valid})
  if inner_stop:
   if valid<best-1e-6:best=valid;beststate=copy.deepcopy(m.state_dict());bestepoch=ep;stale=0
   else:stale+=1
   if stale>=8:break
 if inner_stop:m.load_state_dict(beststate)
 else:bestepoch=epochs
 assert any(not torch.equal(initial[k],m.state_dict()[k]) for k in initial),'Weights did not change'
 pack={'state_dict':m.state_dict(),'mu':mu,'sd':sd,'kind':kind,'end':end,'epochs':bestepoch}
 if checkpoint is not None:torch.save(pack,checkpoint)
 return m,pack,trace,{'kind':kind,'origin':end,'epochs_run':len(trace),'selected_epochs':bestepoch,'parameter_count':sum(p.numel() for p in m.parameters()),'training_sequences':len(xx),'normalization_mean':mu,'normalization_std':sd,'train_seconds':time.perf_counter()-tic,'max_training_target_year':max(y for _,y in ident),'inner_stop_year':2009 if inner_stop else None}
def predict_batch(model,pack,X):
 model.eval()
 with torch.no_grad():return model(torch.tensor(((np.asarray(X,dtype=np.float32)-pack['mu'])/pack['sd'])[...,None])).numpy().astype(float)*pack['sd']+pack['mu']
def verify_checkpoint(path,X):
 p=torch.load(path,map_location='cpu',weights_only=True);m=TinySequence(p['kind']);m.load_state_dict(p['state_dict']);return predict_batch(m,p,X)
