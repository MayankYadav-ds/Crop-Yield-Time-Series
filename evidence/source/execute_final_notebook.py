import os,time,datetime,sys,json,asyncio
from pathlib import Path
import nbformat
from nbclient import NotebookClient
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
root=Path('work/final_audit').resolve();out=Path('outputs/FINAL_EVALUATION_SUBMISSION');out.mkdir(exist_ok=True)
for env,folder in [('JUPYTER_RUNTIME_DIR','kernel_tmp'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','mpl_cache'),('JUPYTER_CONFIG_DIR','jupyter_config')]:
 p=root/folder;p.mkdir(exist_ok=True);os.environ[env]=str(p)
os.environ['JUPYTER_PATH']=str(root/'jupyter_runtime/share/jupyter');os.environ['APY_CSV']=str(Path('work/crop_production.csv').resolve());os.environ['APY_OUTPUT']=str(root/'notebook_results')
nb=nbformat.read(root/'final_unexecuted.ipynb',as_version=4);start=time.perf_counter();events=[]
def started(cell,cell_index,**kwargs):
 if cell.cell_type=='code':print('Starting notebook cell',cell_index+1,flush=True)
def finished(cell,cell_index,**kwargs):
 if cell.cell_type=='code':
  events.append({'cell_index':cell_index+1,'execution_count':cell.get('execution_count'),'error_outputs':sum(o.output_type=='error' for o in cell.get('outputs',[]))});print('Completed notebook cell',cell_index+1,flush=True)
client=NotebookClient(nb,kernel_name='crop-final-audit',timeout=900,allow_errors=False,resources={'metadata':{'path':str(Path.cwd())}},on_cell_start=started,on_cell_executed=finished,record_timing=True)
try:client.execute()
finally:nbformat.write(nb,out/'Complete_Time_Series_Project_EXECUTED.ipynb')
errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error'];assert not errors
codes=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in codes)
manifest={'execution_method':'real Jupyter kernel via nbclient','notebook_cells':len(nb.cells),'code_cells_executed':len(codes),'error_outputs':len(errors),'elapsed_seconds':time.perf_counter()-start,'events':events,'embedded_png_outputs':sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs),'colab_hosted_execution':'not performed: browser signed out; no account credentials provided','model_failures_are_explicitly_logged':True}
(root/'notebook_execution.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
