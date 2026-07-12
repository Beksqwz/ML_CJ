"""Stage 13 independent 1h research experiments over frozen Stage 7D data."""
from __future__ import annotations
import json,gc,time,argparse
from datetime import UTC,datetime
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,CatBoostRegressor
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/stage13'; SEEDS=(20260711,20260712,20260713)
CHECK=OUT/'checkpoints'; ART=ROOT/'models/stage13'
def args():
 p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');return p.parse_args()
def cp(key,seed): return CHECK/f'{key}_seed{seed}.json'
def valid(path):
 try:
  q=json.loads(path.read_text());return q if q.get('status')=='completed' and 'validation_pr_auc' in q else None
 except Exception:return None
def prep(d,f,c):
 x=d[f].copy()
 for z in c:x[z]=x[z].astype('string').fillna('__MISSING__').astype(str)
 return x
def pr(y,p):
 g=pd.DataFrame({'p':p,'y':y}).groupby('p').y.agg(['count','sum']).sort_index(ascending=False);r=g['sum'].cumsum()/y.sum();q=g['sum'].cumsum()/g['count'].cumsum();return float((q*r.diff().fillna(r)).sum())
def metric(y,p):
 top=np.argsort(-p)[:int(np.ceil(.1*len(y)))];b=np.clip(np.digitize(p,np.linspace(0,1,11),right=True)-1,0,9);e=sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).sum() for i in range(10) if (b==i).any())/len(y);return {'pr_auc':pr(y,p),'recall_at_top_10pct':float(y[top].sum()/y.sum()),'lift_at_top_10pct':float(y[top].mean()/y.mean()),'brier_score':float(np.mean((p-y)**2)),'expected_calibration_error_10_bins':float(e)}
def cat(x,y,v,vy,idx,seed,loss='Logloss',weight=None):
 p={'iterations':1500,'learning_rate':.05,'depth':7,'l2_leaf_reg':5.,'loss_function':loss,'eval_metric':'RMSE' if loss=='Poisson' else 'PRAUC','random_seed':seed,'allow_writing_files':False,'task_type':'GPU','devices':'0'}
 if weight:p['scale_pos_weight']=weight
 m=(CatBoostRegressor if loss=='Poisson' else CatBoostClassifier)(**p);m.fit(x,y,cat_features=idx,eval_set=(v,vy),early_stopping_rounds=100,verbose=False);return m,p
def run_cat(name,x,y,idx,loss='Logloss',weight=None,force=False):
 runs=[];m0=None;params=None
 for seed in SEEDS:
  path=cp(name.replace('.','_'),seed); cached=None if force else valid(path); model_path=ART/f'{name.replace(".","_")}_seed{seed}.cbm'
  if cached and model_path.exists():
   print(f'[resume] {name} seed={seed}'); runs.append(cached); params=cached['params'];
   if seed==SEEDS[0]:m0=(CatBoostRegressor() if loss=='Poisson' else CatBoostClassifier());m0.load_model(model_path)
   continue
  print(f'[start] {name} seed={seed}'); started=time.perf_counter(); CHECK.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
  try:
   m,p=cat(x['train'],y['train'],x['validation'],y['validation'],idx,seed,loss,weight); q=(m.predict(x['validation'],prediction_type='Exponent').reshape(-1) if loss=='Poisson' else m.predict_proba(x['validation'])[:,1]); row={'status':'completed','seed':seed,'validation_pr_auc':pr(y['validation'],q),'best_iteration':int(m.get_best_iteration()),'params':p,'seconds':time.perf_counter()-started};m.save_model(model_path);path.write_text(json.dumps(row,ensure_ascii=False,indent=2));runs.append(row);m0=m if seed==SEEDS[0] else m0;params=p;print(f'[done] {name} seed={seed} {row["seconds"]:.1f}s')
  except Exception as exc:
   row={'status':'failed','seed':seed,'error':str(exc),'seconds':time.perf_counter()-started};path.write_text(json.dumps(row,ensure_ascii=False,indent=2));runs.append(row);print(f'[failed] {name} seed={seed}: {exc}')
  finally:
   if 'm' in locals():del m
   gc.collect();time.sleep(2)
 if m0 is None:return {'name':name,'failed':True,'stability_three_seeds':{'runs':runs,'pr_auc_std':None},'metrics':{}}
 # Poisson rate to event probability: 1-exp(-rate), monotonic and count-process calibrated.
 conv=lambda a:1-np.exp(-np.maximum(a,0)) if loss=='Poisson' else a
 raw=lambda split:(np.clip(m0.predict(x[split]).reshape(-1),0,20) if loss=='Poisson' else m0.predict_proba(x[split])[:,1])
 out={'validation':metric(y['validation'],conv(raw('validation'))),'test':metric(y['test'],conv(raw('test')))}
 return {'name':name,'winner':{'params':params,'best_iteration':runs[0]['best_iteration']},'stability_three_seeds':{'runs':runs,'pr_auc_std':float(np.std([r['validation_pr_auc'] for r in runs]))},'metrics':out}
def accept(r,b):
 m=r['metrics'];return m['validation']['pr_auc']>=b['validation']['pr_auc']*1.03 and m['test']['pr_auc']>=b['test']['pr_auc'] and r.get('stability_three_seeds',{}).get('pr_auc_std',0)<=.01 and all(m[s][k]<=b[s]['calibration'][k] for s in ('validation','test') for k in ('brier_score','expected_calibration_error_10_bins'))
def main():
 force=args().force
 OUT.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'reports/stage7d/1h/stage7d_feature_config.json').read_text());f=cfg['numerical_features']+cfg['categorical_features'];c=cfg['categorical_features'];d={s:pd.read_parquet(ROOT/f'data/processed/stage7d/training_dataset_1h_{s}.parquet') for s in ('train','validation','test')};y={s:d[s].target_1h.to_numpy(np.int8) for s in d};base=json.loads((ROOT/'reports/stage7d/1h/stage7d_weather_experiment_report.json').read_text())['experimental_metrics']; results=[]
 # A
 da={s:z.copy() for s,z in d.items()};
 for z in da.values():z['accidents_per_length']=z.segment_accidents_total_prior/(z.road_length.fillna(0).clip(lower=0)+20.0)
 fa=f+['accidents_per_length'];xa={s:prep(da[s],fa,c) for s in da};a=run_cat('A_exposure_normalization',xa,y,[fa.index(z) for z in c],force=force);a['feature_formula']='segment_accidents_total_prior / (road_length + alpha=20)';results.append(a)
 # B
 xb={s:prep(d[s],f,c) for s in d};b=run_cat('B_poisson',xb,y,[f.index(z) for z in c],loss='Poisson',force=force);b['output_transform']='p_event = 1 - exp(-rate)';results.append(b)
 # C: each full 3 seed, select validation PR then top recall
 cw=[]
 for w in (1.5,2.,3.):
  r=run_cat(f'C_class_weight_{w}',xb,y,[f.index(z) for z in c],weight=w,force=force);r['scale_pos_weight']=w;cw.append(r)
 cwin=max(cw,key=lambda r:(r['metrics']['validation']['pr_auc'],r['metrics']['validation']['recall_at_top_10pct']));results+=cw
 # D read-only models; XGB train-only ordinal mapping matching Stage10 representation
 cm=CatBoostClassifier();cm.load_model(ROOT/'models/production/catboost_1h_weather_experiment.cbm');cp={s:cm.predict_proba(xb[s])[:,1] for s in d};xx={s:d[s][f].copy() for s in d}
 for col in c:
  mp={v:i for i,v in enumerate(pd.unique(xx['train'][col].astype('string').fillna('__MISSING__')))}
  for z in xx.values():z[col]=z[col].astype('string').fillna('__MISSING__').map(mp).fillna(-1).astype('int32')
 xm=XGBClassifier();xm.load_model(ROOT/'models/stage10_experiments/xgboost_1h.json');xp={s:xm.predict_proba(xx[s])[:,1] for s in d};grid=[]
 for q in np.arange(.1,1,.1):grid.append((float(q),pr(y['validation'],q*cp['validation']+(1-q)*xp['validation'])));w=max(grid,key=lambda z:z[1])[0];ep={s:w*cp[s]+(1-w)*xp[s] for s in d};ens={'name':'D_catboost_xgboost_ensemble','ensemble_weight_catboost':w,'validation_grid':grid,'metrics':{s:metric(y[s],ep[s]) for s in ('validation','test')},'comparison_single_xgboost':{s:metric(y[s],xp[s]) for s in ('validation','test')}};results.append(ens)
 # The ensemble grid is recorded per weight; retain only its validation-selected member.
 ensemble_rows=[r for r in results if r['name']=='D_catboost_xgboost_ensemble']
 if ensemble_rows:
  results=[r for r in results if r['name']!='D_catboost_xgboost_ensemble']+[max(ensemble_rows,key=lambda r:r['metrics']['validation']['pr_auc'])]
 for r in results:
  r['accepted_as_candidate']=False if r.get('failed') else (accept(r,base) if r['name']!='D_catboost_xgboost_ensemble' else (r['metrics']['validation']['pr_auc']>=base['validation']['pr_auc']*1.03 and r['metrics']['test']['pr_auc']>=base['test']['pr_auc'] and all(r['metrics'][s][k]<=base[s]['calibration'][k] for s in ('validation','test') for k in ('brier_score','expected_calibration_error_10_bins'))));r['decision_ru']='Кандидат прошёл правило; требуется ручное решение.' if r['accepted_as_candidate'] else 'Кандидат отклонён или прогон failed; Stage 7D остаётся финальной.'; (OUT/(r['name']+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2))
 summary={'generated_at_utc':datetime.now(UTC).isoformat(),'stage':'13','baseline_stage7d':base,'experiments':results,'accepted_candidates':[r['name'] for r in results if r['accepted_as_candidate']],'conclusion_ru':'Stage 7D остаётся production-моделью: ни один Stage 13 кандидат не прошёл правило.' if not any(r['accepted_as_candidate'] for r in results) else 'Перечисленные кандидаты требуют отдельного ручного решения; Stage 7D автоматически не заменяется.'};(OUT/'stage13_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
