"""Create causal Stage 11 graph features on the verified parallel 1h splits."""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/stage11'; REPORTS=ROOT/'reports/stage11'
SPATIAL=['segment_neighbors_accidents_prev_24h','segment_neighbors_accidents_prev_7d','graph_degree','graph_betweenness_centrality']

def adjacency() -> pd.DataFrame:
    # GraphML is the authoritative topology; CSV supplies its stable directed-edge segment IDs.
    graph=nx.read_graphml(ROOT/'data/roads/astana_roads.graphml')
    edges=pd.read_csv(ROOT/'data/roads/astana_edges.csv',usecols=['u','v','key'])
    if graph.number_of_edges()!=len(edges): raise ValueError('GraphML/edge table edge count mismatch')
    edges['road_segment_id']=edges.u.astype(str)+'_'+edges.v.astype(str)+'_'+edges.key.astype(str)
    endpoint={}
    for r in edges.itertuples(index=False):
        endpoint.setdefault(str(r.u),set()).add(r.road_segment_id); endpoint.setdefault(str(r.v),set()).add(r.road_segment_id)
    first={s:set().union(*(endpoint[str(n)] for n in (u,v)))-{s} for s,u,v in edges[['road_segment_id','u','v']].itertuples(index=False,name=None)}
    two={s:(n|set().union(*(first.get(x,set()) for x in n))-{s}) for s,n in first.items()}
    # Deterministic NetworkX approximation; exact Brandes is prohibitively slow here.
    centrality=nx.betweenness_centrality(graph.to_undirected(),k=min(512,graph.number_of_nodes()),seed=20260711,normalized=True)
    return pd.DataFrame([{'road_segment_id':s,'neighbor_segment_ids':sorted(two[s]),'graph_degree':len(two[s]),'graph_betweenness_centrality':float((centrality.get(str(u),0)+centrality.get(str(v),0))/2)} for s,u,v in edges[['road_segment_id','u','v']].itertuples(index=False,name=None)])

def dynamic(frame: pd.DataFrame, adj: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out=frame.merge(adj[['road_segment_id','neighbor_segment_ids','graph_degree','graph_betweenness_centrality']],on='road_segment_id',how='left',validate='many_to_one')
    if out.neighbor_segment_ids.isna().any(): raise ValueError('segments absent from graph adjacency')
    event_map={s:g.sort_values('t') for s,g in events.groupby('road_segment_id')}
    for col,hours in [('segment_neighbors_accidents_prev_24h',24),('segment_neighbors_accidents_prev_7d',168)]:
        values=np.zeros(len(out),dtype=np.int32)
        for seg,ids in out.groupby('road_segment_id').groups.items():
            query=pd.to_datetime(out.loc[ids,'datetime_hour']).astype('datetime64[ns]').astype('int64').to_numpy(); total=np.zeros(len(ids),dtype=np.int32)
            for n in out.loc[ids[0],'neighbor_segment_ids']:
                e=event_map.get(n)
                if e is None: continue
                times=e.t.to_numpy(dtype=np.int64); cum=e['count'].cumsum().to_numpy()
                before=lambda x: np.where((p:=np.searchsorted(times,x,side='left'))>0,cum[p-1],0)
                total += before(query)-before(query-np.int64(hours*3600*10**9))
            values[np.asarray(ids)]=total
        out[col]=values
    return out.drop(columns='neighbor_segment_ids')

def main():
    adj=adjacency(); DATA.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True); adj.to_parquet(DATA/'segment_adjacency.parquet',index=False)
    ready=pd.read_parquet(ROOT/'data/processed/accidents_with_roads_ml_ready.parquet'); ev=ready[['road_segment_id','accident_datetime']].copy(); ev.road_segment_id=ev.road_segment_id.astype(str); ev['t']=pd.to_datetime(ev.accident_datetime).astype('datetime64[ns]').astype('int64'); ev=ev.groupby(['road_segment_id','t'],as_index=False).size().rename(columns={'size':'count'})
    source=json.loads((REPORTS/'stage11_reconstruction_report.json').read_text())
    cfg=json.loads((ROOT/'reports/stage7d/1h/stage7d_feature_config.json').read_text()); cfg['numerical_features']=list(cfg['numerical_features'])+SPATIAL; cfg['excluded_from_model_features']=sorted(set(cfg['excluded_from_model_features'])|{'road_segment_id'}); cfg['spatial_features']={'temporal':SPATIAL[:2],'static':SPATIAL[2:]}; cfg['source']='Stage11 parallel reconstruction plus causal 2-hop GraphML features'
    for split in ('train','validation','test'):
        d=pd.read_parquet(DATA/f'training_dataset_1h_{split}_with_keys.parquet'); dynamic(d,adj,ev).to_parquet(DATA/f'training_dataset_1h_{split}.parquet',index=False)
    (REPORTS/'stage11_feature_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    (REPORTS/'stage11_graph_build_report.json').write_text(json.dumps({'adjacency_path':str((DATA/'segment_adjacency.parquet').resolve()),'graph_source':str((ROOT/'astana_roads.graphml').resolve()),'neighbor_definition':'all segments at graph distance <=2 via shared endpoints','betweenness_method':'networkx sampled betweenness_centrality k=512, seed=20260711','spatial_features':cfg['spatial_features'],'parallel_split_audit':source['row_level_reconstruction_audit']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
