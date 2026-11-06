"""Frozen ChemBERTa representations, pinned snapshots, no remote Python code.

Weights remain in work/. Token/feature caches contain no fitted target model.
Canonical random-SMILES seeds depend on structure, never on row or target.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
import torch
from transformers import AutoModel,AutoTokenizer

import train_v5 as T

MODELS={
 'mtr':('DeepChem/ChemBERTa-77M-MTR','66b895cab8adebea0cb59a8effa66b2020f204ca'),
 'mlm':('DeepChem/ChemBERTa-77M-MLM','ed8a5374f2024ec8da53760af91a33fb8f6a15ff')}
FILES=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
       'added_tokens.json','vocab.json','merges.txt','pytorch_model.bin']


def snapshot(model_id,revision,cache):
    dest=Path(cache)/model_id.replace('/','__')/revision
    dest.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name in FILES:
        path=dest/name
        if not path.exists():
            url=f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
            response=requests.get(url,stream=True,timeout=(10,90))
            response.raise_for_status()
            temp=path.with_suffix(path.suffix+'.part')
            with open(temp,'wb') as f:
                for chunk in response.iter_content(1024*1024):
                    if chunk:f.write(chunk)
            temp.replace(path)
        manifest[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    (dest/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    return dest,manifest


def views(smiles):
    parent=Chem.MolToSmiles(T.std(smiles),isomericSmiles=True)
    seed=int(hashlib.sha256(parent.encode()).hexdigest()[:8],16)%(2**31-1)
    random=list(Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(parent),3,randomSeed=seed))
    return {'raw':[smiles],'canonical':[parent],'augmented':[parent]+random}


def encode_sequences(tok,model,strings,batch_size=1,progress=None):
    means=[];clss=[];truncated=0
    for start in range(0,len(strings),batch_size):
        batch=strings[start:start+batch_size]
        truncated+=sum(len(tok.encode(s,add_special_tokens=True))>512 for s in batch)
        data=tok(batch,padding=True,truncation=True,max_length=512,return_tensors='pt',return_special_tokens_mask=True)
        special=data.pop('special_tokens_mask')
        with torch.inference_mode():
            z=model(**data).last_hidden_state
            mask=(data['attention_mask']*(1-special)).unsqueeze(-1)
            pooled=(z*mask).sum(1)/mask.sum(1).clamp(min=1)
        means.append(pooled.numpy());clss.append(z[:,0,:].numpy())
        if progress and start%(batch_size*20)==0:print('EMBED',progress,start,'/',len(strings),flush=True)
    return np.vstack(means),np.vstack(clss),int(truncated)


def encode_smiles(smiles,encoder,view,cache,batch_size=1,threads=3):
    """Same frozen feature operation used for research and portable inference."""
    torch.set_num_threads(threads);torch.manual_seed(20261107)
    source=list(smiles)
    per_row=[[s] if view=='raw' else views(s)[view] for s in source]
    strings=sorted({s for row in per_row for s in row})
    model_id,revision=MODELS[encoder]
    path,_=snapshot(model_id,revision,cache)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model,info=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,output_loading_info=True,add_pooling_layer=False)
    if any(not k.startswith('pooler.') for k in info.get('missing_keys',[])):
        raise RuntimeError('An encoder weight is missing')
    model.eval()
    means,clss,truncated=encode_sequences(tok,model,strings,batch_size)
    if truncated:raise ValueError('SMILES exceeds frozen encoder token limit; do not silently truncate inference')
    mapping={s:i for i,s in enumerate(strings)}
    result=np.vstack([np.r_[means[[mapping[s] for s in row]].mean(0),clss[[mapping[s] for s in row]].mean(0)] for row in per_row])
    if not np.isfinite(result).all():raise ValueError('Nonfinite frozen features')
    return result.astype(float)


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    p.add_argument('--out',default='results/audit_20261107');p.add_argument('--cache',default='work/chemberta_snapshots')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/a.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(20261107)
    tr=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES']);te=pd.read_csv(root/'data/raw/test.csv',usecols=['SMILES'])
    all_smiles=list(tr.SMILES)+list(te.SMILES)
    variants=[views(s) for s in all_smiles]
    strings=sorted({s for d in variants for values in d.values() for s in values})
    lookup={s:i for i,s in enumerate(strings)}
    for name in a.models:
        model_id,revision=MODELS[name]
        path,download_manifest=snapshot(model_id,revision,root/a.cache)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,output_loading_info=True,add_pooling_layer=False)
        unexpected_missing=[k for k in loading.get('missing_keys',[]) if not k.startswith('pooler.')]
        if unexpected_missing:raise RuntimeError('Uninitialized encoder weights: '+str(unexpected_missing))
        model.eval()
        means,clss,truncated=encode_sequences(tok,model,strings,a.batch_size,name)
        result={}
        for view in ['raw','canonical','augmented']:
            for pool,matrix in [('mean',means),('cls',clss)]:
                result[f'{view}_{pool}']=np.vstack([matrix[[lookup[s] for s in d[view]]].mean(0) for d in variants])
        if not all(np.isfinite(x).all() for x in result.values()):raise ValueError('Nonfinite encoder features')
        np.savez_compressed(out/f'embeddings_{name}.npz',**result)
        report={'model_id':model_id,'revision':revision,'torch':torch.__version__,
                'rows':len(all_smiles),'train_rows':len(tr),'hidden_size':means.shape[1],
                'input_smiles_sha256':hashlib.sha256(('\n'.join(all_smiles)).encode()).hexdigest(),
                'random_smiles_per_parent':3,'pooler_used':False,'encoder_frozen':True,
                'batch_size':a.batch_size,'torch_threads':a.threads,
                'unique_sequences':len(strings),'truncated_sequences':int(truncated),
                'weights_and_tokenizer':download_manifest,'loading_info':loading,
                'note':'Frozen encoder only; no task labels, no assay metadata; checkpoint weights not redistributed.'}
        (out/f'embeddings_{name}.json').write_text(json.dumps(report,indent=2))
        print('EMBED_DONE',name,{k:list(v.shape) for k,v in result.items()},flush=True)
        del model


if __name__=='__main__':main()
