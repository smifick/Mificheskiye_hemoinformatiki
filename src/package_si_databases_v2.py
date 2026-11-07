"""Package published SI labels separately from review-only endpoint records.

No organizer concentration, missingness or ratio is read. External SI targets
come from the stored AUTHOR-REPORTED SI, not newly divided concentrations.
Uploaded competition files are used only locally to audit labels/overlaps.
"""
import argparse,hashlib,io,json,platform,subprocess,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE_REF='f6dab86fcc4c3b4e5f77eadc89598e23ec468816'
NIOCH_REF='29c21fa176f4c6d7470c4a2e9914cc40cbd20a51'
NIOCH_SOURCE='results/si_tail_evidence/external_audit/research_training_bank.csv'
READY_FIELDS=['SMILES','strain','Culture','SI']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot_csv(root,path,columns,ref=SOURCE_REF):
    raw=subprocess.check_output(['git','show',ref+':'+path],cwd=root)
    return pd.read_csv(io.BytesIO(raw),usecols=columns)

def keys(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid public SMILES')
    mol=rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol=rdMolStandardize.Uncharger().uncharge(mol)
    enum=rdMolStandardize.TautomerEnumerator();enum.SetMaxTautomers(100);enum.SetMaxTransforms(100)
    parent=Chem.MolToSmiles(enum.Canonicalize(mol),isomericSmiles=True)
    sc=MurckoScaffold.GetScaffoldForMol(mol)
    scaffold=Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc),isomericSmiles=False) if sc.GetNumAtoms() else 'ACYCLIC'
    return parent,scaffold

def competition(root,archive):
    old=pd.read_csv(root/'data/raw/train.csv',usecols=['SMILES','SI VACV','strain','Culture'],keep_default_na=False)
    oldtest=pd.read_csv(root/'data/raw/test.csv',usecols=['index','SMILES','strain','Culture'],keep_default_na=False)
    with zipfile.ZipFile(archive) as z:
        train_name=next(n for n in z.namelist() if Path(n).name=='train_v2.csv')
        test_name=next(n for n in z.namelist() if Path(n).name=='test.csv')
        train=pd.read_csv(z.open(train_name),usecols=['SMILES','SI VACV','strain','Culture'],keep_default_na=False)
        test=pd.read_csv(z.open(test_name),usecols=['index','SMILES','strain','Culture'],keep_default_na=False)
    assert train.SMILES.tolist()==old.SMILES.tolist()
    pd.testing.assert_frame_equal(test,oldtest)
    before=old['SI VACV'].to_numpy(float);after=train['SI VACV'].to_numpy(float);different=before!=after
    audit=dict(train_v2_verified=True,archive_sha256=sha(archive),train_rows=len(train),test_rows=len(test),
        identical_train_SMILES=True,test_allowed_fields_unchanged=True,changed_labels=int(different.sum()),
        old_10000_to_zero=int(((before==10000)&(after==0)).sum()),zero_labels=int((after==0).sum()),
        labels_equal10000=int((after==10000).sum()),max_SI=float(after.max()),
        unchanged_other_labels=bool(np.all((before[different]==10000)&(after[different]==0))),
        user_reported_submission='submission_filtered237.csv',user_reported_MSE=1228652.39224,
        score_verification='user supplied; not independently queried from Kaggle',
        concentrations_loaded=False,ratios_computed=False,uploaded_raw_files_published=False)
    return train,test,audit

def make_public_labels(root):
    columns=['SMILES','SI_reported','doi','cell_line','compound_id','publication_url','validation_status']
    legacy=snapshot_csv(root,'data/external_curated/vacv_si.csv',columns)
    legacy['strain']='';legacy['source_dataset']='reviewed_legacy56';legacy['license']='ChEMBL CC-BY-SA-3.0; primary publication attribution'
    # Pin an already published, manually reviewed additional source snapshot.
    raw=subprocess.check_output(['git','show',NIOCH_REF+':'+NIOCH_SOURCE],cwd=root)
    new=pd.read_csv(io.BytesIO(raw),usecols=columns+['origin','accepted_for_research','structure_confirmation'])
    new=new[new.origin=='new_complete_publication'].copy();assert len(new)==14
    assert new.accepted_for_research.astype(str).str.lower().eq('true').all()
    assert new.structure_confirmation.str.contains('manual graph attribution').all()
    new['validation_status']='reviewed_author_table_and_manually_attributed_structure'
    new['strain']='';new['source_dataset']='nioch_vector2023_reviewed14';new['license']='CC-BY-4.0; article table and manually attributed structures'
    pmc=snapshot_csv(root,'data/external_research/pmc/training_si.csv',[
        'SMILES','SI_reported','doi','cell_line','compound_id','strain','source_url','validation_status','SI_relation'])
    assert pmc.SI_relation.eq('=').all()
    pmc['publication_url']=pmc.source_url;pmc['source_dataset']='PMC10995631_reviewed12'
    pmc['license']='CC-BY-4.0 article; PubChem structure terms'
    common=columns+['strain','source_dataset','license']
    data=pd.concat([legacy[common],new[common],pmc[common]],ignore_index=True)
    data=data.rename(columns={'SI_reported':'SI','cell_line':'Culture','publication_url':'source_url'})
    data['SI']=pd.to_numeric(data.SI,errors='raise')
    assert len(data)==82 and np.isfinite(data.SI).all() and (data.SI>0).all()
    data['organism']='Vaccinia virus';data['relation']='=';data['target_origin']='author_reported_SI'
    data['strain_status']=np.where(data.strain.eq(''),'not recorded in source snapshot; do not impute a named strain','reported')
    data['source_record_id']=[f'published_SI_{i:03d}' for i in range(1,len(data)+1)]
    return data

def package(root,out,archive):
    out.mkdir(parents=True,exist_ok=False)
    train,test,audit=competition(root,archive);(out/'train_v2_audit.json').write_text(json.dumps(audit,indent=2))
    data=make_public_labels(root)
    identity=[keys(s) for s in data.SMILES];data['structure_key']=[x[0] for x in identity];data['scaffold_group']=[x[1] for x in identity]
    train_keys={keys(s)[0] for s in train.SMILES};test_keys={keys(s)[0] for s in test.SMILES}
    data['train_structure_overlap']=data.structure_key.isin(train_keys);data['test_structure_overlap']=data.structure_key.isin(test_keys)
    data.to_csv(out/'external_reported_si_all.csv',index=False)
    no_test=data[~data.test_structure_overlap];no_overlap=data[~data.test_structure_overlap&~data.train_structure_overlap]
    no_test.to_csv(out/'external_si_no_test_overlap.csv',index=False)
    no_overlap.to_csv(out/'external_si_new_structures.csv',index=False)
    # Deliberately omit provenance/overlap/similarity fields from the model input.
    no_overlap[READY_FIELDS].to_csv(out/'augmentation_44_smiles_context_si.csv',index=False)
    nioch=data[data.doi.isin(['10.1039/d0md00108b','10.3390/v15010029'])]
    nioch.to_csv(out/'nioch_vector_reported_si.csv',index=False)
    for name,source in [('legacy56','reviewed_legacy56'),('nioch2023_14','nioch_vector2023_reviewed14'),('pmc2024_12','PMC10995631_reviewed12')]:
        data[data.source_dataset==source].to_csv(out/(name+'.csv'),index=False)
    # These endpoint records contain NO ready SI labels and are review-only.
    pubcols=['SMILES','InChIKey','pubchem_aid','pubchem_cid','endpoint','value_uM','relation','activity_outcome',
             'phenotype','channel','cell_line','strain','doi','source_url','license_url','validation_status']
    pub=snapshot_csv(root,'data/external_research/pubchem/auxiliary_endpoints.csv',pubcols)
    pub['readiness']='review only; not a SI label; no mapping Inactive->0 or 10000'
    pub.to_csv(out/'pubchem_endpoints_review_only.csv',index=False)
    chcols=['SMILES','InChIKey','endpoint','value_uM','relation','cell','strain','doi','assay_id','activity_id','source_url','license','source_release','validation_status']
    chembl=snapshot_csv(root,'data/external_research/chembl/endpoint_bank.csv',chcols)
    chembl['readiness']='review only; separate endpoints are not SI or model input features'
    chembl.to_csv(out/'chembl_endpoints_review_only.csv',index=False)
    anchor=int(np.argmax(train['SI VACV'].to_numpy(float)));query=test[test['index']==233].iloc[0]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    def fp(s):return gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(keys(s)[0])).astype(float)
    a=fp(train.SMILES.iloc[anchor]);q=fp(query.SMILES)
    def sim(a,b):return float(a@b/max(a.sum()+b.sum()-a@b,1.))
    related=no_test.copy();related['similarity_to_test_index233']=[sim(q,fp(s)) for s in related.SMILES]
    related['similarity_to_high_SI_training_anchor']=[sim(a,fp(s)) for s in related.SMILES]
    related.sort_values('similarity_to_test_index233',ascending=False).to_csv(out/'analogue_similarity_review.csv',index=False)
    (out/'query_identity.json').write_text(json.dumps(dict(test_index=233,test_CSV_line=235,training_anchor_source_row=anchor+2,
        training_anchor_SI=float(train['SI VACV'].iloc[anchor]),similarity=sim(a,q),fingerprint='Morgan r2 2048 chirality on standardized tautomer parent',
        test_target_read=False,test_prediction_created=False,interpretation='Similarity only; never assign training-anchor SI to a test molecule'),indent=2))
    summary=dict(reported_SI_rows=len(data),unique_structures=int(data.structure_key.nunique()),publications=int(data.doi.nunique()),
        no_test_overlap_rows=len(no_test),new_structure_rows=len(no_overlap),new_unique_structures=int(no_overlap.structure_key.nunique()),
        nioch_vector_rows=len(nioch),pubchem_review_rows=len(pub),chembl_review_rows=len(chembl),
        labels_recomputed_from_concentrations=False,organizer_concentrations_loaded=False,external_inactive_as_SI_zero=False,
        current_train_label_version='train_v2' if audit.get('train_v2_verified') else 'unverified old snapshot',
        source_refs=dict(external_snapshot=SOURCE_REF,nioch_addition=NIOCH_REF))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    finish_package(root,out,data,no_test,no_overlap,pub,chembl,audit)
    print(json.dumps(summary,indent=2))

def finish_package(root,out,data,no_test,no_overlap,pub,chembl,audit):
    """Validate delivery files, retain provenance, and create a stable ZIP."""
    sources=[
        dict(doi='10.1016/j.ejmech.2023.115601',source_dataset='reviewed_legacy56',
             source_url='https://doi.org/10.1016/j.ejmech.2023.115601',institution_url='',
             structure_provenance='ChEMBL structure attributed by compound and numerical table matching; not independently redrawn'),
        dict(doi='10.1039/d0md00108b',source_dataset='reviewed_legacy56',
             source_url='https://doi.org/10.1039/d0md00108b',institution_url='https://sciact.nioch.nsc.ru/public/article/6306/en',
             structure_provenance='ChEMBL structure attributed by compound and numerical table matching; not independently redrawn'),
        dict(doi='10.3390/v15010029',source_dataset='nioch_vector2023_reviewed14',
             source_url='https://doi.org/10.3390/v15010029',institution_url='https://sciact.nioch.nsc.ru/public/article/11549',
             structure_provenance='Previously reviewed manual graph attribution; table drawings checked; cidofovir from curated ChEMBL graph'),
        dict(doi='10.1099/jgv.0.001971',source_dataset='PMC10995631_reviewed12',
             source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10995631/',institution_url='',
             structure_provenance='Previously reviewed unique structure mapping of author table compounds'),
    ]
    pd.DataFrame(sources).to_csv(out/'source_registry.csv',index=False)
    groups=data.groupby(['doi','Culture'],dropna=False).agg(
        observations=('SI','size'),unique_structures=('structure_key','nunique'),
        minimum_reported_SI=('SI','min'),maximum_reported_SI=('SI','max')).reset_index()
    groups.to_csv(out/'source_summary.csv',index=False)
    ready=pd.read_csv(out/'augmentation_44_smiles_context_si.csv',keep_default_na=False)
    checks={
        'uploaded_train_v2_verified':audit['train_v2_verified'],
        'only_expected_81_label_changes':audit['old_10000_to_zero']==81 and audit['unchanged_other_labels'],
        'external_targets_are_stored_author_SI':data.target_origin.eq('author_reported_SI').all(),
        'external_targets_finite_positive':np.isfinite(data.SI).all() and data.SI.gt(0).all(),
        'external_relation_exact':data.relation.eq('=').all(),
        'source_record_ids_unique':data.source_record_id.is_unique,
        'all_source_validation_status_recorded':data.validation_status.fillna('').ne('').all(),
        'all_public_SMILES_parse':all(Chem.MolFromSmiles(s) is not None for s in data.SMILES),
        'ready_columns_exact_allowlist':ready.columns.tolist()==READY_FIELDS,
        'ready_values_and_row_order_match_provenance':ready.equals(no_overlap[READY_FIELDS].reset_index(drop=True).fillna('')),
        'no_test_overlap_in_augmentation':not no_overlap.test_structure_overlap.any(),
        'no_train_overlap_in_augmentation':not no_overlap.train_structure_overlap.any(),
        'no_test_overlap_in_review_subset':not no_test.test_structure_overlap.any(),
        'augmentation_structure_keys_unique':no_overlap.structure_key.is_unique,
        'no_SI_in_endpoint_review_files':'SI' not in pub.columns and 'SI' not in chembl.columns,
        'no_organizer_concentrations_loaded':not audit['concentrations_loaded'],
        'no_new_concentration_ratios':not audit['ratios_computed'],
    }
    checks={k:bool(v) for k,v in checks.items()}
    if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
    (out/'validation.json').write_text(json.dumps(dict(checks=checks,passed=len(checks),failed=0),indent=2))
    environment=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,rdkit=rdBase.rdkitVersion)
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    (out/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in environment.items() if k!='python')+'\n')
    (out/'README.md').write_text((root/'docs/si_database_package_v2.md').read_text())
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS','manifest.json','si_databases_v2.zip']]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    hashed=sorted(p for p in out.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','si_databases_v2.zip'])
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in hashed))
    with zipfile.ZipFile(out/'si_databases_v2.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(out.iterdir()):
            if not p.is_file() or p.name=='si_databases_v2.zip':continue
            info=zipfile.ZipInfo(p.name,date_time=(2026,11,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes(),compresslevel=9)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/si_databases_v2_20261107');p.add_argument('--dataset-zip',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];package(root,root/a.out,a.dataset_zip)
