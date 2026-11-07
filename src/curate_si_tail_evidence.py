"""Published table values and manually checked chemical graphs, no assay procedures.

Extract the complete VACV Table 1 of Mozhaitsev et al. (2023), including inactive
and bounded entries. Only exact positive ratios enter the research bank.
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

from si_tail import smiles_features
from train_model import standardize
from train_external_v3 import canonical


DOI = '10.3390/v15010029'
XML_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9865139/fullTextXML'


def number(text):
    match = re.match(r'^\s*([<>]=?)?\s*([0-9.,]+)(?:\s*±\s*([0-9.,]+))?', text)
    if not match:
        return None, None, None
    return float(match[2].replace(',', '')), (float(match[3].replace(',', '')) if match[3] else None), match[1] or '='


def table_structure(compound):
    # Checked against all individual compound drawings in the published Table 1.
    a = 'C12CC3CC(CC(C3)C1)C2'       # adamantan-1-yl
    b = 'C1C2CC3CC1CC(C2)C3'        # adamantan-2-yl
    if compound == 'bromantane':
        return Chem.MolToSmiles(Chem.MolFromSmiles('Brc1ccc(N'+b+')cc1'))
    cage = a if compound[-1] == 'a' else b
    aromatic = {'12': 'c1ccccc1', '13': 'c1ccc(F)cc1', '14': 'c1ccc(Cl)cc1',
                '15': 'c1ccc(Br)cc1', '16': 'c1ccccc1[N+](=O)[O-]',
                '17': 'c1cccc([N+](=O)[O-])c1', '18': 'c1ccc([N+](=O)[O-])cc1'}[compound[:2]]
    mol = Chem.MolFromSmiles('O=C(N'+cage+')'+aromatic)
    if mol is None:
        raise ValueError(f'Invalid manually attributed graph: {compound}')
    return Chem.MolToSmiles(mol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--cache', default='work/literature_evidence')
    parser.add_argument('--out', default='results/si_tail_evidence/external_audit')
    args = parser.parse_args()
    root, cache, out = Path(args.root), Path(args.cache), Path(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    path = cache/'PMC9865139.xml'
    if not path.exists():
        response = requests.get(XML_URL, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    xml = ET.fromstring(path.read_bytes())
    table = xml.find(".//table-wrap[@id='viruses-15-00029-t001']")
    rows = table.findall('.//tr')[1:]
    compounds = [f'{i}{letter}' for i in range(12,19) for letter in ['a','b']] + ['bromantane','cidofovir']
    if len(rows) != len(compounds):
        raise ValueError('Publication table schema changed')
    old = pd.read_csv(root/'data/external_curated/vacv_si.csv')
    cidofovir = old[old.compound_id.str.lower().eq('cidofovir')].SMILES.iloc[0]
    candidates = []
    for compound, row in zip(compounds, rows):
        values = [' '.join(''.join(cell.itertext()).split()) for cell in row]
        graphic=row[0].find('inline-graphic')
        image_name=graphic.get('{http://www.w3.org/1999/xlink}href') if graphic is not None else None
        expected_image=f'viruses-15-00029-i{compounds.index(compound)+1:03d}.jpg' if compound!='cidofovir' else None
        if image_name!=expected_image:
            raise ValueError('Publication image/compound ordering changed')
        cc, cc_sd, cc_relation = number(values[1])
        ic, ic_sd, ic_relation = number(values[2])
        reported, _, _ = number(values[3])
        smiles = cidofovir if compound == 'cidofovir' else table_structure(compound)
        exact = cc_relation == '=' and ic_relation == '=' and cc is not None and ic is not None and ic > 0
        candidates.append({'molecule_id': DOI+':'+compound, 'compound_id': compound, 'SMILES': smiles,
                           'doi': DOI, 'table': 'Table 1', 'cell_line': 'Vero', 'organism': 'Vaccinia virus',
                           'CC50_uM': cc, 'CC50_SD_uM': cc_sd, 'CC50_relation': cc_relation,
                           'cytotoxicity_original_name': 'TC50', 'IC50_uM': ic, 'IC50_SD_uM': ic_sd,
                           'IC50_relation': ic_relation, 'concentration_units': 'uM',
                           'SI': cc/ic if exact else None, 'SI_reported': reported,
                           'si_definition': 'TC50/IC50; ratio of reported means; cellular VACV endpoint',
                           'denominator_endpoint': 'IC50', 'cytotoxicity_endpoint': '50% cytotoxic concentration',
                           'relation_status': 'exact reported means' if exact else 'bound or inactive; no exact SI',
                           'structure_confirmation': 'manual graph attribution; all table drawings checked; cidofovir from previously curated ChEMBL graph',
                           'activity_id': None, 'molecule_chembl_id': None,
                           'activity_id_note': 'DOI not indexed in queried ChEMBL document endpoint; publication IDs used',
                           'publication_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9865139/',
                           'published_structure_image':image_name,
                           'denominator_relative_SD': ic_sd/ic if exact and ic_sd is not None else None,
                           'measurement_uncertainty': 'M ± SD, n=3 reported; covariance and original denominator quantification limits unknown',
                           'accepted_for_research': exact,
                           'inclusion_reason': 'complete publication table; both exact endpoints and assigned structure' if exact else 'excluded: bounds or NA not converted to exact labels'})
    new = pd.DataFrame(candidates)
    new.to_csv(out/'new_publication_candidates.csv', index=False)
    accepted = new[new.accepted_for_research].copy()
    accepted.to_csv(out/'new_publication_exact.csv', index=False)
    train = pd.read_csv(root/'data/raw/train.csv', usecols=['SMILES','SI VACV'])
    test = pd.read_csv(root/'data/raw/test.csv', usecols=['SMILES'])
    train_keys = {canonical(s,True)[0] for s in train.SMILES}
    test_keys = {canonical(s,True)[0] for s in test.SMILES}
    old_keys = {canonical(s,True)[0] for s in old.SMILES}
    raw_endpoints=pd.read_csv(root/'data/external/external_endpoints.csv',usecols=['SMILES'])
    previous_raw_keys,previous_raw_groups=set(),set()
    invalid_raw=0
    for smiles in raw_endpoints.SMILES.dropna().unique():
        try:
            previous_raw_keys.add(canonical(smiles,True)[0])
            previous_raw_groups.add(standardize(smiles)[1])
        except ValueError:
            invalid_raw+=1
    new['canonical_smiles']=[canonical(s,True)[0] for s in new.SMILES]
    new['structure_seen_in_previous_raw_endpoints']=new.canonical_smiles.isin(previous_raw_keys)
    new.to_csv(out/'new_publication_candidates.csv',index=False)
    all_external = pd.concat([old[old.cell_line.eq('Vero')].assign(origin='previous_curated_vero'),
                              accepted.assign(origin='new_complete_publication')], ignore_index=True)
    _, external_keys, external_groups = smiles_features(all_external.SMILES)
    _, _, train_groups = smiles_features(train.SMILES)
    all_external['canonical_smiles'] = external_keys
    all_external['scaffold_group'] = external_groups
    all_external['organizer_train_overlap'] = all_external.canonical_smiles.isin(train_keys)
    all_external['organizer_test_overlap'] = all_external.canonical_smiles.isin(test_keys)
    all_external['new_structure_vs_previous_curated'] = ~all_external.canonical_smiles.isin(old_keys)
    all_external['new_scaffold_vs_organizer_train'] = ~all_external.scaffold_group.isin(set(train_groups))
    all_external['eligible_research_after_test_exclusion'] = ~all_external.organizer_test_overlap
    all_external.to_csv(out/'external_bank_audit.csv', index=False)
    all_external[all_external.eligible_research_after_test_exclusion].to_csv(out/'research_training_bank.csv', index=False)
    new_rows = all_external[all_external.origin.eq('new_complete_publication')]
    summary = {'new_source_doi': DOI, 'xml_url': XML_URL, 'xml_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'all_table_rows': len(new), 'exact_rows': len(accepted), 'excluded_bounds_or_inactive': len(new)-len(accepted),
               'previous_curated_rows': len(old), 'previous_vero_rows': int(old.cell_line.eq('Vero').sum()),
               'previous_curated_unique_structures': len(old_keys),
               'new_exact_unique_structures': new_rows.canonical_smiles.nunique(),
               'new_structures_vs_previous_curated': new_rows[new_rows.new_structure_vs_previous_curated].canonical_smiles.nunique(),
               'new_structures_vs_train_and_previous': new_rows[~new_rows.organizer_train_overlap & new_rows.new_structure_vs_previous_curated].canonical_smiles.nunique(),
               'previous_raw_unique_canonical_structures':len(previous_raw_keys),
               'previous_raw_invalid_unique_SMILES':invalid_raw,
               'new_exact_structures_not_in_previous_raw_endpoints':new[new.accepted_for_research & ~new.structure_seen_in_previous_raw_endpoints].canonical_smiles.nunique(),
               'new_exact_structures_already_in_previous_raw_endpoints':new[new.accepted_for_research & new.structure_seen_in_previous_raw_endpoints].canonical_smiles.nunique(),
               'new_scaffolds_vs_organizer_train': new_rows[new_rows.new_scaffold_vs_organizer_train].scaffold_group.nunique(),
               'new_scaffolds_vs_train_and_all_previous_endpoints':new_rows[~new_rows.scaffold_group.isin(set(train_groups)|previous_raw_groups)].scaffold_group.nunique(),
               'new_test_structure_overlaps': int(new_rows.organizer_test_overlap.sum()),
               'new_SI_above_10000': int(accepted.SI.gt(10000).sum()),
               'new_SI_max': float(accepted.SI.max()),
               'bank_research_rows_after_test_exclusion': int(all_external.eligible_research_after_test_exclusion.sum()),
               'competition_external_data_allowed': True,
               'rule_evidence':'User-provided Dataset Description screenshot explicitly requires collecting open-source molecular data; organizer CC50_uM/IC50_uM_VACV forbidden as predictors',
               'use_in_competition_submission': False,
               'sampling_bias': 'Source was found by a tail-focused search; entire VACV table retained, but publication selection remains nonrandom.',
               'origin_of_organizer_extreme': 'Unresolved. The 2023 paper cites SI 31500 from a related 2022 ChemMedChem paper; the primary measurement/structure pair was not independently verified and does not establish provenance of organizer SI 36066.667.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
