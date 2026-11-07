# ChEMBL: audited endpoint bank for the external-data study

Source: ChEMBL / EMBL-EBI, **ChEMBL_37**. Retrieval: 2026-11-07T00:26:32.703344+00:00.
Official API: https://www.ebi.ac.uk/chembl/api/data/ .
The API target census searches `organism__icontains=vaccinia`, retains ORGANISM
targets, and retrieves every endpoint type for those targets, with full pagination.
The frozen repository export additionally includes documented cytotoxicity and
ambiguous assay records, which remain explicitly identified.

The merged audit has **9606 activities**, including
**2735 technically valid declared
endpoints**. The unrestricted target census added 2753
activity IDs to the old export; **none supplied a new eligible exact IC50/EC50/CC50
or compatible direct SI**. Other endpoints such as MIC, ED50, percent inhibition
and concentration-dependent activity are retained in the rejection audit, never
renamed IC50 or assigned a Kaggle technical code.

`endpoint_bank.csv` contains **2160 rows** for separate auxiliary tasks.
All external labels for official test structures were removed before this export.
Rows overlapping official training structures are flagged: each fold must also
exclude its validation structures and scaffolds at training time.

| Separate task | Rows | Molecules | Scaffolds | New molecules vs train/test/curated | New scaffolds vs train/test/curated |
|---|---:|---:|---:|---:|---:|
| VACV_EC50 | 440 | 302 | 127 | 288 | 90 |
| VACV_IC50 | 263 | 238 | 114 | 191 | 68 |
| cytotoxicity_CC50 | 1453 | 728 | 196 | 672 | 136 |
| reported_CC50/EC50 | 4 | 4 | 1 | 4 | 0 |

The four reported `CC50/EC50` ratios have an explicit database assay definition;
they are not CC50/IC50 labels. `verified_si.csv` contains **0** newly
accepted CC50/IC50 labels. `candidate_si_pairs_review_only.csv` contains
329 unapproved candidate ratios; matching molecule/document IDs alone
does not establish comparable biological measurements. The source provides
numeric auxiliary information that was already archived; the earlier model used
most of this archive only to derive source presence. These are not newly discovered
experimental measurements and must not be counted as independent replicates.

Files preserve full canonical structure and stereo, InChIKey, standardized parent
and tautomer/scaffold keys, exact relation and units, cell/strain where provided,
DOI, molecule/assay/activity/document IDs, API URL, source release, quality status,
and inclusion/exclusion reasons. No laboratory procedures are collected or generated.

`archive_manifest.json` identifies the existing source ZIP and member SHA256.
`request_manifest.json` identifies each additional raw API response in the cache,
its retrieval/access dates and SHA256. API pagination count/ID checks, bounded
retry, at least 0.5 seconds between fresh requests and persistent cache support
resumption. The complete source archive already lives at
`data/chembl_package/vacv_chembl_data.zip`; new API responses are in the configured
cache and can be archived by response hash. `files_sha256.json` hashes this export.

From the repository root:

```bash
python src/collect_chembl_research.py \
  --cache work/external_source_cache/chembl/ChEMBL_37 \
  --out data/external_research/chembl
python -m unittest discover -s tests -p test_chembl_research.py -v
```

`--offline` audits only the frozen archive, explicitly marked as such. It does
not claim a fresh target/endpoint census. A changed ChEMBL release requires a
complete reviewed refresh rather than silently mixing releases.

## License and attribution

Adapted ChEMBL data in this directory are **CC BY-SA 3.0 Unported**:
https://creativecommons.org/licenses/by-sa/3.0/ . Attribute ChEMBL / EMBL-EBI,
release 37 and the linked original publications, and preserve the ShareAlike terms
when redistributing adapted ChEMBL tables. The license is confirmed by the official
FAQ: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions .
API access is also subject to EMBL-EBI terms. The scripts are project code; this
statement identifies the data license and does not relicense unrelated project files.
