# PubChem pilot

Official ESearch found53 non-ChEMBL AIDs.13 dose-response AIDs and explicit VACV rows of mixed panel602175 were downloaded; SID/CSV completeness was checked.390 endpoint rows,98 structures and31 scaffolds are new to the prior project. There are208 exact positive numeric values, including199 AC50 and9 TC50, and no approved SI.9 antiviral IC50 values are censored.

Reporter AC50, cytotoxicity AC50 and TC50 stay separate. Active/Inactive do not become organizer code10000. No test-structure labels remain. DOI is blank unless linked to a primary measurement publication unambiguously.

Access:https://pubchem.ncbi.nlm.nih.gov/rest/pug/ ; https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ . Terms:https://pubchem.ncbi.nlm.nih.gov/docs/downloads and depositor-specific restrictions; no universal CC license inferred.

Run:python src/collect_pubchem_research.py --cache work/external_source_cache/pubchem . Rate2.5 requests/sec, retry, cached response SHA/date, search pagination, and per-assay SID counts are implemented. Large primary screen not downloaded. Full API description/protocol responses stay in work cache; only numeric tables and non-operational metadata are exported.
