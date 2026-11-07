# Open-data snapshot for SI regression

The collection contains separately identified physical SI and auxiliary endpoints.

- pmc/training_si.csv:12 author-paired CC50/IC50 SI,12 structures, HeLa; CC-BY4.0.
- chembl/endpoint_bank.csv:2160 exact numerical auxiliary endpoint rows; ChEMBL37, CC-BY-SA3.0. These were already in the earlier source archive.
- pubchem/auxiliary_endpoints.csv:390 rows;208 exact positive numerical endpoints, the remainder includes censored records.98 structures total. PubChem/depositor terms apply.
- training_bank.csv:657 evaluated rows =53 retained legacy physical+12 PMC physical+384 ChEMBL auxiliary+208 PubChem auxiliary. Units:SI is dimensionless, other outputs use micromolar. All numeric training rows have exact relation.

No external test-structure numerical labels enter this bank. Task names preserve endpoint/source context; auxiliary measurements do not become SI or code10000. Missing values are not zero labels. The final model rejected new-source additions and uses only the53 retained legacy physical records.

The ZIP additionally contains inclusion/exclusion audits, selected numerical table evidence, allowed raw API/numeric-table responses, and SHA256SUMS. Full article XML and PubChem description/protocol text remain in the private work cache; their URLs and hashes remain in manifests. Source licenses apply to their separate components; no universal license is asserted for this collection. Pretrained encoder weights are absent.

See docs/external_research_20261107.md and results/external_20261107/source_registry.csv for counts, restrictions, source effects and reproduction commands.
