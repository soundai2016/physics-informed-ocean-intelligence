# Physics-Informed Ocean Intelligence

Evidence tables and Python analysis for the manuscript.

## Data

- `study_corpus.csv`: retained studies.
- `configuration_specs.csv`: editable configuration coding.
- `study_configurations.csv`: generated configuration table.
- `configuration_evidence_ledger.csv`: source anchors and claim boundaries.
- `human_recoding_a02_configurations.csv`, `human_recoding_a03_configurations.csv`: independent two-slot recoding records.
- `taxonomy_decision_rules.md`, `taxonomy_schema.json`: coding rules and schema.

A02/A03 titles and DOI values are checked against the retained corpus and bibliography. Study-level configuration presence and counts are inferred from filled configuration slots; no separate study-level recoding file is required.

## Run

```bash
python3 -m pip install -r requirements.txt
make
```

`make` rebuilds tables, recoding statistics, figures, manuscript numbers, and validation checks.
