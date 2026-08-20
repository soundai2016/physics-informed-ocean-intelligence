from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BIB = ROOT / "references.bib"
RECODING_FILES = (
    DATA / "human_recoding_a02_configurations.csv",
    DATA / "human_recoding_a03_configurations.csv",
)
CODING_FIELDS = ("O", "C", "E", "A", "N", "Physics insertion", "Claim boundary")
INSERTIONS = {
    "none", "objective_or_projection", "inference", "representation", "solver_coupling", "action_constraint"
}
COMPUTATIONAL_INSERTIONS = {"objective_or_projection", "inference", "representation", "solver_coupling"}
BOUNDARY_CODES = {
    "B_OFFLINE_TO_OPERATIONAL", "B_SIMULATION_TO_FIELD",
    "B_CONTROLLED_TO_OPEN_OCEAN", "B_DEVELOPER_TO_INDEPENDENT",
}


def read_csv(path: Path, required: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if required and not required.issubset(fields):
            raise ValueError(f"{path.name}: missing columns {sorted(required - fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return rows


def tokens(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def parse_bibtex(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    position = 0
    while True:
        match = re.search(r"@\w+\s*\{\s*([^,\s]+)\s*,", text[position:], re.I)
        if not match:
            return entries
        start = position + match.start()
        key = match.group(1)
        brace = text.find("{", start)
        depth = 0
        end = None
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None or key in entries:
            raise ValueError(f"invalid/duplicate BibTeX entry: {key}")
        entries[key] = text[start:end]
        position = end


def bib_field(entry: str, field: str) -> str:
    match = re.search(
        rf"\b{re.escape(field)}\s*=\s*(?:\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}|\"([^\"]*)\")",
        entry,
        re.I | re.S,
    )
    return (match.group(1) or match.group(2) or "").strip() if match else ""


def normalize_title(value: str) -> str:
    value = re.sub(r"[{}\\]", "", value).replace('"u', "u").replace("~", " ")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def validate_configurations() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    studies = read_csv(DATA / "study_corpus.csv", {
        "study_id", "citation_key", "title", "year", "persistent_id",
        "configuration_count", "configuration_status",
    })
    configs = read_csv(DATA / "study_configurations.csv", {
        "configuration_id", "study_id", "citation_key", "O_levels", "C_levels",
        "E_levels", "A_levels", "N_levels", "physics_insertion",
        "physics_informed_status", "source_section", "source_page", "source_evidence",
        "anchor_url", "anchor_basis", "locator_status", "verified_on",
        "supported_claim", "claim_boundary_code", "unsupported_claim",
        "extraction_decision", "extraction_basis",
    })
    if len({row["study_id"] for row in studies}) != len(studies):
        raise ValueError("duplicate study identifiers")
    if len({row["citation_key"] for row in studies}) != len(studies):
        raise ValueError("duplicate study citation keys")
    if len({row["configuration_id"] for row in configs}) != len(configs):
        raise ValueError("duplicate configuration identifiers")
    study_by_id = {row["study_id"]: row for row in studies}
    counts = Counter(row["study_id"] for row in configs)
    for study in studies:
        if int(study["configuration_count"]) != counts.get(study["study_id"], 0):
            raise ValueError(f"stale configuration count: {study['citation_key']}")
    for row in configs:
        if row["study_id"] not in study_by_id:
            raise ValueError(f"orphan configuration: {row['configuration_id']}")
        if row["citation_key"] != study_by_id[row["study_id"]]["citation_key"]:
            raise ValueError(f"configuration citation mismatch: {row['configuration_id']}")
        for axis in "OCEAN":
            values = tokens(row[f"{axis}_levels"])
            if not values or any(not re.fullmatch(fr"{axis}[0-3]", value) for value in values):
                raise ValueError(f"invalid {axis}: {row['configuration_id']}")
            if axis in "OEA" and len(values) != 1:
                raise ValueError(f"non-single {axis}: {row['configuration_id']}")
            if axis in "CN" and f"{axis}0" in values and len(values) > 1:
                raise ValueError(f"mixed {axis}0: {row['configuration_id']}")
        insertions = set(tokens(row["physics_insertion"]))
        if not insertions or not insertions.issubset(INSERTIONS) or ("none" in insertions and len(insertions) != 1):
            raise ValueError(f"invalid physics insertion: {row['configuration_id']}")
        established = bool(insertions & COMPUTATIONAL_INSERTIONS)
        if established != (row["physics_informed_status"] == "established"):
            raise ValueError(f"physics status mismatch: {row['configuration_id']}")
        c_values = set(tokens(row["C_levels"]))
        if "C1" in c_values and not insertions & {"objective_or_projection", "inference"}:
            raise ValueError(f"C1 crosswalk mismatch: {row['configuration_id']}")
        if "C2" in c_values and "representation" not in insertions:
            raise ValueError(f"C2 crosswalk mismatch: {row['configuration_id']}")
        if "C3" in c_values and "solver_coupling" not in insertions:
            raise ValueError(f"C3 crosswalk mismatch: {row['configuration_id']}")
        if row["locator_status"] not in {"full_text_page_anchor", "full_text_section_anchor"}:
            raise ValueError(f"non-full-text locator: {row['configuration_id']}")
        if "abstract" in row["source_section"].casefold() or row["source_page"].upper() == "NA":
            raise ValueError(f"invalid locator: {row['configuration_id']}")
        for field in (
            "source_section", "source_page", "source_evidence", "anchor_url", "anchor_basis",
            "verified_on", "supported_claim", "unsupported_claim", "extraction_basis",
        ):
            if not row[field].strip():
                raise ValueError(f"missing {field}: {row['configuration_id']}")
        if row["claim_boundary_code"] not in BOUNDARY_CODES:
            raise ValueError(f"invalid claim boundary: {row['configuration_id']}")
        if row["extraction_decision"] not in {"single", "split"}:
            raise ValueError(f"invalid extraction decision: {row['configuration_id']}")
    return studies, configs


def validate_anchor_ledger(configs: list[dict[str, str]]) -> None:
    ledger = read_csv(DATA / "configuration_evidence_ledger.csv", {
        "configuration_id", "source_section", "source_page", "anchor_url",
        "locator_status", "claim_boundary_code",
    })
    left = {row["configuration_id"]: row for row in configs}
    right = {row["configuration_id"]: row for row in ledger}
    if set(left) != set(right):
        raise ValueError("anchor ledger identifiers do not match configurations")
    for configuration_id in left:
        for field in ("source_section", "source_page", "anchor_url", "locator_status", "claim_boundary_code"):
            if left[configuration_id][field] != right[configuration_id][field]:
                raise ValueError(f"anchor ledger drift: {configuration_id} {field}")


def validate_recoding_file(
    path: Path,
    studies: list[dict[str, str]],
    expected_lookup: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows = read_csv(path, {"Study ID", "title", "doi", "Slot", *CODING_FIELDS})
    study_ids = {row["study_id"] for row in studies}
    keys = [(row["Study ID"], row["Slot"]) for row in rows]
    expected_keys = {(study_id, slot) for study_id in study_ids for slot in ("1", "2")}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError(f"{path.name}: expected exactly two fixed slots per retained study")
    counts: Counter[str] = Counter()
    for row in rows:
        title, doi = expected_lookup[row["Study ID"]]
        if row["title"].strip() != title or row["doi"].strip() != doi:
            raise ValueError(f"{path.name}: bibliographic mismatch at {row['Study ID']}")
        states = [bool(row[field].strip()) for field in CODING_FIELDS]
        if any(states) and not all(states):
            raise ValueError(f"{path.name}: partially filled row at {row['Study ID']} slot {row['Slot']}")
        if not all(states):
            continue
        counts[row["Study ID"]] += 1
        for field, prefix in (("O", "O"), ("C", "C"), ("E", "E"), ("A", "A"), ("N", "N")):
            values = set(tokens(row[field]))
            allowed = {f"{prefix}{i}" for i in range(4)}
            if not values or not values.issubset(allowed):
                raise ValueError(f"{path.name}: invalid {field} at {row['Study ID']} slot {row['Slot']}")
            if field in {"O", "E", "A"} and len(values) != 1:
                raise ValueError(f"{path.name}: {field} must be single-valued")
            if field in {"C", "N"} and f"{prefix}0" in values and len(values) != 1:
                raise ValueError(f"{path.name}: baseline {prefix}0 must be exclusive")
        insertions = set(tokens(row["Physics insertion"]))
        if not insertions or not insertions.issubset(INSERTIONS) or ("none" in insertions and len(insertions) != 1):
            raise ValueError(f"{path.name}: invalid physics insertion at {row['Study ID']} slot {row['Slot']}")
        if row["Claim boundary"] not in BOUNDARY_CODES:
            raise ValueError(f"{path.name}: invalid claim boundary at {row['Study ID']} slot {row['Slot']}")
    return rows, counts


def validate_reliability(studies: list[dict[str, str]]) -> None:
    entries = parse_bibtex(BIB.read_text(encoding="utf-8"))
    expected_lookup = {}
    for study in studies:
        doi = bib_field(entries[study["citation_key"]], "doi").strip()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
        expected_lookup[study["study_id"]] = (study["title"].strip(), doi)
    recodings = [validate_recoding_file(path, studies, expected_lookup) for path in RECODING_FILES]
    study = read_csv(RESULTS / "human_recoding_study_reliability.csv", {
        "dimension", "n_studies", "n_observations", "exact_agreement", "cohen_kappa"
    })
    if {row["dimension"] for row in study} != {"configuration_bearing", "configuration_count"}:
        raise ValueError("human-recoding study reliability dimension set mismatch")
    if any(int(row["n_studies"]) != len(studies) or int(row["n_observations"]) != len(studies) for row in study):
        raise ValueError("study extraction reliability does not cover the retained corpus")
    config = read_csv(RESULTS / "human_recoding_configuration_reliability.csv", {
        "dimension", "n_studies", "n_observations", "exact_agreement", "cohen_kappa", "eligibility_rule"
    })
    if {row["dimension"] for row in config} != {"O", "C", "E", "A", "N", "physics_insertion", "claim_boundary"}:
        raise ValueError("human-recoding configuration reliability dimension set mismatch")
    sizes = {(int(row["n_studies"]), int(row["n_observations"])) for row in config}
    if len(sizes) != 1:
        raise ValueError("configuration reliability rows use inconsistent analysis sets")
    maps = []
    for rows, _ in recodings:
        mapped: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if row["O"].strip():
                mapped[row["Study ID"]].append(row["Slot"])
        maps.append({study_id: sorted(slots) for study_id, slots in mapped.items()})
    eligible = [
        study_id for study_id in sorted(set(maps[0]) | set(maps[1]))
        if len(maps[0].get(study_id, [])) == len(maps[1].get(study_id, [])) > 0
        and maps[0][study_id] == maps[1][study_id]
    ]
    expected_configs = sum(len(maps[0][study_id]) for study_id in eligible)
    if sizes != {(len(eligible), expected_configs)}:
        raise ValueError("configuration reliability analysis set drift")
    facets = read_csv(RESULTS / "human_recoding_facet_reliability.csv", {
        "dimension", "n_studies", "n_observations", "dimension_group", "facet", "a02_positive", "a03_positive"
    })
    if any((int(row["n_studies"]), int(row["n_observations"])) != (len(eligible), expected_configs) for row in facets):
        raise ValueError("facet reliability analysis set drift")


def validate_reviews_and_bibliography(studies: list[dict[str, str]]) -> None:
    read_csv(DATA / "closest_review_scope.csv", {"citation_key", "decision", "rationale"})
    comparison = read_csv(DATA / "closest_review_comparison.csv", {
        "review_key", "usual_comparison_unit", "primary_depth",
        "interfaces_emphasized", "evidence_granularity", "full_text_basis",
    })
    if not any(row["review_key"] == "this_survey" for row in comparison):
        raise ValueError("closest-review comparison is missing this_survey")
    entries = parse_bibtex(BIB.read_text(encoding="utf-8"))
    missing = {row["citation_key"] for row in studies} - set(entries)
    if missing:
        raise ValueError(f"study records missing from BibTeX: {sorted(missing)}")
    for study in studies:
        entry = entries[study["citation_key"]]
        if normalize_title(bib_field(entry, "title")) != normalize_title(study["title"]):
            raise ValueError(f"BibTeX title mismatch: {study['citation_key']}")
        if bib_field(entry, "year") != study["year"]:
            raise ValueError(f"BibTeX year mismatch: {study['citation_key']}")
    for row in comparison:
        if row["review_key"] != "this_survey" and row["review_key"] not in entries:
            raise ValueError(f"review missing from BibTeX: {row['review_key']}")


def validate_schema() -> None:
    schema = json.loads((DATA / "taxonomy_schema.json").read_text(encoding="utf-8"))
    if not schema.get("$defs") or not schema.get("properties"):
        raise ValueError("taxonomy schema is malformed")


def validate_results() -> None:
    rows = read_csv(RESULTS / "manuscript_numbers.csv", {"macro", "value", "format"})
    macros = [row["macro"] for row in rows]
    if len(macros) != len(set(macros)):
        raise ValueError("duplicate manuscript macro")


def validate_figures() -> None:
    stems = {"ocean_taxonomy", "evidence_closure", "corpus_profile", "configuration_heatmap", "evidence_structure"}
    signatures = {"pdf": b"%PDF", "png": b"\x89PNG\r\n\x1a\n"}
    for stem in stems:
        for extension, signature in signatures.items():
            path = FIGURES / f"{stem}.{extension}"
            if not path.exists() or path.stat().st_size < 1024:
                raise ValueError(f"missing or truncated figure: {path.name}")
            with path.open("rb") as handle:
                if handle.read(len(signature)) != signature:
                    raise ValueError(f"invalid figure signature: {path.name}")


def main() -> None:
    studies, configs = validate_configurations()
    validate_anchor_ledger(configs)
    validate_reliability(studies)
    validate_reviews_and_bibliography(studies)
    validate_schema()
    validate_results()
    validate_figures()
    print(f"VALIDATION PASS: {len(studies)} studies; {len(configs)} configurations; recoding, bibliography, results and figures verified")


if __name__ == "__main__":
    main()
