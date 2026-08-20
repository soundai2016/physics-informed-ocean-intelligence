from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STUDY_PATH = DATA / "study_corpus.csv"
SPEC_PATH = DATA / "configuration_specs.csv"
OUT_PATH = DATA / "study_configurations.csv"

AXES = "OCEAN"
EVAL_SETTINGS = {"D", "S", "H", "F"}
LEVEL_TEXT = {
    "O0": "gridded or single-source observations",
    "O1": "multimodal observations",
    "O2": "irregular, delayed, or provenance-aware observations",
    "O3": "action-dependent acquisition",
    "C0": "no computation-altering physical insertion established",
    "C1": "constraint in the objective, projection, or inference step",
    "C2": "physics-structured representation",
    "C3": "online solver coupling",
    "E0": "state reconstruction or same-time inference",
    "E1": "forecasting or emulation",
    "E2": "action-conditioned evolution",
    "E3": "online adaptation with controls",
    "A0": "passive analysis",
    "A1": "decision support",
    "A2": "active sensing or control",
    "A3": "cooperative multi-agent operation",
    "N0": "aggregate or in-domain evaluation",
    "N1": "calibrated uncertainty",
    "N2": "shift, regime, fault, or recovery evaluation",
    "N3": "independent assurance envelope",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def tokens(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def scope_class(domain: str) -> str:
    if domain.startswith("underwater_") or domain in {
        "passive_sonar", "ocean_acoustics_and_sonar", "machine_learning_in_acoustics",
    }:
        return "underwater_acoustics"
    if domain == "marine_robotics_and_active_sensing":
        return "marine_robotics"
    if domain in {
        "learned_ocean_systems_and_benchmarks", "assimilation_and_state_estimation",
        "ocean_parameterization_and_hybrid_models", "ocean_extremes_and_forecasting",
        "compound_ocean_extremes", "sea_ice_forecasting", "sea_ice_subgrid_dynamics",
        "sea_ice_surrogate_modeling", "regional_ocean_forecasting", "ocean_extremes",
        "ocean_extremes_and_subgrid_dynamics", "regional_ocean_sciml",
        "sea_ice_data_assimilation", "sea_ice_data_assimilation_and_prediction",
        "sea_ice_hybrid_assimilation", "ocean_sound_speed_reconstruction",
    }:
        return "direct_ocean"
    if domain in {
        "extreme_weather_ood", "extreme_weather_and_regional_forecasting",
        "scientific_machine_learning", "ood_and_physical_generalization",
    }:
        return "transferable_adjacent"
    if domain == "world_models_and_control":
        return "transferable_control"
    if domain == "uncertainty_safety_and_validation":
        return "transferable_assurance"
    return "other"


def rationale(axis: str, value: str) -> str:
    levels = tokens(value)
    return "; ".join(f"{level}: {LEVEL_TEXT[level]}" for level in levels)


def validate_specs(specs: list[dict[str, str]], studies: dict[str, dict[str, str]]) -> None:
    required = {
        "citation_key", "evaluation_setting", "variant", "configuration_label", "claim_endpoint",
        "O_levels", "C_levels", "E_levels", "A_levels", "N_levels", "physics_insertion",
        "source_section", "source_page", "source_evidence",
        "anchor_url", "anchor_basis", "verification_status", "verified_on",
    }
    if not required.issubset(specs[0]):
        raise ValueError(f"configuration_specs.csv missing columns: {sorted(required - set(specs[0]))}")

    seen: set[tuple[str, str, str]] = set()
    allowed_insertion = {
        "none", "objective_or_projection", "inference", "representation",
        "solver_coupling", "action_constraint",
    }
    for row in specs:
        key = row["citation_key"]
        if key not in studies:
            raise ValueError(f"configuration spec references unknown study: {key}")
        identity = (key, row["evaluation_setting"], row["variant"])
        if identity in seen:
            raise ValueError(f"duplicate configuration spec: {identity}")
        seen.add(identity)
        if row["evaluation_setting"] not in EVAL_SETTINGS:
            raise ValueError(f"invalid evaluation setting for {key}")
        for axis in AXES:
            values = tokens(row[f"{axis}_levels"])
            if not values or any(v not in LEVEL_TEXT or not v.startswith(axis) for v in values):
                raise ValueError(f"invalid {axis} levels for {key}")
            if axis in "OEA" and len(values) != 1:
                raise ValueError(f"{axis} must be single-valued for {key}")
            if axis == "C" and "C0" in values and len(values) != 1:
                raise ValueError(f"C0 is an exclusive baseline for {key}")
            if axis == "N" and "N0" in values and len(values) != 1:
                raise ValueError(f"N0 is an exclusive baseline for {key}")
        insertion = tokens(row["physics_insertion"])
        if not insertion or any(v not in allowed_insertion for v in insertion):
            raise ValueError(f"invalid physics insertion for {key}")
        if "none" in insertion and len(insertion) != 1:
            raise ValueError(f"invalid mixed 'none' physics insertion for {key}")
        if not row["source_section"].strip() or not row["source_page"].strip() or not row["source_evidence"].strip():
            raise ValueError(f"missing source provenance for {key}")


def build_rows(studies: list[dict[str, str]], specs: list[dict[str, str]]) -> list[dict[str, str]]:
    study_map = {row["citation_key"]: row for row in studies}
    study_ids = {row["citation_key"]: f"S{i:03d}" for i, row in enumerate(studies, 1)}
    validate_specs(specs, study_map)

    counters: Counter[str] = Counter()
    source_counts = Counter(spec["citation_key"] for spec in specs)
    rows: list[dict[str, str]] = []
    for spec in specs:
        study = study_map[spec["citation_key"]]
        study_id = study_ids[spec["citation_key"]]
        counters[study_id] += 1
        configuration_id = f"{study_id}-C{counters[study_id]:02d}"
        insertion = tokens(spec["physics_insertion"])
        computational_insertions = {
            "objective_or_projection", "inference", "representation", "solver_coupling",
        }
        physics_status = (
            "established"
            if computational_insertions.intersection(insertion)
            else "not_established"
        )
        signature = "--".join(
            "/".join(tokens(spec[f"{axis}_levels"])) for axis in AXES
        ) + f";{spec['evaluation_setting']}"
        section = spec["source_section"].strip()
        page = spec["source_page"].strip()
        if spec["evaluation_setting"] == "D":
            boundary_code = "B_OFFLINE_TO_OPERATIONAL"
            unsupported = "Boundary: offline/hindcast."
        elif spec["evaluation_setting"] == "S":
            boundary_code = "B_SIMULATION_TO_FIELD"
            unsupported = "Boundary: simulation/replay."
        elif spec["evaluation_setting"] == "H":
            boundary_code = "B_CONTROLLED_TO_OPEN_OCEAN"
            unsupported = "Boundary: controlled-water or hardware."
        else:
            boundary_code = "B_DEVELOPER_TO_INDEPENDENT"
            unsupported = "Boundary: developer-run field evidence."
        extraction_decision = "split" if source_counts[spec["citation_key"]] > 1 else "single"
        extraction_basis = (
            "separate endpoint or setting"
            if extraction_decision == "split"
            else "single evaluated endpoint and setting"
        )

        row = {
            "configuration_id": configuration_id,
            "study_id": study_id,
            "citation_key": study["citation_key"],
            "year": study["year"],
            "title": study["title"],
            "domain": study["domain"],
            "evidence_scope": scope_class(study["domain"]),
            "publication_status": study["publication_status"],
            "persistent_id": study["persistent_id"],
            "configuration_label": spec["configuration_label"],
            "claim_endpoint": spec["claim_endpoint"],
            "evaluation_setting": spec["evaluation_setting"],
            "evidence_independence": "DEV",
            "O_levels": spec["O_levels"],
            "C_levels": spec["C_levels"],
            "E_levels": spec["E_levels"],
            "A_levels": spec["A_levels"],
            "N_levels": spec["N_levels"],
            "physics_insertion": spec["physics_insertion"],
            "physics_informed_status": physics_status,
            "ocean_signature": signature,
            "source_section": section,
            "source_page": page,
            "source_evidence": spec["source_evidence"].strip(),
            "anchor_url": spec["anchor_url"].strip(),
            "anchor_basis": spec["anchor_basis"].strip(),
            "verified_on": spec["verified_on"].strip(),
            "O_evidence_summary": rationale("O", spec["O_levels"]),
            "C_evidence_summary": rationale("C", spec["C_levels"]),
            "E_evidence_summary": rationale("E", spec["E_levels"]),
            "A_evidence_summary": rationale("A", spec["A_levels"]),
            "N_evidence_summary": rationale("N", spec["N_levels"]),
            "locator_status": spec["verification_status"].strip(),
            "coding_basis": "full-text anchor and coding rules",
            "supported_claim": spec["claim_endpoint"],
            "claim_boundary_code": boundary_code,
            "unsupported_claim": unsupported,
            "extraction_decision": extraction_decision,
            "extraction_basis": extraction_basis,
        }
        rows.append(row)
    return rows


def validate_study_index(studies: list[dict[str, str]], configs: list[dict[str, str]]) -> None:
    counts = Counter(row["citation_key"] for row in configs)
    for i, study in enumerate(studies, 1):
        expected_id = f"S{i:03d}"
        expected_count = str(counts.get(study["citation_key"], 0))
        if study["domain"] == "scope_boundary":
            expected_status = "scope_boundary"
        elif int(expected_count):
            expected_status = "configuration_coded"
        else:
            expected_status = "interface_support_only"
        observed = (study.get("study_id", ""), study.get("configuration_count", ""), study.get("configuration_status", ""))
        expected = (expected_id, expected_count, expected_status)
        if observed != expected:
            raise ValueError(f"study index mismatch for {study['citation_key']}: {observed} != {expected}")


def main() -> None:
    studies = read_csv(STUDY_PATH)
    specs = read_csv(SPEC_PATH)
    configs = build_rows(studies, specs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(configs[0]))
        writer.writeheader()
        writer.writerows(configs)
    validate_study_index(studies, configs)
    print(f"Wrote {len(configs)} configurations from {len({row['citation_key'] for row in configs})} studies")


if __name__ == "__main__":
    main()
