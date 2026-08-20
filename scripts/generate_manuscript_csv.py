from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript_numbers.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return rows


def tokens(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def main() -> None:
    studies = read_csv(DATA / "study_corpus.csv")
    configs = read_csv(DATA / "study_configurations.csv")
    closest_reviews = read_csv(DATA / "closest_review_comparison.csv")
    study_rel = {row["dimension"]: row for row in read_csv(RESULTS / "human_recoding_study_reliability.csv")}
    config_rel = {row["dimension"]: row for row in read_csv(RESULTS / "human_recoding_configuration_reliability.csv")}
    a02_configs = read_csv(DATA / "human_recoding_a02_configurations.csv")
    a03_configs = read_csv(DATA / "human_recoding_a03_configurations.csv")
    pub = Counter(row["publication_status"] for row in studies)
    setting = Counter(row["evaluation_setting"] for row in configs)
    physics_status = Counter(row["physics_informed_status"] for row in configs)
    insertion = Counter(item for row in configs for item in tokens(row["physics_insertion"]))
    levels = {
        f"{axis}{index}": sum(f"{axis}{index}" in tokens(row[f"{axis}_levels"]) for row in configs)
        for axis in "OCEAN" for index in range(4)
    }
    boundaries = Counter(row["claim_boundary_code"] for row in configs)
    core = [row for row in configs if row["evidence_scope"] in {"direct_ocean", "underwater_acoustics"}]
    direct = [row for row in configs if row["evidence_scope"] == "direct_ocean"]
    a02_filled = [row for row in a02_configs if row["O"].strip()]
    a03_filled = [row for row in a03_configs if row["O"].strip()]
    bearing = study_rel["configuration_bearing"]
    config_count = study_rel["configuration_count"]
    config_n = config_rel["N"]
    values: list[tuple[str, object, str]] = [
        ("StudyCount", len(studies), "int"),
        ("ConfigurationCount", len(configs), "int"),
        ("ConfigurationStudyCount", len({row["study_id"] for row in configs}), "int"),
        ("MultiConfigStudyCount", sum(int(row["configuration_count"]) > 1 for row in studies), "int"),
        ("IndependentRecodingStudyCount", len(studies), "int"),
        ("IndependentCoderAConfigCount", len(a02_filled), "int"),
        ("IndependentCoderBConfigCount", len(a03_filled), "int"),
        ("HumanRecodingMatchedStudyCount", int(config_rel["O"]["n_studies"]), "int"),
        ("HumanRecodingMatchedConfigCount", int(config_rel["O"]["n_observations"]), "int"),
        ("StudyBearingAgreement", float(bearing["exact_agreement"]), ".3f"),
        ("StudyBearingKappa", float(bearing["cohen_kappa"]), ".3f"),
        ("ConfigurationCountAgreement", float(config_count["exact_agreement"]), ".3f"),
        ("ConfigurationCountKappa", float(config_count["cohen_kappa"]), ".3f"),
        ("ConfigurationCountMeanAbsDiff", float(config_count["mean_absolute_difference"]), ".3f"),
        ("ConfigNAgreement", float(config_n["exact_agreement"]), ".3f"),
        ("ConfigNKappa", float(config_n["cohen_kappa"]), ".3f"),
        ("ClosestReviewCount", len(closest_reviews) - 1, "int"),
        ("PeerReviewedCount", pub["peer_reviewed_or_book"], "int"),
        ("PreprintCount", pub["preprint"], "int"),
        ("OtherRecordCount", len(studies) - pub["peer_reviewed_or_book"] - pub["preprint"], "int"),
        ("SettingD", setting["D"], "int"),
        ("SettingS", setting["S"], "int"),
        ("SettingH", setting["H"], "int"),
        ("SettingF", setting["F"], "int"),
        ("PhysicsEstablished", physics_status["established"], "int"),
        ("PhysicsNotEstablished", physics_status["not_established"], "int"),
        ("InsertionRepresentation", insertion["representation"], "int"),
        ("InsertionInference", insertion["inference"], "int"),
        ("InsertionObjective", insertion["objective_or_projection"], "int"),
        ("InsertionSolver", insertion["solver_coupling"], "int"),
        ("InsertionAction", insertion["action_constraint"], "int"),
        ("IrregularActionObservationCount", sum(any(item in tokens(row["O_levels"]) for item in ("O2", "O3")) for row in configs), "int"),
        ("ForecastDynamicsCount", sum(any(item in tokens(row["E_levels"]) for item in ("E1", "E2", "E3")) for row in configs), "int"),
        ("AcousticEndpointCount", sum(row["evidence_scope"] == "underwater_acoustics" for row in configs), "int"),
        ("DecisionControlCount", sum(any(item in tokens(row["A_levels"]) for item in ("A1", "A2", "A3")) for row in configs), "int"),
        ("ActiveAgencyCount", sum(any(item in tokens(row["A_levels"]) for item in ("A2", "A3")) for row in configs), "int"),
        ("AssuranceCount", sum("N3" in tokens(row["N_levels"]) for row in configs), "int"),
        ("FullTextPageAnchorCount", sum(row["locator_status"] == "full_text_page_anchor" for row in configs), "int"),
        ("FullTextSectionAnchorCount", sum(row["locator_status"] == "full_text_section_anchor" for row in configs), "int"),
        ("SingleExtractionCount", sum(row["extraction_decision"] == "single" for row in configs), "int"),
        ("SplitExtractionCount", sum(row["extraction_decision"] == "split" for row in configs), "int"),
        ("CoreOceanAcousticConfigCount", len(core), "int"),
        ("DirectOceanConfigCount", len(direct), "int"),
        ("CoreOceanAcousticPhysicsCount", sum(row["physics_informed_status"] == "established" for row in core), "int"),
        ("DirectOceanPhysicsCount", sum(row["physics_informed_status"] == "established" for row in direct), "int"),
        ("CoreOceanAcousticDecisionCount", sum(any(item in tokens(row["A_levels"]) for item in ("A1", "A2", "A3")) for row in core), "int"),
        ("CoreOceanAcousticAssuranceCount", sum("N3" in tokens(row["N_levels"]) for row in core), "int"),
        ("BoundaryOfflineOperational", boundaries["B_OFFLINE_TO_OPERATIONAL"], "int"),
        ("BoundarySimulationField", boundaries["B_SIMULATION_TO_FIELD"], "int"),
        ("BoundaryControlledOpenOcean", boundaries["B_CONTROLLED_TO_OPEN_OCEAN"], "int"),
        ("BoundaryDeveloperIndependent", boundaries["B_DEVELOPER_TO_INDEPENDENT"], "int"),
    ]
    words = {0: "Zero", 1: "One", 2: "Two", 3: "Three"}
    for axis in "OCEAN":
        for index in range(4):
            values.append((f"Count{axis}{words[index]}", levels[f"{axis}{index}"], "int"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["macro", "value", "format"])
        writer.writeheader()
        for macro, value, fmt in values:
            writer.writerow({"macro": macro, "value": value, "format": fmt})
    print("Manuscript numbers updated")


if __name__ == "__main__":
    main()
