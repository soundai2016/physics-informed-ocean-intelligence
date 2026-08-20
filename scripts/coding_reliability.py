from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
STUDIES = DATA / "study_corpus.csv"
A02_CONFIGS = DATA / "human_recoding_a02_configurations.csv"
A03_CONFIGS = DATA / "human_recoding_a03_configurations.csv"
STUDY_OUT = RESULTS / "human_recoding_study_reliability.csv"
CONFIG_OUT = RESULTS / "human_recoding_configuration_reliability.csv"
FACET_OUT = RESULTS / "human_recoding_facet_reliability.csv"
BOOTSTRAP = 5000
SEED = 20260819
CONFIG_DIMS = [
    ("O", "O"),
    ("C", "C"),
    ("E", "E"),
    ("A", "A"),
    ("N", "N"),
    ("physics_insertion", "Physics insertion"),
    ("claim_boundary", "Claim boundary"),
]
FACETS = {
    "C": ("C0", "C1", "C2", "C3"),
    "N": ("N0", "N1", "N2", "N3"),
    "physics_insertion": (
        "none", "objective_or_projection", "inference", "representation", "solver_coupling"
    ),
}
CODING_FIELDS = ("O", "C", "E", "A", "N", "Physics insertion", "Claim boundary")
INSERTIONS = {
    "none", "objective_or_projection", "inference", "representation", "solver_coupling", "action_constraint"
}
BOUNDARIES = {
    "B_OFFLINE_TO_OPERATIONAL", "B_SIMULATION_TO_FIELD",
    "B_CONTROLLED_TO_OPEN_OCEAN", "B_DEVELOPER_TO_INDEPENDENT",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm(value: str | None) -> str:
    value = (value or "").strip()
    if ";" not in value:
        return value
    return ";".join(sorted(item.strip() for item in value.split(";") if item.strip()))


def as_set(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").split(";") if item.strip()}


def filled(row: dict[str, str]) -> bool:
    return bool(row["O"].strip())


def agreement(left: list[object], right: list[object]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left)


def kappa(left: list[object], right: list[object]) -> float:
    observed = agreement(left, right)
    counts_left, counts_right = Counter(left), Counter(right)
    labels = set(counts_left) | set(counts_right)
    n = len(left)
    expected = sum(counts_left[label] / n * counts_right[label] / n for label in labels)
    return math.nan if abs(1 - expected) < 1e-15 else (observed - expected) / (1 - expected)


def mean_jaccard(left: list[object], right: list[object]) -> float:
    values = []
    for a, b in zip(left, right):
        sa, sb = as_set(str(a)), as_set(str(b))
        values.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
    return sum(values) / len(values)


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_units(
    unit_ids: list[str],
    observations: dict[str, list[tuple[object, object]]],
    metric: Callable[[list[object], list[object]], float],
) -> tuple[float, float]:
    rng = random.Random(SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP):
        sampled = [unit_ids[rng.randrange(len(unit_ids))] for _ in unit_ids]
        left: list[object] = []
        right: list[object] = []
        for unit in sampled:
            for a, b in observations[unit]:
                left.append(a)
                right.append(b)
        value = metric(left, right)
        if math.isfinite(value):
            estimates.append(float(value))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def metric_row(
    dimension: str,
    units: list[str],
    observations: dict[str, list[tuple[object, object]]],
    include_jaccard: bool = False,
) -> dict[str, str]:
    left = [a for unit in units for a, _ in observations[unit]]
    right = [b for unit in units for _, b in observations[unit]]
    exact = agreement(left, right)
    kap = kappa(left, right)
    exact_low, exact_high = bootstrap_units(units, observations, agreement)
    kappa_low, kappa_high = bootstrap_units(units, observations, kappa)
    row = {
        "dimension": dimension,
        "n_studies": str(len(units)),
        "n_observations": str(len(left)),
        "exact_agreement": f"{exact:.6f}",
        "exact_agreement_ci95_low": f"{exact_low:.6f}",
        "exact_agreement_ci95_high": f"{exact_high:.6f}",
        "cohen_kappa": "" if not math.isfinite(kap) else f"{kap:.6f}",
        "kappa_ci95_low": "" if not math.isfinite(kappa_low) else f"{kappa_low:.6f}",
        "kappa_ci95_high": "" if not math.isfinite(kappa_high) else f"{kappa_high:.6f}",
        "mean_set_jaccard": "",
        "jaccard_ci95_low": "",
        "jaccard_ci95_high": "",
    }
    if include_jaccard:
        jac = mean_jaccard(left, right)
        jac_low, jac_high = bootstrap_units(units, observations, mean_jaccard)
        row.update({
            "mean_set_jaccard": f"{jac:.6f}",
            "jaccard_ci95_low": f"{jac_low:.6f}",
            "jaccard_ci95_high": f"{jac_high:.6f}",
        })
    return row


def validate_config_table(studies: list[dict[str, str]], rows: list[dict[str, str]], label: str) -> None:
    study_ids = {row["study_id"] for row in studies}
    keys = [(row["Study ID"], row["Slot"]) for row in rows]
    expected = {(study_id, slot) for study_id in study_ids for slot in ("1", "2")}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"{label}: expected exactly two fixed slots per retained study")
    for row in rows:
        states = [bool(row[field].strip()) for field in CODING_FIELDS]
        if any(states) and not all(states):
            raise ValueError(f"{label}: partially filled coding row at {row['Study ID']} slot {row['Slot']}")
        if not all(states):
            continue
        for field, prefix in (("O", "O"), ("C", "C"), ("E", "E"), ("A", "A"), ("N", "N")):
            values = as_set(row[field])
            allowed = {f"{prefix}{i}" for i in range(4)}
            if not values or not values.issubset(allowed):
                raise ValueError(f"{label}: invalid {field} code at {row['Study ID']} slot {row['Slot']}")
            if field in {"O", "E", "A"} and len(values) != 1:
                raise ValueError(f"{label}: {field} must be single-valued")
            if field in {"C", "N"} and f"{prefix}0" in values and len(values) != 1:
                raise ValueError(f"{label}: baseline {prefix}0 must be exclusive")
        insertions = as_set(row["Physics insertion"])
        if not insertions or not insertions.issubset(INSERTIONS) or ("none" in insertions and len(insertions) != 1):
            raise ValueError(f"{label}: invalid physics insertion at {row['Study ID']} slot {row['Slot']}")
        if row["Claim boundary"] not in BOUNDARIES:
            raise ValueError(f"{label}: invalid claim boundary at {row['Study ID']} slot {row['Slot']}")


def config_maps(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    mapped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if filled(row):
            mapped[row["Study ID"]].append(row)
    for study_id in mapped:
        mapped[study_id].sort(key=lambda row: int(row["Slot"]))
    return mapped


def score_studies(
    studies: list[dict[str, str]],
    a02_rows: list[dict[str, str]],
    a03_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    study_ids = sorted(row["study_id"] for row in studies)
    a02 = Counter(row["Study ID"] for row in a02_rows if filled(row))
    a03 = Counter(row["Study ID"] for row in a03_rows if filled(row))
    dimensions = {
        "configuration_bearing": lambda counts, study_id: counts[study_id] > 0,
        "configuration_count": lambda counts, study_id: counts[study_id],
    }
    rows = []
    for dimension, getter in dimensions.items():
        observations = {
            study_id: [(getter(a02, study_id), getter(a03, study_id))]
            for study_id in study_ids
        }
        row = metric_row(dimension, study_ids, observations)
        if dimension == "configuration_count":
            mae = sum(abs(a02[study_id] - a03[study_id]) for study_id in study_ids) / len(study_ids)
            row["mean_absolute_difference"] = f"{mae:.6f}"
        else:
            row["mean_absolute_difference"] = ""
        rows.append(row)
    return rows


def score_configurations(
    a02_rows: list[dict[str, str]],
    a03_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str], int]:
    a02, a03 = config_maps(a02_rows), config_maps(a03_rows)
    study_ids = sorted(set(a02) | set(a03))
    eligible = [
        study_id for study_id in study_ids
        if len(a02.get(study_id, [])) == len(a03.get(study_id, [])) > 0
        and [row["Slot"] for row in a02[study_id]] == [row["Slot"] for row in a03[study_id]]
    ]
    matched_count = sum(len(a02[study_id]) for study_id in eligible)
    rows = []
    for dimension, field in CONFIG_DIMS:
        observations = {
            study_id: [
                (norm(left[field]), norm(right[field]))
                for left, right in zip(a02[study_id], a03[study_id])
            ]
            for study_id in eligible
        }
        row = metric_row(dimension, eligible, observations, include_jaccard=True)
        row["eligibility_rule"] = "same positive configuration count and fixed slot alignment"
        rows.append(row)
    return rows, eligible, matched_count


def score_facets(
    a02_rows: list[dict[str, str]],
    a03_rows: list[dict[str, str]],
    eligible: list[str],
) -> list[dict[str, str]]:
    a02, a03 = config_maps(a02_rows), config_maps(a03_rows)
    field_for = {"C": "C", "N": "N", "physics_insertion": "Physics insertion"}
    rows = []
    for dimension, facets in FACETS.items():
        field = field_for[dimension]
        for facet in facets:
            observations = {
                study_id: [
                    (facet in as_set(left[field]), facet in as_set(right[field]))
                    for left, right in zip(a02[study_id], a03[study_id])
                ]
                for study_id in eligible
            }
            row = metric_row(f"{dimension}:{facet}", eligible, observations)
            left_values = [a for study_id in eligible for a, _ in observations[study_id]]
            right_values = [b for study_id in eligible for _, b in observations[study_id]]
            row.update({
                "dimension_group": dimension,
                "facet": facet,
                "a02_positive": str(sum(bool(value) for value in left_values)),
                "a03_positive": str(sum(bool(value) for value in right_values)),
            })
            rows.append(row)
    return rows


def main() -> None:
    studies = read_csv(STUDIES)
    a02_configs = read_csv(A02_CONFIGS)
    a03_configs = read_csv(A03_CONFIGS)
    validate_config_table(studies, a02_configs, "A02")
    validate_config_table(studies, a03_configs, "A03")
    study_rows = score_studies(studies, a02_configs, a03_configs)
    config_rows, eligible, matched_count = score_configurations(a02_configs, a03_configs)
    facet_rows = score_facets(a02_configs, a03_configs, eligible)
    write_csv(STUDY_OUT, study_rows)
    write_csv(CONFIG_OUT, config_rows)
    write_csv(FACET_OUT, facet_rows)
    print(f"Human recoding: {len(studies)} studies; {len(eligible)} matched studies; {matched_count} matched configurations")


if __name__ == "__main__":
    main()
