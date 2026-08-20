# OCEAN configuration coding rules

Code one evaluated endpoint under one materially distinct setting. Use only evidence established by the cited full text.

## Sequence

1. State the evaluated endpoint.
2. Split only materially distinct endpoints or settings.
3. Record setting: D offline/hindcast, S simulation/replay, H controlled hardware/water, F field.
4. Code O/C/E/A/N.
5. Cross-check C against physics insertion.
6. Record full-text anchor, supported claim, and nearest unsupported boundary.

## O — observation interface

Use the first applicable level: `O3` action changes acquisition support; `O2` irregular support, latency, missingness, calibration, pose uncertainty, or provenance affects inference; `O1` multiple identifiable modalities; `O0` homogeneous/single-source or complete gridded input. Generic action-conditioned dynamics are not O3 unless acquisition support changes.

## C — computation-altering physical structure

- `C0`: none established.
- `C1`: physical constraint changes objective, projection, or inference.
- `C2`: representation explicitly encodes physical geometry, operator, invariance, conservation, scale separation, or mechanistic state used by computation.
- `C3`: learned component exchanges state, tendencies, or fields online with a solver.

`C0` is exclusive; C1-C3 may coexist. Physical variable names, geometry as ordinary features, post-hoc diagnostics, or generic graph/Fourier/transformer/neural-operator architecture do not by themselves establish C1-C3.

## Physics insertion

Allowed values: `objective_or_projection`, `inference`, `representation`, `solver_coupling`, `action_constraint`, `none`. The first four establish physics-informed computation. `action_constraint` alone does not. C1 requires objective/projection or inference; C2 requires representation; C3 requires solver coupling. `none` is exclusive.

## E — evolution

- `E0`: same-time reconstruction, inverse inference, or assimilation.
- `E1`: natural evolution, initialized forecasting, or free-running emulation.
- `E2`: counterfactual evolution under actions, forcings, schedules, or interventions.
- `E3`: evaluated online adaptation with an explicit update rule.

Autoregressive rollout is E1 unless actions/interventions alter the transition.

## A — agency

Use the highest evaluated level: `A0` passive estimate/forecast; `A1` decision support with an explicit objective/cost/interface; `A2` closed-loop sensing, navigation, communication, or control; `A3` coordinated multi-platform action. Potential use is not evidence.

## N — nonstationarity and assurance

`N0` is exclusive; N1-N3 may coexist.

- `N0`: aggregate in-domain evaluation only.
- `N1`: calibrated probabilistic quality or proper-score evaluation.
- `N2`: explicitly contrasted or held-out temporal, geographic, event, forcing, resolution, closure, water-mass, sensor/receiver, hardware, corruption, or fault shift.
- `N3`: separately specified and evaluated safety envelope, fallback, hazard logic, intervention rule, or audit.

Uncalibrated confidence is not N1. Pooled task/domain diversity, multiple regions, or horizons are not N2 unless evaluated as a shift/regime test. Evidence independence is separate from N3.

## Evidence independence and boundaries

Evidence independence is `DEV`, `EXT_REPLICATION`, or `EXT_AUDIT`. Field evidence is not automatically independent.

Claim boundary codes: `B_OFFLINE_TO_OPERATIONAL`, `B_SIMULATION_TO_FIELD`, `B_CONTROLLED_TO_OPEN_OCEAN`, `B_DEVELOPER_TO_INDEPENDENT`.

## Reliability

A02 and A03 use two fixed slots per retained study. Configuration presence and count are inferred from filled slots. Taxonomy and claim-boundary reliability uses only studies with the same positive count and identical slot alignment. Confidence intervals use study-cluster bootstrap resampling.
