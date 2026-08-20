from __future__ import annotations
import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ocean-intelligence-mpl"))

import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.edgecolor": "#A7B2BE",
    "axes.labelcolor": "#243447",
    "xtick.color": "#5B6878",
    "ytick.color": "#5B6878",
})

TEXT = "#243447"
SUBTEXT = "#5B6878"
EDGE = "#667788"
GRID = "#D9E1E8"
ACCENT = "#0072B2"

ROW_COLORS = {
    "O": "#EAF4FA",
    "C": "#EAF7F2",
    "E": "#FFF6E5",
    "A": "#F7EDF7",
    "N": "#FDEEE8",
}
UPPER_FILL = "#F4F8FB"
LOWER_FILL = "#FBF8F1"
AXIS_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]


def save_figure(fig, path: Path, *, dpi: int = 300) -> None:
    kwargs = {"dpi": dpi}
    if path.suffix.lower() == ".pdf":
        kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    fig.savefig(path, **kwargs)


def rounded(ax, x, y, w, h, *, fc="white", ec=EDGE, lw=0.88, r=0.012):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(patch)
    return patch


def add_title(ax, title, subtitle):
    ax.text(0.012, 0.968, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.4, fontweight="bold", color=TEXT)
    ax.text(0.012, 0.928, subtitle, transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, color=SUBTEXT)


def fig_ocean(outdir: Path):
    rows = [
        ("O", "Observation\nregime", [
            ("O0", "Gridded or\nsingle source"),
            ("O1", "Multimodal\nobservations"),
            ("O2", "Irregular / delayed /\nprovenance-aware"),
            ("O3", "Action-dependent\nacquisition"),
        ]),
        ("C", "Constraints\nand coupling", [
            ("C0", "No computational\nphysics insertion"),
            ("C1", "Constrained objective /\ninference"),
            ("C2", "Physics-structured\nrepresentation"),
            ("C3", "Online solver\ncoupling"),
        ]),
        ("E", "Evolution", [
            ("E0", "State\nreconstruction"),
            ("E1", "Forecast or\nemulation"),
            ("E2", "Action-conditioned\ncounterfactual"),
            ("E3", "Online adaptation\nwith controls"),
        ]),
        ("A", "Agency", [
            ("A0", "Passive\nanalysis"),
            ("A1", "Decision\nsupport"),
            ("A2", "Active sensing\nor control"),
            ("A3", "Cooperative\nmulti-agent"),
        ]),
        ("N", "Nonstationarity\nand assurance", [
            ("N0", "Aggregate in-domain\nevaluation only"),
            ("N1", "Calibrated\nuncertainty"),
            ("N2", "Declared shift / OOD /\nfault test"),
            ("N3", "Independent assurance\nenvelope"),
        ]),
    ]

    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    add_title(
        ax,
        "OCEAN taxonomy",
        "Separately coded interface dimensions; labels denote evaluated configurations rather than maturity or performance."
    )

    label_x = 0.014
    band_x = 0.165
    band_w = 0.823
    card_x0 = 0.198
    card_gap = 0.012
    card_w = (0.972 - card_x0 - 3 * card_gap) / 4
    card_h = 0.115
    row_gap = 0.026
    top_y = 0.75

    for i, (axis_code, axis_name, cards) in enumerate(rows):
        y = top_y - i * (card_h + row_gap)
        rounded(ax, band_x, y - 0.003, band_w, card_h + 0.006,
                fc=ROW_COLORS[axis_code], ec=ROW_COLORS[axis_code], lw=0.0, r=0.02)
        ax.text(label_x, y + card_h * 0.56, axis_code, transform=ax.transAxes,
                ha="left", va="center", fontsize=9.7, fontweight="bold", color=TEXT)
        ax.text(label_x + 0.036, y + card_h * 0.52, axis_name, transform=ax.transAxes,
                ha="left", va="center", fontsize=6.1, fontweight="bold",
                color=TEXT, linespacing=1.02)
        for j, (code, label) in enumerate(cards):
            x = card_x0 + j * (card_w + card_gap)
            rounded(ax, x, y, card_w, card_h, fc="white", ec=EDGE, lw=0.82, r=0.012)
            ax.text(x + card_w/2, y + card_h*0.68, code, transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.4, fontweight="bold", color=TEXT)
            ax.text(x + card_w/2, y + card_h*0.31, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=5.35, color=TEXT, linespacing=1.03)

    ax.text(0.012, 0.055,
            "D offline/hindcast · S simulation/replay · H hardware/controlled-water · F field. "
            "C0/N0 are exclusive baselines; C1–C3 and N1–N3 may coexist.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.15, color=SUBTEXT)

    for ext in ("pdf", "png"):
        save_figure(fig, outdir / f"ocean_taxonomy.{ext}")
    plt.close(fig)


def fig_closure(outdir: Path):
    fig, ax = plt.subplots(figsize=(7.25, 3.90))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    add_title(
        ax,
        "Evidence closure across interfaces",
        "Claim path above; recurrent transfer conditions below."
    )

    ax.text(
        0.012, 0.845, "Claim path",
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=6.0,
        fontweight="bold",
        color=SUBTEXT
    )

    top_nodes = [
        ("Observation", "support · error\nprovenance"),
        ("Belief", "state ·\nuncertainty"),
        ("Dynamics", "natural /\naction-conditioned"),
        ("Acoustic /\nsensor", "propagation ·\nreceiver · task"),
        ("Decision", "planning ·\nactive sensing"),
        ("Assurance", "constraints ·\nfallback · audit"),
    ]


    x0 = 0.012
    gap = 0.025
    y = 0.62
    w = 0.142
    h = 0.115

    arrow_margin = 0.005

    centers = []

    for i, (title, subtitle) in enumerate(top_nodes):
        x = x0 + i * (w + gap)

        rounded(
            ax,
            x, y, w, h,
            fc=UPPER_FILL,
            ec=EDGE,
            lw=0.82,
            r=0.014
        )

        ax.text(
            x + w / 2,
            y + h * 0.67,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color=TEXT,
            linespacing=1.0
        )

        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.85,
            color=TEXT,
            linespacing=1.0
        )

        centers.append((x + w / 2, y + h / 2))

        if i < len(top_nodes) - 1:
            current_right = x + w
            next_left = x + w + gap

            ax.annotate(
                "",
                xy=(
                    next_left - arrow_margin,
                    y + h / 2
                ),
                xytext=(
                    current_right + arrow_margin,
                    y + h / 2
                ),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="->",
                    color=ACCENT,
                    linewidth=0.75,
                    shrinkA=0,
                    shrinkB=0
                )
            )

    ax.annotate(
        "",
        xy=(centers[0][0], 0.495),
        xytext=(centers[4][0], 0.495),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="->",
            color=EDGE,
            linewidth=0.78,
            connectionstyle="arc3,rad=-0.18"
        )
    )

    ax.text(
        (centers[0][0] + centers[4][0]) / 2,
        0.525,
        "Executed actions change future observations",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.25,
        color=SUBTEXT
    )

    ax.text(
        0.012, 0.335,
        "Evidence-transfer conditions",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=SUBTEXT
    )

    items = [
        (
            "1",
            "Propagation",
            "Errors amplify across\n"
            "rollout and downstream\n"
            "operators."
        ),
        (
            "2",
            "Task sufficiency",
            "Metrics must control\n"
            "task-relevant directions,\n"
            "or test the consequence."
        ),
        (
            "3",
            "Hidden dynamics",
            "Unresolved state can\n"
            "shift the effective dynamics\n"
            "across regimes."
        ),
        (
            "4",
            "Endogeneity",
            "Policies change occupancy\n"
            "and future observations;\n"
            "transfer needs coverage."
        ),
    ]

    bx0 = 0.012
    bgap = 0.016
    by = 0.07
    bw = 0.233
    bh = 0.18

    for i, (n, title, body) in enumerate(items):
        x = bx0 + i * (bw + bgap)

        rounded(
            ax,
            x, by, bw, bh,
            fc=LOWER_FILL,
            ec=EDGE,
            lw=0.82,
            r=0.014
        )

        rounded(
            ax,
            x + 0.012,
            by + bh - 0.048,
            0.036,
            0.029,
            fc="white",
            ec=EDGE,
            lw=0.75,
            r=0.018
        )

        ax.text(
            x + 0.030,
            by + bh - 0.033,
            n,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color=TEXT
        )

        ax.text(
            x + 0.056,
            by + bh - 0.034,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.85,
            fontweight="bold",
            color=TEXT
        )

        ax.text(
            x + 0.014,
            by + 0.044,
            body,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=4.85,
            color=TEXT,
            linespacing=1.02
        )

    for ext in ("pdf", "png"):
        save_figure(fig, outdir / f"evidence_closure.{ext}")

    plt.close(fig)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fig_corpus_profile(outdir: Path, corpus_path: Path, config_path: Path):
    studies = _read_csv(corpus_path)
    configs = _read_csv(config_path)
    if not studies or not configs:
        raise ValueError("study or configuration table is empty")

    years = Counter(int(r["year"]) for r in studies)
    year_min, year_max = min(years), max(years)
    year_values = list(range(year_min, year_max + 1))
    year_counts = [years.get(y, 0) for y in year_values]

    scope_order = ["direct_ocean", "underwater_acoustics", "marine_robotics", "transferable_adjacent", "transferable_control", "transferable_assurance"]
    scope_labels = ["Ocean", "Acoustics", "Robotics", "Adjacent", "Control", "Assurance"]
    scope_counts = Counter(r["evidence_scope"] for r in configs)
    evidence_counts = Counter(r["evaluation_setting"] for r in configs)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75), gridspec_kw={"width_ratios": [1.52, 1.12, 0.86]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.22, wspace=0.40)
    fig.suptitle("Study provenance and evaluated-configuration profile", x=0.055, y=0.965, ha="left", fontsize=8.4, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.885, f"{len(studies)} study records are separated from {len(configs)} source-anchored configurations; counts describe the retained map, not field prevalence.", ha="left", va="top", fontsize=5.5, color=SUBTEXT)

    ax=axes[0]
    ax.bar(year_values, year_counts, color=ACCENT, width=0.82)
    ax.set_title("Records by publication year",loc="left",fontsize=6.4,fontweight="bold",color=TEXT)
    ax.set_xlim(year_min,year_max); ax.set_ylim(bottom=0)
    ax.set_xticks([1995,2000,2005,2010,2015,2020,2026]); ax.tick_params(axis="x",labelrotation=35,labelsize=5.2); ax.tick_params(axis="y",labelsize=5.2)
    ax.grid(axis="y",color=GRID,linewidth=0.45,alpha=0.7)

    ax=axes[1]
    vals=[scope_counts[x] for x in scope_order]
    bars=ax.barh(scope_labels[::-1], vals[::-1], color=AXIS_COLORS[0], height=0.62)
    ax.set_title("Configuration scope",loc="left",fontsize=6.4,fontweight="bold",color=TEXT)
    ax.set_xlim(0,max(vals)*1.18); ax.tick_params(axis="both",labelsize=5.0); ax.grid(axis="x",color=GRID,linewidth=0.45,alpha=0.7)
    for bar,value in zip(bars,vals[::-1]): ax.text(value+0.5,bar.get_y()+bar.get_height()/2,str(value),va="center",fontsize=5.1,color=TEXT)

    ax=axes[2]
    labels=list("DSHF"); vals=[evidence_counts[x] for x in labels]
    bars=ax.barh(labels[::-1], vals[::-1], color=["#D55E00", "#CC79A7", "#009E73", "#0072B2"], height=0.62)
    ax.set_title("Evaluation setting",loc="left",fontsize=6.4,fontweight="bold",color=TEXT)
    ax.set_xlim(0,max(vals)*1.18); ax.tick_params(axis="both",labelsize=5.2); ax.grid(axis="x",color=GRID,linewidth=0.45,alpha=0.7)
    for bar,value in zip(bars,vals[::-1]): ax.text(value+0.7,bar.get_y()+bar.get_height()/2,str(value),va="center",fontsize=5.1,color=TEXT)

    for ax in axes:
        ax.spines[["top","right","left"]].set_visible(False); ax.spines["bottom"].set_color("#94A3B8"); ax.set_axisbelow(True)
    fig.text(0.055,0.045,"D offline/hindcast · S simulation/replay · H hardware/controlled-water · F field. Evidence independence is an orthogonal field in the configuration table.",ha="left",va="bottom",fontsize=5.0,color=SUBTEXT)
    for ext in ("pdf","png"): save_figure(fig, outdir / f"corpus_profile.{ext}")
    plt.close(fig)


def fig_configuration_heatmap(outdir: Path, config_path: Path):
    configs=_read_csv(config_path)
    levels=[f"{a}{i}" for a in "OCEAN" for i in range(4)]
    configs=sorted(configs,key=lambda r:(r["evidence_scope"],r["citation_key"],r["configuration_id"]))
    matrix=[]
    for r in configs:
        present=set()
        for a in "OCEAN": present.update(x.strip() for x in r[f"{a}_levels"].split(";") if x.strip())
        matrix.append([1 if level in present else 0 for level in levels])

    fig,ax=plt.subplots(figsize=(7.25,4.20)); fig.patch.set_facecolor("white")
    ax.imshow(matrix,aspect="auto",interpolation="nearest",cmap=mpl.colors.ListedColormap(["#F7FAFC", ACCENT]),vmin=0,vmax=1)
    ax.set_xticks(range(len(levels)),labels=levels,fontsize=5.5)
    ax.set_yticks([])
    ax.set_title("Configuration-level OCEAN evidence matrix",loc="left",fontsize=8.4,fontweight="bold",color=TEXT,pad=10)
    ax.set_xlabel(f"Complete O0-N3 assignments for each evaluated configuration ({len(configs)} rows)",fontsize=5.5,color=SUBTEXT,labelpad=6)
    for x in [3.5,7.5,11.5,15.5]: ax.axvline(x,color="white",linewidth=1.4)
    scopes=[]; last=None; start=0
    for i,r in enumerate(configs+[{"evidence_scope":None}]):
        cur=r["evidence_scope"]
        if last is None: last=cur; start=i
        elif cur!=last:
            scopes.append((last,start,i-1)); ax.axhline(i-0.5,color=GRID,linewidth=0.65)
            last=cur; start=i
    label_map={"direct_ocean":"direct ocean","underwater_acoustics":"underwater acoustics","marine_robotics":"marine robotics","transferable_adjacent":"adjacent transfer","transferable_control":"control transfer","transferable_assurance":"assurance transfer"}
    for scope,a,b in scopes:
        ax.text(-0.7,(a+b)/2,label_map.get(scope,scope),ha="right",va="center",fontsize=4.7,color=SUBTEXT,clip_on=False)
    ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.subplots_adjust(left=0.18,right=0.99,top=0.90,bottom=0.12)
    for ext in ("pdf","png"): save_figure(fig, outdir / f"configuration_heatmap.{ext}")
    plt.close(fig)



def fig_evidence_structure(outdir: Path, config_path: Path):
    configs = _read_csv(config_path)

    boundary_order = [
        "B_OFFLINE_TO_OPERATIONAL",
        "B_SIMULATION_TO_FIELD",
        "B_CONTROLLED_TO_OPEN_OCEAN",
        "B_DEVELOPER_TO_INDEPENDENT",
    ]
    boundary_labels = [
        "Offline -> operational",
        "Simulation -> field",
        "Controlled water -> open ocean",
        "Developer -> independent evidence",
    ]
    boundary_counts = Counter(row["claim_boundary_code"] for row in configs)
    boundary_values = [boundary_counts[key] for key in boundary_order]

    def has_any(row, axis, wanted):
        values = {item.strip() for item in row[f"{axis}_levels"].split(";") if item.strip()}
        return bool(values.intersection(wanted))

    coverage_labels = [
        "Irregular/action-dependent\nobservation (O2/O3)",
        "Forecast/action-conditioned\ndynamics (E1-E3)",
        "Underwater acoustic\nendpoint scope",
        "Decision/control\n(A1-A3)",
        "Assurance envelope\n(N3)",
    ]
    coverage = [
        sum(has_any(row, "O", {"O2", "O3"}) for row in configs),
        sum(has_any(row, "E", {"E1", "E2", "E3"}) for row in configs),
        sum(row["evidence_scope"] == "underwater_acoustics" for row in configs),
        sum(has_any(row, "A", {"A1", "A2", "A3"}) for row in configs),
        sum(has_any(row, "N", {"N3"}) for row in configs),
    ]

    fig = plt.figure(figsize=(7.25, 3.15))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Claim boundaries and interface-level evidence coverage",
        x=0.055, y=0.965, ha="left", fontsize=8.4, fontweight="bold", color=TEXT,
    )
    fig.text(
        0.055, 0.885,
        "The left panel shows where the nearest unsupported extension lies; the right panel shows how far evaluated evidence reaches.",
        ha="left", va="top", fontsize=5.4, color=SUBTEXT,
    )

    left = fig.add_axes([0.07, 0.18, 0.39, 0.60])
    right = fig.add_axes([0.60, 0.18, 0.36, 0.60])

    y = np.arange(len(boundary_labels))
    bars = left.barh(y, boundary_values, height=0.58)
    left.set_yticks(y, labels=boundary_labels)
    left.invert_yaxis()
    left.set_xlim(0, max(boundary_values) * 1.16)
    left.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.9)
    left.set_axisbelow(True)
    left.set_title("A  Nearest unsupported claim boundary", loc="left", fontsize=6.4, fontweight="bold", color=TEXT, pad=5)
    left.set_xlabel("configuration count", fontsize=5.3)
    left.tick_params(axis="x", labelsize=5.0, length=2.5)
    left.tick_params(axis="y", labelsize=4.9, length=0, pad=5)
    for bar, value in zip(bars, boundary_values):
        left.text(value + 0.6, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=5.0, color=TEXT)
    left.spines[["top", "right", "left"]].set_visible(False)
    left.spines["bottom"].set_color("#A7B2BE")

    y = np.arange(len(coverage_labels))
    bars = right.barh(y, coverage, height=0.58)
    right.set_yticks(y, labels=coverage_labels)
    right.invert_yaxis()
    right.set_xlim(0, max(coverage) * 1.18)
    right.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.9)
    right.set_axisbelow(True)
    right.set_title("B  Where evaluated evidence reaches", loc="left", fontsize=6.4, fontweight="bold", color=TEXT, pad=5)
    right.set_xlabel("configuration count (non-exclusive)", fontsize=5.3)
    right.tick_params(axis="x", labelsize=5.0, length=2.5)
    right.tick_params(axis="y", labelsize=4.9, length=0, pad=5)
    for bar, value in zip(bars, coverage):
        right.text(value + 0.5, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=5.0, color=TEXT)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.spines["bottom"].set_color("#A7B2BE")

    for extension in ("pdf", "png"):
        save_figure(fig, outdir / f"evidence_structure.{extension}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "study_corpus.csv")
    parser.add_argument("--configurations", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "study_configurations.csv")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_ocean(args.outdir)
    fig_closure(args.outdir)
    fig_corpus_profile(args.outdir, args.corpus, args.configurations)
    fig_configuration_heatmap(args.outdir, args.configurations)
    fig_evidence_structure(args.outdir, args.configurations)
    print("Figures updated")

if __name__ == "__main__":
    main()
