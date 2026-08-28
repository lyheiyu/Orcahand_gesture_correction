from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import evaluate_orca_dimension_control as dimension_control
import generate_shot_sweep_figures as shot_sweep


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "diagnostics" / "orca_compact_selection_20260827"
OUTPUT = ROOT / "paper_final_compact_orca_20260827"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

PRIMARY_REPS = (
    "joint_angle11", "corrected17", "optimized_action17", "corrected_flex11",
    "optimized_action_flex11", "compact_corrected7", "compact_optimized_action7",
)
DISPLAY = {
    "joint_angle11": "JointAngle-11", "corrected17": "Actuator Projection-17",
    "optimized_action17": "Refined ORCA-17", "corrected_flex11": "Projection Flex-11",
    "optimized_action_flex11": "Refined Flex-11", "compact_corrected7": "Compact Projection-7",
    "compact_optimized_action7": "Compact Refined-7", "corrected_pca11": "Projection PCA-11",
    "optimized_action_pca11": "Refined PCA-11",
}
COLORS = {
    "joint_angle11": "#D89B1D", "corrected17": "#4B8B62", "optimized_action17": "#E46B25",
    "corrected_flex11": "#80B98B", "optimized_action_flex11": "#F3A15D",
    "compact_corrected7": "#246B52", "compact_optimized_action7": "#C8432F",
    "corrected_pca11": "#4A7FA7", "optimized_action_pca11": "#6B8299",
}
FROZEN = (3, 6, 9, 11, 12, 15, 16)
ACTUATOR_BOUNDS = (
    (-1.13446403, 0.610865235), (-0.523598790, 0.523598790),
    (-0.436332315, 1.745329260), (-0.261799395, 1.867502330),
    (-0.471238911, 0.471238911), (-0.436332315, 1.745329260),
    (-0.261799395, 1.867502330), (-0.471238911, 0.471238911),
    (-0.436332315, 1.745329260), (-0.261799395, 1.867502330),
    (-0.436332315, 0.523598790), (-0.436332315, 1.745329260),
    (-0.261799395, 1.867502330), (-0.785398185, 0.575958669),
    (-0.314159274, 0.959931076), (-0.436332315, 1.745329260),
    (-0.261799395, 1.867502330),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(old, new)
    return text


def write_latex_table(path: Path, columns: list[str], rows: list[list[object]], align: str | None = None) -> None:
    alignment = align or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(latex_escape(value) for value in columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 8, "figure.dpi": 120,
    })


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 2.0, 1.7, 1.0, "MediaPipe\n21x3 landmarks", "#E7EEF3"),
        (2.5, 2.0, 1.8, 1.0, "Actuator Projection-17", "#D9ECDD"),
        (4.8, 2.0, 1.8, 1.0, "MuJoCo-constrained\ntemporal refinement", "#F8DFCC"),
        (7.1, 2.0, 1.6, 1.0, "Refined ORCA-17", "#F4C9AC"),
        (9.2, 2.0, 1.6, 1.0, "Compact Refined-7", "#F2B7A9"),
        (11.2, 2.0, 1.5, 1.0, "Resample-16\n+ classifier", "#E7EEF3"),
        (2.5, 0.3, 1.8, 0.9, "JointAngle-11\nexternal baseline", "#F4E2A7"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="#334", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold", fontsize=9)
    for start, end in (((2.0, 2.5), (2.5, 2.5)), ((4.3, 2.5), (4.8, 2.5)),
                       ((6.6, 2.5), (7.1, 2.5)), ((8.7, 2.5), (9.2, 2.5)),
                       ((10.8, 2.5), (11.2, 2.5)), ((1.15, 2.0), (2.5, 1.2)),
                       ((4.3, 0.75), (11.95, 2.0))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     color="#4C566A", linewidth=1.4))
    ax.text(5.7, 3.55, "Huber fitting + kinematic consistency + bounds\n+ first/second-order temporal regularization",
            ha="center", va="center", fontsize=9, color="#7A3B19")
    ax.text(10.0, 1.45, "K=7 frozen on development data before final testing",
            ha="center", fontsize=9, color="#8D2E20")
    ax.set_title("Final Structured Temporal Representation Pipeline", weight="bold", pad=12)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_01_final_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def table_actuators() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inventory = read_rows(SOURCE / "orca_actuator_inventory.csv")
    all_rows, selected_rows = [], []
    for row in inventory:
        index = int(row["index"])
        low, high = ACTUATOR_BOUNDS[index]
        output = {"index": index, "actuator": row["actuator_name"],
                  "role": row["anatomical_semantic_role"], "finger": row["finger_group"],
                  "type": row["type"], "lower_rad": f"{low:.4f}", "upper_rad": f"{high:.4f}",
                  "compact7": "yes" if index in FROZEN else "no"}
        all_rows.append(output)
        if index in FROZEN:
            selected_rows.append(output)
    write_rows(TABLES / "table_01_orca_actuators.csv", all_rows)
    write_rows(TABLES / "table_02_compact7_actuators.csv", selected_rows)
    write_latex_table(TABLES / "table_01_orca_actuators.tex",
                      ["Index", "Actuator", "Role", "Group", "Type", "Low (rad)", "High (rad)", "Compact7"],
                      [[r["index"], r["actuator"], r["role"], r["finger"], r["type"],
                        r["lower_rad"], r["upper_rad"], r["compact7"]] for r in all_rows],
                      "rllllrrc")
    write_latex_table(TABLES / "table_02_compact7_actuators.tex",
                      ["Index", "Actuator", "Role"],
                      [[r["index"], r["actuator"], r["role"]] for r in selected_rows], "rll")
    return all_rows, selected_rows


def figure_actuators(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.2))
    y = np.arange(len(rows))
    colors = ["#C8432F" if int(row["index"]) in FROZEN else "#D8DEE5" for row in rows]
    ax.barh(y, np.ones(len(rows)), color=colors, edgecolor="white")
    labels = [f"{row['index']:>2}  {row['role']}" for row in rows]
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([])
    for position, row in enumerate(rows):
        ax.text(0.98, position, "SELECTED" if int(row["index"]) in FROZEN else row["type"],
                ha="right", va="center", color="white" if int(row["index"]) in FROZEN else "#4C566A",
                fontsize=8, weight="bold" if int(row["index"]) in FROZEN else "normal")
    ax.set_title("ORCA-17 Actuator Inventory and Frozen Compact Refined-7 Readout", weight="bold")
    ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_02_orca_actuator_structure.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def figure_trajectory() -> dict[str, object]:
    dataset = ROOT / "diagnostics" / "updated_6class_20260820" / "gesture_sequence_dataset_chinese_dance_6class_after_fix.csv"
    ids, labels, sequences = dimension_control._load_base(dataset)
    improvements = []
    for index, (corrected, optimized) in enumerate(zip(sequences["corrected17"], sequences["optimized_action17"], strict=True)):
        if len(corrected) < 3:
            continue
        corr_acc = float(np.mean(np.linalg.norm(corrected[2:] - 2 * corrected[1:-1] + corrected[:-2], axis=1)))
        opt_acc = float(np.mean(np.linalg.norm(optimized[2:] - 2 * optimized[1:-1] + optimized[:-2], axis=1)))
        improvements.append((corr_acc - opt_acc, index, corr_acc, opt_acc))
    positive = sorted(value for value in improvements if value[0] > 0)
    chosen = positive[len(positive) // 2]
    _, index, corr_acc, opt_acc = chosen
    corrected, optimized = sequences["corrected17"][index], sequences["optimized_action17"][index]
    selected_variance = [(float(np.std(corrected[:, actuator])), actuator) for actuator in FROZEN]
    plotted = [actuator for _, actuator in sorted(selected_variance, reverse=True)[:3]]
    fig, axes = plt.subplots(len(plotted), 1, figsize=(10, 6.5), sharex=True)
    for ax, actuator in zip(axes, plotted, strict=True):
        ax.plot(corrected[:, actuator], color="#4B8B62", alpha=0.8, linewidth=1.2, label="Actuator Projection-17")
        ax.plot(optimized[:, actuator], color="#C8432F", linewidth=1.7, label="Refined ORCA-17")
        ax.set_ylabel(dimension_control.ACTUATORS[actuator].replace("right_", "").replace("_actuator", ""))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Median Positive Acceleration-Improvement Sequence", weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_03_representative_trajectory.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    metadata = {"selection_rule": "median positive sequence-level acceleration improvement",
                "sequence_id": ids[index], "label": labels[index], "frames": len(corrected),
                "corrected_acceleration": corr_acc, "optimized_acceleration": opt_acc,
                "plotted_actuator_indices": ";".join(map(str, plotted))}
    write_rows(TABLES / "figure_03_selection_metadata.csv", [metadata])
    return metadata


def table_and_figure_stability() -> list[dict[str, object]]:
    rows = [
        {"representation": "Actuator Projection-17", "velocity_mean": 0.4590213503,
         "velocity_std": 0.2075952379, "acceleration_mean": 0.7099963695, "acceleration_std": 0.3476120304},
        {"representation": "Refined ORCA-17", "velocity_mean": 0.2469635930,
         "velocity_std": 0.1080174742, "acceleration_mean": 0.2703253537, "acceleration_std": 0.1291777744},
    ]
    write_rows(TABLES / "table_06_temporal_stability.csv", rows)
    write_latex_table(TABLES / "table_06_temporal_stability.tex",
                      ["Actuator representation", "Velocity", "Acceleration", "Reduction"],
                      [["Actuator Projection-17", "0.4590", "0.7100", "--"],
                       ["Refined ORCA-17", "0.2470", "0.2703", "46.2% / 61.9%"]], "lrrr")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
    for ax, metric, title in zip(axes, ("velocity_mean", "acceleration_mean"),
                                 ("Mean actuator velocity", "Mean actuator acceleration"), strict=True):
        values = [float(row[metric]) for row in rows]
        ax.bar(["Projection", "Refined"], values, color=["#4B8B62", "#C8432F"])
        ax.set_title(title); ax.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values): ax.text(i, value + max(values) * 0.025, f"{value:.4f}", ha="center")
    fig.suptitle("Actuator-space Temporal Stability (571 Sequences)", weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_04_temporal_stability.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return rows


def final_results() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary = read_rows(SOURCE / "final_test_results.csv")
    paired = read_rows(SOURCE / "final_test_paired_comparisons_holm.csv")
    for row in summary:
        if row["representation"] in DISPLAY:
            row["display_name"] = DISPLAY[row["representation"]]
    write_rows(TABLES / "table_04_final_classification.csv", summary)
    joint_paired = [row for row in paired if row["comparison"] == "compact_optimized_action7_minus_joint_angle11"]
    write_rows(TABLES / "table_05_compact_oa_vs_jointangle.csv", joint_paired)
    table_rows = []
    lookup = {(row["classifier"], row["representation"]): row for row in summary}
    for representation in PRIMARY_REPS:
        table_rows.append([DISPLAY[representation]] + [
            f"{float(lookup[(classifier, representation)]['accuracy_mean']):.4f} / "
            f"{float(lookup[(classifier, representation)]['macro_f1_mean']):.4f} / "
            f"{float(lookup[(classifier, representation)]['kappa_mean']):.4f}"
            for classifier in dimension_control.CLASSIFIERS
        ])
    write_latex_table(TABLES / "table_04_final_classification.tex",
                      ["Representation", "SVM", "KNN", "RF", "MLP"], table_rows, "lrrrr")
    stats_rows = []
    for row in joint_paired:
        if row["metric"] in {"accuracy", "macro_f1"}:
            difference = float(row["mean_difference"])
            half_width = float(row["ci95_difference"])
            interval = f"[{difference - half_width:+.4f}, {difference + half_width:+.4f}]"
            stats_rows.append([shot_sweep.DISPLAY_CLASSIFIERS[row["classifier"]], row["metric"],
                               f"{difference:+.4f}", interval, f"{float(row['wilcoxon_p_holm']):.4g}",
                               f"{float(row['cohen_dz']):+.3f}"])
    write_latex_table(TABLES / "table_05_compact_oa_vs_jointangle.tex",
                      ["Classifier", "Metric", "Difference", "95% CI", "Holm p", "dz"], stats_rows, "lllrrr")
    return summary, paired


def figure_dimension_control(summary: list[dict[str, str]]) -> None:
    selected = ("joint_angle11", "corrected17", "optimized_action17", "corrected_flex11",
                "optimized_action_flex11", "corrected_pca11", "optimized_action_pca11")
    lookup = {(row["classifier"], row["representation"]): row for row in summary}
    x = np.arange(4); width = 0.82 / len(selected)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, representation in enumerate(selected):
        means = [float(lookup[(classifier, representation)]["macro_f1_mean"]) for classifier in dimension_control.CLASSIFIERS]
        cis = [float(lookup[(classifier, representation)]["macro_f1_ci95"]) for classifier in dimension_control.CLASSIFIERS]
        positions = x - 0.41 + width / 2 + offset * width
        ax.bar(positions, means, width, yerr=cis, capsize=2, color=COLORS[representation], label=DISPLAY[representation])
    ax.set_xticks(x, [shot_sweep.DISPLAY_CLASSIFIERS[c] for c in dimension_control.CLASSIFIERS])
    ax.set_ylim(0.6, 0.9); ax.set_ylabel("Macro-F1"); ax.grid(axis="y", alpha=0.2)
    ax.set_title("Dimension-controlled Representation Comparison", weight="bold")
    ax.legend(ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_05_dimension_control.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def figure_development_k() -> None:
    rows = sorted(read_rows(SOURCE / "compact_dimension_selection.csv"), key=lambda row: int(row["k"]))
    x = [int(row["k"]) for row in rows]; y = [float(row["combined_mean"]) for row in rows]
    ci = [float(row["combined_ci95"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.errorbar(x, y, yerr=ci, color="#4A7FA7", marker="o", capsize=4)
    ax.axvline(7, color="#C8432F", linestyle="--", label="K=7 frozen before final test")
    ax.set_xticks(x); ax.set_xlabel("Candidate dimensions K"); ax.set_ylabel("Development combined score")
    ax.set_title("Development-only Compactness Selection", weight="bold"); ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_06_development_selection.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def figure_final_joint(summary: list[dict[str, str]], paired: list[dict[str, str]]) -> None:
    lookup = {(row["classifier"], row["representation"]): row for row in summary}
    classifiers = dimension_control.CLASSIFIERS; x = np.arange(4); width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, representation in enumerate(("joint_angle11", "compact_optimized_action7")):
        means = [float(lookup[(c, representation)]["accuracy_mean"]) for c in classifiers]
        cis = [float(lookup[(c, representation)]["accuracy_ci95"]) for c in classifiers]
        ax.bar(x + (offset - .5) * width, means, width, yerr=cis, capsize=4,
               color=COLORS[representation], label=DISPLAY[representation])
    p_lookup = {(row["classifier"], row["metric"]): float(row["wilcoxon_p_holm"]) for row in paired
                if row["comparison"] == "compact_optimized_action7_minus_joint_angle11"}
    for position, classifier in enumerate(classifiers):
        if p_lookup[(classifier, "accuracy")] < 0.05:
            height = max(float(lookup[(classifier, rep)]["accuracy_mean"]) + float(lookup[(classifier, rep)]["accuracy_ci95"])
                         for rep in ("joint_angle11", "compact_optimized_action7"))
            ax.text(position, height + 0.015, "*", ha="center", fontsize=16, weight="bold")
    ax.set_xticks(x, [shot_sweep.DISPLAY_CLASSIFIERS[c] for c in classifiers]); ax.set_ylim(0.7, 0.9)
    ax.set_ylabel("Accuracy"); ax.set_title("Frozen Final Test: Compact Refined-7 vs JointAngle-11", weight="bold")
    ax.text(0.99, 0.02, "* Holm-adjusted p < 0.05", transform=ax.transAxes, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_07_final_compact_vs_jointangle.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def figure_dimension_efficiency(summary: list[dict[str, str]]) -> None:
    reps = ("compact_optimized_action7", "joint_angle11", "optimized_action_flex11", "optimized_action17")
    dimensions = {"compact_optimized_action7": 7, "joint_angle11": 11,
                  "optimized_action_flex11": 11, "optimized_action17": 17}
    lookup = {(row["classifier"], row["representation"]): row for row in summary}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, metric in zip(axes, ("accuracy_mean", "macro_f1_mean"), strict=True):
        for representation in reps:
            values = [float(lookup[(classifier, representation)][metric]) for classifier in dimension_control.CLASSIFIERS]
            ax.scatter([dimensions[representation]] * len(values), values, color=COLORS[representation], s=35, alpha=.7)
            ax.plot(dimensions[representation], np.mean(values), marker="D", color=COLORS[representation], markersize=8,
                    label=DISPLAY[representation])
        ax.set_xticks([7, 11, 17]); ax.set_xlabel("Frame-level dimensions"); ax.grid(alpha=0.2)
        ax.set_title(metric.replace("_mean", "").replace("_", " ").title())
    axes[0].set_ylabel("Final-test score"); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Recognition Utility vs Representation Dimension", weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "figure_08_performance_vs_dimension.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def table_and_figure_perturbation() -> list[dict[str, object]]:
    rows = [
        {"condition": "Overall", "projection": 0.0291631566, "optimized_action": 0.0182004129},
        {"condition": "Gaussian", "projection": 0.1020472236, "optimized_action": 0.0605009190},
        {"condition": "Spike", "projection": 0.0060458862, "optimized_action": 0.0049729056},
        {"condition": "Dropout", "projection": 0.0058790240, "optimized_action": 0.0045924240},
    ]
    for row in rows:
        row["relative_reduction"] = 1 - float(row["optimized_action"]) / float(row["projection"])
    write_rows(TABLES / "table_07_controlled_perturbation.csv", rows)
    write_latex_table(TABLES / "table_07_controlled_perturbation.tex",
                      ["Condition", "Projection-17", "Refined ORCA-17", "Reduction"],
                      [[r["condition"], f"{r['projection']:.4f}", f"{r['optimized_action']:.4f}",
                        f"{100*r['relative_reduction']:.1f}%"] for r in rows], "lrrr")
    x = np.arange(len(rows)); width = .36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width/2, [r["projection"] for r in rows], width, color="#4B8B62", label="Actuator Projection-17")
    ax.bar(x + width/2, [r["optimized_action"] for r in rows], width, color="#C8432F", label="Refined ORCA-17")
    ax.set_xticks(x, [r["condition"] for r in rows]); ax.set_ylabel("Actuator clean-reference sensitivity")
    ax.set_title("Controlled Perturbation Robustness in Actuator Space", weight="bold")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_09_controlled_perturbation.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return rows


def remaining_tables() -> None:
    dimensions = [
        ["Raw landmarks", 63, 1008, "Cartesian observations"], ["JointAngle-11", 11, 176, "3D included angles"],
        ["Actuator Projection-17", 17, 272, "Frame-wise ORCA state"], ["Refined ORCA-17", 17, 272, "Refined ORCA state"],
        ["Projection/Refined Flex-11", 11, 176, "Predefined semantic subset"], ["Compact Projection/Refined-7", 7, 112, "Development-selected frozen subset"],
        ["Projection/Refined PCA-11", 11, 176, "Training-only statistical control"],
    ]
    write_rows(TABLES / "table_03_representation_dimensions.csv",
               [{"representation": r[0], "frame_dimensions": r[1], "encoded_features": r[2], "role": r[3]} for r in dimensions])
    write_latex_table(TABLES / "table_03_representation_dimensions.tex",
                      ["Representation", "Frame dim.", "Resample16", "Role"], dimensions, "lrrl")
    stability = read_rows(ROOT / "figures" / "paper_updated_6class_20260820" / "loss_ablation_stability_6class.csv")
    classification = read_rows(ROOT / "figures" / "paper_updated_6class_20260820" / "loss_ablation_summary_6class.csv")
    class_lookup = {(row["classifier"], row["feature_set"]): row for row in classification}
    ablation = []
    for row in stability:
        key = row["feature_set"]
        ablation.append({"variant": key, "velocity_mean": row["velocity_mean"],
                         "acceleration_mean": row["acceleration_mean"],
                         "svm_accuracy": class_lookup[("svm", key)]["accuracy_mean"],
                         "rf_accuracy": class_lookup[("rf", key)]["accuracy_mean"]})
    write_rows(TABLES / "table_08_ablation.csv", ablation)
    write_latex_table(TABLES / "table_08_ablation.tex",
                      ["Variant", "Velocity", "Acceleration", "SVM Acc.", "RF Acc."],
                      [[r["variant"], f"{float(r['velocity_mean']):.4f}", f"{float(r['acceleration_mean']):.4f}",
                        f"{float(r['svm_accuracy']):.4f}", f"{float(r['rf_accuracy']):.4f}"] for r in ablation], "lrrrr")
    runtime = json.loads((ROOT / "diagnostics" / "updated_6class_20260820" / "paper_completion" /
                          "runtime_summary_6class.json").read_text(encoding="utf-8"))
    runtime_rows = [{"frames": runtime["frames"], "mean_ms": runtime["solve_time_mean_ms"],
                     "median_ms": runtime["solve_time_median_ms"], "p95_ms": runtime["solve_time_p95_ms"],
                     "iterations_mean": runtime["iterations_mean"], "success_rate": runtime["success_rate"],
                     "finite_rate": runtime["finite_rate"]}]
    write_rows(TABLES / "table_09_runtime.csv", runtime_rows)
    write_latex_table(TABLES / "table_09_runtime.tex",
                      ["Frames", "Mean ms", "Median ms", "P95 ms", "Iterations", "Success"],
                      [[runtime["frames"], f"{runtime['solve_time_mean_ms']:.2f}", f"{runtime['solve_time_median_ms']:.2f}",
                        f"{runtime['solve_time_p95_ms']:.2f}", f"{runtime['iterations_mean']:.2f}", "100%"]], "rrrrrr")

    optimizer_rows = [
        {"parameter": "landmark weight", "symbol": "lambda_l", "value": "1.00"},
        {"parameter": "palm-normal weight", "symbol": "lambda_n", "value": "0.20"},
        {"parameter": "projection-prior weight", "symbol": "lambda_p", "value": "0.30"},
        {"parameter": "first-order temporal weight", "symbol": "lambda_s", "value": "0.10"},
        {"parameter": "acceleration weight", "symbol": "lambda_a", "value": "0.15"},
        {"parameter": "neutral-pose weight", "symbol": "lambda_d", "value": "0.15"},
        {"parameter": "soft-boundary weight", "symbol": "lambda_b", "value": "0.05"},
        {"parameter": "Huber threshold", "symbol": "delta", "value": "0.08"},
        {"parameter": "optimizer", "symbol": "--", "value": "SciPy L-BFGS-B"},
        {"parameter": "maximum iterations per frame", "symbol": "--", "value": "120"},
        {"parameter": "initialization", "symbol": "--", "value": "clipped actuator projection"},
        {"parameter": "other stopping tolerances", "symbol": "--", "value": "SciPy defaults"},
    ]
    write_rows(TABLES / "table_10_optimizer_parameters.csv", optimizer_rows)
    write_latex_table(TABLES / "table_10_optimizer_parameters.tex",
                      ["Parameter", "Symbol", "Value"],
                      [[r["parameter"], r["symbol"], r["value"]] for r in optimizer_rows], "lll")

    classifier_rows = [
        {"classifier": "SVM", "frozen_setting": "RBF kernel; C=5; gamma=scale"},
        {"classifier": "KNN", "frozen_setting": "k=3; distance weighting"},
        {"classifier": "RandomForest", "frozen_setting": "200 trees; unrestricted depth"},
        {"classifier": "MLP", "frozen_setting": "hidden=(128,64); alpha=1e-4; lr=1e-3; max_iter=1200"},
    ]
    write_rows(TABLES / "table_11_classifier_parameters.csv", classifier_rows)
    write_latex_table(TABLES / "table_11_classifier_parameters.tex",
                      ["Classifier", "Frozen setting"],
                      [[r["classifier"], r["frozen_setting"]] for r in classifier_rows], "ll")

    corruption_rows = [
        {"type": "Gaussian", "levels": "sigma=0.01, 0.03, 0.06", "scope": "all 21 landmarks and coordinates"},
        {"type": "Spike", "levels": "magnitude=0.75; duration=1,2,3 frames", "scope": "one randomly selected distal finger group"},
        {"type": "Dropout", "levels": "duration=3,5 frames", "scope": "freeze one distal finger group at last visible frame"},
    ]
    write_rows(TABLES / "table_12_corruption_protocol.csv", corruption_rows)
    write_latex_table(TABLES / "table_12_corruption_protocol.tex",
                      ["Corruption", "Levels", "Affected observations"],
                      [[r["type"], r["levels"], r["scope"]] for r in corruption_rows], "lll")


def main() -> None:
    setup(); figure_pipeline(); all_actuators, _ = table_actuators(); figure_actuators(all_actuators)
    figure_trajectory(); table_and_figure_stability(); summary, paired = final_results()
    figure_dimension_control(summary); figure_development_k(); figure_final_joint(summary, paired)
    figure_dimension_efficiency(summary); table_and_figure_perturbation(); remaining_tables()
    print(f"output={OUTPUT}")
    print(f"figures={len(list(FIGURES.glob('*.png')))} tables={len(list(TABLES.glob('table_*.csv')))}")


if __name__ == "__main__":
    main()
