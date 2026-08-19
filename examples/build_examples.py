"""Write the example datasets used by the skill walkthrough and the test suite.

Run with ``python examples/build_examples.py``. The generated files are small,
committed, and deliberately messy in the ways real experimental exports are:
bold winners, a units row, percent signs, ``mean ± sd`` cells and a legacy sheet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_benchmark_workbook() -> Path:
    from openpyxl import Workbook

    path = HERE / "forecasting_mse.xlsx"
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "MSE"
    sheet.append(["Dataset", "iTransformer", "Pathformer", "PatchTST", "PDF", "MoFo"])
    rows = [
        ("Traffic", 0.445, 0.452, 0.435, 0.438, 0.424),
        ("ETTh1", 0.495, 0.450, 0.457, 0.456, 0.447),
        ("ETTh2", 0.424, 0.413, 0.406, 0.398, 0.379),
        ("ETTm1", 0.429, 0.428, 0.416, 0.408, 0.388),
        ("ETTm2", 0.375, 0.361, 0.362, 0.349, 0.342),
        ("Weather", 0.320, 0.318, 0.312, 0.323, 0.312),
        ("Electricity", 0.214, 0.211, 0.214, 0.199, 0.191),
        ("Solar", 0.223, 0.208, 0.215, 0.212, 0.193),
    ]
    for row in rows:
        sheet.append(list(row))

    notes = workbook.create_sheet("notes")
    notes.append(["key", "value"])
    notes.append(["horizon", 720])
    notes.append(["metric", "mean squared error, lower is better"])

    workbook.save(path)
    return path


def write_auroc_markdown() -> Path:
    path = HERE / "pathology_auroc.md"
    header = ["Task", "EAGLE", "CHIEF", "GigaPath", "CTransPath", "Virchow2"]
    rows = [
        ("BRCA subtyping", "**0.94**", "0.91", "0.90", "0.88", "0.92"),
        ("NSCLC subtyping", "0.96", "0.94", "**0.97**", "0.91", "0.95"),
        ("RCC subtyping", "**0.98**", "0.96", "0.97", "0.94", "0.97"),
        ("EGFR mutation", "0.72", "0.68", "**0.74**", "0.65", "0.71"),
        ("KRAS mutation", "**0.69**", "0.64", "0.66", "0.61", "0.67"),
        ("MSI status", "**0.86**", "0.81", "0.83", "0.78", "0.84"),
        ("HRD status", "0.77", "0.73", "**0.79**", "0.70", "0.76"),
        ("Grade prediction", "**0.83**", "0.79", "0.81", "0.76", "0.80"),
        ("Overall survival", "0.68", "0.65", "**0.70**", "0.62", "0.67"),
        ("Recurrence risk", "**0.72**", "0.68", "0.70", "0.66", "0.71"),
    ]
    lines = [
        "# Pathology foundation model AUROC",
        "",
        "Slide-level AUROC across ten tasks. Best model per task in bold.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_energy_long_csv() -> Path:
    path = HERE / "energy_mix_long.csv"
    rows = [
        ("Sector", "Fuel", "Energy_EJ"),
        ("Industry", "Coal", 3.10),
        ("Industry", "Oil", 0.62),
        ("Industry", "Gas", 0.74),
        ("Industry", "Electricity", 1.85),
        ("Buildings", "Coal", 0.44),
        ("Buildings", "Oil", 0.21),
        ("Buildings", "Gas", 0.96),
        ("Buildings", "Electricity", 1.52),
        ("Road transport", "Coal", 0.02),
        ("Road transport", "Oil", 2.41),
        ("Road transport", "Gas", 0.18),
        ("Road transport", "Electricity", 0.33),
        ("Agriculture", "Coal", 0.15),
        ("Agriculture", "Oil", 0.48),
        ("Agriculture", "Gas", 0.09),
        ("Agriculture", "Electricity", 0.27),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_variance_json() -> Path:
    path = HERE / "variance_contributions.json"
    payload = {
        "study": "drivers of canopy greenness",
        "unit": "percent of explained variance",
        "factors": [
            {"factor": "Solar radiation", "contribution": 21.4},
            {"factor": "Land surface temperature", "contribution": 18.2},
            {"factor": "Precipitation", "contribution": 15.9},
            {"factor": "Soil moisture", "contribution": 12.7},
            {"factor": "Elevation", "contribution": 9.8},
            {"factor": "Nitrogen deposition", "contribution": 7.3},
            {"factor": "Slope", "contribution": 6.1},
            {"factor": "Aspect", "contribution": 4.4},
            {"factor": "Residual", "contribution": 4.2},
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_flux_timeseries_csv() -> Path:
    path = HERE / "n2o_flux_timeseries.csv"
    rows = [
        ("Date", "LF_flux", "LF_flux_sem", "HF_flux", "HF_flux_sem"),
        ("2024-10-20", 16.0, 3.0, 26.0, 4.0),
        ("2024-10-26", 48.0, 6.0, 45.0, 6.0),
        ("2024-10-31", 26.0, 4.0, 34.0, 4.0),
        ("2024-11-05", 9.0, 2.0, 25.0, 4.0),
        ("2024-11-13", 6.0, 2.0, 6.0, 2.0),
        ("2024-12-31", 8.0, 2.0, 12.0, 3.0),
        ("2025-03-12", 10.0, 3.0, 34.0, 4.0),
        ("2025-03-20", 23.0, 4.0, 39.0, 5.0),
        ("2025-03-28", 85.0, 8.0, 152.0, 12.0),
        ("2025-04-05", 58.0, 6.0, 79.0, 8.0),
        ("2025-04-15", 14.0, 3.0, 15.0, 3.0),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_otu_overlap_csv() -> Path:
    path = HERE / "otu_overlap.csv"
    rows = [
        ("Treatment", "Total_OTUs", "Unique_OTUs", "Core_OTUs"),
        ("CK", 1842, 118, 936),
        ("Cu-L", 1790, 96, 936),
        ("Cu-M", 1655, 84, 936),
        ("Cu-H", 1498, 71, 936),
        ("Cr-L", 1774, 103, 936),
        ("Cr-M", 1612, 88, 936),
        ("Cr-H", 1441, 64, 936),
        ("Mix-L", 1728, 99, 936),
        ("Mix-M", 1566, 79, 936),
        ("Mix-H", 1387, 58, 936),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_variant_membership_csv() -> Path:
    path = HERE / "variant_concordance.csv"
    rows = [
        ("Concordance", "Plasma_specific", "Tissue_specific", "Shared", "Patients"),
        ("Disconcordant", "no", "yes", "no", 420),
        ("Disconcordant", "yes", "no", "no", 7),
        ("Disconcordant", "yes", "yes", "no", 76),
        ("Complete concordant", "no", "no", "no", 8),
        ("Complete concordant", "no", "no", "yes", 42),
        ("Partially concordant", "yes", "no", "yes", 99),
        ("Partially concordant", "no", "yes", "yes", 209),
        ("Partially concordant", "yes", "yes", "yes", 250),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_temperature_pairs_csv() -> Path:
    path = HERE / "surface_temperature_pairs.csv"
    rng_state = 20260819
    rows: list[tuple[str, float, float]] = []
    covers = {
        "Grass": (0.82, 3.1),
        "Cropland": (0.74, 5.4),
        "Water": (0.41, 9.8),
        "Urban": (0.93, 1.2),
    }
    # A deterministic linear congruential generator keeps the file reproducible
    # without importing numpy here.
    seed = rng_state
    for cover, (slope, intercept) in covers.items():
        for index in range(28):
            seed = (1103515245 * seed + 12345) % (2**31)
            jitter = (seed / (2**31) - 0.5) * 3.0
            lst = 18.0 + index * 0.55
            gst = intercept + slope * lst + jitter
            rows.append((cover, round(lst, 2), round(gst, 2)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Land_cover", "LST_C", "GST_C"))
        writer.writerows(rows)
    return path


def write_policy_mix_tsv() -> Path:
    path = HERE / "policy_mix.tsv"
    rows = [
        ("Scenario", "Carbon price", "Subsidy", "Standard", "Trade-off", "Synergy"),
        ("Price only", 48, 0, 0, 0, 0),
        ("Subsidy only", 0, 41, 0, 0, 0),
        ("Standard only", 0, 0, 37, 0, 0),
        ("Price + subsidy", 44, 38, 0, 9, 14),
        ("Price + standard", 45, 0, 34, 11, 12),
        ("Subsidy + standard", 0, 39, 33, 7, 16),
        ("All three", 42, 36, 31, 15, 22),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    return path


def write_residual_samples_csv() -> Path:
    path = HERE / "residual_strain.csv"
    seed = 771
    rows: list[tuple[str, float]] = []
    for model, centre, spread in (
        ("Zhang", 0.0, 42.0),
        ("Kioumarsi", 12.0, 61.0),
        ("Xue", -6.0, 33.0),
    ):
        for _ in range(140):
            total = 0.0
            for _ in range(6):
                seed = (1103515245 * seed + 12345) % (2**31)
                total += seed / (2**31)
            # Sum of six uniforms approximates a normal draw.
            rows.append((model, round(centre + spread * (total - 3.0) / 0.707, 2)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Model", "Residual_strain_microstrain"))
        writer.writerows(rows)
    return path


BUILDERS = (
    write_benchmark_workbook,
    write_auroc_markdown,
    write_energy_long_csv,
    write_variance_json,
    write_flux_timeseries_csv,
    write_otu_overlap_csv,
    write_variant_membership_csv,
    write_temperature_pairs_csv,
    write_policy_mix_tsv,
    write_residual_samples_csv,
)


def main() -> int:
    for builder in BUILDERS:
        print(builder())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
