"""Run the proposed detector and upstream CoDiT on the same GAIT records.

CoDiT is launched as its own process because its published GAIT implementation
uses an older PyTorch environment. This module does not vendor or modify that
third-party source tree.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_CODIT = ROOT.parent / "time-series-OOD-main"
_EXTERNAL_CODIT = ROOT / "benchmarks" / "external" / "time-series-OOD"
DEFAULT_CODIT = (_WORKSPACE_CODIT if
                 (_WORKSPACE_CODIT / "ours" / "gait" / "check_OOD_gait.py").is_file()
                 else _EXTERNAL_CODIT)
DEFAULT_DATA = (DEFAULT_CODIT / "ours" / "gait" / "data" /
                "gait-in-neurodegenerative-disease-database-1.0.0")
GAIT_DATA_URL = "https://drive.google.com/drive/folders/1Z-3YnlhcCxI_KlFF6FF7tMp5MSEZURRH?usp=sharing"
WINDOW_LENGTHS = (16, 18, 20)
HEALTHY_SPLITS = {
    "train": tuple("control{}".format(i) for i in range(1, 7)),
    "calibration": tuple("control{}".format(i) for i in range(7, 12)),
    "test": tuple("control{}".format(i) for i in range(12, 17)),
}
SEVERE_OOD = {
    "PD": tuple("park{}".format(i) for i in (1, 4, 7, 8, 10, 11, 12, 13, 14)),
    "ALS": tuple("als{}".format(i) for i in (1, 12, 13, 4, 5, 6, 7, 3, 9)),
    "HD": tuple("hunt{}".format(i) for i in (3, 4, 7, 10, 13, 15, 16, 18, 19)),
}
CODIT_OOD_ORDER = tuple((subject, group) for group in ("PD", "ALS", "HD")
                        for subject in SEVERE_OOD[group])
REQUIRED_RECORDS = tuple(subject + ".ts" for subject in
                         sum((list(group) for group in HEALTHY_SPLITS.values()), []) +
                         [subject for subjects in SEVERE_OOD.values() for subject in subjects])


def _dataset_is_complete(data_root):
    data_root = Path(data_root)
    return all((data_root / filename).is_file() for filename in REQUIRED_RECORDS)


def _safe_extract(zip_path, destination):
    """Extract a downloaded archive while rejecting paths outside destination."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(str(zip_path)) as archive:
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if os.path.commonpath((str(destination), str(target))) != str(destination):
                raise ValueError("unsafe path in GAIT archive: {}".format(item.filename))
        archive.extractall(str(destination))


def ensure_gait_dataset(data_root=DEFAULT_DATA, dataset_url=GAIT_DATA_URL):
    """Download and extract CoDiT's GAIT records once; reuse a complete local copy.

    Returns True when records were downloaded and False when the existing data
    directory was complete and reused.
    """
    data_root = Path(data_root).resolve()
    if _dataset_is_complete(data_root):
        print("GAIT dataset is already complete; skipping download: {}".format(data_root))
        return False

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "GAIT data are missing. Install the benchmark extra with "
            "`pip install -e '.[benchmarks]'` (includes gdown), or download "
            "the archive manually from {}.".format(dataset_url)) from exc

    data_root.parent.mkdir(parents=True, exist_ok=True)
    print("GAIT records are missing or incomplete; downloading CoDiT's archive.", flush=True)
    with tempfile.TemporaryDirectory(prefix="codit_gait_download_",
                                     dir=str(data_root.parent)) as temporary:
        download_dir = Path(temporary) / "download"
        extract_dir = Path(temporary) / "extracted"
        download_dir.mkdir()
        extract_dir.mkdir()
        try:
            gdown.download_folder(url=dataset_url, output=str(download_dir),
                                  quiet=False, use_cookies=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not download CoDiT's GAIT archive. Download it manually "
                "from {} and extract its .ts records into {}.".format(
                    dataset_url, data_root)) from exc

        for archive_path in download_dir.rglob("*.zip"):
            _safe_extract(archive_path, extract_dir)
        search_roots = [download_dir, extract_dir]
        found = {}
        for filename in REQUIRED_RECORDS:
            match = next((path for root in search_roots
                          for path in root.rglob(filename) if path.is_file()), None)
            if match is not None:
                found[filename] = match
        missing = sorted(set(REQUIRED_RECORDS) - set(found))
        if missing:
            raise RuntimeError(
                "Downloaded GAIT archive did not contain all CoDiT records; "
                "missing {}. Check the archive at {}.".format(
                    ", ".join(missing), dataset_url))

        data_root.mkdir(parents=True, exist_ok=True)
        for filename, source in found.items():
            shutil.copy2(str(source), str(data_root / filename))

    if not _dataset_is_complete(data_root):
        raise RuntimeError("GAIT archive extraction did not produce a complete dataset at {}".format(data_root))
    print("GAIT data ready at {}".format(data_root), flush=True)
    return True


def _read_trace(path):
    """Read a CoDiT ``.ts`` record, dropping its leading time column."""
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError("{}:{} is not a numeric CoDiT record".format(
                    path, line_number)) from exc
            if len(row) != 13:
                raise ValueError("{}:{} must contain a time value and 12 gait features; found {} columns".format(
                    path, line_number, len(row)))
            rows.append(row[1:])
    if not rows or not np.isfinite(rows).all():
        raise ValueError("{} must contain finite gait measurements".format(path))
    return np.asarray(rows, dtype=float)


def _load_protocol(data_root):
    """Load CoDiT's published healthy split and severe patient subset."""
    traces, labels = {}, {}
    for split, subjects in HEALTHY_SPLITS.items():
        for subject in subjects:
            traces[subject] = _read_trace(Path(data_root) / (subject + ".ts"))
            labels[subject] = "HEALTHY"
    for group, subjects in SEVERE_OOD.items():
        for subject in subjects:
            traces[subject] = _read_trace(Path(data_root) / (subject + ".ts"))
            labels[subject] = group
    return traces, labels


def _contexts(series, window, warmup):
    """Build length-matched contexts and their corresponding raw window starts."""
    left = warmup + window - 1
    if len(series) < left + window:
        raise ValueError("record is too short for W={} and warmup={}".format(window, warmup))
    starts = np.arange(left, len(series) - window + 1, dtype=int)
    segments = np.stack([series[i - left:i + window] for i in starts])
    return segments, starts


def _windows_for_subjects(traces, labels, subjects, window, warmup):
    values, targets, metadata = [], [], []
    for subject in subjects:
        chunks, starts = _contexts(traces[subject], window, warmup)
        values.append(chunks)
        targets.extend([0 if labels[subject] == "HEALTHY" else 1] * len(chunks))
        metadata.extend((subject, int(start), labels[subject]) for start in starts)
    if not values:
        raise ValueError("no subjects selected for this detector split")
    return np.concatenate(values), np.asarray(targets, dtype=np.int8), metadata


def _run_codit(codit_root, data_root, checkpoint_dir, output_dir, python,
               n_randomizations, seed):
    """Invoke the official upstream GAIT evaluator once for each window length."""
    gait_dir = Path(codit_root) / "ours" / "gait"
    script = gait_dir / "check_OOD_gait.py"
    if not script.is_file():
        raise FileNotFoundError("CoDiT GAIT evaluator not found: {}".format(script))
    if not Path(python).is_file() and not (Path(python).name.lower() in ("python", "python.exe")
                                           and _which(python)):
        raise FileNotFoundError("CoDiT Python executable not found: {}".format(python))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    for window in WINDOW_LENGTHS:
        checkpoint = Path(checkpoint_dir) / "gait_{}.pt".format(window)
        if not checkpoint.is_file():
            raise FileNotFoundError("missing official CoDiT checkpoint: {}".format(checkpoint))
        result_dir = output_dir / "codit_raw" / "w{}".format(window)
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), str(script), "--save_dir", str(result_dir),
            "--ckpt", str(checkpoint), "--transformation_list",
            "high_pass", "low_high", "high_low", "identity",
            "--wl", str(window), "--n", str(n_randomizations),
            "--trials", "1", "--seed", str(seed), "--disease_type", "all",
            "--cal_root_dir", str(data_root),
            "--in_test_root_dir", str(data_root),
            "--out_test_root_dir", str(data_root),
        ]
        print("Running official CoDiT W={} (upstream selects CPU/CUDA)".format(window), flush=True)
        subprocess.run(command, cwd=str(gait_dir), check=True)
        in_path = result_dir / "in_fisher_iter{}.npz".format(n_randomizations)
        out_path = result_dir / "out_fisher_iter{}.npz".format(n_randomizations)
        if not in_path.is_file() or not out_path.is_file():
            raise RuntimeError("CoDiT finished without its documented Fisher score files")
        with np.load(in_path) as archive:
            in_flat = archive["in_fisher_values_win"].astype(float).reshape(-1)
        with np.load(out_path) as archive:
            out_flat = archive["out_fisher_values_win"].astype(float).reshape(-1)
        ordered = {}
        cursor = 0
        for subject in HEALTHY_SPLITS["test"]:
            count = max(0, len(_read_trace(Path(data_root) / (subject + ".ts"))) - window + 1)
            ordered[subject] = in_flat[cursor:cursor + count]
            cursor += count
        if cursor != len(in_flat):
            raise ValueError("CoDiT healthy score count does not match its 5 test records for W={}".format(window))
        cursor = 0
        for subject, _group in CODIT_OOD_ORDER:
            count = max(0, len(_read_trace(Path(data_root) / (subject + ".ts"))) - window + 1)
            ordered[subject] = out_flat[cursor:cursor + count]
            cursor += count
        if cursor != len(out_flat):
            raise ValueError("CoDiT OOD score count does not match its severe patient records for W={}".format(window))
        raw[window] = ordered
    return raw


def _which(command):
    import shutil
    return shutil.which(command)


def _auc(labels, scores):
    from scipy.stats import rankdata
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        return None
    ranks = rankdata(scores, method="average")
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def _write_summary_artifacts(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [entry["window_length"] for entry in results["window_results"]]
    metrics = [entry["metrics_by_group"]["ALL"] for entry in results["window_results"]]
    x = np.arange(len(windows), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for offset, key, label, color in (
            (-width, "proposed_low_auroc", "Proposed low level", "#245b8f"),
            (0, "proposed_high_auroc", "Proposed high level", "#d07a27"),
            (width, "codit_auroc", "CoDiT", "#4d8a57")):
        ax.bar(x + offset, [item[key] for item in metrics], width,
               label=label, color=color)
    ax.set_xticks(x, [str(value) for value in windows])
    ax.set_xlabel("Window length")
    ax.set_ylabel("Window-level AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_8_auroc.pdf")
    plt.close(fig)

    lines = [r"\begin{tabular}{lcrrr}", r"\toprule",
             r"Group & $W$ & Proposed low & Proposed high & CoDiT \\",
             r"\midrule"]
    for entry in results["window_results"]:
        for group in ("ALL",) + tuple(SEVERE_OOD):
            values = entry["metrics_by_group"][group]
            fmt = lambda value: "--" if value is None else "{:.3f}".format(value)
            lines.append("{} & {} & {} & {} & {} \\\\".format(
                group, entry["window_length"], fmt(values["proposed_low_auroc"]),
                fmt(values["proposed_high_auroc"]), fmt(values["codit_auroc"])))
        lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "experiment_8_auroc.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run_experiment_8(data_root, codit_root=DEFAULT_CODIT, checkpoint_dir=None,
                     output_dir=None, codit_python=sys.executable,
                     n_randomizations=100, seed=1729,
                     warmup=9, k_low=1, filter_cutoff=0.08,
                     filter_order=3):
    """Run both methods on CoDiT's native, shared GAIT subject protocol."""
    from hierarchical_conformal import RandomizedWindowDetector

    data_root = Path(data_root).resolve()
    codit_root = Path(codit_root).resolve()
    checkpoint_dir = (Path(checkpoint_dir).resolve() if checkpoint_dir else
                      codit_root / "ours" / "gait" / "saved_models")
    output_dir = (Path(output_dir).resolve() if output_dir else
                  ROOT / "benchmarks" / "outputs" / "experiment_8")
    ensure_gait_dataset(data_root)
    traces, labels = _load_protocol(data_root)
    codit_scores = _run_codit(codit_root, data_root, checkpoint_dir, output_dir,
                              codit_python, n_randomizations, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "CoDiT GAIT shared-machine, shared-record comparison",
        "status": "completed_descriptive_run",
        "protocol": "official CoDiT healthy split and severe patient subset; descriptive window-level AUROC",
        "codit_source": str(codit_root), "data_root": str(data_root),
        "codit_checkpoint_dir": str(checkpoint_dir),
        "subjects": {"train": 6, "calibration": 5, "healthy_test": 5,
                     "ood": {group: len(ids) for group, ids in SEVERE_OOD.items()}},
        "settings": {"seed": seed, "window_lengths": list(WINDOW_LENGTHS),
                     "codit_randomizations": n_randomizations, "warmup": warmup,
                     "k_low": k_low, "filter_cutoff": filter_cutoff,
                     "filter_order": filter_order},
        "window_results": [],
    }

    train_subjects = HEALTHY_SPLITS["train"]
    calibration_subjects = HEALTHY_SPLITS["calibration"]
    test_subjects = HEALTHY_SPLITS["test"] + tuple(
        subject for subject, _group in CODIT_OOD_ORDER)
    for window in WINDOW_LENGTHS:
        train, _, _ = _windows_for_subjects(
            traces, labels, train_subjects, window, warmup)
        calibration, _, _ = _windows_for_subjects(
            traces, labels, calibration_subjects, window, warmup)
        test, target, metadata = _windows_for_subjects(
            traces, labels, test_subjects, window, warmup)
        detector = RandomizedWindowDetector(
            window=window, k_low=k_low, rate_window=window, k_high=1,
            delta1_low=0.9, delta1_high=0.9, warmup=warmup,
            filter_cutoff=filter_cutoff, filter_order=filter_order, seed=seed + window)
        detector.fit(train, calibration, calibration)
        proposed = detector.score(test)
        rows = []
        grouped_rows = defaultdict(list)
        for (subject, start, group), label, low, high in zip(
                metadata, target, proposed["low_score"], proposed["high_score"]):
            upstream = codit_scores[window][subject]
            if start >= len(upstream):
                raise ValueError("CoDiT did not score {} window {} for W={}".format(
                    subject, start, window))
            row = {"subject_id": subject, "group": group,
                   "window_length": window, "window_index": start,
                   "label": int(label), "proposed_low_score": float(low),
                   "proposed_high_score": float(high),
                   # CoDiT's Fisher score is larger for nominal windows.
                   "codit_anomaly_score": float(-upstream[start])}
            rows.append(row)
            grouped_rows[group].append(row)
        with (output_dir / "experiment_8_windows_w{}.csv".format(window)).open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        metrics = {}
        for group in ("ALL",) + tuple(SEVERE_OOD):
            subset = (rows if group == "ALL" else
                      [row for row in rows if row["group"] in ("HEALTHY", group)])
            metrics[group] = {
                "windows": len(subset),
                "subjects": len({row["subject_id"] for row in subset}),
                "proposed_low_auroc": _auc(
                    [r["label"] for r in subset], [r["proposed_low_score"] for r in subset]),
                "proposed_high_auroc": _auc(
                    [r["label"] for r in subset], [r["proposed_high_score"] for r in subset]),
                "proposed_high_alarm_rate": float(np.mean([
                    r["proposed_high_score"] > detector.high.threshold_ for r in subset])),
                "codit_auroc": _auc(
                    [r["label"] for r in subset], [r["codit_anomaly_score"] for r in subset]),
            }
        results["window_results"].append({
            "window_length": window, "matched_windows": len(rows),
            "metrics_by_group": metrics,
            "window_scores_csv": "experiment_8_windows_w{}.csv".format(window),
        })

    with (output_dir / "experiment_8_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    _write_summary_artifacts(results, output_dir)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--codit-root", type=Path, default=DEFAULT_CODIT)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--codit-python", default=sys.executable,
                        help="Python executable from CoDiT's isolated environment")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks" / "outputs" / "experiment_8")
    parser.add_argument("--n-randomizations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--warmup", type=int, default=9)
    args = parser.parse_args(argv)
    if args.n_randomizations < 1:
        parser.error("--n-randomizations must be positive")
    results = run_experiment_8(
        data_root=args.data, codit_root=args.codit_root,
        checkpoint_dir=args.checkpoint_dir, output_dir=args.output,
        codit_python=args.codit_python,
        n_randomizations=args.n_randomizations,
        seed=args.seed, warmup=args.warmup)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
