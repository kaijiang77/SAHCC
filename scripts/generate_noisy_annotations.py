#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str
    level: float
    label: str


PERTURBATIONS = (
    Perturbation("clean", "clean", 0.0, "0"),
    Perturbation("abs_jitter_2px", "abs_jitter", 2.0, "2 px"),
    Perturbation("abs_jitter_4px", "abs_jitter", 4.0, "4 px"),
    Perturbation("abs_jitter_6px", "abs_jitter", 6.0, "6 px"),
    Perturbation("abs_jitter_8px", "abs_jitter", 8.0, "8 px"),
    Perturbation("rel_jitter_005d", "rel_jitter", 0.05, "0.05 d_i"),
    Perturbation("rel_jitter_010d", "rel_jitter", 0.10, "0.10 d_i"),
    Perturbation("rel_jitter_020d", "rel_jitter", 0.20, "0.20 d_i"),
    Perturbation("drop_005", "drop", 0.05, "5%"),
    Perturbation("drop_010", "drop", 0.10, "10%"),
    Perturbation("drop_020", "drop", 0.20, "20%"),
)
PERTURBATION_BY_NAME = {p.name: p for p in PERTURBATIONS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GT-noise annotation roots and run sanity checks for unified crowd datasets."
    )
    parser.add_argument("--root", type=Path, default=Path("data/unified/SHHA"), help="Clean unified dataset root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Parent directory for generated dataset roots.")
    parser.add_argument("--train-split", default="train", help="Split whose annotations will be perturbed.")
    parser.add_argument(
        "--eval-splits",
        nargs="*",
        default=None,
        help="Clean splits to copy and check. Defaults to all splits except train.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Perturbation seed used per setting.")
    parser.add_argument("--knn-k", type=int, default=2, help="Nearest neighbors used to recompute mean_mnn.")
    parser.add_argument("--settings", nargs="*", default=None, choices=sorted(PERTURBATION_BY_NAME))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated annotation roots.")
    parser.add_argument("--check-only", action="store_true", help="Only run sanity checks on existing outputs.")
    return parser.parse_args()


def read_split_ids(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_eval_splits(root: Path, train_split: str) -> list[str]:
    splits = sorted(path.stem for path in (root / "splits").glob("*.txt"))
    return [split for split in splits if split != train_split]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def expected_drop_count(num_points: int, ratio: float) -> int:
    return int(np.rint(ratio * num_points))


def compute_mean_mnn(points: np.ndarray, fallback: np.ndarray | None = None, k: int = 2, chunk_size: int = 1024) -> np.ndarray:
    points = points.astype(np.float32, copy=False).reshape(-1, 2)
    num_points = len(points)
    if num_points == 0:
        return np.empty((0,), dtype=np.float32)
    if num_points == 1:
        if fallback is not None and len(fallback) == 1:
            return fallback.astype(np.float32, copy=True).reshape(1)
        return np.ones((1,), dtype=np.float32)

    k_eff = min(k, num_points - 1)
    out = np.empty((num_points,), dtype=np.float32)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        block = points[start:end]
        diff = block[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        dist[rows, cols] = np.inf
        nearest = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        out[start:end] = nearest.mean(axis=1)
    return out


def annotation_path(root: Path, sample_id: str) -> Path:
    return root / "annotations" / f"{sample_id}.npz"


def setting_root(output_root: Path, perturbation: Perturbation, seed: int) -> Path:
    if perturbation.kind == "clean":
        return output_root / "clean"
    return output_root / f"{perturbation.name}_seed{seed}"


def link_dataset_dirs(clean_root: Path, out_root: Path) -> None:
    for dirname in ("images", "splits"):
        src = clean_root / dirname
        dst = out_root / dirname
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve(), target_is_directory=True)


def perturb_arrays(arrays: dict[str, np.ndarray], perturbation: Perturbation, rng: np.random.Generator, k: int) -> dict[str, np.ndarray]:
    out = {key: value.copy() for key, value in arrays.items()}
    points = out["points"].astype(np.float32, copy=True).reshape(-1, 2)
    labels = out["labels"].astype(np.int64, copy=True).reshape(-1)
    fallback_mnn = out["mean_mnn"].astype(np.float32, copy=True).reshape(-1)
    image_wh = out["image_wh"].astype(np.int64).reshape(2)
    width, height = int(image_wh[0]), int(image_wh[1])

    if perturbation.kind == "clean" or len(points) == 0:
        noisy_points = points
        noisy_labels = labels
        noisy_mnn = fallback_mnn
    elif perturbation.kind == "abs_jitter":
        eps = rng.uniform(-perturbation.level, perturbation.level, size=points.shape).astype(np.float32)
        noisy_points = points + eps
        noisy_points[:, 0] = np.clip(noisy_points[:, 0], 0, width - 1)
        noisy_points[:, 1] = np.clip(noisy_points[:, 1], 0, height - 1)
        noisy_labels = labels
        noisy_mnn = compute_mean_mnn(noisy_points, fallback=fallback_mnn, k=k)
    elif perturbation.kind == "rel_jitter":
        radius = (perturbation.level * fallback_mnn).reshape(-1, 1)
        eps = rng.uniform(-radius, radius, size=points.shape).astype(np.float32)
        noisy_points = points + eps
        noisy_points[:, 0] = np.clip(noisy_points[:, 0], 0, width - 1)
        noisy_points[:, 1] = np.clip(noisy_points[:, 1], 0, height - 1)
        noisy_labels = labels
        noisy_mnn = compute_mean_mnn(noisy_points, fallback=fallback_mnn, k=k)
    elif perturbation.kind == "drop":
        num_drop = expected_drop_count(len(points), perturbation.level)
        if num_drop == 0:
            keep = np.ones((len(points),), dtype=bool)
        else:
            drop_idx = rng.choice(len(points), size=num_drop, replace=False)
            keep = np.ones((len(points),), dtype=bool)
            keep[drop_idx] = False
        noisy_points = points[keep]
        noisy_labels = labels[keep]
        noisy_mnn = compute_mean_mnn(noisy_points, fallback=fallback_mnn[keep], k=k)
    else:
        raise ValueError(f"unsupported perturbation kind: {perturbation.kind}")

    out["points"] = noisy_points.astype(np.float32, copy=False)
    out["mean_mnn"] = noisy_mnn.astype(np.float32, copy=False)
    out["labels"] = noisy_labels.astype(np.int64, copy=False)
    if "count" in out:
        out["count"] = np.asarray([len(noisy_points)], dtype=out["count"].dtype)
    return out


def prepare_setting_root(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and not overwrite:
        raise FileExistsError(f"output already exists, pass --overwrite to replace: {out_root}")
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "annotations").mkdir(parents=True)


def generate_setting(
    clean_root: Path,
    out_root: Path,
    perturbation: Perturbation,
    train_ids: list[str],
    eval_ids: list[str],
    seed: int,
    k: int,
    overwrite: bool,
) -> dict[str, object]:
    prepare_setting_root(out_root, overwrite=overwrite)
    link_dataset_dirs(clean_root, out_root)

    rng = np.random.default_rng(seed)
    total_clean_train = 0
    total_noisy_train = 0
    for sample_id in train_ids:
        arrays = load_npz(annotation_path(clean_root, sample_id))
        total_clean_train += int(arrays["points"].reshape(-1, 2).shape[0])
        noisy = perturb_arrays(arrays, perturbation, rng, k=k)
        total_noisy_train += int(noisy["points"].reshape(-1, 2).shape[0])
        save_npz(annotation_path(out_root, sample_id), noisy)

    for sample_id in eval_ids:
        arrays = load_npz(annotation_path(clean_root, sample_id))
        save_npz(annotation_path(out_root, sample_id), arrays)

    manifest = {
        "clean_root": str(clean_root),
        "output_root": str(out_root),
        "perturbation": perturbation.name,
        "kind": perturbation.kind,
        "level": perturbation.level,
        "label": perturbation.label,
        "seed": None if perturbation.kind == "clean" else seed,
        "train_files": len(train_ids),
        "eval_files": len(eval_ids),
        "clean_train_points": total_clean_train,
        "noisy_train_points": total_noisy_train,
        "dropped_train_points": total_clean_train - total_noisy_train,
        "mean_mnn_knn_k": k,
        "notes": "Only train split annotations are perturbed; eval split annotations are clean copies.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def assert_equal_array(a: np.ndarray, b: np.ndarray, message: str) -> None:
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError(message)


def assert_close_array(a: np.ndarray, b: np.ndarray, message: str, atol: float = 1e-3) -> None:
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(message)


def is_subsequence_rows(sub: np.ndarray, full: np.ndarray) -> bool:
    pos = 0
    for row in sub:
        while pos < len(full) and not np.array_equal(row, full[pos]):
            pos += 1
        if pos == len(full):
            return False
        pos += 1
    return True


def sanity_check_train_sample(clean: dict[str, np.ndarray], noisy: dict[str, np.ndarray], perturbation: Perturbation, k: int, sample_id: str) -> None:
    clean_points = clean["points"].astype(np.float32).reshape(-1, 2)
    noisy_points = noisy["points"].astype(np.float32).reshape(-1, 2)
    clean_labels = clean["labels"].astype(np.int64).reshape(-1)
    noisy_labels = noisy["labels"].astype(np.int64).reshape(-1)
    noisy_mnn = noisy["mean_mnn"].astype(np.float32).reshape(-1)
    width, height = noisy["image_wh"].astype(np.int64).reshape(2)

    if len(noisy_points) != len(noisy_labels) or len(noisy_points) != len(noisy_mnn):
        raise AssertionError(f"{sample_id}: points, labels, and mean_mnn lengths differ")
    if len(noisy_points) and not np.isfinite(noisy_points).all():
        raise AssertionError(f"{sample_id}: noisy points contain non-finite values")
    if len(noisy_mnn) and not np.isfinite(noisy_mnn).all():
        raise AssertionError(f"{sample_id}: mean_mnn contains non-finite values")
    if len(noisy_points) and perturbation.kind in {"abs_jitter", "rel_jitter"}:
        if noisy_points[:, 0].min() < -1e-4 or noisy_points[:, 0].max() > width - 1 + 1e-4:
            raise AssertionError(f"{sample_id}: x coordinates are out of image bounds")
        if noisy_points[:, 1].min() < -1e-4 or noisy_points[:, 1].max() > height - 1 + 1e-4:
            raise AssertionError(f"{sample_id}: y coordinates are out of image bounds")
    if "count" in noisy and int(noisy["count"].reshape(-1)[0]) != len(noisy_points):
        raise AssertionError(f"{sample_id}: count does not match noisy points")

    if perturbation.kind == "clean":
        assert_equal_array(clean["points"], noisy["points"], f"{sample_id}: clean points changed")
        assert_equal_array(clean["mean_mnn"], noisy["mean_mnn"], f"{sample_id}: clean mean_mnn changed")
        assert_equal_array(clean["labels"], noisy["labels"], f"{sample_id}: clean labels changed")
        return

    if perturbation.kind in {"abs_jitter", "rel_jitter"}:
        assert_equal_array(clean_labels, noisy_labels, f"{sample_id}: jitter changed labels")
        if len(clean_points) != len(noisy_points):
            raise AssertionError(f"{sample_id}: jitter changed point count")
        if perturbation.kind == "abs_jitter":
            limit = np.full((len(clean_points), 1), perturbation.level + 1e-4, dtype=np.float32)
        else:
            clean_mnn = clean["mean_mnn"].astype(np.float32).reshape(-1, 1)
            limit = perturbation.level * clean_mnn + 1e-4
        # Some clean datasets contain a few out-of-bound GT points. Jitter is
        # applied first and then clamped, so validate the post-clamp interval.
        upper_bound = np.asarray([width - 1, height - 1], dtype=np.float32)
        min_allowed = np.clip(clean_points - limit, 0, upper_bound)
        max_allowed = np.clip(clean_points + limit, 0, upper_bound)
        if len(noisy_points) and (
            np.any(noisy_points < min_allowed - 1e-4) or np.any(noisy_points > max_allowed + 1e-4)
        ):
            raise AssertionError(f"{sample_id}: jitter exceeds configured radius")
    elif perturbation.kind == "drop":
        expected = len(clean_points) - expected_drop_count(len(clean_points), perturbation.level)
        if len(noisy_points) != expected:
            raise AssertionError(f"{sample_id}: drop count mismatch")
        if not is_subsequence_rows(noisy_points, clean_points):
            raise AssertionError(f"{sample_id}: dropped points are not an ordered subset of clean points")
        if not is_subsequence_rows(noisy_labels.reshape(-1, 1), clean_labels.reshape(-1, 1)):
            raise AssertionError(f"{sample_id}: dropped labels are not an ordered subset of clean labels")

    if len(noisy_points) > 1:
        recomputed = compute_mean_mnn(noisy_points, fallback=noisy_mnn, k=k)
        assert_close_array(noisy_mnn, recomputed, f"{sample_id}: mean_mnn is not consistent with noisy points")


def sanity_check_setting(
    clean_root: Path,
    out_root: Path,
    perturbation: Perturbation,
    train_ids: list[str],
    eval_ids: list[str],
    k: int,
) -> dict[str, int]:
    checked_train = 0
    checked_eval = 0
    for sample_id in train_ids:
        clean = load_npz(annotation_path(clean_root, sample_id))
        noisy = load_npz(annotation_path(out_root, sample_id))
        sanity_check_train_sample(clean, noisy, perturbation, k=k, sample_id=sample_id)
        checked_train += 1

    for sample_id in eval_ids:
        clean = load_npz(annotation_path(clean_root, sample_id))
        noisy = load_npz(annotation_path(out_root, sample_id))
        for key in ("points", "mean_mnn", "labels", "count"):
            if key in clean:
                assert_equal_array(clean[key], noisy[key], f"{sample_id}: eval {key} is not clean")
        checked_eval += 1

    return {"checked_train_files": checked_train, "checked_eval_files": checked_eval}


def main() -> None:
    args = parse_args()
    clean_root = args.root.resolve()
    output_root = args.output_root.resolve() if args.output_root else clean_root.with_name(f"{clean_root.name}_gt_noise")
    eval_splits = args.eval_splits if args.eval_splits is not None else discover_eval_splits(clean_root, args.train_split)
    train_ids = read_split_ids(clean_root, args.train_split)
    eval_ids = []
    for split in eval_splits:
        eval_ids.extend(read_split_ids(clean_root, split))
    eval_ids = sorted(set(eval_ids))
    settings = [PERTURBATION_BY_NAME[name] for name in (args.settings or PERTURBATION_BY_NAME.keys())]

    summaries = []
    for perturbation in settings:
        out_root = setting_root(output_root, perturbation, seed=args.seed)
        if not args.check_only:
            manifest = generate_setting(
                clean_root=clean_root,
                out_root=out_root,
                perturbation=perturbation,
                train_ids=train_ids,
                eval_ids=eval_ids,
                seed=args.seed,
                k=args.knn_k,
                overwrite=args.overwrite,
            )
        else:
            manifest = {"output_root": str(out_root), "perturbation": perturbation.name}

        check = sanity_check_setting(
            clean_root=clean_root,
            out_root=out_root,
            perturbation=perturbation,
            train_ids=train_ids,
            eval_ids=eval_ids,
            k=args.knn_k,
        )
        manifest.update(check)
        summaries.append(manifest)
        print(
            f"[OK] {perturbation.name}: root={out_root} "
            f"train={check['checked_train_files']} eval={check['checked_eval_files']}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
