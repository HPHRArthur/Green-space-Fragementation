#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urban Green Space Monte Carlo Simulation
=========================================
Input:
  - A raster-like image: green pixels = existing green space; white pixels = non-green.
  - Optional planning mask image: white = allowed, black = prohibited.
  - Optional region ID mask (CSV/PNG) can be added in a future extension.

Model:
  - 10 annual steps by default.
  - Green growth: only candidate pixels adjacent to existing green pixels.
  - Ordinary green loss: randomly remove eligible existing green pixels.
  - Explicit fragmentation: attempt cuts through existing patches; a cut is accepted
    only when it splits a connected patch into >=2 components.
  - Planning mask constrains where growth/loss/cuts can happen.
  - Monte Carlo repetitions estimate pixel-level green probability and metric distributions.

Important:
  - The rates are scenario parameters, not automatically inferred from one image.
  - Fragmentation cuts remove green pixels, so cut pixels also reduce total green area.
  - Planning constraints here are spatial masks (hard constraints). If you have
    numeric regional quotas or a formal LP objective, this needs an LP/MILP extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------- Image processing ---------------------------

def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def classify_green(rgb: np.ndarray,
                   green_min: int = 60,
                   green_dominance: int = 20,
                   green_ratio: float = 1.10) -> np.ndarray:
    """
    Classify green pixels robustly:
      G >= green_min, G >= R + green_dominance, G >= B + green_dominance,
      G >= green_ratio * R and G >= green_ratio * B.
    For a clean green/white image this should work well.
    If your green has a specific RGB value, exact-color classification can be used instead.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (
        (g >= green_min)
        & (g >= r + green_dominance)
        & (g >= b + green_dominance)
        & (g >= green_ratio * np.maximum(r, 1))
        & (g >= green_ratio * np.maximum(b, 1))
    )


def load_binary_map(path: str,
                    green_min: int = 60,
                    green_dominance: int = 20,
                    green_ratio: float = 1.10) -> np.ndarray:
    rgb = load_rgb(path)
    mask = classify_green(rgb, green_min, green_dominance, green_ratio)
    if mask.sum() == 0:
        raise ValueError(
            "No green pixels detected. Check that the input image contains green pixels, "
            "or adjust --green-min / --green-dominance / --green-ratio."
        )
    return mask


def load_allowed_mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    """
    Planning mask interpretation:
      - white/light pixels = allowed
      - black/dark pixels = prohibited
    If omitted, all pixels are allowed by the planning mask; growth still follows
    the edge-only rule.
    """
    if path is None:
        return np.ones(shape, dtype=bool)
    rgb = load_rgb(path)
    if rgb.shape[:2] != shape:
        raise ValueError(f"Planning mask size {rgb.shape[:2]} differs from input size {shape}.")
    gray = rgb.mean(axis=2)
    return gray >= 128


def save_binary_map(mask: np.ndarray, path: str, green=(0, 180, 0), white=(255, 255, 255)):
    arr = np.empty((*mask.shape, 3), dtype=np.uint8)
    arr[:] = white
    arr[mask] = green
    Image.fromarray(arr).save(path)


# --------------------------- Spatial operations ---------------------------

def neighbor_count(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Count green neighbors, excluding the cell itself."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    h, w = mask.shape
    count = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in offsets:
        count += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return count


def sample_without_replacement(candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if n <= 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    n = min(int(n), candidates.size)
    return rng.choice(candidates, size=n, replace=False)


def connected_components(mask: np.ndarray, connectivity: int = 8):
    """
    Return list of components, each an array of flattened pixel indices.
    Python BFS implementation: correct but can be slow for very large dense images.
    """
    h, w = mask.shape
    seen = np.zeros(mask.size, dtype=bool)
    flat = mask.ravel()
    if connectivity == 8:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    for start in np.flatnonzero(flat):
        if seen[start]:
            continue
        q = deque([int(start)])
        seen[start] = True
        comp = []
        while q:
            idx = q.popleft()
            comp.append(idx)
            y, x = divmod(idx, w)
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    ni = ny * w + nx
                    if flat[ni] and not seen[ni]:
                        seen[ni] = True
                        q.append(ni)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def edge_candidates(mask: np.ndarray, allowed: np.ndarray, connectivity: int = 8) -> np.ndarray:
    neighbors = neighbor_count(mask, connectivity)
    return np.flatnonzero((~mask & allowed & (neighbors > 0)).ravel())


def add_green(mask: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out.ravel()[flat_indices] = True
    return out


def remove_green(mask: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out.ravel()[flat_indices] = False
    return out


# --------------------------- Simulation mechanics ---------------------------

def attempt_patch_splits(mask: np.ndarray,
                         allowed_loss: np.ndarray,
                         split_probability: float,
                         rng: np.random.Generator,
                         connectivity: int = 8,
                         max_attempts_per_patch: int = 3) -> tuple[np.ndarray, int, int]:
    """
    For each existing patch, attempt a split with probability split_probability.
    To create a simple local cut:
      - pick a random pixel in the patch that is allowed to be removed;
      - temporarily remove it;
      - accept only if the number of connected components increases.
    For wider cuts, this function can be replaced by line/path-based cutting.
    Returns (updated_mask, accepted_split_count, removed_pixel_count).
    This single-pixel version only splits a patch where an articulation pixel exists.
    """
    if split_probability <= 0:
        return mask, 0, 0

    out = mask.copy()
    accepted = 0
    removed = 0
    components = connected_components(out, connectivity)

    for comp in components:
        if comp.size < 3 or rng.random() >= split_probability:
            continue

        # Try random eligible pixels in this component.
        eligible = comp[allowed_loss.ravel()[comp]]
        if eligible.size == 0:
            continue
        rng.shuffle(eligible)
        success = False

        for idx in eligible[:max_attempts_per_patch]:
            candidate = out.copy()
            candidate.ravel()[idx] = False
            new_components = connected_components(candidate, connectivity)
            # Count components touching the original patch after removing candidate.
            comp_set = set(comp.tolist())
            touched = 0
            for new_comp in new_components:
                if np.any(np.isin(new_comp, comp)):
                    touched += 1
            if touched >= 2:
                out = candidate
                accepted += 1
                removed += 1
                success = True
                break
        # If no articulation pixel is found, no split is recorded.
    return out, accepted, removed


def apply_spatial_constraints(old: np.ndarray,
                              proposed: np.ndarray,
                              allowed_growth: np.ndarray,
                              allowed_loss: np.ndarray) -> np.ndarray:
    """
    Enforce spatial masks:
      - newly green cells must be allowed growth locations;
      - removed green cells must be allowed loss locations.
    Also preserves old state at forbidden-to-change pixels.
    """
    out = proposed.copy()
    grew = (~old) & out
    lost = old & (~out)
    out[grew & (~allowed_growth)] = False
    out[lost & (~allowed_loss)] = True
    return out


def metrics(mask: np.ndarray, connectivity: int = 8) -> dict:
    area = int(mask.sum())
    comps = connected_components(mask, connectivity)
    sizes = [int(c.size) for c in comps]
    n_patches = len(sizes)
    mean_patch = float(area / n_patches) if n_patches else 0.0
    largest_share = float(max(sizes) / mask.size) if sizes else 0.0

    # Count boundary edges between green and non-green/outside.
    # Edge density is reported in pixel-edge units per image pixel.
    padded = np.pad(mask, 1, mode="constant")
    h, w = mask.shape
    edge = 0
    edge += np.count_nonzero(padded[1:1+h, 1:1+w] != padded[1:1+h, 2:2+w+1])
    edge += np.count_nonzero(padded[1:1+h, 1:1+w] != padded[1:1+h, 0:w])
    edge += np.count_nonzero(padded[1:1+h, 1:1+w] != padded[2:2+h+1, 1:1+w])
    edge += np.count_nonzero(padded[1:1+h, 1:1+w] != padded[0:h, 1:1+w])
    return {
        "green_pixels": area,
        "green_fraction": area / mask.size,
        "patch_count": n_patches,
        "mean_patch_pixels": mean_patch,
        "largest_patch_fraction_of_landscape": largest_share,
        "boundary_edge_count_4_neighbor": int(edge),
    }


def simulate_one(initial: np.ndarray,
                 allowed_growth: np.ndarray,
                 allowed_loss: np.ndarray,
                 growth_rate: float,
                 loss_rate: float,
                 split_probability: float,
                 years: int,
                 seed: int,
                 connectivity: int = 8):
    rng = np.random.default_rng(seed)
    current = initial.copy()
    maps = [current.copy()]
    yearly_metrics = []
    split_events_total = 0
    split_pixels_total = 0

    for year in range(1, years + 1):
        area_before = int(current.sum())

        # A. Growth: rate is based on green area at start of the year.
        target_growth = int(math.floor(growth_rate * area_before))
        candidates = edge_candidates(current, allowed_growth, connectivity)
        growth_sites = sample_without_replacement(candidates, target_growth, rng)
        after_growth = add_green(current, growth_sites)

        # B. Ordinary loss: rate based on the post-growth area.
        area_after_growth = int(after_growth.sum())
        target_loss = int(math.floor(loss_rate * area_after_growth))
        loss_candidates = np.flatnonzero((after_growth & allowed_loss).ravel())
        loss_sites = sample_without_replacement(loss_candidates, target_loss, rng)
        after_loss = remove_green(after_growth, loss_sites)

        # C. Explicit splitting. Cut pixels must be allowed loss locations.
        after_split, accepted_splits, split_removed = attempt_patch_splits(
            after_loss, allowed_loss, split_probability, rng, connectivity
        )
        split_events_total += accepted_splits
        split_pixels_total += split_removed

        # D. Enforce the hard spatial planning masks.
        current = apply_spatial_constraints(
            old=current,
            proposed=after_split,
            allowed_growth=allowed_growth,
            allowed_loss=allowed_loss
        )

        row = {
            "year": year,
            "target_growth_pixels": target_growth,
            "actual_growth_pixels": int(np.count_nonzero((~maps[-1]) & after_growth)),
            "target_loss_pixels": target_loss,
            "accepted_split_events_this_year": accepted_splits,
            "split_pixels_removed_this_year": split_removed,
        }
        row.update(metrics(current, connectivity))
        yearly_metrics.append(row)
        maps.append(current.copy())

    return maps, yearly_metrics, split_events_total, split_pixels_total


# --------------------------- Output ---------------------------

def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of urban green-space growth and fragmentation."
    )
    parser.add_argument("--input", required=True, help="Input image: green pixels = green space, white = non-green.")
    parser.add_argument("--output", default="results", help="Output directory.")
    parser.add_argument("--planning-mask", default=None,
                        help="Optional mask image: white/light = allowed to change; black/dark = prohibited.")
    parser.add_argument("--growth-rate", type=float, default=0.03,
                        help="Annual growth target as fraction of current green area (default 0.03).")
    parser.add_argument("--loss-rate", type=float, default=0.02,
                        help="Annual ordinary green loss target as fraction of post-growth green area (default 0.02).")
    parser.add_argument("--split-probability", type=float, default=0.10,
                        help="Annual probability of attempting a split per patch (default 0.10).")
    parser.add_argument("--years", type=int, default=10, help="Simulation horizon (default 10 years).")
    parser.add_argument("--runs", type=int, default=100,
                        help="Monte Carlo repetitions (default 100; increase after testing).")
    parser.add_argument("--seed", type=int, default=20261029, help="Random seed for reproducibility.")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Pixel connectivity used to define patches (default 8).")
    parser.add_argument("--green-min", type=int, default=60, help="Minimum green-channel threshold.")
    parser.add_argument("--green-dominance", type=int, default=20,
                        help="Green channel must exceed red/blue by this amount.")
    parser.add_argument("--green-ratio", type=float, default=1.10,
                        help="Green channel must be at least this ratio of red and blue.")
    parser.add_argument("--save-every-run", action="store_true",
                        help="Save each run's year-10 map; otherwise save only the first run's yearly maps.")
    args = parser.parse_args()

    if not (0 <= args.growth_rate <= 1):
        raise ValueError("--growth-rate must be between 0 and 1.")
    if not (0 <= args.loss_rate <= 1):
        raise ValueError("--loss-rate must be between 0 and 1.")
    if not (0 <= args.split_probability <= 1):
        raise ValueError("--split-probability must be between 0 and 1.")
    if args.years < 1 or args.runs < 1:
        raise ValueError("--years and --runs must be >= 1.")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    maps_dir = outdir / "maps"
    maps_dir.mkdir(exist_ok=True)

    initial = load_binary_map(
        args.input, args.green_min, args.green_dominance, args.green_ratio
    )
    allowed = load_allowed_mask(args.planning_mask, initial.shape)

    # In this simple version the same planning mask constrains both additions and removals.
    # If policy allows growth and loss in different areas, use separate masks.
    allowed_growth = allowed.copy()
    allowed_loss = allowed.copy()

    save_binary_map(initial, str(outdir / "classified_initial.png"))
    initial_summary = {
        "input_image": str(Path(args.input).resolve()),
        "shape_height_width": list(initial.shape),
        "initial_green_pixels": int(initial.sum()),
        "initial_green_fraction": float(initial.mean()),
        "growth_rate": args.growth_rate,
        "loss_rate": args.loss_rate,
        "split_probability_per_patch_per_year": args.split_probability,
        "years": args.years,
        "runs": args.runs,
        "connectivity": args.connectivity,
        "seed": args.seed,
        "planning_mask": args.planning_mask,
        "note": "Rates are user-specified scenario parameters, not estimated from one map."
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(initial_summary, f, ensure_ascii=False, indent=2)

    all_metric_rows = []
    all_final_maps = np.zeros((args.runs, *initial.shape), dtype=np.uint8)
    first_run_maps = None

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        maps, rows, total_splits, total_split_pixels = simulate_one(
            initial=initial,
            allowed_growth=allowed_growth,
            allowed_loss=allowed_loss,
            growth_rate=args.growth_rate,
            loss_rate=args.loss_rate,
            split_probability=args.split_probability,
            years=args.years,
            seed=run_seed,
            connectivity=args.connectivity,
        )
        if run_idx == 0:
            first_run_maps = maps
        for row in rows:
            row["run"] = run_idx + 1
            row["seed"] = run_seed
            all_metric_rows.append(row)

        all_final_maps[run_idx] = maps[-1].astype(np.uint8)

        if args.save_every_run:
            save_binary_map(
                maps[-1],
                str(maps_dir / f"run_{run_idx+1:04d}_year_{args.years}.png")
            )

    # Save year-by-year maps for the first realization.
    for year, mask in enumerate(first_run_maps):
        save_binary_map(mask, str(maps_dir / f"example_run_year_{year:02d}.png"))

    write_csv(all_metric_rows, str(outdir / "yearly_metrics_all_runs.csv"))

    # Pixel-level probability of being green at the final year.
    probability = all_final_maps.mean(axis=0)
    prob_img = np.clip(probability * 255, 0, 255).astype(np.uint8)
    Image.fromarray(prob_img, mode="L").save(outdir / "year10_green_probability_grayscale.png")

    # Probability threshold map: pixel is green if green in >=50% of simulations.
    consensus = probability >= 0.5
    save_binary_map(consensus, str(outdir / "year10_consensus_50pct.png"))

    # Summary of each metric by year, using all runs.
    summary_rows = []
    metric_names = [
        "green_pixels", "green_fraction", "patch_count",
        "mean_patch_pixels", "largest_patch_fraction_of_landscape",
        "boundary_edge_count_4_neighbor"
    ]
    for year in range(1, args.years + 1):
        year_rows = [r for r in all_metric_rows if r["year"] == year]
        entry = {"year": year}
        for name in metric_names:
            values = np.asarray([float(r[name]) for r in year_rows], dtype=float)
            entry[name + "_mean"] = float(values.mean())
            entry[name + "_median"] = float(np.median(values))
            entry[name + "_p05"] = float(np.quantile(values, 0.05))
            entry[name + "_p95"] = float(np.quantile(values, 0.95))
        summary_rows.append(entry)
    write_csv(summary_rows, str(outdir / "metrics_summary_by_year.csv"))

    # Plot trajectories of core metrics.
    years = np.arange(1, args.years + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("green_pixels", "Green pixels"),
        ("patch_count", "Number of patches"),
        ("mean_patch_pixels", "Mean patch size (pixels)"),
        ("largest_patch_fraction_of_landscape", "Largest patch / landscape"),
    ]
    for ax, (metric, title) in zip(axes.flat, plot_specs):
        means, lo, hi = [], [], []
        for y in years:
            vals = np.asarray([
                r[metric] for r in all_metric_rows if r["year"] == y
            ], dtype=float)
            means.append(vals.mean())
            lo.append(np.quantile(vals, 0.05))
            hi.append(np.quantile(vals, 0.95))
        ax.plot(years, means, marker="o")
        ax.fill_between(years, lo, hi, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "metric_trajectories.png", dpi=160)
    plt.close(fig)

    print("\nSimulation completed.")
    print(f"Image size: {initial.shape[1]} x {initial.shape[0]}")
    print(f"Initial green pixels: {int(initial.sum())} ({initial.mean():.2%})")
    print(f"Runs: {args.runs}; years: {args.years}")
    print(f"Outputs saved to: {outdir.resolve()}")
    print("Key files:")
    print("  classified_initial.png")
    print("  maps/example_run_year_00.png ... maps/example_run_year_10.png")
    print("  year10_green_probability_grayscale.png")
    print("  year10_consensus_50pct.png")
    print("  yearly_metrics_all_runs.csv")
    print("  metrics_summary_by_year.csv")
    print("  metric_trajectories.png")
    print("  run_config.json")


if __name__ == "__main__":
    main()
