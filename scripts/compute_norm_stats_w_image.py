"""Compute normalization statistics for a config (+ full-data histograms).

This script computes normalization statistics (mean/std, etc.) and saves them to the
config assets directory. Additionally, it computes FULL-DATA, FULL-RANGE histograms
for each dimension of `state` and `actions`, and writes them to multi-page PDFs.

Histogram method:
- Pass 1: compute per-dim min/max over the ENTIRE dataset
- Pass 2: accumulate exact histogram bin counts over the ENTIRE dataset
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def to_2d_lastdim(arr: np.ndarray) -> np.ndarray:
    """Flatten everything except last dim. Return [N, D]."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    D = arr.shape[-1]
    return arr.reshape(-1, D)


def finite_rows(x2: np.ndarray) -> np.ndarray:
    """Keep rows with all finite values to avoid NaN/Inf breaking min/max/hist."""
    if x2.size == 0:
        return x2
    mask = np.all(np.isfinite(x2), axis=1)
    return x2[mask]


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def make_dataloader(config, data_config, max_frames: int | None):
    """Helper to recreate dataloader (needed for pass2)."""
    if data_config.rlds_data_dir is not None:
        return create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    return create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )


def write_fullrange_histograms(
    config,
    data_config,
    num_batches: int,
    max_frames: int | None,
    keys: list[str],
    bins: int,
):
    # -------- Pass 1: full min/max over entire dataset --------
    data_loader1, _ = make_dataloader(config, data_config, max_frames)

    minmax: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for batch in tqdm.tqdm(data_loader1, total=num_batches, desc="Pass1 min/max (FULL)"):
        for key in keys:
            x2 = finite_rows(to_2d_lastdim(np.asarray(batch[key])))
            if x2.size == 0:
                continue
            if key not in minmax:
                minmax[key] = (np.min(x2, axis=0), np.max(x2, axis=0))
            else:
                mn, mx = minmax[key]
                minmax[key] = (
                    np.minimum(mn, np.min(x2, axis=0)),
                    np.maximum(mx, np.max(x2, axis=0)),
                )

    # Prepare edges + counts
    bin_edges: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}

    for key in keys:
        if key not in minmax:
            continue
        mn, mx = minmax[key]
        D = mn.shape[0]

        # Handle degenerate dims where mn==mx
        same = (mx <= mn)
        span = np.where(same, 1.0, (mx - mn))
        mn2 = np.where(same, mn - 0.5 * span, mn)
        mx2 = np.where(same, mx + 0.5 * span, mx)

        edges = np.stack([np.linspace(mn2[d], mx2[d], bins + 1) for d in range(D)], axis=0)  # [D, bins+1]
        bin_edges[key] = edges
        counts[key] = np.zeros((D, bins), dtype=np.int64)

    # -------- Pass 2: exact histogram counts over entire dataset --------
    data_loader2, _ = make_dataloader(config, data_config, max_frames)

    for batch in tqdm.tqdm(data_loader2, total=num_batches, desc="Pass2 hist (FULL)"):
        for key in keys:
            if key not in counts:
                continue
            x2 = finite_rows(to_2d_lastdim(np.asarray(batch[key])))
            if x2.size == 0:
                continue

            edges = bin_edges[key]  # [D, bins+1]
            cnt = counts[key]       # [D, bins]
            D = cnt.shape[0]

            for d in range(D):
                e = edges[d]
                idx = np.searchsorted(e, x2[:, d], side="right") - 1
                idx = np.clip(idx, 0, bins - 1)
                cnt[d] += np.bincount(idx, minlength=bins)

    # -------- Write PDFs --------
    output_path = config.assets_dirs / data_config.repo_id
    hist_dir = output_path / "histograms_full"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        if key not in counts:
            continue
        out_pdf = hist_dir / f"{key}_fullrange_histograms.pdf"
        with PdfPages(out_pdf) as pdf:
            D = counts[key].shape[0]
            for d in range(D):
                e = bin_edges[key][d]
                c = counts[key][d]
                centers = 0.5 * (e[:-1] + e[1:])

                plt.figure()
                plt.bar(centers, c, width=(e[1] - e[0]))
                plt.title(f"{key} dim {d} (full range)")
                plt.xlabel("Value")
                plt.ylabel("Count")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Wrote full-range histograms: {out_pdf}")


def main(
    config_name: str,
    max_frames: int | None = None,
    hist_bins: int = 80,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = make_dataloader(config, data_config, max_frames)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Full-data full-range histograms
    write_fullrange_histograms(
        config=config,
        data_config=data_config,
        num_batches=num_batches,
        max_frames=max_frames,
        keys=keys,
        bins=hist_bins,
    )


if __name__ == "__main__":
    tyro.cli(main)
