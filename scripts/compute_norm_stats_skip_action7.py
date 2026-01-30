"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


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
    print(f"Creating Torch dataset from repo_id: {data_config.repo_id}")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    print(f"Dataset length: {len(dataset)}")
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    print(f"Transformed dataset length: {len(dataset)}")
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    print(f"Number of batches: {num_batches}, shuffle: {shuffle}")
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    print(f"Created Torch data loader with {num_workers} workers")
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
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
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


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"Computing normalization stats for data config: {data_config}")
    
    if data_config.rlds_data_dir is not None:
        print("Using RLDS data loader")
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        print(f"RLDS data dir: {data_config.rlds_data_dir}")
    else:
        print("Using Torch data loader")
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
        print(f"Torch data repo id: {data_config.repo_id}") 
    print(f"Number of batches: {num_batches}")
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    
    # =========================================================================
    # 手动覆盖 Gripper 统计量 (不做数据驱动的归一化，改为固定范围)
    if "actions" in norm_stats:
        print("\n[Manual Override] Detecting 'actions' stats...")
        act_stats = norm_stats["actions"]
        
        # 假设夹爪是动作的第x个维度 (第7维)
        gripper_idx = 6 
        
        print(f"[Manual Override] Forcing Gripper (dim {gripper_idx}) stats to safe values [-1, 1].")
        
        # 为什么设为 q01=-1, q99=1 ?
        # 归一化公式: x_norm = (x - q01) / (q99 - q01) * 2 - 1
        # 当 x=1 时: (1 - (-1)) / 2 * 2 - 1 = 1
        # 当 x=-1 时: (-1 - (-1)) / 2 * 2 - 1 = -1
        # 当 x=0 时: (0 - (-1)) / 2 * 2 - 1 = 0
        # 结论：这等效于不做归一化 (Identity)，或者说把 [-1, 1] 映射到 [-1, 1]
        
        print("length of action stats arrays:", len(act_stats.q01), len(act_stats.q99))
        act_stats.q01[gripper_idx] = -1.0
        act_stats.q99[gripper_idx] = 1.0
        
        # 均值设为0，方差设为1 (防止 Z-Score 归一化爆炸)
        act_stats.mean[gripper_idx] = 0.0
        act_stats.std[gripper_idx] = 1.0
        
    # =========================================================================

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
