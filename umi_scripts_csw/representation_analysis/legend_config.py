"""内部 ``--groups`` / ``*.npy`` 文件名 → 论文图例显示名（不改变磁盘上的组键）。"""


GROUP_LEGEND_LABELS: dict[str, str] = {
    "umi_model_umi_data": "umi_handover_data",
    "umi_model_robot_data": "robot_handover_data",
    "umi_model_umi_bagging_data": "umi_bagging_data",
    "umi_model_umi_other_data": "umi_duck_data",
}


def legend_labels_for_groups(internal_names: list[str]) -> list[str]:
    return [GROUP_LEGEND_LABELS.get(n, n) for n in internal_names]
