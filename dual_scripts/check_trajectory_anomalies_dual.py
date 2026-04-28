#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹异常检测脚本 (dual_franka 格式适配版)

检测所有episodes的state轨迹异常，包括：
- CHECK-004: 同步问题 (增强版，分类严重/中等/轻微)
- CHECK-007: 静态轨迹 (SLAM丢失)
- CHECK-008: 少帧异常
- CHECK-009: 末端跳变
- CHECK-010: Clamp异常

适配 dual_franka parquet 格式:
- observation/state/left_ee_pos_quat
- observation/state/right_ee_pos_quat
- observation/state/gripper_pose

Usage:
    python check_trajectory_anomalies_dual.py \
        --data_dir ~/Downloads/dual_franka_cylinder_handover_20260424_intermediate \
        --output_dir ./anomaly_reports
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AnomalySeverity(Enum):
    """异常严重程度"""
    CRITICAL = "critical"    # 必须剔除
    WARNING = "warning"      # 建议剔除
    MINOR = "minor"          # 可保留但记录
    OK = "ok"


class AnomalyType(Enum):
    """异常类型"""
    # 同步问题 (CHECK-004增强)
    SYNC_CRITICAL = "sync_critical"    # 单手<5帧或静态
    SYNC_WARNING = "sync_warning"      # 差异>10帧
    SYNC_MINOR = "sync_minor"          # 差异3-10帧

    # State异常 (CHECK-007~010)
    STATIC_TRAJECTORY = "static_trajectory"    # CHECK-007
    TOO_FEW_FRAMES = "too_few_frames"          # CHECK-008
    END_JUMP = "end_jump"                      # CHECK-009
    CLAMP_ABNORMAL = "clamp_abnormal"          # CHECK-010

    OK = "ok"


@dataclass
class AnomalyReport:
    """单个episode的异常报告"""
    episode: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    details: Dict

    def to_dict(self):
        # 转换details中的numpy类型为Python原生类型
        clean_details = {}
        for k, v in self.details.items():
            if isinstance(v, np.bool_):
                clean_details[k] = bool(v)
            elif isinstance(v, (np.integer, np.int64)):
                clean_details[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean_details[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_details[k] = v.tolist()
            else:
                clean_details[k] = v

        return {
            "episode": self.episode,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "details": clean_details
        }


# ============== 检测函数 ==============

def check_sync_anomaly(left_info: Dict, right_info: Dict,
                       avg_frames: float) -> Tuple[AnomalyType, str, Dict]:
    """
    CHECK-004增强: 同步问题检测 (分类严重/中等/轻微)

    判定标准:
    - CRITICAL: 单手帧数<5 或 位置方差<0.001 (静态轨迹)
    - WARNING: 帧数差异>10帧
    - MINOR: 帧数差异3-10帧
    """
    left_frames = left_info["num_frames"]
    right_frames = right_info["num_frames"]

    # 加载轨迹数据检查静态
    left_static = left_info.get("is_static", False)
    right_static = right_info.get("is_static", False)

    # CRITICAL: 少帧或静态
    if left_frames < 5 or right_frames < 5 or left_static or right_static:
        return (
            AnomalyType.SYNC_CRITICAL,
            f"严重同步问题: 左手{left_frames}帧{'(静态)' if left_static else ''}, "
            f"右手{right_frames}帧{'(静态)' if right_static else ''}",
            {"left_frames": left_frames, "right_frames": right_frames,
             "left_static": left_static, "right_static": right_static}
        )

    # 计算差异
    frame_diff = abs(left_frames - right_frames)
    threshold_warning = 10  # 绝对帧数差
    threshold_minor = 3

    if frame_diff > threshold_warning:
        return (
            AnomalyType.SYNC_WARNING,
            f"中等同步问题: 差异{frame_diff}帧 (>{threshold_warning})",
            {"left_frames": left_frames, "right_frames": right_frames, "diff": frame_diff}
        )
    elif frame_diff > threshold_minor:
        return (
            AnomalyType.SYNC_MINOR,
            f"轻微同步问题: 差异{frame_diff}帧 ({threshold_minor}~{threshold_warning})",
            {"left_frames": left_frames, "right_frames": right_frames, "diff": frame_diff}
        )

    return AnomalyType.OK, "", {}


def check_static_trajectory(positions: np.ndarray) -> Tuple[bool, float]:
    """
    CHECK-007: 静态轨迹检测

    判定: 位置方差 < 0.001
    说明: 轨迹几乎无变化，疑似SLAM追踪丢失
    """
    if len(positions) < 2:
        return True, 0.0

    pos_variance = np.var(positions, axis=0)
    max_variance = np.max(pos_variance)

    threshold = 0.001
    is_static = max_variance < threshold

    return is_static, max_variance


def check_end_jump(positions: np.ndarray) -> Tuple[bool, float]:
    """
    CHECK-009: 末端跳变检测

    判定: 最后3帧位移 > 0.1m
    说明: 轨迹末端有异常跳变，可能录制提前终止或SLAM丢失
    """
    if len(positions) < 5:
        return False, 0.0

    # 计算最后3帧的位移
    last_pos = positions[-1]
    prev_pos = positions[-4]  # 倒数第4帧

    displacement = np.linalg.norm(last_pos - prev_pos)
    threshold = 0.1  # 10cm

    has_jump = displacement > threshold

    return has_jump, displacement


def check_clamp_abnormal(clamps: np.ndarray) -> Tuple[bool, int]:
    """
    CHECK-010: Clamp异常检测

    判定: 夹爪变化次数 < 2
    说明: handover任务应有2-3次开合(张→合→张)
    """
    if len(clamps) < 3:
        return True, 0

    # 检测显著变化 (阈值=0.01，排除噪声)
    changes = 0
    for i in range(1, len(clamps)):
        if abs(clamps[i] - clamps[i-1]) > 0.01:
            changes += 1

    # handover应有: 张开→闭合(取物)→张开(递送)→闭合(放下)→张开
    # 最少应有2次变化
    threshold = 2
    is_abnormal = changes < threshold

    return is_abnormal, changes


def analyze_single_hand(positions: np.ndarray, clamps: np.ndarray) -> Dict:
    """分析单只手的数据"""
    num_frames = len(positions)

    # CHECK-007: 静态轨迹
    is_static, pos_variance = check_static_trajectory(positions)

    # CHECK-009: 末端跳变
    has_end_jump, end_displacement = check_end_jump(positions)

    # CHECK-010: Clamp异常
    clamp_abnormal, clamp_changes = check_clamp_abnormal(clamps)

    return {
        "num_frames": num_frames,
        "is_static": is_static,
        "pos_variance": float(pos_variance),
        "has_end_jump": has_end_jump,
        "end_displacement": float(end_displacement),
        "clamp_abnormal": clamp_abnormal,
        "clamp_changes": clamp_changes,
        "valid": True
    }


def load_episode_data(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """
    从parquet文件加载episode数据

    Returns:
        left_positions: [N, 3] 左手位置
        right_positions: [N, 3] 右手位置
        left_clamps: [N] 左手夹爪
        right_clamps: [N] 右手夹爪
        valid: 是否成功加载
        error: 错误信息
    """
    try:
        df = pd.read_parquet(parquet_path)

        # 提取左手位置 (前3维)
        left_pos_quat = np.array(df['observation/state/left_ee_pos_quat'].tolist())
        left_positions = left_pos_quat[:, :3]

        # 提取右手位置 (前3维)
        right_pos_quat = np.array(df['observation/state/right_ee_pos_quat'].tolist())
        right_positions = right_pos_quat[:, :3]

        # 提取夹爪数据
        gripper_pose = np.array(df['observation/state/gripper_pose'].tolist())
        left_clamps = gripper_pose[:, 0]
        right_clamps = gripper_pose[:, 1]

        return left_positions, right_positions, left_clamps, right_clamps, True, ""
    except Exception as e:
        return None, None, None, None, False, str(e)


def analyze_episode(data_dir: Path, episode_name: str, avg_frames: float) -> List[AnomalyReport]:
    """分析单个episode的所有异常"""
    reports = []

    parquet_path = data_dir / f"{episode_name}.parquet"

    # 加载数据
    left_positions, right_positions, left_clamps, right_clamps, valid, error = load_episode_data(parquet_path)

    if not valid:
        reports.append(AnomalyReport(
            episode=episode_name,
            anomaly_type=AnomalyType.SYNC_CRITICAL,
            severity=AnomalySeverity.CRITICAL,
            description=f"无法加载数据: {error}",
            details={"error": error}
        ))
        return reports

    # 分析左右手
    left_info = analyze_single_hand(left_positions, left_clamps)
    right_info = analyze_single_hand(right_positions, right_clamps)

    # CHECK-004: 同步问题
    sync_type, sync_desc, sync_details = check_sync_anomaly(left_info, right_info, avg_frames)
    if sync_type != AnomalyType.OK:
        severity = {
            AnomalyType.SYNC_CRITICAL: AnomalySeverity.CRITICAL,
            AnomalyType.SYNC_WARNING: AnomalySeverity.WARNING,
            AnomalyType.SYNC_MINOR: AnomalySeverity.MINOR
        }.get(sync_type, AnomalySeverity.WARNING)

        reports.append(AnomalyReport(
            episode=episode_name,
            anomaly_type=sync_type,
            severity=severity,
            description=sync_desc,
            details=sync_details
        ))

    # 分别检查每只手的state异常
    for hand, info in [("left", left_info), ("right", right_info)]:
        if not info.get("valid", False):
            continue

        # CHECK-007: 静态轨迹
        if info["is_static"] and info["num_frames"] >= 5:  # 排除已在sync中报告的
            reports.append(AnomalyReport(
                episode=episode_name,
                anomaly_type=AnomalyType.STATIC_TRAJECTORY,
                severity=AnomalySeverity.WARNING,
                description=f"{hand}手轨迹静态，方差{info['pos_variance']:.6f}",
                details={"hand": hand, "variance": info['pos_variance'], "frames": info['num_frames']}
            ))

        # CHECK-008: 少帧
        if info["num_frames"] < 5 and not any(r.anomaly_type == AnomalyType.SYNC_CRITICAL for r in reports if r.episode == episode_name):
            reports.append(AnomalyReport(
                episode=episode_name,
                anomaly_type=AnomalyType.TOO_FEW_FRAMES,
                severity=AnomalySeverity.CRITICAL,
                description=f"{hand}手仅{info['num_frames']}帧",
                details={"hand": hand, "frames": info['num_frames']}
            ))

        # CHECK-009: 末端跳变
        if info["has_end_jump"]:
            reports.append(AnomalyReport(
                episode=episode_name,
                anomaly_type=AnomalyType.END_JUMP,
                severity=AnomalySeverity.WARNING,
                description=f"{hand}手末端跳变{info['end_displacement']:.3f}m",
                details={"hand": hand, "displacement": info['end_displacement']}
            ))

        # CHECK-010: Clamp异常
        if info["clamp_abnormal"]:
            reports.append(AnomalyReport(
                episode=episode_name,
                anomaly_type=AnomalyType.CLAMP_ABNORMAL,
                severity=AnomalySeverity.MINOR,
                description=f"{hand}手夹爪变化异常，仅{info['clamp_changes']}次",
                details={"hand": hand, "changes": info['clamp_changes']}
            ))

    # 如果没有异常，添加OK记录
    if not reports:
        reports.append(AnomalyReport(
            episode=episode_name,
            anomaly_type=AnomalyType.OK,
            severity=AnomalySeverity.OK,
            description="无异常",
            details={}
        ))

    return reports


def main():
    parser = argparse.ArgumentParser(description="检测所有episodes的state轨迹异常 (dual_franka格式)")
    parser.add_argument("--data_dir", required=True, help="数据目录")
    parser.add_argument("--output_dir", default="./anomaly_reports", help="报告输出目录")
    parser.add_argument("--min_frames", type=int, default=5, help="最少帧数阈值")
    parser.add_argument("--static_variance_threshold", type=float, default=0.001, help="静态轨迹方差阈值")
    parser.add_argument("--end_jump_threshold", type=float, default=0.1, help="末端跳变阈值(m)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描所有episodes
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    print(f"[INFO] 发现 {len(parquet_files)} 个episodes")

    # 先计算平均帧数
    frame_counts = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            frame_counts.append(len(df))
        except:
            pass

    avg_frames = np.mean(frame_counts) if frame_counts else 0
    print(f"[INFO] 平均帧数: {avg_frames:.1f}")

    # 分析所有episodes
    all_reports = []
    for i, parquet_file in enumerate(parquet_files):
        episode_name = parquet_file.stem  # e.g., "episode_000000"
        reports = analyze_episode(data_dir, episode_name, avg_frames)
        all_reports.extend(reports)

        if (i + 1) % 50 == 0:
            print(f"[PROGRESS] 已处理 {i+1}/{len(parquet_files)} episodes")

    # 生成报告
    summary = {
        "total_episodes": len(parquet_files),
        "avg_frames": float(avg_frames),
        "thresholds": {
            "min_frames": args.min_frames,
            "static_variance": args.static_variance_threshold,
            "end_jump": args.end_jump_threshold
        },
        "statistics": {
            "critical": len([r for r in all_reports if r.severity == AnomalySeverity.CRITICAL]),
            "warning": len([r for r in all_reports if r.severity == AnomalySeverity.WARNING]),
            "minor": len([r for r in all_reports if r.severity == AnomalySeverity.MINOR]),
            "ok": len([r for r in all_reports if r.severity == AnomalySeverity.OK])
        },
        "by_type": {}
    }

    # 按类型统计
    for anomaly_type in AnomalyType:
        count = len([r for r in all_reports if r.anomaly_type == anomaly_type])
        if count > 0:
            summary["by_type"][anomaly_type.value] = count

    # 按严重程度分类报告
    classified_reports = {
        "critical": [r.to_dict() for r in all_reports if r.severity == AnomalySeverity.CRITICAL],
        "warning": [r.to_dict() for r in all_reports if r.severity == AnomalySeverity.WARNING],
        "minor": [r.to_dict() for r in all_reports if r.severity == AnomalySeverity.MINOR],
        "ok": [r.to_dict() for r in all_reports if r.severity == AnomalySeverity.OK]
    }

    # 保存完整报告
    full_report = {
        "report_version": "1.1-dual",
        "report_type": "trajectory_anomaly_detection",
        "data_format": "dual_franka_parquet",
        "data_source": str(data_dir),
        "summary": summary,
        "classified_reports": classified_reports
    }

    report_path = output_dir / "trajectory_anomaly_report.json"
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "="*70)
    print("轨迹异常检测报告 (dual_franka格式)")
    print("="*70)
    print(f"总episodes: {summary['total_episodes']}")
    print(f"平均帧数: {summary['avg_frames']:.1f}")
    print(f"\n异常统计:")
    print(f"  CRITICAL: {summary['statistics']['critical']}")
    print(f"  WARNING: {summary['statistics']['warning']}")
    print(f"  MINOR: {summary['statistics']['minor']}")
    print(f"  OK: {summary['statistics']['ok']}")

    print(f"\n按类型统计:")
    for t, c in summary["by_type"].items():
        print(f"  - {t}: {c}")

    print(f"\n报告已保存: {report_path}")

    # 输出严重异常的episodes列表
    if classified_reports["critical"]:
        print("\n" + "="*70)
        print("严重异常列表 (建议剔除):")
        print("="*70)
        for r in classified_reports["critical"][:20]:  # 只显示前20
            print(f"  {r['episode']}: {r['anomaly_type']} - {r['description']}")
        if len(classified_reports["critical"]) > 20:
            print(f"  ... 还有 {len(classified_reports['critical']) - 20} 个")


if __name__ == "__main__":
    main()
