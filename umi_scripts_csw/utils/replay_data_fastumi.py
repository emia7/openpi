"""
Replay recorded demo episodes on the robot (offline delta).

This version assumes:
- The saved action is ABSOLUTE in physical units:
    action = [x, y, z, rx, ry, rz, gripper]
  where position is in meters, and rotation is a 3D rotation-vector (axis-angle / rotvec) in radians.
- We convert ABSOLUTE -> DELTA offline relative to the previous frame:
    delta_pos = pos_t - pos_{t-1}
    delta_rot = log( R_{t-1}^T * R_t )
- Gripper is kept ABSOLUTE (passed through as action[6]).

Usage:
    python replay_data.py --task_name=pick-place-0 --episode=1
    python replay_data.py --task_name=pick-place-0 --episode=1 --use_filtered
    python replay_data.py --task_name=pick-place-0 --episode=1 --dry_run
"""

import os
import time
import json
import numpy as np
import pandas as pd
from absl import app, flags
from scripts.pose_repr_util import convert_pose_mat_rep
from scripts.pose_util import pose6_to_mat, mat_to_pose6,pose7_to_pos_rotvec
from scipy.spatial.transform import Rotation as R
from scripts.visualize import visualize_frames

from experiments import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "souhu_fastumi_base", "Name of experiment (from experiments/ directory).")
flags.DEFINE_string("task_name", None, "Task name prefix for episode files.", required=True)
flags.DEFINE_integer("episode", None, "Episode number to replay.", required=True)
flags.DEFINE_string("data_dir", None, "Directory containing parquet files. Defaults to ./demo_data")
flags.DEFINE_bool("use_filtered", False, "Use filtered episode from processed folder.")
flags.DEFINE_bool("dry_run", False, "Dry run without robot (fake_env=True).")
flags.DEFINE_bool("skip_reset", False, "Skip initial reset (assume robot is already at start position).")
flags.DEFINE_float("speed_factor", 1.0, "Speed factor for replay (1.0 = original speed).")
flags.DEFINE_bool("confirm_each_step", False, "Wait for Enter key before each step (for debugging).")


def load_replay_episode(data_dir, task_name, episode_num, use_filtered=False):
    """Load episode from parquet file."""

    # filepath = "/home/ubuntu/qiuyi/fastumi_data/batch_data_replay_test_lerobot/data/chunk-000/episode_000001.parquet"
    filepath = "/home/ubuntu/qiuyi/fastumi_data/replay_test_lerobot/data/chunk-000/episode_000000.parquet"
    

    df = pd.read_parquet(filepath)

    return df, filepath

def get_transform_matrix():
    """
    定义坐标变换矩阵:
    New X = Old Z
    New Y = - Old X
    New Z = - Old Y
    """
    return np.array([
        [ 0,  0,  1, 0],
        [-1,  0,  0, 0],
        [ 0, -1,  0, 0],
        [ 0,  0,  0, 1]
    ])

def main(_):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = FLAGS.data_dir or os.path.join(project_root, "demo_data")

    # Load episode
    df, filepath = load_replay_episode(data_dir, FLAGS.task_name, FLAGS.episode, FLAGS.use_filtered)
    print(f"Loaded episode from: {filepath}")
    print(f"Total steps: {len(df)}")

    # Create environment
    config = get_config(FLAGS.exp_name)
    env = config.get_replay_env(fake_env=FLAGS.dry_run)

    print("\n" + "=" * 60)
    print("Episode Replay (ABS->DELTA offline)")
    print("=" * 60)
    print(f"Task: {FLAGS.task_name}")
    print(f"Episode: {FLAGS.episode}")
    print(f"Steps: {len(df)}")
    print(f"Filtered: {FLAGS.use_filtered}")
    print(f"Dry run: {FLAGS.dry_run}")
    print(f"Skip reset: {FLAGS.skip_reset}")
    print(f"Speed factor: {FLAGS.speed_factor}")
    print("=" * 60 + "\n")

    # Reset environment
    if not FLAGS.skip_reset:
        print("Resetting environment...")
        obs, info = env.reset()
        print("Reset complete.")

    user_input = input("Press Enter to start replay, or 'q' to quit: ").strip().lower()
    if user_input == "q":
        env.close()
        return

    hz = float(env.hz)
    step_time = 1.0 / hz / float(FLAGS.speed_factor)
    print(f"\nEnv hz: {hz}")
    print(f"Replaying at {hz * float(FLAGS.speed_factor):.1f} Hz...")

    sum = np.zeros(7)

    for idx in range(0,int(len(df)),6):
        start_time = time.time()

        row_cur = df.iloc[idx]
        row_next = df.iloc[idx + 6] if idx + 6 < len(df) else None


        cur_abs = np.array(row_cur["actions"], dtype=np.float32).reshape(-1)
        next_abs = np.array(row_next["actions"], dtype=np.float32).reshape(-1) if row_next is not None else None

        # First frame: no previous reference
        cur_pose = cur_abs[:6]
        next_pose = next_abs[:6] if next_abs is not None else None


        # breakpoint()
        
        cur_gripper = cur_abs[6]
        next_gripper = next_abs[6] if next_abs is not None else None

        cur_pose_mat = pose6_to_mat(cur_pose)  # (4,4)
        # compute relative action
        if next_pose is not None:
            next_pose_mat = pose6_to_mat(next_pose)  # (4,4)
            # Convert ABS -> DELTA offline
            delta_pose_mat = convert_pose_mat_rep(
                next_pose_mat,
                cur_pose_mat,
                pose_rep='relative',
                backward =False,
            )  # (4,4)
            # visualize_frames([cur_pose_mat,next_pose_mat,delta_pose_mat], labels=["A","B1","B2"], axis_length=0.15)
        if idx == 0:
            # print(obs.keys())
            pos,rotvec = pose7_to_pos_rotvec(obs["state"]["tcp_pose"])  # (pos(3,), rotvec(3,))
        else:
            pos,rotvec = pose7_to_pos_rotvec(next_obs["state"]["tcp_pose"])  # (pos(3,), rotvec(3,))

        cur_franka_pose6 = np.concatenate([pos, rotvec], axis=-1)  # shape (6,)
        cur_franka_pose_mat = pose6_to_mat(cur_franka_pose6)




        # fastumi_2_franka = get_transform_matrix()
        # delta_pose_mat = fastumi_2_franka @ delta_pose_mat @ fastumi_2_franka.T

        if next_pose is not None:
            next_franka_pose_mat = convert_pose_mat_rep(
                delta_pose_mat,
                cur_franka_pose_mat,
                pose_rep='relative',
                backward =True,
            )  # (4,4)

        # visualize_frames([cur_franka_pose_mat,next_franka_pose_mat], labels=["A","B"], axis_length=0.15)


        R_cur = cur_franka_pose_mat[:3, :3]
        t_cur = cur_franka_pose_mat[:3, 3]

        R_tgt = next_franka_pose_mat[..., :3, :3]
        t_tgt = next_franka_pose_mat[..., :3, 3]

        # base-frame translational delta
        dpos_base = t_tgt - t_cur  # (3)

        # base-frame rotational delta: R_tgt * R_cur^{-1}
        dR = np.matmul(R_tgt, R_cur.T)  # (3,3)
        drot_base = R.from_matrix(dR).as_rotvec()  # (3)
      
        if next_gripper is not None:
            if next_gripper/88 > 0.8:
                dgripper = -1.0
            else:
                dgripper = 1.0
        dpos_base = np.asarray(dpos_base, dtype=np.float32).reshape(-1)   # (3,)
        drot_base = np.asarray(drot_base, dtype=np.float32).reshape(-1)   # (3,)
        dgripper  = np.asarray(dgripper,  dtype=np.float32).reshape(1)    # (1,)


        action_7d = np.concatenate([dpos_base, drot_base, dgripper], axis=0)  # (7,)
        # print(action_7d)

        sum = sum + action_7d
        print(sum[2])


        next_obs, reward, done, truncated, info = env.step(action_7d)

        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"Step {idx + 1}/{len(df)}", end="\r")

        time.sleep(0.5)

        elapsed = time.time() - start_time
        sleep_time = step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"\n\nReplay complete! ({len(df)} steps)")
    env.close()


if __name__ == "__main__":
    app.run(main)

  