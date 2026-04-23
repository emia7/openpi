import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def transform_pose_7d(pose_7d):
    """
    输入: [x, y, z, qx, qy, qz, qw]
    输出: 变换后的 [x, y, z, qx, qy, qz, qw]
    变换逻辑: x->-y, y->-z, z->x
    """
    # 1. 提取位置和四元数
    pos = np.array(pose_7d[:3])  # x, y, z
    quat = np.array(pose_7d[3:]) # qx, qy, qz, qw

    # 2. 定义变换矩阵
    # New X = - Old Y
    # New Y = - Old Z
    # New Z =   Old X
    T_matrix = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ])

    # 3. 变换位置 (矩阵乘向量)
    # P_new = T * P_old
    pos_new = T_matrix @ pos

    # 4. 变换姿态 (四元数旋转)
    # Q_new = Q_trans * Q_old
    # 先将变换矩阵转换为旋转对象
    r_trans = R.from_matrix(T_matrix)
    r_old = R.from_quat(quat)
    
    # 执行旋转组合
    r_new = r_trans * r_old
    quat_new = r_new.as_quat() # 返回 [x, y, z, w]

    # 5. 拼接结果并转换为 list
    return np.concatenate((pos_new, quat_new)).tolist()

def main(input_path, output_path):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查基本结构
    if "records" not in data:
        print("Error: JSON format incorrect (missing 'records').")
        return

    processed_count = 0
    
    # 遍历处理每一条记录
    for record in data["records"]:
        if "pose" in record and len(record["pose"]) == 7:
            # 执行变换
            original_pose = record["pose"]
            new_pose = transform_pose_7d(original_pose)
            
            # 更新 pose
            record["pose"] = new_pose
            processed_count += 1

    print(f"Processed {processed_count} poses.")

    # 写入新文件
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Pose coordinate system in JSON.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    main(args.input, args.output)