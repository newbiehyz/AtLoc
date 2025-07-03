import os
import json
import glob
import re
import math
import shutil
import numpy as np

def clean_json_text(text):
    return re.sub(r',\s*([\]\}])', r'\1', text)

def get_pose_matrix(x, y, yaw_deg):
    theta = math.radians(90 - yaw_deg)
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0,               0,                1]
    ])
    x = x * 0.001  # mm → m
    y = y * 0.001
    t = np.array([[x], [y], [0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def backup_png_files():
    backup_dir = os.path.abspath(os.path.join("..", "output_fisheye_backup"))
    os.makedirs(backup_dir, exist_ok=True)
    png_files = glob.glob("*.jpg")
    print(f"\n📁 正在备份 {len(png_files)} 个 .png 文件到 {backup_dir}")
    for f in png_files:
        shutil.copy2(f, os.path.join(backup_dir, os.path.basename(f)))
    print("✅ 备份完成。")

def process_json_poses():
    json_files = glob.glob("patac_app2emap*.json")
    print(f"\n📂 处理 JSON 文件，共 {len(json_files)} 个")
    for file in json_files:
        print(f"  - {file}")
        try:
            with open(file, 'r') as f:
                raw_text = f.read()
                cleaned_text = clean_json_text(raw_text)
                data = json.loads(cleaned_text)

            for entry in data:
                try:
                    x = entry["App2emap_DR.x"]
                    y = entry["App2emap_DR.y"]
                    yaw = entry.get("App2emap_DR.yaw", entry.get("App2emap_DR.canAng", 0.0))
                    timestamp = int(entry["App2emap_DR.timeStamp"])
                    T = get_pose_matrix(x, y, yaw)
                    pose_filename = f"pose_{timestamp}.txt"
                    with open(pose_filename, 'w') as f_pose:
                        for row in T:
                            f_pose.write(' '.join(f"{v:.7f}" for v in row) + '\n')
                except KeyError as e:
                    print(f"    ⚠️ JSON字段缺失，跳过: {e}")
        except Exception as e:
            print(f"    ❌ JSON 解析失败: {file}, 错误: {e}")

def rename_images_by_timestamp():
    pattern = re.compile(r'front_camera_fov195_(\d+)_\d+\.jpg')
    image_files = glob.glob("front_camera_fov195_*.jpg")
    print(f"\n📷 图像重命名：找到 {len(image_files)} 个")
    for old_path in image_files:
        filename = os.path.basename(old_path)
        match = pattern.match(filename)
        if match:
            timestamp = match.group(1)
            new_filename = f"frame-{timestamp}.color.png"
            os.rename(old_path, new_filename)
            print(f"  ✅ {filename} → {new_filename}")
        else:
            print(f"  ⚠️ 跳过未匹配图像: {filename}")

def extract_timestamp(filename, pattern):
    match = pattern.match(os.path.basename(filename))
    return int(match.group(1)) if match else None

def find_closest_pose(timestamp, pose_ts_list):
    return min(pose_ts_list, key=lambda t: abs(t - timestamp))

def sync_and_rename_frames_and_poses():
    frame_files = glob.glob("frame-*.color.png")
    pose_files = glob.glob("pose_*.txt")

    frame_pattern = re.compile(r'frame-(\d+)\.color\.png')
    pose_pattern = re.compile(r'pose_(\d+)\.txt')

    frame_data = []
    for f in frame_files:
        ts = extract_timestamp(f, frame_pattern)
        if ts is not None:
            frame_data.append((ts, f))
    frame_data.sort()

    pose_map = {}
    pose_ts_list = []
    for p in pose_files:
        ts = extract_timestamp(p, pose_pattern)
        if ts is not None:
            pose_map[ts] = p
            pose_ts_list.append(ts)

    print(f"\n🔗 时间同步中：帧图像 {len(frame_data)}，姿态 {len(pose_ts_list)}")
    used_pose_files = set()

    for idx, (frame_ts, frame_file) in enumerate(frame_data):
        closest_pose_ts = find_closest_pose(frame_ts, pose_ts_list)
        pose_file = pose_map[closest_pose_ts]
        new_base = f"frame-{idx:06d}"

        frame_new = f"{new_base}.color.png"
        os.rename(frame_file, frame_new)

        pose_new = f"{new_base}.pose.txt"
        shutil.copyfile(pose_file, pose_new)
        used_pose_files.add(pose_file)

        print(f"  🎯 配对: {frame_file} + {pose_file} → {new_base}.*")

    # 删除未使用的 pose_xxx.txt
    print(f"\n🧹 清理未匹配的 pose_*.txt 文件...")
    for p in pose_files:
        if p not in used_pose_files:
            os.remove(p)
            print(f"  🗑️ 删除: {p}")

def delete_rear_left_right_files():
    patterns = ["rear_*", "left_*", "right_*", "pose_*"]
    print("\n🧽 删除 rear-/left-/right- 开头的文件...")
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            os.remove(f)
            print(f"  🗑️ 删除: {f}")
            
def move_frames_to_seq_folder():
    print("\n📂 整理输出文件到 ../seq-all")
    seq_dir = os.path.abspath(os.path.join("..", "seq-all"))
    os.makedirs(seq_dir, exist_ok=True)
    frame_files = glob.glob("frame-*.color.png") + glob.glob("frame-*.pose.txt")
    for f in frame_files:
        shutil.move(f, os.path.join(seq_dir, f))
    print(f"✅ 已移动 {len(frame_files)} 个 frame-* 文件到 {seq_dir}")


def main():
    backup_png_files()
    process_json_poses()
    rename_images_by_timestamp()
    sync_and_rename_frames_and_poses()
    delete_rear_left_right_files()
    move_frames_to_seq_folder()

    print("\n✅ 所有处理完成。")

if __name__ == "__main__":
    main()

