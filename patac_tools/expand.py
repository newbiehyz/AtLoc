import os
import shutil
import re
from PIL import Image
import numpy as np

def expand_and_split_dataset():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    color_files = sorted([f for f in os.listdir(current_dir) if f.endswith('.color.png')])
    pose_files = sorted([f for f in os.listdir(current_dir) if f.endswith('.pose.txt')])
    assert len(color_files) == len(pose_files), "图像和位姿文件数量不一致"

    def extract_index(filename):
        match = re.search(r'frame-(\d{6})\.', filename)
        return int(match.group(1)) if match else -1

    max_index = max(extract_index(f) for f in color_files)
    start_index = max_index + 1
    original_color_files = list(color_files)
    original_pose_files = list(pose_files)

    while True:
        try:
            repeat = int(input(f"当前数据帧数为 {len(color_files)}，请输入扩充倍数（也将生成该数量的子目录）: "))
            if repeat < 1:
                raise ValueError
            break
        except ValueError:
            print("请输入一个大于0的整数")

    print(f"\n将每个样本复制 {repeat} 次，总生成 {len(color_files) * repeat} 帧")
    print(f"扩展数据将从 frame-{start_index:06d} 开始编号")

    generated_files = []
    new_idx = start_index
    for r in range(repeat):
        for i in range(len(original_color_files)):
            color_src = os.path.join(current_dir, original_color_files[i])
            pose_src = os.path.join(current_dir, original_pose_files[i])

            color_dst = os.path.join(current_dir, f"frame-{new_idx:06d}.color.png")
            pose_dst = os.path.join(current_dir, f"frame-{new_idx:06d}.pose.txt")

            shutil.copyfile(color_src, color_dst)
            shutil.copyfile(pose_src, pose_dst)

            generated_files.append((color_dst, pose_dst))
            new_idx += 1

    print(f"\n扩展完成，共生成 {len(generated_files)} 对文件")

    # 计算图像 stats.txt
    print("开始计算 stats.txt ...")
    means = []
    stds = []

    for color_path, _ in generated_files:
        img = Image.open(color_path).convert('RGB')
        img_np = np.asarray(img).astype(np.float32) / 255.0
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            means.append(np.mean(img_np, axis=(0, 1)))
            stds.append(np.std(img_np, axis=(0, 1)))

    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)

    stats_path = os.path.join(parent_dir, 'stats.txt')
    np.savetxt(stats_path, np.vstack((mean_rgb, std_rgb)), fmt='%8.7f')
    print(f"✓ 已保存图像 stats.txt 到 {stats_path}")

    # 计算位姿 pose_stats.txt
    print("开始计算 pose_stats.txt ...")
    poses_xyz = []
    for _, pose_path in generated_files:
        pose_matrix = np.loadtxt(pose_path)
        if pose_matrix.shape != (4, 4):
            print(f"⚠️ 警告：跳过无效 pose 文件 {pose_path}")
            continue
        translation = pose_matrix[:3, 3]  # 提取 tx, ty, tz
        poses_xyz.append(translation)

    poses_xyz = np.array(poses_xyz)
    mean_t = np.mean(poses_xyz, axis=0)
    std_t = np.std(poses_xyz, axis=0)

    pose_stats_path = os.path.join(parent_dir, 'pose_stats.txt')
    np.savetxt(pose_stats_path, np.vstack((mean_t, std_t)), fmt='%8.7f')
    print(f"✓ 已保存位姿 pose_stats.txt 到 {pose_stats_path}")


    # 拆分到子目录
    files_per_folder = len(generated_files) // repeat
    for i in range(repeat):
        folder_name = f"seq-{i+1:02d}"
        folder_path = os.path.join(parent_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        start = i * files_per_folder
        end = (i + 1) * files_per_folder if i < repeat - 1 else len(generated_files)
        for j, (color_path, pose_path) in enumerate(generated_files[start:end]):
            new_index = j
            new_color_name = f"frame-{new_index:06d}.color.png"
            new_pose_name  = f"frame-{new_index:06d}.pose.txt"

            shutil.move(color_path, os.path.join(folder_path, new_color_name))
            shutil.move(pose_path, os.path.join(folder_path, new_pose_name))


        print(f"✓ 已移动到：{folder_name} ({end - start} 对文件)")

    print("\n🎉 所有扩展文件处理完成：图像、位姿、stats.txt 和 pose_stats.txt 已生成")

if __name__ == "__main__":
    expand_and_split_dataset()

