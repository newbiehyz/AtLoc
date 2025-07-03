import cv2
import os
from glob import glob

# 获取当前目录下所有 .jpg 文件并排序
images = sorted(glob('./*.jpg'))

# 检查是否有图片
if not images:
    raise ValueError("当前目录下没有找到任何 .jpg 文件")

# 读取第一张图确定尺寸
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# 输出视频设置
output_video = 'output_video.mp4'
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 写入每一帧
for img_path in images:
    frame = cv2.imread(img_path)
    video_writer.write(frame)

video_writer.release()
print(f'视频已保存为 {output_video}')
