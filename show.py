import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# 输入参数
qualify_feedback_file = "quality_feedback.txt"
video_path = "./renames/step1"

actions_chinese_to_english_dict = {
    # 跌倒类型
    "原地软倒": "stillfall",
    "行进软倒": "walkingfall",
    "推倒": "pushoverfall",
    "绊倒": "tripfall",
    # 其他动作
    "吃药": "medicine",
    "吃饭": "eating",
    "喝水": "drinking",
    "拿手机": "takephone",
    "拿水杯": "takecup",
    "磕碰": "knock",
    "关门": "close",
    "开门": "open",
}


def _add_chinese_in_image(img, context, color=(255, 0, 0)):
    """
    在图像中增加中文

    默认使用 cv2.putText 会有问题
    建议使用 PIL 添加中文
    需要指定字体，在Windows下已经测试，在Linux下不清楚
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.array(img))

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("MSYH.TTF", 20, encoding="utf-8")
    draw.text((0, 0), context, color, font=font)

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def display_single_video(
    video_path, context, color=(255, 0, 0), resize_size=(720, 480),
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("video {} doesn't exists.".format(video_path))
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_size is not None:
            frame = cv2.resize(frame, resize_size)
        frame = _add_chinese_in_image(frame, context, color)

        cv2.imshow("Demo", frame)
        cv2.waitKey(1)
    cap.release()


if __name__ == "__main__":
    # 获取所有已有视频，构建字典
    # key 为 `{pid}_{camera}_{action}`
    # vavlue 为对应视频的路径
    video_index_name_to_full_path_dict = {}
    for dirname, _, file_names in os.walk(video_path):
        for file_name in file_names:
            if not file_name.endswith(".mp4"):
                continue
            row = file_name.split("_")
            key = row[-1][:5] + "_" + row[1] + "_" + row[0]
            video_index_name_to_full_path_dict[key] = os.path.join(
                dirname, file_name
            )

    # 读取反馈数据文件，根据类别进行分类
    useless_list = []
    remade_list = []
    typo_list = []
    with open(qualify_feedback_file, "r") as f:
        lines = f.readlines()
    sample_type = -1
    for line in lines:
        if line.startswith("无效数据"):
            sample_type = 1
        elif line.startswith("需要返工"):
            sample_type = 2
        elif line.startswith("有小问题"):
            sample_type = 3
        elif line.startswith("P"):
            row = line.split(" ")
            key = row[0] + "_" + actions_chinese_to_english_dict[row[1]]
            details = line
            pair = (video_index_name_to_full_path_dict[key], details)
            if sample_type == 1:
                useless_list.append(pair)
            elif sample_type == 2:
                remade_list.append(pair)
            elif sample_type == 3:
                typo_list.append(pair)

    # 分别展示三类数据
    for sample in useless_list:
        display_single_video(sample[0], sample[1])

    for sample in remade_list:
        display_single_video(sample[0], sample[1])

    for sample in typo_list:
        display_single_video(sample[0], sample[1])
