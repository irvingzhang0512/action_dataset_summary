import os
import shutil

# 输入数据
src_video_path = "./splits/0729"
target_video_path = "./renames"

# 数据目录初始化
useless_video_dir = os.path.join(target_video_path, "useless")
step1_video_path = os.path.join(target_video_path, "step1")
step2_video_path = os.path.join(target_video_path, "step2")
if not os.path.exists(target_video_path):
    os.makedirs(target_video_path)
if not os.path.exists(step1_video_path):
    os.makedirs(step1_video_path)
if not os.path.exists(step2_video_path):
    os.makedirs(step2_video_path)
if not os.path.exists(useless_video_dir):
    os.makedirs(useless_video_dir)

# 用到的一些参数
start_pid_num = 0
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
    # 人体姿态
    "站": "stand",
    "坐": "sit",
    "蹲": "squat",
    "躺": "lie",
}
light_chinese_to_english = {
    # 光线
    "暗淡光": "dimlight",
    "冷色光": "coldlight",
    "暖色光": "warmlight",
    "自然光": "naturallight",
}
pose_chinese_to_english = {
    # 姿态
    "站姿": "stand",
    "坐姿": "sit",
}
sleeve_chinese_to_english = {
    # 衣着
    "短袖": "shortsleeve",
    "长袖": "longsleeve",
}
shot_chinese_to_english = {
    # 视角
    "跟拍": "followshot",
    "背面": "backshot",
    "正面": "frontshot",
    "侧面": "sideshot",
}
shelter_chinese_to_english = {
    # 遮挡
    "遮挡": "withshelter",
    "无遮挡": "withoutshelter",
}
camera_chinese_to_english = {
    # 摄像头
    "2米": "2m",
    "3米": "3m",
    "4米": "4m",
    "reid01": "reid01",
    "reid02": "reid02",
    "reid03": "reid03",
    "reid04": "reid04",
    "reid05": "reid05",
}


def _handle_single_video(video_path):
    conds = os.path.basename(video_path).split(".")[0].split("_")

    # 选择需要的数据，剔除 reid 数据
    action_name = conds[-1]
    if action_name not in actions_chinese_to_english_dict.keys():
        if "行走" not in action_name:
            print("unknown action {} for {}".format(action_name, video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return

    # 选择命名合法的视频
    if len(conds) != 9:
        print("rename error {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return

    # 从文件名中提取各个元素
    pid_num = int(conds[0].split("P")[1]) + start_pid_num
    pid_num = "%04d" % pid_num  # add 0
    pid = "P" + str(pid_num)
    pose = conds[1]
    light = conds[2]
    sleeve = conds[3]
    shot = conds[4]
    shelter = conds[5]
    camera = conds[6]
    sceneid = conds[7]

    # 判断输入数据是否合法
    if pose not in pose_chinese_to_english.keys():
        print("unknown pose {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return
    if light not in light_chinese_to_english.keys():
        print("unknown light {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return
    if sleeve not in sleeve_chinese_to_english.keys():
        print("unknown sleeve {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return
    if shot not in shot_chinese_to_english.keys():
        print("unknown shot {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return
    if shelter not in shelter_chinese_to_english.keys():
        print("unknown shelter {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return
    if camera not in camera_chinese_to_english.keys():
        print("unknown camera {}".format(video_path))
        shutil.copy(
            video_path,
            os.path.join(useless_video_dir, os.path.basename(video_path)),
        )
        return

    # 根据 camera 类别拆分数据，并分别保存
    if camera.startswith("reid"):
        target_dir = os.path.join(step2_video_path, pid)
    else:
        target_dir = os.path.join(step1_video_path, pid)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    new_file_name = (
        "_".join(
            [
                actions_chinese_to_english_dict[action_name],
                camera_chinese_to_english[camera],
                pose_chinese_to_english[pose],
                light_chinese_to_english[light],
                sleeve_chinese_to_english[sleeve],
                shot_chinese_to_english[shot],
                shelter_chinese_to_english[shelter],
                sceneid,
                pid,
            ]
        )
        + ".mp4"
    )
    shutil.copy(video_path, os.path.join(target_dir, new_file_name))


if __name__ == "__main__":
    for cur_path, _, file_names in os.walk(src_video_path):
        for file_name in file_names:
            if file_name.endswith(".mp4"):
                _handle_single_video(os.path.join(cur_path, file_name))
    print("run over!")
