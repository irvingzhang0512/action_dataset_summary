import pandas as pd
import os
import numpy as np
from collections import defaultdict

# 输入
csv_dir = "labels"  # 标注结果所在文件夹（会寻找子目录），后续处理所有`.csv`文件
video_dir = "renames/step1"  # 标注视频所在文件夹（会寻找子目录），后续处理所有`.mp4`文件

# 一些参数
avaiable_cameras = ["2m", "3m", "4m"]  # 处理指定摄像头的数据，若为空则处理所有
avaiable_person_ids = [33]
# avaiable_person_ids = range(13, 34)
avaiable_persons = [
    "P%04d" % i for i in avaiable_person_ids
]  # 处理指定人物数据，若为空则处理所有

# 导出标记错误
enable_labeling_err = True
labeling_err_file = "./labeling_error.txt"

# 导出统计结果
enable_summary = True
summary_file = "./summary.csv"

# 导出质量管理反馈结果
enable_quality_feedback = True
qualify_feedback_file = "./quality_feedback.txt"

# 所有labels的集合
frame_type = {"medium", "end"}
pose_type = {"stand", "sit", "squat", "lie", "half_lie"}
action_type = {
    "stillfall",
    "walkingfall",
    "pushoverfall",
    "tripfall",
    "takephone",
    "takecup",
    "eating",
    "drinking",
    "medicine",
    "knock",
    "close",
    "open",
}
quality_type = {
    "qualified",
    "err_camera_lacking",
    "err_action_lacking",
    "err_light",
    "err_pose",
    "err_sleeve",
    "err_view",
    "err_shelter",
    "err_unknown",
}
total_type = frame_type | pose_type | action_type | quality_type

series_error_type = {"err_camera_lacking", "err_action_lacking", "err_unknown"}
remade_error_type = {"err_action_lacking"}
useless_error_type = {"err_camera_lacking", "err_unknown"}
typo_error_type = {
    "err_light",
    "err_pose",
    "err_sleeve",
    "err_view",
    "err_shelter",
}

translations_dict = {
    "qualified": "合格",
    "err_camera_lacking": "画面缺失",
    "err_action_lacking": "动作不完整或动作标注错误",
    "err_light": "光照与文件名不符",
    "err_pose": "姿态与文件名不符",
    "err_sleeve": "衣着与文件名不符",
    "err_view": "视角与文件名不符",
    "err_shelter": "遮挡与文件名不符",
    "err_unknown": "未知错误",
    "medium": "中间帧",
    "end": "结尾帧",
    "stand": "站",
    "sit": "坐",
    "squat": "蹲",
    "lie": "躺",
    "half_lie": "过渡",
    "stillfall": "原地软倒",
    "walkingfall": "行进软倒",
    "pushoverfall": "推倒",
    "tripfall": "绊倒",
    "takephone": "拿手机",
    "takecup": "拿水杯",
    "eating": "吃饭",
    "drinking": "喝水",
    "medicine": "吃药",
    "knock": "磕碰",
    "close": "关门",
    "open": "开门",
    "Not_Labeled_or_Wrong_Labeled": "未标注或标注无效",
    "No_Data": "无数据",
}


def _trans_func(context):
    """
    将英文结果转换为中文
    """
    if len(context) > 0:
        segs = set(context.split(" "))
        res = []
        for seg in segs:
            if seg == "":
                continue
            if seg not in translations_dict:
                print("unknown english `{}`".format(seg))
            else:
                res.append(translations_dict[seg])
        context = " ".join(res)
    return context


def _get_error_labeling(row):
    """
    每个 row 代表一个bbox以及对应的所有标签
    本函数根据每一行的信息判断是否出现标记错误
    """

    # 对输入数据进行预处理
    # 主要工作就是获取每一类的标签，并构建为set对象
    if len(row["action_labels"]) > 1:
        action_labels = set(row["action_labels"].split(" "))
    else:
        action_labels = {}
    if len(row["filename_action_labels"]) > 1:
        filename_action_labels = set(row["filename_action_labels"].split(" "))
    else:
        filename_action_labels = {}
    if len(row["quality_labels"]) > 1:
        quality_labels = set(row["quality_labels"].split(" "))
    else:
        quality_labels = {}
    if len(row["pose_labels"]) > 1:
        pose_labels = set(row["pose_labels"].split(" "))
    else:
        pose_labels = {}
    if len(row["unknown_labels"]) > 1:
        unknown_labels = set(row["unknown_labels"].split(" "))
    else:
        unknown_labels = {}
    is_medium = row["is_medium"]
    is_end = row["is_end"]

    result = ""

    # 对通用错误进行集中处理
    # 通用错误1: 对于unknown label集中处理
    if len(unknown_labels) > 0:
        result += "存在未知标签 {}。".format(unknown_labels)
    # 通用错误2: 帧类别不能同时存在
    if is_medium and is_end:
        result += "同时存在中间帧标签与结尾帧标签。"

    # 对于三类 bbox 分别处理
    # 第一类bbox：无中间帧标签+无结尾帧标签
    if not is_medium and not is_end:
        if len(pose_labels) != 1:
            result += "非目标bbox - 姿态标签不存在或不止一个。"
        return result
    # 第二类bbox：中间帧
    if is_medium:
        if len(quality_labels) == 0:
            result += "中间帧：漏标质量管理标签。"
        elif len(quality_labels) > 1 or list(quality_labels)[0] != "qualified":
            if len(quality_labels) > 1 and "qualified" in quality_labels:
                result += "中间帧：质量管理标签中同时存在 `qualified` 以及其他错误标签。"
            # 质量存在问题，需要判断质量问题严重性
            if len(quality_labels & series_error_type) > 0:
                # 对于严重错误，不需要考虑其他问题
                return result
        # 无质量问题，或存在质量问题但不需要反工时
        # 判断行为标签、姿态标签数量只能有一个
        # 行为标签与文件名中的行为标签保持一致
        if len(action_labels) != 1:
            result += "中间帧 - 行为标签遗漏或不止一个。"
        elif list(action_labels)[0] != list(filename_action_labels)[0]:
            result += "中间帧 - 行为标签与文件名中的行为标签不同。"
        if len(pose_labels) != 1:
            result += "中间帧 - 姿态标签遗漏或不止一个。"
        return result
    # 第三类bbox：结尾帧
    # 主要就是考虑行为标签、姿态标签数量必须是1个
    # 判断行为标签与姿态标签
    if len(action_labels) != 1:
        result += "结尾帧 - 行为标签遗漏或不止一个。"
    elif list(action_labels)[0] != list(filename_action_labels)[0]:
        result += "结尾帧 - 行为标签与文件名中的行为标签不同。"
    if len(pose_labels) != 1:
        result += "结尾帧 - 姿态标签遗漏或不止一个。"

    return result


def _get_and_concat_all_csvs(csv_dir):
    """获取并拼接所有 csv 结果"""
    csv_list = []
    for cur_path, _, file_names in os.walk(csv_dir):
        csv_list += [
            os.path.join(cur_path, file_name)
            for file_name in file_names
            if file_name.endswith(".csv")
        ]
    return pd.concat([pd.read_csv(csv_file) for csv_file in csv_list])


def _output_labeling_err_results(
    bbox_df, err_labeling_df, video_list, labeling_err_file
):
    """
    输出标注错误相关结果
    输出一：print输出
    输出二：写入文件中
    """

    # 获取漏标数据
    missing_index_list = []
    labeled_samples = set(
        list(bbox_df[bbox_df["is_medium"] > 0]["index_name"].unique())
    )
    for video_name in video_list:
        row = video_name.split("_")
        pid = row[-1][:5]
        if len(avaiable_persons) > 1:
            if pid not in avaiable_persons:
                continue
        cur_index_name = "{}_{}_{}".format(pid, row[1], row[0])
        if cur_index_name not in labeled_samples:
            missing_index_list.append(cur_index_name)

    labeling_err_writer = open(labeling_err_file, "w")

    # 漏标数据输出
    context = "漏标视频（可能没有标视频，也可能漏了`中间帧`标签）共有 {} 个，分别是：".format(
        len(missing_index_list)
    )
    labeling_err_writer.write(context + "\n")
    print(context)

    for sample in missing_index_list:
        print(sample)
        labeling_err_writer.write(sample + "\n")
    print("\n")
    labeling_err_writer.write("\n")

    # 错标数据输出
    context = "其他错标视频共有 {} 个，其编号与错误类别如下：".format(len(err_labeling_df))
    labeling_err_writer.write(context + "\n")
    print(context)
    for row in err_labeling_df.iterrows():
        context = "{}: {}".format(
            row[1]["index_name"], row[1]["labeling_err_details"]
        )
        print(context)
        labeling_err_writer.write(context + "\n")


def _filter_df_by_camera_and_person(df, avaiable_cameras, avaiable_persons):
    """
    根据 camera 编号和 person 编号筛选数据
    """
    df["camera"] = df["image"].str.split("_").map(lambda x: x[1])
    df["person"] = df["image"].str.split("_").map(lambda x: x[-1][:5])
    if len(avaiable_cameras) > 0:
        df = df[df["camera"].isin(avaiable_cameras)]
    if len(avaiable_persons) > 0:
        df = df[df["person"].isin(avaiable_persons)]
    return df


def _get_bbox_df(df):
    """
    根据bbox执行group操作，汇总每个bbox的信息
    """

    def _group_by_bbox(x):
        """
        根据 bbox 进行group操作，获取group后每行的结果
        'image', 'action_labels', 'quality_labels',
        'pose_labels', 'unknown_labels', 'is_medium'
        文件名，要求行为标签，质量标签，
        姿态标签，未知标签，是否是中间帧，打标签的错误

        """
        img_name = x.iloc[0, 0]
        cur_filename_action_labels = {img_name.split("_")[0]} & action_type

        # 获取各类标签
        labels = set(list(x["label"]))
        cur_action_labels = labels & action_type
        cur_quality_labels = labels & quality_type
        cur_pose_labels = labels & pose_type
        cur_unknown_labels = labels - total_type

        # 当前bbox是否是中间帧
        is_medium = np.array(x["is_medium"]).sum()
        is_end = np.array(x["is_end"]).sum()

        return pd.Series(
            [
                img_name,
                " ".join(cur_filename_action_labels),
                " ".join(cur_action_labels),
                " ".join(cur_quality_labels),
                " ".join(cur_pose_labels),
                " ".join(cur_unknown_labels),
                is_medium,
                is_end,
            ]
        )

    # 增加新列 bbox 用于 groupby
    df["bbox"] = (
        df["image"]
        + df["xmin"].astype(str)
        + df["ymin"].astype(str)
        + df["xmax"].astype(str)
        + df["ymax"].astype(str)
    )
    df["is_medium"] = df["label"].map(lambda x: 1 if x == "medium" else 0)
    df["is_end"] = df["label"].map(lambda x: 1 if x == "end" else 0)
    bbox_df = df.groupby("bbox").apply(_group_by_bbox)
    bbox_df.columns = [
        "image",
        "filename_action_labels",
        "action_labels",
        "quality_labels",
        "pose_labels",
        "unknown_labels",
        "is_medium",
        "is_end",
    ]
    bbox_df["index_name"] = (
        bbox_df["image"]
        .str.split("_")
        .apply(lambda x: x[-1][:5] + "_" + x[1] + "_" + x[0])
    )
    return bbox_df


def _get_summary_df(bbox_df, video_list, actions, total_samples, summary_file):
    """
    获取标注结果汇总信息
    将英文汇总信息转换为中文，保存为本地文件
    """

    def _generate_action_results(x):
        """
        获取每行(pid+camera)、每列(12个动作)的质量管理结果
        """
        data = [""] * 12
        img_split = list(x["image"])[0].split("_")
        key = img_split[-1][:5] + "_" + img_split[1]
        for row in x.iterrows():
            data[actions.index(row[1][1])] += row[1][3] + " "
        for idx, ele in enumerate(data):
            if ele == "":
                if actions[idx] in total_samples[key]:
                    data[idx] = "Not_Labeled_or_Wrong_Labeled"
                else:
                    data[idx] = "No_Data"
        return pd.Series(data)

    # 英文版summary
    bbox_df["index_name"] = (
        bbox_df["image"].str.split("_").apply(lambda x: x[-1][:5] + "_" + x[1])
    )
    bbox_df = bbox_df[(bbox_df["is_medium"] > 0)]
    summary_df = bbox_df.groupby("index_name").apply(_generate_action_results)
    summary_df.columns = actions
    for column in actions:
        summary_df[column] = summary_df[column].str.strip()

    # 中文版 summary
    summary_ch_df = summary_df.copy()
    for col in summary_ch_df.columns:
        summary_ch_df[col] = summary_ch_df[col].apply(_trans_func)
    summary_ch_df.columns = [
        translations_dict[k] for k in list(summary_ch_df.columns)
    ]
    summary_ch_df.to_csv(summary_file)

    return summary_df, summary_ch_df


def _outputs_quality_feedback(qualify_feedback_file, summary_df):
    """
    根据标记汇总结果，输出质量管理反馈信息
    """
    qualify_feedback_writer = open(qualify_feedback_file, "w")
    remade_list = []
    useless_list = []
    typo_list = []

    # 遍历标记汇总结果，保存所有质量问题
    # 并根据质量问题的严重程度进行分类
    for row in summary_df.iterrows():
        index_name = row[0]
        for action in row[1].index:
            qualities = set(row[1][action].split(" "))
            context = " ".join(
                [
                    index_name,
                    translations_dict[action],
                    _trans_func(row[1][action]),
                ]
            )
            if len(qualities & useless_error_type) > 0:
                useless_list.append(context)
            if len(qualities & remade_error_type) > 0:
                remade_list.append(context)
            if len(qualities & typo_error_type) > 0:
                typo_list.append(context)

    # 按严重程度分别输出相关质量问题
    # 1. 无法使用的数据
    print("\n")
    context = "无效数据有 {} 个，分别是：".format(len(useless_list))
    print(context)
    qualify_feedback_writer.write(context + "\n")
    for context in useless_list:
        print(context)
        qualify_feedback_writer.write(context + "\n")
    print()
    qualify_feedback_writer.write("\n")

    # 2. 需要返工的数据
    context = "需要返工数据有 {} 个，分别是：".format(len(remade_list))
    print(context)
    qualify_feedback_writer.write(context + "\n")
    for context in remade_list:
        print(context)
        qualify_feedback_writer.write(context + "\n")
    print()
    qualify_feedback_writer.write("\n")

    # 有小问题，但不影响使用的数据
    context = "有小问题但不影响使用的数据有 {} 个，分别是：".format(len(typo_list))
    print(context)
    qualify_feedback_writer.write(context + "\n")
    for context in typo_list:
        print(context)
        qualify_feedback_writer.write(context + "\n")

    qualify_feedback_writer.close()


def main(args):
    # 获取所有视频
    video_list = []
    for cur_path, _, file_names in os.walk(video_dir):
        for file_name in file_names:
            if file_name.endswith(".mp4"):
                row = file_name.split("_")
                if (
                    row[1] in avaiable_cameras
                    and row[-1][:5] in avaiable_persons
                ):
                    video_list.append(file_name)

    # 获取视频 dict
    # key: {pid}_{camera}
    # value: list(actions)
    total_samples = defaultdict(list)
    for file_name in video_list:
        row = file_name.split("_")
        action = row[0]
        camera = row[1]
        person = row[-1][:5]
        key = "{}_{}".format(person, camera)
        total_samples[key].append(action)

    # 获取原始df，并根据 camera 和 person 筛选数据
    df = _get_and_concat_all_csvs(csv_dir)
    df = _filter_df_by_camera_and_person(df, avaiable_cameras, avaiable_persons)

    # 根据 bbox 汇总数据
    bbox_df = _get_bbox_df(df)

    # 处理标记错误相关信息
    bbox_df["labeling_err_details"] = bbox_df.apply(
        lambda row: _get_error_labeling(row), axis=1
    )
    bbox_df["labeling_err"] = bbox_df["labeling_err_details"].apply(
        lambda x: len(x) != 0
    )
    err_labeling_df = bbox_df[bbox_df["labeling_err"]]

    # 导出标记错误信息
    if enable_labeling_err:
        _output_labeling_err_results(
            bbox_df, err_labeling_df, video_list, labeling_err_file
        )

    # 获取sumamry结果
    if enable_summary:
        actions = list(action_type)
        summary_df, summary_ch_df = _get_summary_df(
            bbox_df, video_list, actions, total_samples, summary_file
        )

        # 获取质量管理反馈结果
        if enable_quality_feedback:
            _outputs_quality_feedback(qualify_feedback_file, summary_df)


if __name__ == "__main__":
    main(None)
