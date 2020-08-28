import pandas as pd
import os
from collections import defaultdict

# 输入
csv_dir = "./data/raw_labels"  # 标注结果所在文件夹（会寻找子目录），后续处理所有`.csv`文件
video_dir = "./data/action_dataset/step1"  # 标注视频所在文件夹（会寻找子目录），后续处理所有`.mp4`文件

# 一些参数
avaiable_cameras = ["2m", "3m", "4m"]  # 处理指定摄像头的数据，若为空则处理所有
avaiable_person_ids = [319, 322]
# avaiable_person_ids = range(207, 215)
avaiable_persons = [
    "P%04d" % i for i in avaiable_person_ids
]  # 处理指定人物数据，若为空则处理所有

# 导出标记错误
enable_labeling_err = True
labeling_err_file = "./labeling_err.txt"

# 导出统计结果
enable_output_summary = True
quality_summary_file = "./quality_summary.csv"
bbox_summary_file = "./bbox_summary.csv"

# 导出质量管理反馈结果
enable_quality_feedback = True
qualify_feedback_file = "./quality_feedback.txt"

# 所有labels的集合
frame_type = {"medium", "end"}
pose_type = {"stand", "sit", "squat", "lie", "half_lie", "other_pose"}
actions = [
    "close",
    "drinking",
    "eating",
    "knock",
    "medicine",
    "open",
    "pushoverfall",
    "stillfall",
    "takecup",
    "takephone",
    "tripfall",
    "walkingfall",
]
action_type = set(actions)
quality_type = {
    "qualified",
    "err_remade_action_lacking",
    "err_useless",
    "err_name",
    "err_rename",
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

series_error_type = {
    "err_camera_lacking",
    "err_action_lacking",
    "err_unknown",
    "err_remade_action_lacking",
    "err_useless",
}
remade_error_type = {"err_action_lacking", "err_remade_action_lacking"}
useless_error_type = {"err_camera_lacking", "err_unknown", "err_useless"}
typo_error_type = {
    "err_light",
    "err_pose",
    "err_sleeve",
    "err_view",
    "err_shelter",
    "err_rename",
    "err_name",
}


# 中英文转换相关
translations_dict = {
    "qualified": "合格",
    "err_remade_action_lacking": "需返工",
    "err_useless": "无效数据",
    "err_rename": "命名错误",
    "err_name": "命名错误",
    "err_camera_lacking": "无效数据",
    "err_action_lacking": "需返工",
    "err_light": "命名错误",
    "err_pose": "命名错误",
    "err_sleeve": "命名错误",
    "err_view": "命名错误",
    "err_shelter": "命名错误",
    "err_unknown": "无效数据",
    "medium": "中间帧",
    "end": "结尾帧",
    "stand": "站",
    "sit": "坐",
    "squat": "蹲",
    "lie": "躺",
    "half_lie": "过渡",
    "other_pose": "其他姿态",
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
    "Not_Labeled": "未标注",
    "No_Data": "无数据",
}


def _trans_func(context):
    """将英文结果转换为中文"""

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


def _filter_df_by_camera_and_person(df, avaiable_cameras, avaiable_persons):
    """根据 camera 编号和 person 编号筛选数据"""

    df["camera"] = df["image"].str.split("_").map(lambda x: x[1])
    df["person"] = df["image"].str.split("_").map(lambda x: x[-1][:5])
    if len(avaiable_cameras) > 0:
        df = df[df["camera"].isin(avaiable_cameras)]
    if len(avaiable_persons) > 0:
        df = df[df["person"].isin(avaiable_persons)]
    return df


def _quality_group_by_sample(x):
    """根据样本groupby，统计质量管理标签"""

    labels = list(x["label"].unique())
    cur_labels = " ".join(labels)
    row = x.iloc[0, 0].split("_")
    pid_camera = "{}_{}".format(row[-1][:5], row[1])
    action = row[0]

    labeling_err = False
    if len(labels) > 1 and "qualified" in labels:
        # 质量管理标签标记错误只有一种情况，同时存在“合格”与“不合格”标签
        labeling_err = True

    most_series_err = ""
    most_series_err_type = ""
    cur_labels_set = set(labels)
    if len(cur_labels_set & useless_error_type) > 0:
        most_series_err = " ".join(list(cur_labels_set & useless_error_type))
        most_series_err_type = "useless"
    elif len(cur_labels_set & remade_error_type) > 0:
        most_series_err = " ".join(list(cur_labels_set & remade_error_type))
        most_series_err_type = "remade"
    elif len(cur_labels_set & typo_error_type) > 0:
        most_series_err = " ".join(list(cur_labels_set & typo_error_type))
        most_series_err_type = "typo"

    return pd.Series(
        [
            pid_camera,
            action,
            cur_labels,
            most_series_err,
            most_series_err_type,
            labeling_err,
        ]
    )


def _quality_summary_groupby(x):
    """统计质量管理结果"""

    index_name = x["index_name"].unique()[0]
    res = [""] * len(actions)
    for row in x.iterrows():
        res[actions.index(row[1]["action"])] = row[1]["quality_labels"]
    for i in range(len(res)):
        if len(res[i]) == 0:
            res[i] = (
                "No_Data"
                if actions[i] not in total_samples[index_name]
                else "Not_Labeled"
            )
    return pd.Series(res)


def _outputs_quality_feedback(qualify_feedback_file, summary_df):
    """根据标记汇总结果，输出质量管理反馈信息"""

    qualify_feedback_writer = open(qualify_feedback_file, "w")
    remade_list = []
    useless_list = []
    typo_list = []
    missing_video_list = []

    # 遍历标记汇总结果，保存所有质量问题
    # 并根据质量问题的严重程度进行分类
    for row in summary_df.iterrows():
        index_name = row[0]
        for action in row[1].index:
            context = " ".join(
                [
                    index_name,
                    translations_dict[action],
                    # _trans_func(row[1][action]),
                ]
            )
            if row[1][action] == "No_Data":
                missing_video_list.append(context)
            qualities = set(row[1][action].split(" "))
            if len(qualities & useless_error_type) > 0:
                useless_list.append(context)
            if len(qualities & remade_error_type) > 0:
                remade_list.append(context)
            if len(qualities & typo_error_type) > 0:
                typo_list.append(context)

    # 1. 无法使用的数据
    if len(useless_list) > 0:
        context = "无效数据有 {} 个，分别是：".format(len(useless_list))
        print(context)
        qualify_feedback_writer.write(context + "\n")
        for context in useless_list:
            print(context)
            qualify_feedback_writer.write(context + "\n")
        print()
        qualify_feedback_writer.write("\n")

    # 2. 需要返工的数据
    if len(remade_list) > 0:
        context = "需要返工数据有 {} 个，分别是：".format(len(remade_list))
        print(context)
        qualify_feedback_writer.write(context + "\n")
        for context in remade_list:
            print(context)
            qualify_feedback_writer.write(context + "\n")
        print()
        qualify_feedback_writer.write("\n")

    # 3. 有小问题，但不影响使用的数据
    if len(typo_list) > 0:
        context = "有小问题但不影响使用的数据有 {} 个，分别是：".format(len(typo_list))
        print(context)
        qualify_feedback_writer.write(context + "\n")
        for context in typo_list:
            print(context)
            qualify_feedback_writer.write(context + "\n")
        print()
        qualify_feedback_writer.write("\n")

    # 4. 原始视频不存在
    # 4.1. {person}_{camera} 缺少某几个视频
    if len(missing_video_list) > 0:
        context = "原始视频不存在的情况有{}个，分别是：".format(len(missing_video_list))
        print(context)
        qualify_feedback_writer.write(context + "\n")
        for context in missing_video_list:
            print(context)
            qualify_feedback_writer.write(context + "\n")
        print()
        qualify_feedback_writer.write("\n")

    # 4.2. 缺少所有{person}_{camera} 的视频
    all_person_camera = {
        "{}_{}".format(person, camera)
        for person in avaiable_persons
        for camera in avaiable_cameras
    }
    cur_person_camera = set(summary_df.index.to_list())
    missing_person_camera = all_person_camera - cur_person_camera
    if len(missing_person_camera) > 0:
        for row in missing_person_camera:
            row = row.split("_")
            context = "人物{}中所有{}摄像头的数据都不存在".format(row[0], row[1])
            print(context)
            qualify_feedback_writer.write(context + "\n")
        print()
        qualify_feedback_writer.write("\n")
    
    # 5. 数据合格率（不包括原始数据不存在）
    qualified_cnt = (summary_df == 'qualified').sum().sum()
    total_cnt = (quality_summary_df.isin(quality_type)).sum().sum()
    accuracy = 100.0 * qualified_cnt / total_cnt
    context = "数据合格率（不包括原始数据缺失的样本）为{:.1f}%".format(accuracy)
    print(context)
    qualify_feedback_writer.write(context + "\n")
    print()
    qualify_feedback_writer.write("\n")

    qualify_feedback_writer.close()


def _sample_group_by_bbox(x):
    """
    在gourpby sample后再执行groupby bbox

    记录当前bbox类别（中间帧、结尾帧、其他帧）
    记录当前bbox标注的错误：
    1. 姿态标签数量不为一（如果不存在严重质量问题，那么要求所有bbox都对应一个人，要求有一个对应的姿态）
    2. 帧类别标签数量不为一（同时为中间帧、结尾帧是不允许的）
    """
    labels_list = list(x["label"])
    labels_set = set(labels_list)
    medium = labels_list.count("medium")
    end = labels_list.count("end")
    other = 0
    labeling_err = []
    if medium == 0 and end == 0:
        other = 1

    if medium != 0 and end != 0:
        labeling_err.append("同一个bbox不能同时作为中间帧与结尾帧")

    if len(labels_set & pose_type) != 1:
        labeling_err.append("同一个bbox的姿态标签不为一")

    return pd.Series([medium, end, other, " ".join(labeling_err)])


def _group_by_sample(x):
    labels_list = list(x["label"])
    labels_set = set(labels_list)
    row = x.iloc[0, 0].split("_")
    pid_camera = "{}_{}".format(row[-1][:5], row[1])
    action = row[0]

    labeling_err_list = []
    res_list = [pid_camera, action]

    # 1. 处理质量管理标签
    # 筛选不存在严重问题的数据
    # 忽略其他可能出现的质量管理问题
    if len(labels_set & series_error_type) != 0:
        res_list.append(
            "SeriesErrorData {}".format(list(labels_set & series_error_type))
        )
        res_list.append("")
        res_list += [-1, -1, -1]
        return pd.Series(res_list)

    # 2. 统计帧类别标签数量，一个样本中间帧、结尾帧的数量可能大于1

    # 3. 根据bbox进行group by操作
    # 记录可能出现的标注错误：姿态标签数量不为一，帧类别标签数量不为一，有帧类别标签或质量管理标签、但没有姿态标签
    x["bbox"] = (
        x["image"]
        + x["xmin"].astype(str)
        + x["ymin"].astype(str)
        + x["xmax"].astype(str)
        + x["ymax"].astype(str)
    )
    bbox_df = x.groupby("bbox").apply(_sample_group_by_bbox)
    bbox_df.columns = ["medium", "end", "other", "labeling_err_details"]
    medium_cnt, end_cnt, other_cnt = (
        bbox_df["medium"].sum(),
        bbox_df["end"].sum(),
        bbox_df["other"].sum(),
    )
    res_list.append("{}-{}-{}".format(medium_cnt, end_cnt, other_cnt))

    for cur_labeling_err in bbox_df["labeling_err_details"].unique():
        if len(cur_labeling_err) > 0:
            labeling_err_list += cur_labeling_err.split(" ")

    res_list.append(" ".join(list(set(labeling_err_list))))
    res_list += [medium_cnt, end_cnt, other_cnt]
    return pd.Series(res_list)


def _sample_group_by_index_name(x):
    index_name = x["index_name"].unique()[0]
    res = [""] * len(actions)
    for row in x.iterrows():
        labeling_err_details = row[1]["labeling_err_details"]
        if len(labeling_err_details) > 0:
            r = "LabelingError: " + labeling_err_details
        else:
            r = row[1]["data"]
        res[actions.index(row[1]["action"])] = r
    for i in range(len(res)):
        if len(res[i]) == 0:
            res[i] = (
                "No_Data"
                if actions[i] not in total_samples[index_name]
                else "Not_Labeled"
            )
    return pd.Series(res)


def _outputs_labeling_err(quality_df, quality_summary_df, sample_df, writer):
    # 导出标记错误
    # 1. 同时存在“合格”与“不合格”质量管理标签。
    quality_labeling_err_df = quality_df[quality_df["labeling_error"]]
    if len(quality_labeling_err_df) > 0:
        context = "共有{}个样本同时存在“合格”与“不合格”标签，分别是：".format(
            len(quality_labeling_err_df)
        )
        print(context)
        writer.write(context + "\n")
        for row in quality_labeling_err_df.iterrows():
            context = " ".join([row[1]["index_name"], row[1]["action"]])
            print(context)
            writer.write(context + "\n")
        print()
        writer.write("\n")

    # 2. 有视频没有质量管理标注
    missing_quality_list = []
    for row in quality_summary_df.iterrows():
        for cur_action in actions:
            if row[1][cur_action] == "Not_Labeled":
                missing_quality_list.append(" ".join([row[0], cur_action]))
    if len(missing_quality_list) > 0:
        context = "漏标质量管理标签的样本共有{}个，分别是：".format(len(missing_quality_list))
        print(context)
        writer.write(context + "\n")
        for context in missing_quality_list:
            print(context)
            writer.write(context + "\n")
        print()
        writer.write("\n")

    # 3. 漏标中间帧
    missing_medium_df = sample_df[sample_df["medium"] == 0]
    if len(missing_medium_df) > 0:
        context = "漏标中间帧的样本共有{}个，分别是：".format(len(missing_medium_df))
        print(context)
        writer.write(context + "\n")
        for row in missing_medium_df.iterrows():
            context = " ".join([row[1]["index_name"], row[1]["action"]])
            print(context)
            writer.write(context + "\n")
        print()
        writer.write("\n")

    # 4. 其他标注错误
    other_labeling_err_df = sample_df[sample_df["has_other_labeling_err"]]
    if len(other_labeling_err_df) > 0:
        context = "存在其他标注错误的样本共有{}个，分别是".format(len(other_labeling_err_df))
        print(context)
        writer.write(context + "\n")
        for row in other_labeling_err_df.iterrows():
            context = " ".join(
                [
                    row[1]["index_name"],
                    row[1]["action"],
                    row[1]["labeling_err_details"],
                ]
            )
            print(context)
            writer.write(context + "\n")
        print()
        writer.write("\n")


if __name__ == "__main__":
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
    df["sample"] = df["image"].str.split("#").apply(lambda x: x[0])

    # 获取质量相关数据
    # 根据sample执行 groupby，统计质量管理标签
    quality_df = df[df["label"].isin(quality_type)]
    quality_df = quality_df.groupby("sample").apply(_quality_group_by_sample)
    quality_df.columns = [
        "index_name",
        "action",
        "quality_labels",
        "most_series_error",
        "most_series_error_type",
        "labeling_error",
    ]

    # 获取质量统计结果
    quality_summary_df = quality_df.groupby("index_name").apply(
        _quality_summary_groupby
    )
    quality_summary_df.columns = actions

    # 中文版质量管理 summary
    if enable_output_summary:
        quality_summary_ch_df = quality_summary_df.copy()
        for col in quality_summary_ch_df.columns:
            quality_summary_ch_df[col] = quality_summary_ch_df[col].apply(
                _trans_func
            )
        quality_summary_ch_df.columns = [
            translations_dict[k] for k in list(quality_summary_ch_df.columns)
        ]
        quality_summary_ch_df.to_csv(quality_summary_file)

    # 获取质量管理反馈结果
    if enable_quality_feedback:
        _outputs_quality_feedback(qualify_feedback_file, quality_summary_df)

    # 根据样本统计bbox的一些信息
    sample_df = df.groupby("sample").apply(_group_by_sample)
    sample_df.columns = [
        "index_name",
        "action",
        "data",
        "labeling_err_details",
        "medium",
        "end",
        "other",
    ]
    sample_df["has_other_labeling_err"] = sample_df[
        "labeling_err_details"
    ].apply(lambda x: len(x) > 0)

    # 统计每个sample的bbox信息
    sample_summary_df = sample_df.groupby("index_name").apply(
        _sample_group_by_index_name
    )
    sample_summary_df.columns = actions
    if enable_output_summary:
        sample_summary_df.to_csv(bbox_summary_file)

    # 导出标注错误反馈结果
    if enable_labeling_err:
        labeling_err_writer = open(labeling_err_file, "w")
        _outputs_labeling_err(
            quality_df, quality_summary_df, sample_df, labeling_err_writer
        )
        labeling_err_writer.close()
