# 数据标注与质量管理

+ [0. 前言](#0-前言)
+ [1. 数据标注](#1数据标注)
  + [1.1. 标注期望实现的目标](#11-标注期望实现的目标)
  + [1.2. 质量管理标签细则](#12-质量管理标签细则)
  + [1.3. 标注可能出现的错误](#13-标注可能出现的错误)
  + [1.4. 需要标注的内容](#14-需要标注的内容)
+ [2. 脚本介绍以及使用](#2-脚本介绍以及使用)
  + [2.1. 视频重命名脚本](#21-视频重命名脚本)
  + [2.2. 统计汇总脚本](#22-统计汇总脚本)
  + [2.3. 错误样本可视化脚本](#23-错误样本可视化脚本)
+ [3. 注意事项](#3-注意事项)

## 0. 前言
+ 希望大家仔细阅读这个文档。
+ 在上传标注借过前，先运行 `summary.py` 脚本，修正 `labeling_error.txt` 中的错误。

## 1. 数据标注

### 1.1. 标注期望实现的目标
+ 质量管理：及时向曼孚反馈数据质量。
+ 为后续行为识别模型提供输入数据。

### 1.2. 质量管理标签细则
+ 质量管理标签一共可分为四个大类：
  + 数据合格
  + 数据质量有问题，不影响使用，无需返工，后续拍摄需要注意：一般是文件名命名
  + 数据质量有问题，需要返工处理：例如视频长度不足、视频内容与文件名行为不符
  + 数据质量有问题，返工也无法修复
+ 标注人物分类
  + 目标人物：即做`目标动作`的人员。
  + 非目标人物：即工作人员。
+ 质量管理相关标签包括：
  + `qualified`：数据合格
  + `err_remade_action_lacking`：**需要返工**的数据
  + `err_useless`：无效数据
  + `err_name`：命名错误
  + `err_unknown`：其他**返工也无法修复**的错误

### 1.3. 标注可能出现的错误
+ 一个样本（视频）中，同时存在“合格”与“不合格”标签。
+ 一个样本（视频）中，缺少 质量管理 标签。
+ 一个样本（视频）中，缺少 中间帧 标签。
+ 同一个框不能存在多个姿态。
+ 同一个框不能同时对应中间帧与结尾帧。

### 1.4. 需要标注的内容
+ 对于每个样本，都需要标注质量管理标签。
  + 质量管理标签随便标在哪个bbox上。
  + 对于存在严重问题（需要返工，或无效数据）的样本，只需要标注质量管理标签，其他标签直接忽略。
+ 对于合格样本，或存在小问题的样本
  + 需要标注中间帧，即在目标人物上标注中间帧+姿态，非目标人物标注姿态。

## 2. 脚本介绍以及使用

### 2.1. 视频重命名脚本
+ 脚本：`rename_videos.py`
+ 输入参数：
  + `src_video_path`：需要重命名的视频路径。会寻找所有子目录中的 .mp4 文件进行重命名
  + `target_video_path`：重命名结果保存路径。
+ 详细功能：
  + 原始视频名称使用的都是中文，为了方便后续工作，重命名为英文。
  + 根据需求，将原始视频分为三类：
    + `step1`：行为识别摄像头拍摄的数据。
    + `step2`：ReID摄像头拍摄的数据。
    + `useless`：非行为识别数据或不符合行为识别命名要求的数据。

### 2.2. 统计汇总脚本
+ 脚本：`summary.ipynb` 和 `summary.py`
+ 输入参数：
  + `csv_dir`：标注结果路径。会寻找当前目录以及所有子目录中的 `.csv` 文件。
  + `video_dir`：标注原始视频路径。会寻找所有目录以及子目录中的 `.mp4` 文件。
  + `avaiable_cameras`：需要统计的摄像头集合，默认为step1中的 `['2m', '3m', '4m']`。
  + `avaiable_person_ids`：需要统计的人员编号，需要手动设置，例如 `[7, 8, 9, 18, 19, 20, 21, 22]`
  + `enable_labeling_err` & `labeling_err_file`
    + 前者表示 `是否导出标注错误结果`，后者表示 `标注结果错误导出文件路径`
  + `enable_output_summary` & `quality_summary_file` & `bbox_summary_file`
    + 前者表示 `是否导出汇总文件`，后两者表示两个 `汇总结果导出文件路径`。
    + 两个汇总文件分别表示 质量管理汇总与bbox汇总。
  + `enable_quality_feedback` & `qualify_feedback_file`
    + 前者表示 `是否导出质量管理反馈结果`，后者表示 `质量管理反馈结果文件路径`
+ 质量管理统计结果
  + 每一行表示 `{person}_{camera}` 
  + 每一列表示一个动作，共12个动作
  + 每一个单元格代表质量管理结果，也可能是“无数据”或“未标注”
+ bbox统计结果
  + 每一行表示 `{person}_{camera}` 
  + 每一列表示一个动作，共12个动作
  + 如果存在严重错误，则相关单元格表示为 `SeriesErrorData: [error_list]`
  + 如果存在标注问题，则相关单元格表示为 `LabelingError: str`
  + 如果相关单元格没别的问题，则表示为 `{medium_cnt}-{end_cnt}-{other_cnt}`，统计每个样本各类bbox的数量
+ 质量管理反馈功能
  + 将上一步的 `质量管理统计结果` 的结果作为输入，将所有存在质量问题的标签进行分类，并输出结果。
    + 所有质量问题可以按照 `1.3.`
  + 除了统计标注的质量错误类别外，还会统计 `没有原始视频` 的样本。

### 2.3. 错误样本可视化脚本
+ 脚本：`show.py` 或 `show.ipynb`
+ 输入参数：
  + `qualify_feedback_file`：`统计汇总脚本` 中输出的 `质量管理反馈结果文件`
  + `video_path`：视频保存路径
+ 功能详解
  + 首先，会通过字典保存 `video_path` 中所有视频的路径。
    + key 为 `{person}_{camera}_{action}`
    + value 为对应视频的路径
  + 其次，解析 `qualify_feedback_file`，按照三类质量问题进行保存，并保存质量问题细节。
    + 每个样本对应一个元组，元组有两个元素。
    + 第一个元素保存对应视频的路径。
    + 第二个元素保存对应质量问题细节，即 `qualify_feedback_file` 对应行中所有字符
  + 最后，视频可视化
    + 根据视频路径读取视频。
    + 在视频上添加文字，并通过 `IPython`(jupyter中) 或 `cv2.imshow`(py脚本中) 输出。

## 3. 注意事项
+ 一个视频某个动作多次出现，每次出现都标注一次。**如果其中一次行为合格，则所有行为都标注为合格**。
  + 例如，一个视频有两个动作时，一般是坐着做一次，站着做一次，总有一次姿态可能就跟文件名不符了，这时候就忽略姿态不符的情况，标注一个 `qualified` 即可。
  + 每个动作都对应一个中间帧，每个中间帧都需要标注 `qualified`。
+ 需要返工的数据包括：
  + 视频截取不完整或太多。
  + 视频中行为与文件名行为不符。
+ 无效数据主要是**肉眼无法分辨行为类别**的数据，造成的原因可能是：
  + 演员演技浮夸。
  + 演员被遮挡。
