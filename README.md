# 数据标注与质量管理

+ [0. 前言](#0-前言)
+ [1. 数据标注](#1数据标注)
  + [1.1. 标注目标](#11-标注目标)
  + [1.2. 质量管理标签细则](#12-质量管理标签细则)
  + [1.3. 标注可能出现的错误](#13-标注可能出现的错误)
+ [2. 脚本介绍以及使用](#2-脚本介绍以及使用)
  + [2.1. 视频重命名脚本](#21-视频重命名脚本)
  + [2.2. 统计汇总脚本](#22-统计汇总脚本)
  + [2.3. 错误样本可视化脚本](#23-错误样本可视化脚本)

## 0. 前言
+ 希望大家仔细阅读这个文档。
+ 在上传标注借过前，先运行 `suumary.py` 脚本，修正 `labeling_error.txt` 中的错误。

## 1. 数据标注

### 1.1. 标注目标
+ 质量管理：及时向曼孚反馈数据质量。
+ 为后续行为识别模型提供输入数据。

### 1.2. 质量管理标签细则
+ 质量管理标签一共可分为四个大类：
  + 数据合格
  + 数据质量有问题，不影响使用，无需返工，后续拍摄需要注意：一般是文件名命名
  + 数据质量有问题，需要返工处理：例如视频长度不足、视频内容与文件名行为不符
  + 数据质量有问题，返工也无法修复
+ 对于不同数据质量的标注要求：
  + 数据合格：标注中间帧（四类标签）与结尾帧（三类标签）
  + 有问题，但不影响使用：标注中间帧（四类标签）与结尾帧（三类标签）
  + 有问题，需要返工：标注中间帧（随便标就行，只需要标错误类型与帧类别）
  + 有问题，返工也无法修复：标注中间帧（只需要标错误类型与帧类别）
+ 质量管理相关标签包括：
  + `qualified`：数据合格
  + `err_camera_lacking`：视频中画面缺失导致**视频不可用**，**返工也无法修复**
  + `err_action_lacking`：视频中行为不完整或行为类别与视频不符，**需要返工**
  + `err_light`：文件名命名中 “光照” 类别错误，属于**有问题但不影响使用**的数据
  + `err_pose`：文件名命名中 “姿态” 类别错误，属于**有问题但不影响使用**的数据
  + `err_sleeve`：文件名命名中 “衣着” 类别错误，属于**有问题但不影响使用**的数据
  + `err_view`：文件名命名中 “视角” 类别错误，属于**有问题但不影响使用**的数据
  + `err_shelter`：文件名命名中 “遮挡” 类别错误，属于**有问题但不影响使用**的数据
  + `err_unknown`：其他**返工也无法修复**的错误

### 1.3. 标注可能出现的错误
+ 忘记标注：可能是忘记标注，也可能是忘记指定`中间帧`。
+ 标签数量不符合要求
  + 中间帧、结尾帧行为标签不止一个
  + 姿态标签没有或不止一个
  + 中间帧、结尾帧的帧种类标签不止一个
  + 中间帧不存在质量标签
+ 存在未知标签
+ 质量标签异常：即同时存在合格与错误

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
  + `enable_summary` & `summary_file`
    + 前者表示 `是否对标记结果进行汇总`，后者表示 `汇总结果导出文件路径`
  + `enable_quality_feedback` & `qualify_feedback_file`
    + 前者表示 `是否导出质量管理反馈结果`，后者表示 `质量管理反馈结果文件路径`
+ 标注错误结果统计功能
  + 统计漏标以及错标数据
  + bbox的类型可分为三类：
    + 第一类：中间帧bbox
    + 第二类：结尾帧bbox
    + 第三类：非目标bbox（即视频中除了目标人物外，其他人物的bbox）
  + 所谓“漏标”，指的是某个视频没有第一类bbox，即 `中间帧` bbox。
    + 错误可能是漏标了视频，也可能是标了视频但漏标了 `中间帧` 标签。
  + 所谓“错标”，可能有以下情况：
    + 存在未知标签
    + 同时存在中间帧标签与结尾帧标签
    + 非目标bbox
      + 姿态标签不存在或不止一个。
    + 中间帧：
      + 漏标质量管理标签
      + 质量管理标签中同时存在 `qualified` 以及其他错误标签
      + 行为标签遗漏或不止一个
      + 行为标签与文件名中的行为标签不同
      + 姿态标签遗漏或不止一个
    + 结尾帧：
      + 行为标签遗漏或不止一个
      + 行为标签与文件名中的行为标签不同
      + 姿态标签遗漏或不止一个
+ 标注结果统计功能
  + 按照 `{person}_{camera}` 分别统计12个动作的质量
  + 注意：只统计没有标注错误的样本。有标注错误的样本会被归类为 `未标注或标注无效`
+ 质量管理反馈功能
  + 将上一步的 `标注结果统计` 的结果作为输入，将所有存在质量问题的标签进行分类，并输出结果。
  + 所有质量问题可以按照 `1.3.` 分为三类
    + 无效数据
    + 需要返工
    + 有小问题但不影响使用

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