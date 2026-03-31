LCCTV 使用说明
================

一、当前正式入口

1. tracking/run_inference.py
   适用于已经准备好的图片序列目录，输入格式为：
   DATA/<sequence_name>/img 或 DATA/<sequence_name>/imgs
   DATA/<sequence_name>/groundtruth.txt

2. tracking/run_video_inference.py
   适用于直接输入一个视频文件。
   这是当前推荐的视频推理入口，会自动完成：
   - 视频切帧
   - 自动选择初始跟踪目标
   - 生成 groundtruth.txt
   - 调用 LCCTV 推理
   - 输出 tracking 和地震烈度结果

二、推荐目录结构

同级目录建议如下：

dz/
  LCCTV/
  DATA/

三、推荐 CPU 环境

conda create -n lcctv-cpu python=3.10 -y
conda activate lcctv-cpu
pip install --force-reinstall numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops

四、直接输入视频的运行方式

最简命令：

python tracking/run_video_inference.py /path/to/video.mp4

推荐命令：

python tracking/run_video_inference.py /path/to/video.mp4 --device cpu --reference-data-dir /path/to/DATA

常用参数：

--tracker-param            模型配置名，默认 B9_cae_center_all_ep300
--epoch                    权重 epoch，默认 300
--device                   cpu 或 cuda:0
--sequence-name            输出工作区里的序列名
--reference-data-dir       历史 DATA 根目录，用于辅助自动选择目标
--max-frames               仅处理前 N 帧
--every-n                  每 N 帧抽一帧
--output-root              指定输出目录
--checkpoint               指定权重文件
--skip-metrics             跳过地震烈度计算
--allow-size-change        允许跟踪框宽高变化
--reuse-workspace          复用已有切帧结果，不清空工作区

五、输出结果

默认输出目录：

LCCTV/output/video_inference/<video_name>/<tracker_param>_epXXXX/

其中包含：

1. workspace/<sequence_name>/
   - img/                      切帧结果
   - frame_manifest.json       切帧摘要
   - groundtruth.txt           自动生成的初始框
   - auto_target_debug.jpg     自动选目标可视化图
   - auto_target_patch.jpg     目标 patch
   - auto_target_result.json   自动选目标结果明细

2. results/<sequence_name>/
   - bboxes.txt                每帧跟踪框
   - time.txt                  每帧耗时
   - summary.json              该视频的完整推理结果
   - report.txt                文本摘要

3. 根目录文件
   - video_pipeline_summary.json  整体流水线摘要
   - results/summary.json         本次运行汇总
   - results/run_report.txt       本次运行文本汇总

六、如果你已经有图片序列

直接使用：

python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu

七、说明

tracking/test.py 是旧入口，不再建议使用。
更详细的命令说明见：

tracking/inference_usage.md
