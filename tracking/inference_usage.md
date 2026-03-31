# LCCTV Inference Usage

## Official entrypoints

The repository now provides two official inference commands:

1. `tracking/run_inference.py`
   Use this when your input is already a prepared sequence directory such as `DATA/1/img` plus `groundtruth.txt`.
2. `tracking/run_video_inference.py`
   Use this when your input is a single video file. The script will:
   - extract frames
   - automatically select the initial tracking target
   - write `groundtruth.txt`
   - run LCCTV inference
   - save the final tracking and earthquake metrics output

`tracking/test.py` is legacy code and should no longer be used for deployment.

## Recommended CPU environment

```bash
conda create -n lcctv-cpu python=3.10 -y
conda activate lcctv-cpu
pip install --force-reinstall numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

If your environment already uses NumPy 2.x and a package reports an ABI conflict, repair it with:

```bash
pip install --force-reinstall numpy==1.26.4
```

## Sequence directory inference

Run inference on an existing `DATA` directory:

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300
```

Useful options:

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --sequence 1
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --max-frames 100
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu --sequence-size 1=0.4 2_all=0.1 3=0.1
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu --output-dir output/inference/cpu_full
```

## Single video inference

### What the new script does

`tracking/run_video_inference.py` accepts one video file and produces a complete output package.

Pipeline:

1. Read the source video.
2. Extract zero-padded frames into a workspace sequence.
3. Automatically choose a stable wall-mounted target from the early frames.
4. Write the initial target box into `groundtruth.txt`.
5. Run LCCTV on the generated sequence.
6. Save tracking results, metrics, and a pipeline summary.

### Minimum command

Only the video path is required. The tracker parameter and epoch default to the current production model.

```bash
python tracking/run_video_inference.py /path/to/video.mp4
```

### Recommended command

```bash
python tracking/run_video_inference.py /path/to/video.mp4 \
  --device cpu \
  --reference-data-dir /path/to/DATA
```

### Common options

```bash
python tracking/run_video_inference.py /path/to/video.mp4 --sequence-name room4_case1
python tracking/run_video_inference.py /path/to/video.mp4 --max-frames 200
python tracking/run_video_inference.py /path/to/video.mp4 --every-n 2
python tracking/run_video_inference.py /path/to/video.mp4 --output-root /tmp/lcctv_video_run
python tracking/run_video_inference.py /path/to/video.mp4 --checkpoint /path/to/lcctv_ep0300.pth.tar
python tracking/run_video_inference.py /path/to/video.mp4 --skip-metrics
python tracking/run_video_inference.py /path/to/video.mp4 --allow-size-change
python tracking/run_video_inference.py /path/to/video.mp4 --reuse-workspace
```

## Output layout for single video inference

Default output root:

```text
output/video_inference/<video_name>/<tracker_param>_epXXXX/
```

Generated files:

```text
output/video_inference/<video_name>/<tracker_param>_epXXXX/
  video_pipeline_summary.json
  workspace/
    <sequence_name>/
      img/
        00000001.jpg
        ...
      frame_manifest.json
      groundtruth.txt
      auto_target_debug.jpg
      auto_target_patch.jpg
      auto_target_result.json
  results/
    summary.json
    run_report.txt
    <sequence_name>/
      bboxes.txt
      time.txt
      summary.json
      report.txt
```

## Key output files

- `workspace/<sequence_name>/groundtruth.txt`
  The automatically generated initial box in `x,y,w,h` format.
- `workspace/<sequence_name>/auto_target_debug.jpg`
  Visualization of the selected target and top candidates.
- `results/<sequence_name>/summary.json`
  Final tracking output for this sequence.
- `results/summary.json`
  Run-level summary for the current command.
- `video_pipeline_summary.json`
  End-to-end pipeline summary including extraction, auto target selection, and output paths.

## Notes

- If you already have clean frame data, use `run_inference.py`.
- If you only have a video file, use `run_video_inference.py`.
- The auto target selector is designed for the historical LCCTV scenes and prefers stable wall-mounted objects that match the previous annotation style.
