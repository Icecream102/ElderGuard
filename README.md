# ElderGuard

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Status: Prototype](https://img.shields.io/badge/Status-Prototype-orange?style=flat-square)](#elderguard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)


ElderGuard is a real-time elder-care activity recognition prototype. It combines person detection, human pose estimation, multi-object tracking, and spatiotemporal graph convolutional networks (ST-GCN) to identify falls and other potentially dangerous behaviours in camera streams.

> **Important:** This is a research and prototype project. It must not be used as the sole basis for emergency medical response.

## Alert notifications

<table>
  <thead>
    <tr>
      <th>Monitoring software</th>
      <th>SMS</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="33.33%"><img src="results/告警-监控软件.jpg" alt="Monitoring software alert" width="100%"></td>
      <td width="33.33%"><img src="results/告警-短信.jpg" alt="SMS alert" width="100%"></td>
      <td width="33.33%"><img src="results/告警-邮件.jpg" alt="Email alert" width="100%"></td>
    </tr>
  </tbody>
</table>

## Detected abnormal events

<table>
  <thead>
    <tr>
      <th>Physical assault</th>
      <th>Fall</th>
      <th>No movement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="33.33%"><img src="results/异常画面-殴打.jpg" alt="Detected physical assault" width="100%"></td>
      <td width="33.33%"><img src="results/异常画面-跌倒.jpg" alt="Detected fall" width="100%"></td>
      <td width="33.33%"><img src="results/异常画面-静止不动.jpg" alt="Detected no movement" width="100%"></td>
    </tr>
  </tbody>
</table>


## Project structure

```text
├── main.py                    # Command-line real-time detection entry point
├── App.py                     # Tkinter desktop application
├── main_server.py             # Alert-service entry point
├── Detection/                 # YOLO detector
├── SPPE/                      # Pose-estimation implementation
├── Track/                     # Multi-object tracking
├── Actionsrecognition/        # ST-GCN training and evaluation code
├── alert/                     # Alerts and abnormal-frame capture
└── Models/                    # Local pretrained-model directory
```

## Features

- Single-class Tiny-YOLO person detection
- AlphaPose (FastPose) human keypoint estimation
- Kalman-filter and IoU-based multi-object tracking
- ST-GCN action recognition for standing, walking, sitting, lying down, standing up, falling, and physical assault
- Input from a live camera, local video file, RTSP stream, or RTMP stream
- Optional alerts and abnormal-frame capture

## Requirements

- Python 3.8 or later
- PyTorch; use a version compatible with your local CUDA installation for GPU inference
- NVIDIA GPU recommended; CPU inference is available through `--device cpu` but is slower

## Quick start

```bash
git clone https://github.com/Icecream102/ElderGuard.git
cd ElderGuard
pip install -r requirements.txt
```

Download the pretrained weights and place them in the following locations:

```text
Models/
├── yolo-tiny-onecls/best-model.pth
├── sppe/fast_res50_256x192.pth
└── TSSTG/tsstg-model.pth
```

Model downloads:

- [Tiny-YOLO one-class weights and configuration](https://huggingface.co/Luanneee/yolo-tiny-onecls/tree/main)
- [AlphaPose FastPose ResNet-50 weights](https://huggingface.co/Luanneee/sppe/tree/main)
- [TSSTG action-recognition weights](https://huggingface.co/Luanneee/TSSTG/tree/main)

Run with the default camera (device `0`):

```bash
python main.py --camera 0 --device cuda
```

Run with a local video file or network stream:

```bash
python main.py --camera path/to/video.mp4 --device cuda
python main.py --camera "rtsp://user:password@camera-host/stream" --device cuda
```

The desktop application entry point is `App.py`. The alert-service entry point is `main_server.py`.

## Alert-service configuration

The alert service requires MySQL. SMS alerts require Twilio and email alerts require SMTP.

## Future work

- - Prepare and deploy ElderGuard as an online application, with monitoring and operational safeguards.



## Acknowledgements

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [COCO](https://cocodataset.org/)
