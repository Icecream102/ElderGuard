# ElderGuard

实时老人看护行为识别项目结合人员检测、人体姿态估计、多目标跟踪与时空图卷积网络（ST-GCN），从摄像头视频流中识别跌倒等行为。


>这是研究/原型项目，不应作为紧急医疗响应系统的唯一依据。

## 功能

- 单类别 Tiny-YOLO 人体检测
- AlphaPose（FastPose）人体关键点估计
- 卡尔曼滤波与 IoU 多目标跟踪
- ST-GCN 行为识别：站立、行走、坐下、躺下、起立、跌倒及暴力殴打行为
- 可选的实时摄像头、视频文件或 RTSP/RTMP 流输入
- 可选告警与异常帧保存

## 环境要求

- Python 3.8+
- PyTorch（GPU 推理建议安装与本机 CUDA 匹配的版本）
- NVIDIA GPU（可改用 `--device cpu`，速度会较慢）

## 快速开始

```bash
git clone https://github.com/<your-account>/ElderGuard-gpu.git
cd ElderGuard-gpu
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

将预训练权重放入以下路径：

```text
Models/
├── yolo-tiny-onecls/best-model.pth
├── sppe/fast_res50_256x192.pth
└── TSSTG/tsstg-model.pth
```

模型下载来源：

- [Tiny-YOLO one-class 权重与配置](https://drive.google.com/file/d/1obEbWBSm9bXeg10FriJ7R2cGLRsg-AfP/view?usp=sharing)
- [AlphaPose FastPose ResNet-50 权重](https://drive.google.com/file/d/1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU/view?usp=sharing)
- [TSSTG 行为识别权重](https://drive.google.com/file/d/1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u/view?usp=sharing)

运行摄像头（默认设备 0）：

```bash
python main.py --camera 0 --device cuda
```

运行本地视频或网络流：

```bash
python main.py --camera path/to/video.mp4 --device cuda
python main.py --camera "rtsp://user:password@camera-host/stream" --device cuda
```

桌面界面入口为 `App.py`。部署端的告警流程入口为 `main_server.py`。

## 告警服务配置

告警功能需要 MySQL；短信功能需要 Twilio，邮件功能需要 SMTP。未配置相应变量时，该通道会被跳过。

## 项目结构

```text
├── main.py                    # 命令行实时检测入口
├── App.py                     # Tkinter 桌面界面
├── main_server.py             # 告警服务入口
├── Detection/                 # YOLO 检测器
├── SPPE/                      # 姿态估计实现
├── Track/                     # 多目标跟踪
├── Actionsrecognition/        # ST-GCN 训练与评估代码
├── alert/                     # 告警和异常帧保存
└── Models/                    # 本地模型目录
```


## 致谢

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [COCO](https://cocodataset.org/)
