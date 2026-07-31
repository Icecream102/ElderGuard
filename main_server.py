import os
import cv2
import time
import torch
import argparse
import numpy as np
from threading import Thread

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from alert import alertor
from alert import pic_to_db


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# source:Source of camera or video file path
# source = '../Data/test_video/test7.mp4'
# source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
# source = './data/Videos/cam7.avi'
# source = './data/Videos/video (7).avi'
# source = '0'
# source = 'D:/learn/ciscn/data/FallDataset/Home_01/Videos/video (27).avi'
# source = r'D:\learn\ciscn\data\violence_self_build\videos\v7.mp4'

# source = 'rtmp://192.168.137.1:1935/live/test'
stream_path = os.getenv('ELDERGUARD_STREAM_PATH', 'default')
source = os.getenv('ELDERGUARD_SOURCE', '0')
# source = r'D:\UNI\IT\Code\Python\ElderGuard-gpu\Data\videos\video7.avi'
still_time = 5  # 设置静止时间阈值


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def process_alarm(behavior, when):
    # alarm = alertor.Alarm(source, behavior)
    alarm = alertor.Alarm(stream_path, behavior, when)
    alarm.phone_alert()
    alarm.email_alert()


def save_frame(behavior, img, b, ti):
    # pic_saver = pic_to_db.Saver(source, behavior, img, b, ti)
    pic_saver = pic_to_db.Saver(stream_path, behavior, img, b, ti)  # 3
    pic_saver.save()


if __name__ == '__main__':
    device = os.getenv('ELDERGUARD_DEVICE', 'cuda')  # cuda or cpu

    # DETECTION MODEL.
    inp_dets = 384  # Size of input in detection model in square must be divisible by 32 (int).
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = [224, 160]  # (224x160) Size of input in pose model must be divisible by 32 (h, w)
    # inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1],
                               device=device)  # restnet50:Backbone model for SPPE FastPose model.

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = source
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    fps_time = 0
    f = 0
    s_time = time.time()  # 开始检测的时间
    fall_flag = 1  # 是第一次跌倒
    fall_time = time.time()
    viol_flag = 1  # 是第一次被殴打
    viol_time = time.time()
    lying_flag = 1  # 某个时间段内的第一次跌倒
    lying_start = 0
    while cam.grabbed():  # grab:指向下一个帧
        if f == 0:
            print('==========begin!===========')
        f += 1
        frame = cam.getitem()  # shape=(384,384,3)

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)  # tensor:(1,7)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            # clr = (0, 255, 0)  # green
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (205, 51, 51)  # red:255, 0, 0
                    # 开启新线程保存异常帧
                    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    save_img = frame.copy()
                    p1 = Thread(target=save_frame, args=('fall', save_img, bbox, t,))
                    p1.start()

                    cur_time = time.time()
                    if cur_time - fall_time > 120 or fall_flag == 1:  # 告警条件：两次异常间隔大于2min或者是第一次跌倒
                        fall_time = cur_time
                        fall_flag = 0
                        p = Thread(target=process_alarm, args=('fall down', t,))
                        p.start()
                elif action_name == 'violence':
                    print('###########violence!############')
                    clr = (72, 118, 255)  # blue:0, 0, 225
                    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    save_img = frame.copy()
                    p3 = Thread(target=save_frame, args=('viol', save_img, bbox, t,))
                    p3.start()

                    cur_time = time.time()
                    if cur_time - viol_time > 120 or viol_flag == 1:
                        viol_time = cur_time
                        viol_flag = 0
                        p = Thread(target=process_alarm, args=('violence', t,))
                        p.start()
                elif action_name == 'Lying Down':  # 长时间未活动检测
                    if lying_flag == 1:  # 是第一次检测到躺下
                        lying_start = time.time()
                        lying_flag = 0  # 下一次就不是第一次
                        print('...lying...')
                    else:  # 不是第一次检测到躺下
                        lying_end = time.time()
                        lying_time = lying_end - lying_start
                        if lying_time > still_time:  # 自定义静止时间 6h:21600,测试时设置为5s
                            lying_flag = 1  # 避免连续告警，若一直静止，则相当于每still_time告警一次
                            # 存储
                            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                            save_img = frame.copy()
                            p2 = Thread(target=save_frame, args=('still', save_img, bbox, t,))
                            p2.start()
                            # 告警
                            p = Thread(target=process_alarm, args=('still', t,))
                            p.start()
                else:
                    lying_flag = 1  # 连续几帧的跌倒被中断，重置静止事件
    e_time = time.time()
    total_time = e_time - s_time
    print('this video needs {:.4f} to detect'.format(total_time))

    # Clear resource.
    cam.stop()
    cv2.destroyAllWindows()
