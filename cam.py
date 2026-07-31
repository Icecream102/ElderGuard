import os
import cv2
import queue
import threading

q = queue.Queue()


def Receive():
    print("start Reveive")
    source = os.getenv('ELDERGUARD_SOURCE', '0')
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    # 按帧读取视频
    # ret是布尔型，正确读取则返回True，读取失败或读取视频结尾则会返回False
    # frame为每一帧的图像
    ret, frame = cap.read()
    # frame放入队列

    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
