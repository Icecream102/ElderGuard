"""
保存异常画面到数据库和文件夹中
"""
import pymysql.cursors
import cv2
import os


def to_db(pic_name, stream_path, abnormal_type, date, time):
    db = pymysql.connect(host=os.getenv('ELDERGUARD_DB_HOST', 'localhost'),
                         user=os.getenv('ELDERGUARD_DB_USER', 'elderguard'),
                         password=os.getenv('ELDERGUARD_DB_PASSWORD', ''),
                         database=os.getenv('ELDERGUARD_DB_NAME', 'elderguard'))
    cursor = db.cursor()
    # 查询摄像头的流媒体路径获取对应的用户名、环境名、摄像头名称
    sql_select = "select user_name,env_name,camera_name from user_info where stream_path='{}'".format(stream_path)
    cursor.execute(sql_select)
    result = cursor.fetchall()[0]
    user_name = result[0]
    env_name = result[1]
    camera_name = result[2]
    group_id = int(date[6:].replace('-', '') + time[:5].replace(':', ''))
    # print(group_id)
    # print(user_name)
    # print(env_name)
    # print(camera_name)

    # 异常画面保存的路径
    frame_path = os.path.join(os.getenv('ELDERGUARD_ALERT_OUTPUT', 'abnormal_pic'),
                              user_name, env_name, camera_name)
    web_path = '/abn_frame/' + user_name + '/' + env_name + '/' + camera_name + '/' + pic_name + '-' + abnormal_type + '.png'  # 网络访问的路径
    # print(frame_path)
    # print(web_path)
    # 若数据库表设置唯一索引，则可以忽略时间相同的帧
    sql_insert = "INSERT INTO abnormal_frame(group_id,user_name,camera_name,date,time,abnormal_type,frame_path) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    args = (group_id, user_name, camera_name, date, time, abnormal_type, web_path)
    cursor.execute(sql_insert, args)
    # 提交事务
    db.commit()
    cursor.close()
    db.close()

    return frame_path


'''
t2 = t1.split(' ')[0]  # 日期2023-05-02
                    t3 = t1.split(' ')[1]  # 时间11:19:40
                    t4 = t1.replace(' ', '-').replace(':', '-')  # 作为文件名的一部分
'''


class Saver(object):
    def __init__(self, stream_path, behavior, frame, bbox, pre_time):
        # self.s = source_url
        self.stream_path = stream_path
        self.behavior = behavior
        # self.path = 'D:/learn/ciscn/data/detected_pic/'
        self.frame = frame
        self.bbox = bbox
        self.pre_time = pre_time
        self.date = pre_time.split(' ')[0]  # 日期2023-05-02
        self.time = pre_time.split(' ')[1]  # 时间11:19:40
        self.time_name = pre_time.replace(' ', '-').replace(':', '-')  # 作为文件名的一部分

    def save(self):
        save_img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)  # 存入数据库中不用进行色道转换
        if self.behavior == 'fall':
            col = (51, 51, 205)  # 51, 51, 205:RGB的蓝
            save_img = cv2.putText(save_img, 'fall down', (self.bbox[0] + 5, self.bbox[1] + 15),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   0.4, col, 1)
        elif self.behavior == 'viol':
            col = (255, 118, 72)  # 255, 118, 72：RGB的红
            save_img = cv2.putText(save_img, 'violence', (self.bbox[0] + 5, self.bbox[1] + 15),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   0.4, col, 1)
        elif self.behavior == 'still':
            col = (255, 140, 0)  #
            save_img = cv2.putText(save_img, 'still', (self.bbox[0] + 5, self.bbox[1] + 15),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   0.4, col, 1)

        save_img = cv2.rectangle(save_img, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), col, 1)

        # 保存
        try:
            # 保存相关异常信息到数据库,并获取保存图片的路径
            # to_db(pic_name, stream_path, abnormal_type, date, time)
            d = to_db(self.time_name, self.stream_path, self.behavior, self.date, self.time)
            # 保存异常画面到对应文件夹
            if not os.path.exists(d):
                os.makedirs(d)
            p = d + '/' + self.time_name + '-' + self.behavior + '.png'
            cv2.imwrite(p, save_img)
        except:
            print('ignore a duplicate frame...')

# test
# t = '2023-05-02-11-19-40'
# try:
#     to_db(t, 'yu_home1', 'fall', '2023-05-02', '11:19:42')
# except:
#     print('ignore a duplicate frame...')
# to_db(t, 'yu_home1', 'fall', '2023-05-02', '11:19:42')
