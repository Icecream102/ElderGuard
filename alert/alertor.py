from twilio.rest import Client
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import pymysql.cursors


def get_info(info, stream_path):
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='root',
                         database='elderguard')
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    sql = "select {} from user_info where stream_path='{}'".format(info, stream_path)
    cursor.execute(sql)
    result = cursor.fetchall()[0][0]
    print(result)
    db.close()
    return result


class Alarm(object):
    def __init__(self, stream_path, behavior, when):
        # self.s = source_url
        self.behavior = behavior
        self.phone_number = get_info('phone', stream_path)
        # self.phone_number = '+8617208274142'
        self.email = get_info('email', stream_path)
        self.when = when
        # self.email = '736028517@qq.com'
        # print(stream_path, self.behavior)

    def phone_alert(self):
        if self.phone_number != '':
            account_sid = "AC42770385d76f542b1b41a75a5b1e4187"
            auth_token = "c15cca1798f293087231e7f3b5c4d064"
            client = Client(account_sid, auth_token)
            # notice = "Attention！ “{}” has occurred！".format(self.behavior)
            message = client.messages.create(
                # body="Attention！",
                body='{}!At {}'.format(self.behavior, self.when),
                # body="Hello from Twilio",
                from_="+16053163162",
                # to="+8617208274142"
                to=self.phone_number
            )
            print(message.sid)

    def email_alert(self):
        if self.email != '':
            host_server = 'smtp.qq.com'  # qq邮箱smtp服务器
            sender_qq = '3041951088@qq.com'  # 发件人邮箱
            receiver = [self.email]  # 收件人邮箱
            pwd = 'tjcleaxbpugpdeai'
            mail_title = 'ElderGuard Alert'  # 邮件标题
            mail_content = "Attention！ “{}” has occurred at {}！".format(self.behavior, self.when)  # 邮件正文内容
            # 初始化一个邮件主体
            msg = MIMEMultipart()
            msg["Subject"] = Header(mail_title, 'utf-8')
            msg["From"] = sender_qq
            # msg["To"] = Header("测试邮箱",'utf-8')
            msg['To'] = ";".join(receiver)
            # 邮件正文内容
            msg.attach(MIMEText(mail_content, 'plain', 'utf-8'))

            smtp = SMTP_SSL(host_server)  # ssl登录

            # login(user,password):
            # user:登录邮箱的用户名。
            # password：登录邮箱的密码，像笔者用的是网易邮箱，网易邮箱一般是网页版，需要用到客户端密码，需要在网页版的网易邮箱中设置授权码，该授权码即为客户端密码。
            smtp.login(sender_qq, pwd)

            # sendmail(from_addr,to_addrs,msg,...):
            # from_addr:邮件发送者地址
            # to_addrs:邮件接收者地址。字符串列表['接收地址1','接收地址2','接收地址3',...]或'接收地址'
            # msg：发送消息：邮件内容。一般是msg.as_string():as_string()是将msg(MIMEText对象或者MIMEMultipart对象)变为str。
            smtp.sendmail(sender_qq, receiver, msg.as_string())

            # quit():用于结束SMTP会话。
            smtp.quit()


# url = 'rtsp://admin:VNJDBT@192.168.137.47:554/h264/ch1/main/av_stream'
# url = '111'
# phone = get_info('phone', url)
# print('phone:{}'.format(phone))
# email = get_info('email', url)
# import time
#
# url = 'yu_home1'
# t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
# test = Alarm(url, 'fall down', t)
# test.phone_alert()
# test.email_alert()
