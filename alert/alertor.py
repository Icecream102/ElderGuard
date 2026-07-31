"""Alert delivery helpers configured exclusively through environment variables."""

import os
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP_SSL

import pymysql.cursors
from twilio.rest import Client


def _db_connection():
    return pymysql.connect(
        host=os.getenv('ELDERGUARD_DB_HOST', 'localhost'),
        user=os.getenv('ELDERGUARD_DB_USER', 'elderguard'),
        password=os.getenv('ELDERGUARD_DB_PASSWORD', ''),
        database=os.getenv('ELDERGUARD_DB_NAME', 'elderguard'),
    )


def get_info(info, stream_path):
    """Read a recipient field for a stream from the configured ElderGuard database."""
    if info not in {'phone', 'email'}:
        raise ValueError('Unsupported user-info field: {}'.format(info))
    db = _db_connection()
    try:
        with db.cursor() as cursor:
            cursor.execute('SELECT {} FROM user_info WHERE stream_path=%s'.format(info), (stream_path,))
            row = cursor.fetchone()
            return row[0] if row else ''
    finally:
        db.close()


class Alarm:
    def __init__(self, stream_path, behavior, when):
        self.behavior = behavior
        self.phone_number = get_info('phone', stream_path)
        self.email = get_info('email', stream_path)
        self.when = when

    def phone_alert(self):
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_FROM_NUMBER')
        if not (self.phone_number and account_sid and auth_token and from_number):
            return
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body='{}! At {}'.format(self.behavior, self.when),
            from_=from_number,
            to=self.phone_number,
        )
        print(message.sid)

    def email_alert(self):
        smtp_host = os.getenv('ELDERGUARD_SMTP_HOST')
        smtp_user = os.getenv('ELDERGUARD_SMTP_USER')
        smtp_password = os.getenv('ELDERGUARD_SMTP_PASSWORD')
        sender = os.getenv('ELDERGUARD_SMTP_FROM', smtp_user)
        if not (self.email and smtp_host and smtp_user and smtp_password and sender):
            return

        msg = MIMEMultipart()
        msg['Subject'] = Header('ElderGuard Alert', 'utf-8')
        msg['From'] = sender
        msg['To'] = self.email
        msg.attach(MIMEText('Attention: {} occurred at {}.'.format(self.behavior, self.when), 'plain', 'utf-8'))

        smtp = SMTP_SSL(smtp_host, int(os.getenv('ELDERGUARD_SMTP_PORT', '465')))
        try:
            smtp.login(smtp_user, smtp_password)
            smtp.sendmail(sender, [self.email], msg.as_string())
        finally:
            smtp.quit()
