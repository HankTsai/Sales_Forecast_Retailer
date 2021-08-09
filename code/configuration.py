
import os
import logging
from pymssql import connect
from datetime import datetime
from configparser import ConfigParser

config = ConfigParser()
config.read('setting.ini')


class CodeLogger:
    """log儲存設定模組"""
    def __init__(self):
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.formatter = logging.Formatter(
            '["%(asctime)s - %(levelname)s - %(name)s - %(message)s" - function:%(funcName)s - line:%(lineno)d]')
        self.log_name = config['filepath']['log_path'] + datetime.now().strftime("forecast_%Y-%m-%d_%H-%M-%S.log")
        logging.basicConfig(level=logging.INFO, datefmt='%Y%m%d_%H:%M:%S',)

    def store_logger(self):
        """設定log儲存"""
        handler = logging.FileHandler(self.log_name, "w", encoding = "UTF-8")
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def show_logger(self):
        """設定log在終端機顯示"""
        console = logging.StreamHandler()
        console.setLevel(logging.FATAL)
        console.setFormatter(self.formatter)
        self.logger.addHandler(console)


class DBConnect:
    """繼承並設計DB連線處理"""
    def __init__(self):
        self.host = config['connect']['server']
        self.user = config['connect']['username']
        self.password = config['connect']['password']
        self.database = config['connect']['database']
        self.conn = connect(host=self.host, user=self.user, password=self.password, database=self.database, autocommit=True)

    def query(self, sql, as_dict=False, para=()):
        """查詢DB數據"""
        # as_dict 是讓數據呈現key/value型態
        try:
            cursor = self.conn.cursor(as_dict)
            if para:
                cursor.execute(sql,para)
                return cursor
            else:
                cursor.execute(sql)
                return cursor
        except Exception as me:
            CodeLogger().logger.error(me)

    def insert(self, sql, para=()):
        """新增DB數據"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,para)
        except Exception as me:
            CodeLogger().logger.error(me)

    def delete(self, sql, para=()):
        """刪除DB數據"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,para)
        except Exception as me:
            CodeLogger().logger.error(me)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()