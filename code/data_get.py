

from datetime import datetime
import pandas as pd
from configparser import ConfigParser
from dateutil.relativedelta import relativedelta
from configuration import DBConnect, CodeLogger

conn = DBConnect()
logg = CodeLogger()
config = ConfigParser()
config.read('setting.ini')


class DataGet:

    def __init__(self, pass_day, run_day, date):
        self.pass_day = pass_day
        self.run_day = run_day
        self.date = date
        self.row_train_query = config['filepath']['train_query']
        self.row_predict_query = config['filepath']['predict_query']
        self.storeid_query = "select LOC_ID from SFI_I21_STORE;"

    def verify_para(self, pass_day='0', run_day='7', input_date=datetime.now()):
        if not pass_day.isdigit() or not run_day.isdigit():
            print('pass 與 run 輸入格是錯誤，必須是數字')
        if int(pass_day) < 0:
            print('pass 輸入不可小於0')
            pass_day = 0
        if int(run_day) < 1:
            print('run 輸入不可小於1')
            run_day = 1
        if not input_date.strftime("%Y/%m/%d"):
            print("--date後的日期格是輸入錯誤，必須是:%Y/%m/%d")

    def get_query(self):
        """將基礎sql指令帶入"""
        try:
            with open(self.row_train_query, 'r') as file:
                T_query = file.read()
            with open(self.row_predict_query, 'r') as file:
                P_query = file.read()
            data = conn.query(self.storeid_query)
            store_ids = [str(item[0]) for item in data]
            return T_query, P_query, store_ids
        except Exception as me:
            logg.logger.error(me)

    def query_modify(self, query, store_id, data_type=""):
        """修改SQL指令"""
        try:
            start_date = ""; end_date=""
            if data_type == 'train':
                start_date = (self.date + relativedelta(years=-3)).strftime("%Y/%m/%d")
                end_date = (self.date + relativedelta(months=-3)).strftime("%Y/%m/%d")
            elif data_type == 'verify':
                start_date = (self.date + relativedelta(months=-3)).strftime("%Y/%m/%d")
                end_date = self.date.strftime("%Y/%m/%d")
            elif data_type == 'predict':
                start_date = (self.date + relativedelta(days= (self.pass_day+1))).strftime("%Y%m%d")
                end_date = (self.date + relativedelta(days= (self.pass_day+self.run_day))).strftime("%Y%m%d")

            query = query.replace("@StartDate@", start_date)
            query = query.replace("@EndDate@", end_date)
            query = query.replace("@StoreId@", store_id)
            return query
        except Exception as me:
            logg.logger.error(me)

    def get_dataframe(self,T_query, P_query, store_ids):
        """使用sql指令獲取數據"""
        data_set = {}
        try:
            for store_id in store_ids:
                data = []
                train_query = self.query_modify(T_query, store_id, data_type='train')
                cursor = conn.query(train_query, as_dict=True)
                train_data = pd.DataFrame(cursor.fetchall())

                verify_query = self.query_modify(T_query, store_id, data_type='verify')
                cursor = conn.query(verify_query, as_dict=True)
                verify_data = pd.DataFrame(cursor.fetchall())

                predict_query = self.query_modify(P_query, store_id, data_type='predict')
                cursor = conn.query(predict_query, as_dict=True)
                predict_data = pd.DataFrame(cursor.fetchall())

                data = [train_data, verify_data, predict_data]
                data_set[store_id] = data
            return data_set
        except Exception as me:
            logg.logger.error(me)


