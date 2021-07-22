

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

    def __init__(self, pass_day, run_day, input_date):
        self.pass_day = pass_day
        self.run_day = run_day
        self.input_date = input_date
        self.row_train_query = config['filepath']['train_query']
        self.row_predict_query = config['filepath']['predict_query']
        self.row_date_update = config['filepath']['date_update']
        self.storeid_query = "select LOC_ID from SFI_I21_STORE;"

    def get_query(self):
        """將基礎sql指令帶入"""
        try:
            with open(self.row_train_query, 'r',encoding="utf-8") as file:
                T_query = file.read()
            with open(self.row_predict_query, 'r',encoding="utf-8") as file:
                P_query = file.read()
            with open(self.row_date_update, 'r',encoding="utf-8") as file:
                D_update = file.read()
            data = conn.query(self.storeid_query)
            store_ids = [str(item[0]) for item in data]
            return T_query, P_query, D_update, store_ids
        except Exception as me:
            logg.logger.error(me)

    def query_modify(self, query, store_id, data_type=""):
        """修改SQL指令"""
        try:
            start_date = ""; end_date=""
            if data_type == 'train':
                start_date = (self.input_date + relativedelta(years=-3)).strftime("%Y/%m/%d")
                end_date = (self.input_date + relativedelta(months=-3)).strftime("%Y/%m/%d")
            elif data_type == 'verify':
                start_date = (self.input_date + relativedelta(months=-3)).strftime("%Y/%m/%d")
                end_date = self.input_date.strftime("%Y/%m/%d")
            elif data_type == 'predict':
                start_date = (self.input_date + relativedelta(days= (self.pass_day+1))).strftime("%Y%m%d")
                end_date = (self.input_date + relativedelta(days= (self.pass_day+self.run_day))).strftime("%Y%m%d")
            elif data_type == 'sdate':
                start_date = (self.input_date + relativedelta(years=-3)).strftime("%Y/%m/%d")
                end_date = (self.input_date + relativedelta(years=+1)).strftime("%Y/%m/%d")

            query = query.replace("@StartDate@", start_date)
            query = query.replace("@EndDate@", end_date)
            query = query.replace("@StoreId@", store_id)
            return query
        except Exception as me:
            logg.logger.error(me)

    def get_dataframe(self,T_query, P_query, D_update, store_ids):
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
                date_update = self.query_modify(D_update, store_id, data_type='sdate')
                conn.query(date_update, as_dict=True)

                data = [train_data, verify_data, predict_data]
                data_set[store_id] = data
            return data_set
        except Exception as me:
            logg.logger.error(me)


