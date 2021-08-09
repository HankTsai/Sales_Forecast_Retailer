

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
    """獲取DB數據模組"""
    def __init__(self, pass_day, run_day, date):
        self.pass_day = pass_day
        self.run_day = run_day
        self.date = date
        self.row_train_query = config['filepath']['train_query']
        self.row_predict_query = config['filepath']['predict_query']
        self.row_date_update = config['filepath']['date_update']

    @staticmethod
    def get_stage():
        """選出已啟用的門店號"""
        stage_query = "select LOC_ID, STATUS from SFI_I21_STORE order by LOC_ID "
        cursor = conn.query(stage_query, as_dict=True)
        store_stage = pd.DataFrame(cursor.fetchall())
        launch_list = []
        for idx, row in store_stage.iterrows():
            if row[1] == 1:
                launch_list.append(str(row[0]))
        return launch_list

    def get_query(self):
        """將基礎sql.txt帶入"""
        try:
            with open(self.row_train_query, 'r',encoding="utf-8") as file:
                T_query = file.read()
            with open(self.row_predict_query, 'r',encoding="utf-8") as file:
                P_query = file.read()
            with open(self.row_date_update, 'r',encoding="utf-8") as file:
                D_update = file.read()
            store_ids = self.get_stage()
            return T_query, P_query, D_update, store_ids
        except Exception as me:
            logg.logger.error(me)

    def query_modify(self, query, store_id, data_type=""):
        """修改讀取的SQL指令"""
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
            elif data_type == 'sdate':
                start_date = (self.date + relativedelta(years=-3)).strftime("%Y/%m/%d")
                end_date = (self.date + relativedelta(years=+1)).strftime("%Y/%m/%d")

            query = query.replace("@StartDate@", start_date)
            query = query.replace("@EndDate@", end_date)
            query = query.replace("@StoreId@", store_id)
            return query
        except Exception as me:
            logg.logger.error(me)

    def get_dataframe(self,T_query, P_query, D_update, store_ids):
        """使用調整過的sql指令獲取數據"""
        data_set = {}
        try:
            for store_id in store_ids:
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

