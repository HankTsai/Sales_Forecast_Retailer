
import os
import uuid
import logging
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pymssql import connect
from configparser import ConfigParser
from dateutil.relativedelta import relativedelta

from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class CodeLogger:
    def __init__(self):
        """make logger"""
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.formatter = logging.Formatter(
            '["%(asctime)s - %(levelname)s - %(name)s - %(message)s" - function:%(funcName)s - line:%(lineno)d]')
        self.log_name = config['filepath']['log_path'] + datetime.now().strftime("forecast_%Y-%m-%d_%H-%M-%S.log")
        logging.basicConfig(level=logging.INFO, datefmt='%Y%m%d_%H:%M:%S',)

    def store_logger(self):
        """definite log"""
        handler = logging.FileHandler(self.log_name, "w", encoding = "UTF-8")
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def show_logger(self):
        console = logging.StreamHandler()
        console.setLevel(logging.FATAL)
        console.setFormatter(self.formatter)
        self.logger.addHandler(console)


class DBConnect:
    def __init__(self):
        self.host = config['connect']['server']
        self.user = config['connect']['username']
        self.password = config['connect']['password']
        self.database = config['connect']['database']
        self.conn = connect(host=self.host, user=self.user, password=self.password, database=self.database, autocommit=True)

    def query(self, sql, as_dict=False, para=()):
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
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,para)
        except Exception as me:
            CodeLogger().logger.error(me)

    def delete(self, sql, para=()):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,para)
        except Exception as me:
            CodeLogger().logger.error(me)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


class DataGet:

    def __init__(self, pass_day, run_day, date):
        self.pass_day = pass_day
        self.run_day = run_day
        self.date = date
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
        """使用sql指令獲取數據"""
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


def data_patch(df):
    """補足缺失值"""
    try:
        num = 0
        for idx_row, row in df.iterrows():
            if row.isnull().T.any():
                df.iloc[idx_row, 4:10] = df.iloc[idx_row - 1, 4:10]
            if row.isnull().T.any():
                df.iloc[idx_row, 9] = 1
        for col in df.columns:
            if len(df[df[col].isnull()]) == len(df[col]):
                return ""
        return df
    except Exception as me:
        logg.logger.error(me)

def date_format(df):
    """日期轉換"""
    try:
        df['SDATE'] = df['SDATE'].apply(lambda x: datetime.strptime(x, "%Y/%m/%d"))
        df['Year'] = df['SDATE'].apply(lambda x: x.strftime("%Y"))
        df['Month'] = df['SDATE'].apply(lambda x: x.strftime("%m"))
        df['Day'] = df['SDATE'].apply(lambda x: x.strftime("%d"))
        df['Week'] = df['SDATE'].apply(lambda x: x.isoweekday())
        return df
    except Exception as me:
        logg.logger.error(me)


def data_encode(df):
    """編碼轉換"""
    try:
        # label_encode
        label_encoder = preprocessing.LabelEncoder()
        for col in ['HOLIDAY', 'CELEBRATION', 'HIGH_TEMP', 'LOW_TEMP', 'SKY']:
            df[col] = label_encoder.fit_transform(df[col])
        new_df = df.drop(['SDATE', 'EXIT'], axis=1)
        return new_df, df['SDATE'].apply(lambda x: x.strftime("%Y/%m/%d"))
    #     # onehot_encode
    #     onehot = df[['HOLIDAY','CELEBRATION','HIGH_TEMP','LOW_TEMP','SKY']]
    #     df_dum = pd.get_dummies(data=onehot)
    #     df_dum = pd.concat([df, df_dum], axis=1)
    #     df_dum = df_dum.drop([onehot,'SDATE','EIXT'], axis=1)
    #     return df_dum
    except Exception as me:
        logg.logger.error(me)


def type_transform(df):
    try:
        for col in df.columns:
            if df.dtypes[col] == 'object':
                df[col] = df[col].astype('int32')
        return df
    except Exception as me:
        logg.logger.error(me)

def df_null(df_set, idx, key):
    if idx == 0:
        logg.logger.error(f'storeid {key} 訓練集為空')
        store.store_log('error',f'storeid {key} 訓練集為空')
        df_set.append("")
    elif idx == 1:
        logg.logger.error(f'storeid {key} 驗證集為空')
        store.store_log('error',f'storeid {key} 驗證集為空')
        df_set.append("")
    elif idx == 2:
        logg.logger.error(f'storeid {key} 預測集為空')
        store.store_log('error',f'storeid {key} 預測集為空')
        df_set.append("")
        df_set.append("")
    return df_set

def deal_dataframe(value, key):
    """處理表格數據"""
    df_set = []
    try:
        for idx, df in enumerate(value):
            if len(df) > 1:
                df = data_patch(df)
                if len(df) < 1:
                    df_set = df_null(df_set, idx, key)
                    continue
                df = date_format(df)
                df, date = data_encode(df)
                df = type_transform(df)
                df_set.append(df)
                if idx == 2:
                    df_set.append(date)
            else:
                df_set = df_null(df_set, idx, key)
        while len(df_set) < 4:
            df_set.append("")
        for idx, item in enumerate(df_set):
            if item is None:
                df_set[idx] = ""
        return df_set
    except Exception as me:
        logg.logger.error(me)

class ModelProcess:

    def __init__(self, train, verify, predict, store_id):
        self.train = train
        self.verify =verify
        self.predict = predict
        self.store_id = store_id
        self.dir_path = config['filepath']['model_path']
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def get_paras(self, start_date, end_date):
        paras_query_normal = f"select * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s and  START_DATE<=%s and END_DATE>=%s"
        paras_query_best = f"select top 1 * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s order by MODEL_SCORE desc"
        try:
            cursor = conn.query(paras_query_normal, para=(self.store_id, start_date, end_date)).fetchone()
            if cursor is not None:
                log = '時間範圍內模型參數存在，直接採用DB當周最優'
                logg.logger.info(log)
                return cursor
            else:
                cursor = conn.query(paras_query_best, para=self.store_id).fetchone()
                if cursor is not None:
                    log = '時間範圍內模型參數不存在，將採用DB中分數最優'
                    logg.logger.info(log)
                    return cursor
        except Exception as me:
            logg.logger.error(me)

    @staticmethod
    def para_produce(X, y):
        try:
            cv_params = [{'eta': [0.1, 0.132, 0.3],
                          'gamma': [0],
                          'max_depth': [1,2,3],
                          'subsample': [0.3,0.5,1],
                          'reg_lambda': [1],
                          'reg_alpha': [0],
                          'n_estimators': [200, 250, 290, 300],
                          'min_child_weight': [13, 15, 17, 19],
                          'colsample_bytree': [0.15, 0.222, 0.3, 0.4],
                          'colsample_bylevel': [1]}]
            model_xgb = XGBRegressor()
            tss = TimeSeriesSplit(max_train_size=None, n_splits=10)
            gs = GridSearchCV(model_xgb, cv_params, verbose=2, refit=True, cv=tss, n_jobs=-1)
            gs.fit(X, y)
            return gs.best_params_
        except Exception as me:
            logg.logger.error(me)

    @staticmethod
    def para_select(paras):
        if type(paras) is tuple:
            model = XGBRegressor(n_estimators=paras[10], max_depth=paras[6], gamma=paras[5],
                                 subsample=paras[7], reg_alpha=paras[9], reg_lambda=paras[8],
                                 colsample_bytree=paras[12], colsample_bylevel=paras[13],
                                 min_child_weight=paras[11], eta=paras[4], nthread=4)
            return model
        elif type(paras) is dict:
            model = XGBRegressor(n_estimators=paras['n_estimators'], max_depth=paras['max_depth'],gamma=paras['gamma'],
                                 subsample=paras['subsample'], reg_alpha=paras['reg_alpha'],reg_lambda=paras['reg_lambda'],
                                 colsample_bytree=paras['colsample_bytree'],colsample_bylevel=paras['colsample_bylevel'],
                                 min_child_weight=paras['min_child_weight'], eta=paras['eta'], nthread=4)
            return model

    def model_train(self, paras_use, start_date, end_date):
        try:
            X = self.train.drop(['TARGET'], axis=1)
            y = self.train['TARGET']
            if paras_use == 1:
                paras = self.get_paras(start_date, end_date)
                model = self.para_select(paras)
            else:
                paras = self.para_produce(X, y)
                model = self.para_select(paras)
            model.fit(X, y)
            joblib.dump(model, f'{self.dir_path}/{self.store_id}.pkl')
            logg.logger.info(f'已儲存每周最優化模型 model_{self.store_id}.pkl')
            return model
        except Exception as me:
            logg.logger.error(me)

    def model_matrices(self, y_true, y_pred, row_num, date):
        """模型評分指標"""
        try:
            rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)            # RMSE
            mae = round(mean_absolute_error(y_true, y_pred), 2)                     # MAE
            r2 = round(r2_score(y_true, y_pred), 2)                                 # R2
            mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)      # MAPE
            mape2 = mape * 100                                                      # MAPE2
            name = f'model_{self.store_id}'                                          # ModelFileName
            gdate = date.strftime("%Y/%m/%d")                                       # GDATE
            return [self.store_id, row_num, rmse, mae, r2, mape, mape2, name, gdate]
        except Exception as me:
            logg.logger.error(me)

    def customer_quatity(self, y_true, y_pred, sdate):
        """來客數預測"""
        try:
            customer_df = pd.DataFrame(columns=['SDATE', 'WDAY', 'LOC_ID', 'FORECAST_QTY', 'ACTUAL_QTY'])
            customer_df['SDATE'] = sdate
            customer_df['WDAY'] = self.predict['Week']
            customer_df['LOC_ID'] = self.store_id
            customer_df['FORECAST_QTY'] = [round(num, 2) for num in y_pred]
            customer_df['ACTUAL_QTY'] = [round(float(num), 2) for num in y_true]
            return customer_df
        except Exception as me:
            logg.logger.error(me)

    def model_verify(self, model, row_num, date):
        try:
            X = self.verify.drop(['TARGET'], axis=1)
            y_true = self.verify['TARGET']
            y_pred = model.predict(X)
            evaluate = self.model_matrices(y_true, y_pred, row_num, date)
            return evaluate
        except Exception as me:
            logg.logger.error(me)

    def model_predict(self, model, sdate):
        """模型預測值"""
        try:
            X = self.predict.drop(['TARGET'], axis=1)
            y_true = self.predict['TARGET']
            y_pred = model.predict(X)
            customer_df = self.customer_quatity(y_true, y_pred, sdate)
            return customer_df
        except Exception as me:
            logg.logger.error(me)


class StoreData:

    def __init__(self,store_id='',store_num=0,log_order=0):
        self.store_num = store_num
        self.store_id = store_id
        self.log_order = log_order

    def store_log(self,stage,log):
        log_insert = "insert into SMXCAAA values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        log_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if stage == 'info':
            log_title = '預測功能正常執行'
        else:
            log_title = '預測功能出現異常'
        try:
            self.log_order += 1
            order = str(self.log_order)
            if len(order) == 1: order = f'00{order}.'
            elif len(order) == 2: order = f'0{order}.'
            elif len(order) == 3: order = f'{order}.'

            conn.insert(log_insert, para=(str(uuid.uuid4()), 'MART', '銷售預測系統', 'Batch', '批次', 'Batch_51',
                                          '模型預測來客數', 11, log_time, order+log_title, log, 'administrator',
                                          '系統管理員', '', 'python', 'SYSTEM', log_time, '', ''))
        except Exception as me:
            logg.logger.error(me)

    def store_metrics(self, eva, delete=0):
        """儲存模型評價指標"""
        evaluate_del = "delete from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
        evaluate_insert = "insert into SFI_F01_MODEL values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            if delete > 0:
                conn.delete(evaluate_del, para=(eva[0], eva[8 ]))
                logg.logger.info(f'已刪除舊 model_{self.store_id} 模型評估指標')
            conn.insert(evaluate_insert, para=(eva[0], eva[1], eva[2], eva[3], eva[4], eva[5], eva[6], eva[7], eva[8]))
            logg.logger.info(f'已儲存新 model_{self.store_id} 模型評估指標')
        except Exception as me:
            logg.logger.error(me)
            self.store_log('error', f'店號{self.store_id}模型指標儲存異常 - 程式:data_store,函式:store_metrics')

    def store_customer(self, cus, delete=0):
        """儲存來客數預測"""
        customer_query = "select SDATE, LOC_ID from SFI_F07_CUS_NBR_FORECAST where SDATE=%s and LOC_ID=%s"
        customer_delete = "delete from SFI_F07_CUS_NBR_FORECAST where SDATE=%s and LOC_ID=%s"
        customer_insert = "insert into SFI_F07_CUS_NBR_FORECAST values(%s,%s,%s,%s,%s)"
        try:
            for idx, row in cus.iterrows():
                cursor = conn.query(customer_query, para=(str(row[0])[:10], self.store_id)).fetchone()
                if cursor is not None or delete>0:
                    conn.delete(customer_delete, para=(str(row[0])[:10], self.store_id))
                    logg.logger.info(f'已刪除舊 店號:{self.store_id},日期:{str(row[0])[:10]} 的來客數預測')
                conn.insert(customer_insert, para=(str(row[0])[:10], int(row[1]), int(row[2]), row[3], row[4]))
                logg.logger.info(f'已儲存新 店號:{self.store_id},日期:{str(row[0])[:10]} 的來客數預測')
        except Exception as me:
            logg.logger.error(me)
            self.store_log('error', f'店號{self.store_id}來客數預測值儲存異常 - 程式:data_store,函式:store_customer')

    def store_data(self, eva, cus):
        """以R2和 MAE判別是否儲存數據"""
        try:
            evaluate_query = "select * from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
            cursor = conn.query(evaluate_query,  para=(eva[0], eva[8])).fetchone()
            if cursor is None:
                self.store_metrics(eva)
                self.store_customer(cus)
                self.store_num += 1
                return self.store_num, len(cus)
            else:
                if cursor[4]<eva[4] and cursor[3]>eva[3]:
                    self.store_metrics(eva, delete=1)
                    self.store_customer(cus, delete=1)
                    self.store_num += 1
                    return self.store_num, len(cus)
                else:
                    logg.logger.info(f'已存在較佳 model_{self.store_id} 模型評估指標與來客數預測')
                    return self.store_num, 0
        except Exception as me:
            logg.logger.error(me)

def arguments_set():
    parser = argparse.ArgumentParser(description="program para settings")
    parser.add_argument('-p', type=int, default=0,  help='輸入想要從幾天後開始,ex. -p 5 等於今日的五天後開始計算')
    parser.add_argument('-r', type=int, default=7,help='輸入想要取幾天做預測,ex. -r 5 等於開始的那天往後計算')
    parser.add_argument('-d', type=str, default=datetime.now().strftime("%Y/%m/%d"), help='輸入開始計算天數的日期,ex. -d 2021/01/01 等於從2021/01/01往後計算')
    parser.add_argument('-m', type=int, default=1, help='是否使用已計算的模型參數?預設為使用，如需自行訓練參數，請輸入"-m 0"')
    args = parser.parse_args()
    if int(args.p) < 0:
        raise ValueError('-p 的參數輸入不可小於0')
    if int(args.r) < 1:
        raise ValueError('-r 的參數輸入不可小於1')
    if int(args.m) == 1:
        print('預計使用當周既有模型參數，如參數找到則10分鐘內完成預測，請等候...')
    elif int(args.m) == 0:
        print('預計重新訓練新模型參數，至少40分鐘起完成預測，請等候...')
    else: raise ValueError('-m 的參數輸入只能為0或1')
    try:
        args.d = datetime.strptime(args.d, "%Y/%m/%d")
    except ValueError:
        raise ValueError("-d 的參數輸入格式必須是:Y/m/d (2021/01/01)")
    return args.p, args.r, args.d, args.m


config = ConfigParser()
config.read('setting.ini')
logg = CodeLogger()
logg.store_logger()
conn = DBConnect()
store = StoreData()


def main():
    # 物件化類別並導入參數
    logg = CodeLogger()
    conn = DBConnect()
    store = StoreData()
    logg.store_logger()

    pass_day, run_day, input_date, paras_use = arguments_set()
    start_date =(input_date + relativedelta(days=(pass_day+1))).strftime("%Y/%m/%d")
    end_date = (input_date + relativedelta(days=(pass_day + run_day))).strftime("%Y/%m/%d")

    # 從DB導入數據
    gdata = DataGet(pass_day, run_day, input_date)
    T_query, P_query, D_update, store_ids = gdata.get_query()
    dataset = gdata.get_dataframe(T_query, P_query, D_update, store_ids)

    store_num = 0; cus_num = 0
    for key, value in dataset.items():
        # 處理數據

        train, verify, predict, sdate = deal_dataframe(value, key)
        if len(train)>1 and len(verify)>1 and len(predict)>1:
            # 透過網格搜索查找模型最優參數
            MP = ModelProcess(train, verify, predict, key)
            model = MP.model_train(paras_use, start_date, end_date)
            evaluate = MP.model_verify(model, len(train), input_date)
            customer_df = MP.model_predict(model, sdate)

            # 數據儲存
            store.store_id = key
            store.store_num = store_num
            store_num, cus_num = store.store_data(evaluate,customer_df)
        else: continue

    finish_info = f'程式運作正常，已存入 {store_num} 間店各 {cus_num} 則數據'
    if store_num == 0:  finish_info = f'程式運作正常，已有較佳模型與數據'
    store.store_log('info', finish_info)
    conn.close()

    print(f'已完成 {start_date} - {end_date} 所有數據預測')
    print(f'已存入 {store_num} 間店各 {cus_num} 則數據')


if __name__=='__main__':
    main()
