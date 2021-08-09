
import os
import uuid
import logging
import joblib
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from pymssql import connect
from configparser import ConfigParser
from dateutil.relativedelta import relativedelta

from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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


def data_patch(df):
    """補足表格的缺失值"""
    try:
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
    """表格數據日期轉換"""
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
    """表格數據編碼轉換"""
    try:
        # label_encode
        label_encoder = preprocessing.LabelEncoder()
        for col in ['HOLIDAY', 'CELEBRATION', 'HIGH_TEMP', 'LOW_TEMP', 'SKY']:
            df[col] = label_encoder.fit_transform(df[col])
        new_df = df.drop(['SDATE', 'EXIT'], axis=1)
        return new_df, df['SDATE'].apply(lambda x: x.strftime("%Y/%m/%d"))
    except Exception as me:
        logg.logger.error(me)

def type_transform(df):
    """表格數據儲存格式轉換"""
    # 將物件格式數據改成數值格式，便於計算
    try:
        for col in df.columns:
            if df.dtypes[col] == 'object':
                df[col] = df[col].astype('int32')
        return df
    except Exception as me:
        logg.logger.error(me)

def df_null(df_set, idx, key):
    """檢驗門店數據是否存在，若無則儲存錯誤log"""
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
    """模型預測模組"""
    def __init__(self, train, verify, predict, store_id):
        self.train = train
        self.verify =verify
        self.predict = predict
        self.store_id = store_id
        self.dir_path = config['filepath']['model_path']
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def get_paras(self, start_date, end_date):
        """獲取模型超參數"""
        paras_query_normal = f"select * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s and  START_DATE<=%s and END_DATE>=%s"
        paras_query_best = f"select top 1 * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s order by MODEL_SCORE desc"
        try:
            cursor = conn.query(paras_query_normal, para=(self.store_id, start_date, end_date)).fetchone()
            if cursor is not None:
                log = '時間範圍內模型參數存在，直接採用DB當周最優'
                print(log)
                logg.logger.info(log)
                return cursor
            else:
                cursor = conn.query(paras_query_best, para=self.store_id).fetchone()
                if cursor is not None:
                    log = '時間範圍內模型參數不存在，將採用DB中分數最優'
                    print(log)
                    logg.logger.info(log)
                    return cursor
        except Exception as me:
            logg.logger.error(me)

    @staticmethod
    def para_produce(X, y):
        """模型超參數產生"""
        try:
            cv_params = [{'eta': [0.05,0.1,0.2,0.3],
                          'gamma': [0],
                          'max_depth': [1,3,5],
                          'subsample': [0.3,0.5,1],
                          'reg_lambda': [1],
                          'reg_alpha': [0],
                          'n_estimators': [20,50,70,100,150,200],
                          'min_child_weight': [5,10,15,20],
                          'colsample_bytree': [0.1,0.3,0.5,1],
                          'colsample_bylevel': [1]}]
            # 測試用參數
            # cv_params = [{'eta': [0.132],
            #               'gamma': [0],
            #               'max_depth': [1],
            #               'subsample': [1],
            #               'reg_lambda': [1],
            #               'reg_alpha': [0],
            #               'n_estimators': [290],
            #               'min_child_weight': [17],
            #               'colsample_bytree': [0.222],
            #               'colsample_bylevel': [1]}]
            model_xgb = XGBRegressor()
            tss = TimeSeriesSplit(max_train_size=None, n_splits=10)
            gs = GridSearchCV(model_xgb, cv_params, verbose=2, refit=True, cv=tss, n_jobs=-1)
            gs.fit(X, y)
            return gs.best_params_
        except Exception as me:
            logg.logger.error(me)

    @staticmethod
    def para_select(paras):
        """依照模型超參數選擇訓練的數據帶入格式"""
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

    @staticmethod
    def feature_import(X, model):
        """彙整特徵重要性"""
        feature = {}
        for idx,col in enumerate(X.columns):
            feature[col] = model.feature_importances_[idx]
        feature = sorted(feature.items(), key=lambda e: e[1], reverse=True)
        feature = [item[0]+'_'+str(item[1]) for item in feature]
        return feature

    def model_train(self, paras_use, start_date, end_date):
        """切分數據、模型訓練和儲存模型"""
        try:
            X_train = self.train.drop(['TARGET'], axis=1)
            y_train = self.train['TARGET']
            X_test = self.verify.drop(['TARGET'], axis=1)
            y_test = self.verify['TARGET']

            if paras_use == 1:
                paras = self.get_paras(start_date, end_date)
                model = self.para_select(paras)
            else:
                paras = self.para_produce(X_train, y_train)
                model = self.para_select(paras)

            eval_set = [(X_train,y_train),(X_test,y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, eval_metric="mae",early_stopping_rounds=5)
            joblib.dump(model, f'{self.dir_path}/{self.store_id}.pkl')
            feature = self.feature_import(X_train, model)
            logg.logger.info(f'已儲存當次最優化模型 model_{self.store_id}.pkl')
            store.store_log
            return model,feature
        except Exception as me:
            logg.logger.error(me)

    def model_matrices(self, y_true, y_pred, row_num, date):
        """模型評分指標"""
        try:
            rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
            mae = round(mean_absolute_error(y_true, y_pred), 2)
            r2 = round(r2_score(y_true, y_pred), 2)
            mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
            mape2 = mape * 100
            name = f'model_{self.store_id}'
            gdate = date.strftime("%Y/%m/%d")
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

    @staticmethod
    def learn_curve(model):
        """繪製學習曲線圖"""
        results = model.evals_result()
        pyplot.style.use("ggplot")
        pyplot.figure(figsize=(8, 8))
        pyplot.plot(results['validation_0']['mae'], label='train')
        pyplot.plot(results['validation_1']['mae'], label='test')
        pyplot.xlabel('Iteration')
        pyplot.ylabel('mae')
        pyplot.title("learning_curve")
        pyplot.legend(labels=["train","test"], loc = 'best')
        pyplot.savefig("123.png")

    def model_verify(self, model, row_num, date):
        """模型驗證"""
        try:
            X = self.verify.drop(['TARGET'], axis=1)
            y_true = self.verify['TARGET']
            y_pred = model.predict(X)
            evaluate = self.model_matrices(y_true, y_pred, row_num, date)
            # self.learn_curve(model)
            return evaluate
        except Exception as me:
            logg.logger.error(me)

    def model_predict(self, model, sdate):
        """模型預測"""
        try:
            X = self.predict.drop(['TARGET'], axis=1)
            y_true = self.predict['TARGET']
            y_pred = model.predict(X)
            customer_df = self.customer_quatity(y_true, y_pred, sdate)
            return customer_df
        except Exception as me:
            logg.logger.error(me)

class StoreData:
    """預測資料儲存模組"""
    def __init__(self,store_id='',store_num=0,log_order=0,start_date='',end_date=''):
        self.store_num = store_num
        self.store_id = store_id
        self.start_date = start_date
        self.end_date = end_date
        self.log_order = log_order

    def store_log(self,stage,log):
        """ECP_SMXCAAA的log儲存設定"""
        log_insert = "insert into SMXCAAA values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        log_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        level = {'info':11,'major':15,'warning':17,'error':19}
        if stage == 'info':
            log_title = '預測功能正常'
            stage_code = level[stage]
        elif stage == 'major':
            log_title = '預測統計顯示'
            stage_code = level[stage]
        elif stage == 'warning':
            log_title = '特殊情況警告'
            stage_code = level[stage]
        else:
            log_title = '預測功能異常'
            stage_code = level[stage]

        try:
            self.log_order += 1
            order = str(self.log_order)
            if len(order) == 1: order = f'00{order}.'
            elif len(order) == 2: order = f'0{order}.'
            elif len(order) == 3: order = f'{order}.'

            conn.insert(log_insert, para=(str(uuid.uuid4()), 'MART', '銷售預測系統', 'Batch', '批次', 'Batch_51',
                                          '模型預測來客數', stage_code, log_time, order+log_title, log, 'administrator',
                                          '系統管理員', '', 'python', 'SYSTEM', log_time, '', ''))
        except Exception as me:
            logg.logger.error(me)

    def store_feature(self, fea, delete=0):
        """儲存模行特徵重要性"""
        feature_del = "delete from SFI_F17_FEATURE_INPORTANT where STOREID=%s and START_DATE=%s and END_DATE=%s"
        feature_insert = "insert into SFI_F17_FEATURE_INPORTANT values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            if delete > 0:
                conn.delete(feature_del, para=(self.store_id, self.start_date, self.end_date))
                logg.logger.info(f'已刪除舊 model_{self.store_id} 模型特徵權重')
            conn.insert(feature_insert, para=(self.store_id, self.start_date, self.end_date,
                                              datetime.now().strftime("%Y/%m/%d"),
                                              fea[0],fea[1],fea[2],fea[3],fea[4],fea[5],fea[6],
                                              fea[7],fea[8],fea[9],fea[10],fea[11]))
            logg.logger.info(f'已儲存新 model_{self.store_id} 模型特徵權重')
        except Exception as me:
            logg.logger.error(me)
            self.store_log('error', f'店號{self.store_id}模型特徵權重儲存異常 - 程式:data_store,函式:store_feature')

    def store_metrics(self, eva, delete=0):
        """儲存模型評價指標"""
        evaluate_del = "delete from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
        evaluate_insert = "insert into SFI_F01_MODEL values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            if delete > 0:
                conn.delete(evaluate_del, para=(eva[0], eva[8]))
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

    def store_data(self, eva, cus, fea):
        """以R2和 MAE判別是否儲存數據"""
        try:
            evaluate_query = "select * from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
            cursor = conn.query(evaluate_query,  para=(eva[0], eva[8])).fetchone()
            log = f'已儲存 model_{self.store_id} 模型特徵權重、評估指標與來客數預測'
            if cursor is None:
                self.store_feature(fea)
                self.store_metrics(eva)
                self.store_customer(cus)
                self.store_num += 1
                logg.logger.info(log)
                self.store_log('info', log)
                return self.store_num, len(cus)
            else:
                if cursor[4]<eva[4] and cursor[3]>eva[3]:
                    self.store_feature(fea,delete=1)
                    self.store_metrics(eva, delete=1)
                    self.store_customer(cus, delete=1)
                    self.store_num += 1
                    logg.logger.info(log)
                    self.store_log('info',log)
                    return self.store_num, len(cus)
                else:
                    log = f'已存在較佳 model_{self.store_id} 模型特徵權重、評估指標與來客數預測'
                    logg.logger.info(log)
                    self.store_log('info',log)
                    return self.store_num, 0
        except Exception as me:
            logg.logger.error(me)

def arguments_set():
    """設定程式啟動時帶入的參數"""
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


def main():
    # 物件化類別並導入參數
    pass_day, run_day, input_date, paras_use = arguments_set()
    start_date =(input_date + relativedelta(days=(pass_day+1))).strftime("%Y/%m/%d")
    end_date = (input_date + relativedelta(days=(pass_day + run_day))).strftime("%Y/%m/%d")
    program_start_time = datetime.strptime(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

    # 從DB導入數據
    gdata = DataGet(pass_day, run_day, input_date)
    T_query, P_query, D_update, store_ids = gdata.get_query()
    dataset = gdata.get_dataframe(T_query, P_query, D_update, store_ids)

    store_num = 0; cus_num = 0
    for key, value in dataset.items():
        # 處理數據

        train, verify, predict, sdate = deal_dataframe(value, key)
        if len(train)>1 and len(verify)>1 and len(predict)>1:
            model_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logg.logger.info(f'{key} store model_start_time: {model_start_time}')

            # 透過網格搜索查找模型最優參數
            MP = ModelProcess(train, verify, predict, key)
            model,feature = MP.model_train(paras_use, start_date, end_date)
            evaluate = MP.model_verify(model, len(train), input_date)
            customer_df = MP.model_predict(model, sdate)

            # 數據儲存
            store.store_id = key
            store.store_num = store_num
            store.start_date = start_date
            store.end_date = end_date
            store_num, cus_num = store.store_data(evaluate,customer_df,feature)

            model_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logg.logger.info(f'{key} store model_end_time:{model_end_time}')
        else: continue

    program_end_time = datetime.strptime(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
    program_cost_time = (program_end_time - program_start_time).seconds

    if store_num == 0:
        finish_info = f'程式運作正常，已存在既有較佳模型與數據'
        store.store_log('info', finish_info)
    else:
        finish_info = f'程式運作正常，已存入 {store_num} 間店各 {cus_num} 則數據'
        store.store_log('major', finish_info)
    conn.close()

    print(f'已完成 {start_date} - {end_date} 所有數據預測')
    print(f'已存入 {store_num} 間店各 {cus_num} 則數據')
    print(f'共耗時 {program_cost_time} 秒')


if __name__=='__main__':
    config = ConfigParser()
    config.read('setting.ini')
    logg = CodeLogger()
    logg.store_logger()
    conn = DBConnect()
    store = StoreData()
    main()
