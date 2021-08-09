
import os
import uuid
import logging
import argparse
import pandas as pd
from datetime import datetime
from pymssql import connect
from configparser import ConfigParser
from dateutil.relativedelta import relativedelta

from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class CodeLogger:
    """log設定儲存模組"""
    def __init__(self):
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.formatter = logging.Formatter(
            '["%(asctime)s - %(levelname)s - %(name)s - %(message)s" - function:%(funcName)s - line:%(lineno)d]')
        self.log_name = config['filepath']['log_path'] + datetime.now().strftime("hypeparam_%Y-%m-%d_%H-%M-%S.log")
        logging.basicConfig(level=logging.INFO, datefmt='%Y%m%d_%H:%M:%S',)

    def store_logger(self):
        """儲存設定"""
        handler = logging.FileHandler(self.log_name, "w", encoding = "UTF-8")
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def show_logger(self):
        """終端機顯示設定"""
        console = logging.StreamHandler()
        console.setLevel(logging.FATAL)
        console.setFormatter(self.formatter)
        self.logger.addHandler(console)


class DBConnect:
    """DB連線設定模組"""
    def __init__(self):
        self.host = config['connect']['server']
        self.user = config['connect']['username']
        self.password = config['connect']['password']
        self.database = config['connect']['database']
        self.conn = connect(host=self.host, user=self.user, password=self.password, database=self.database, autocommit=True)

    def query(self, sql, as_dict=False, para=()):
        """DB資料查詢"""
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
        """資料插入DB"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,para)
        except Exception as me:
            CodeLogger().logger.error(me)

    def delete(self, sql, para=()):
        """刪除DB資料"""
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
    """數據獲取模組"""
    def __init__(self, pass_day, run_day, date):
        self.pass_day = pass_day
        self.run_day = run_day
        self.date = date
        self.row_train_query = config['filepath']['train_query']
        self.row_predict_query = config['filepath']['predict_query']
        self.row_date_update = config['filepath']['date_update']

    @staticmethod
    def get_stage():
        """確認有需要執行的店號"""
        stage_query = "select LOC_ID, STATUS from SFI_I21_STORE order by LOC_ID "
        cursor = conn.query(stage_query, as_dict=True)
        store_stage = pd.DataFrame(cursor.fetchall())
        launch_list = []
        for idx, row in store_stage.iterrows():
            if row[1] == 1:
                launch_list.append(str(row[0]))
        return launch_list

    def get_query(self):
        """將基礎sql.txt指令帶入"""
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
        label_encoder = preprocessing.LabelEncoder()
        for col in ['HOLIDAY', 'CELEBRATION', 'HIGH_TEMP', 'LOW_TEMP', 'SKY']:
            df[col] = label_encoder.fit_transform(df[col])
        new_df = df.drop(['SDATE', 'EXIT'], axis=1)
        return new_df, df['SDATE'].apply(lambda x: x.strftime("%Y/%m/%d"))
    except Exception as me:
        logg.logger.error(me)


def type_transform(df):
    """數據格式轉換"""
    try:
        for col in df.columns:
            if df.dtypes[col] == 'object':
                df[col] = df[col].astype('int32')
        return df
    except Exception as me:
        logg.logger.error(me)

def df_null(df_set, idx, key):
    """確認參與預測的店是否能獲取訓練、驗證與預測數據集"""
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
    """模型處理模程"""
    def __init__(self, train, verify, predict, store_id):
        self.train = train
        self.verify =verify
        self.predict = predict
        self.store_id = store_id
        self.dir_path = config['filepath']['model_path']
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def para_select(self):
        """產生模型超參數"""
        try:
            X = self.train.drop(['TARGET'], axis=1)
            y = self.train['TARGET']
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
            # 測試用單一參數
            # cv_params = [{'eta': [0.05],
            #               'gamma': [0],
            #               'max_depth': [1],
            #               'subsample': [0.3],
            #               'reg_lambda': [1],
            #               'reg_alpha': [0],
            #               'n_estimators': [100],
            #               'min_child_weight': [10],
            #               'colsample_bytree': [0.3],
            #               'colsample_bylevel': [1]}]
            model_xgb = XGBRegressor()
            tss = TimeSeriesSplit(max_train_size=None, n_splits=10)
            gs = GridSearchCV(model_xgb, cv_params, verbose=2, refit=True, cv=tss, n_jobs=-1)
            gs.fit(X, y)
            return gs.best_score_, gs.best_params_
        except Exception as me:
            logg.logger.error(me)

class StoreData:
    """預測資料儲存模組"""
    def __init__(self,store_id='',store_num=0,log_order=0):
        self.store_num = store_num
        self.store_id = store_id
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

            conn.insert(log_insert, para=(str(uuid.uuid4()), 'MART', '銷售預測系統', 'Batch', '批次', 'Batch_52',
                                          '模型預測來客數', stage_code, log_time, order+log_title, log, 'administrator',
                                          '系統管理員', '', 'python', 'SYSTEM', log_time, '', ''))
        except Exception as me:
            logg.logger.error(me)

    def store_paras(self, score, paras, start_date, end_date):
        """儲存當周模型最優參數"""
        record_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        paras_query = "select * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s and START_DATE=%s and END_DATE=%s"
        paras_insert = "insert into SFI_F16_WEEKLY_HYPEPARAM_OPT values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor = conn.query(paras_query, para=(self.store_id, start_date, end_date)).fetchone()
            if cursor is None:
                conn.insert(paras_insert, para=(self.store_id, start_date, end_date, round(score,3),
                                                round(paras['eta'],3), paras['gamma'], paras['max_depth'],
                                                paras['subsample'], paras['reg_lambda'],paras['reg_alpha'],
                                                paras['n_estimators'], paras['min_child_weight'],
                                                round(paras['colsample_bytree'],3), paras['colsample_bylevel'],record_date))
                logg.logger.info(f'已儲存新 model_{self.store_id} 模型優化超參數')
                self.store_num += 1
                return self.store_num
            else: return self.store_num
        except Exception as me:
            logg.logger.error(me)
            self.store_log('error', f'店號{self.store_id}模型超參數儲存異常 - 程式:data_store,函式:store_paras')


def arguments_set():
    """設置程式啟動的參數"""
    parser = argparse.ArgumentParser(description="program para settings")
    parser.add_argument('-p', type=int, default=0,  help='輸入想要從幾天後開始,ex. -p 5 等於今日的五天後開始計算')
    parser.add_argument('-r', type=int, default=7,help='輸入想要取幾天做預測,ex. -r 5 等於開始的那天往後計算')
    parser.add_argument('-d', type=str, default=datetime.now().strftime("%Y/%m/%d"), help='輸入開始計算天數的日期,ex. -d 2021/01/01 等於從2021/01/01往後計算')
    args = parser.parse_args()
    if int(args.p) < 0:
        raise ValueError('-p輸入不可小於0')
    if int(args.r) < 1:
        raise ValueError('-r輸入不可小於1')
    try:
        args.d = datetime.strptime(args.d, "%Y/%m/%d")
    except ValueError:
        raise ValueError("-d格式必須是:Y/m/d (2021/01/01)")
    return args.p, args.r, args.d


def main():
    # 物件化類別並導入參數
    pass_day, run_day, input_date = arguments_set()
    start_date =(input_date + relativedelta(days=(pass_day+1))).strftime("%Y/%m/%d")
    end_date = (input_date + relativedelta(days=(pass_day + run_day))).strftime("%Y/%m/%d")
    program_start_time = datetime.strptime(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

    # 從DB導入數據
    gdata = DataGet(pass_day, run_day, input_date)
    T_query, P_query, D_update, store_ids = gdata.get_query()
    dataset = gdata.get_dataframe(T_query, P_query, D_update, store_ids)

    store_num = 0
    for key, value in dataset.items():
        # 處理數據
        train, verify, predict, sdate = deal_dataframe(value, key)

        if len(train)>1 and len(verify)>1 and len(predict)>1:
            model_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logg.logger.info(f'{key} store model_start_time: {model_start_time}')

            # 透過網格搜索查找模型最優參數
            MP = ModelProcess(train, verify, predict, key)
            score, paras = MP.para_select()

            # 數據儲存
            store.store_id = key
            store_num = store.store_paras(score,paras,start_date,end_date)

            model_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logg.logger.info(f'{key} store model_end_time:{model_end_time}')
        else: continue

    program_end_time = datetime.strptime(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
    program_cost_time = (program_end_time - program_start_time).seconds

    if store_num == 0:
        finish_info = f'程式運作正常，模型參數皆已最優'
        store.store_log('info', finish_info)
    else:
        finish_info = f'程式運作正常，已存入 {store_num} 間店的最優模型參數'
        store.store_log('major', finish_info)
    conn.close()

    print(f'已完成 {start_date} - {end_date} 最佳模型參數儲存')
    print(f'已存入 {store_num} 間店最佳模型參數')
    print(f'共耗時 {program_cost_time} 秒')


if __name__=='__main__':
    config = ConfigParser()
    config.read('setting.ini')
    logg = CodeLogger()
    logg.store_logger()
    logg.store_logger()
    conn = DBConnect()
    store = StoreData()
    main()
