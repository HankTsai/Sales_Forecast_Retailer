
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from matplotlib import pyplot
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from configuration import CodeLogger, DBConnect
from configparser import ConfigParser
from data_store import StoreData

store = StoreData()
conn = DBConnect()
logg = CodeLogger()
config = ConfigParser()
config.read('setting.ini')


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