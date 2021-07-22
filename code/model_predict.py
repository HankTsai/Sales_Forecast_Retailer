
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBRegressor,DMatrix,cv
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score,TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from configuration import CodeLogger, DBConnect
from configparser import ConfigParser

conn = DBConnect()
logg = CodeLogger()
config = ConfigParser()
config.read('setting.ini')


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
        paras_query = f"select * from SFI_F16_WEEKLY_HYPEPARAM_OPT where STOREID=%s and  START_DATE<=%s and END_DATE>=%s"
        try:
            cursor = conn.query(paras_query, para=(self.store_id, start_date, end_date)).fetchone()
            if cursor is not None:
                return cursor
            else: return 0
        except Exception as me:
            logg.logger.error(me)

    @staticmethod
    def para_produce(X, y):
        try:
            cv_params = [{'eta': [0.132],
                          'gamma': [0],
                          'max_depth': [1],
                          'subsample': [1],
                          'reg_lambda': [1],
                          'reg_alpha': [0],
                          'n_estimators': [290],
                          'min_child_weight': [17],
                          'colsample_bytree': [0.222],
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
                if paras == 0:
                    logg.logger.info('查無時間範圍內模型參數，將重新訓練')
                    paras = self.para_produce(X, y)
                    model = self.para_select(paras)
                else:
                    logg.logger.info('時間範圍內模型參數存在，直接採用')
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
