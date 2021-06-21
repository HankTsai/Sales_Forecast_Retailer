
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from configuration import CodeLogger
from configparser import ConfigParser

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

    def model_train(self):
        try:
            X = self.train.drop(['TARGET'], axis=1)
            y = self.train['TARGET']
            model = GradientBoostingRegressor(n_estimators=100, max_depth=10, loss='lad')
            tss = TimeSeriesSplit(max_train_size=None, n_splits=10)
            cv_score = cross_val_score(model, X, y, scoring='r2', cv=tss)
            model.fit(X, y)
            #     print(f"storeid:{key}'s cv_r2:{sum(cv_score)/10}")
            joblib.dump(model, f'{self.dir_path}/{self.store_id}.pkl')
            logg.logger.info(f'已儲存模型 GBR_{self.store_id}.pkl')
            return model

        except Exception as me:
            logg.logger.error(me)

    def model_verify(self, model, row_num):
        """模型測試與評分"""
        try:
            X = self.verify.drop(['TARGET'], axis=1)
            y_true = self.verify['TARGET']
            y_pred = model.predict(X)

            rmse = round(np.sqrt(mean_squared_error(y_true,y_pred)), 2)           # RMSE
            mae = round(mean_absolute_error(y_true,y_pred), 2)                    # MAE
            r2 = round(r2_score(y_true, y_pred), 2)                                 # R2
            mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)      # MAPE
            mape2 = mape * 100                                                      # MAPE2
            # accuracy = 100% - (mape2)                                             # ACCURACY
            name = f'GBR_{self.store_id}'                                           # ModelFileName
            data = datetime.now().strftime("%Y/%m/%d")                     # GDATE
            return [self.store_id, row_num, rmse, mae, r2, mape, mape2, name, data]

        except Exception as me:
            logg.logger.error(me)

    def model_predict(self, model, sdate):
        """模型預測值"""
        try:
            X = self.predict.drop(['TARGET'], axis=1)
            y_true = self.predict['TARGET']
            y_pred = model.predict(X)

            customer_df = pd.DataFrame(columns=['SDATE', 'WDAY', 'LOC_ID', 'FORECAST_QTY', 'ACTUAL_QTY'])
            customer_df['SDATE'] = sdate
            customer_df['WDAY'] = self.predict['Week']
            customer_df['LOC_ID'] = self.store_id
            customer_df['FORECAST_QTY'] = [round(num, 2) for num in y_pred]
            customer_df['ACTUAL_QTY'] = [round(float(num), 2) for num in y_true]
            #     customer_df['GenDate'] = date
            return customer_df

        except Exception as me:
            logg.logger.error(me)
