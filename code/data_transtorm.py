
from datetime import datetime
from sklearn import preprocessing
from configuration import CodeLogger

logg = CodeLogger()


def data_patch(df):
    """補足缺失值"""
    try:
        for idx_row, row in df.iterrows():
            if row.isnull().T.any():
                df.iloc[idx_row, 4:10] = df.iloc[idx_row - 1, 4:10]
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

def deal_dataframe(value, key):
    """處理表格數據"""
    try:
        df_set = []
        for idx, df in enumerate(value):
            if len(df) > 1:
                df = data_patch(df)
                df = date_format(df)
                df, date = data_encode(df)
                df_set.append(df)
                if idx == 2:
                    df_set.append(date)
            else:
                if idx == 0:
                    logg.logger.error(f'storeid {key} 訓練集為空')
                    df_set.append('')
                elif idx == 1:
                    logg.logger.error(f'storeid {key} 驗證集為空')
                    df_set.append('')
                elif idx == 2:
                    logg.logger.error(f'storeid {key} 預測集為空')
                    df_set.append('')
        return df_set
    except Exception as me:
        logg.logger.error(me)


