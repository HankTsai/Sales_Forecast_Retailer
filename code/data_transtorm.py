
from datetime import datetime
from sklearn import preprocessing
from configuration import CodeLogger
from data_store import StoreData

store = StoreData()
logg = CodeLogger()


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