
import uuid
from datetime import datetime
from configuration import DBConnect, CodeLogger
logg = CodeLogger()
conn = DBConnect()


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