
from configuration import DBConnect, CodeLogger

logg = CodeLogger()
conn = DBConnect()


class StoreData:

    def __init__(self,store_id):
        self.store_id = store_id

    def store_metrics(self, eva, delete=True):
        """儲存模型評價指標"""
        evaluate_del = "delete from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
        evaluate_insert = f"insert into SFI_F01_MODEL values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        try:
            if delete is True:
                conn.delete(evaluate_del, para=(eva[0], eva[8]))
            conn.insert(evaluate_insert, para=(eva[0], eva[1], eva[2], eva[3], eva[4], eva[5], eva[6], eva[7], eva[8]))
            logg.logger.info(f'已儲存最新 GBR_{self.store_id} 模型評估指標')
        except Exception as me:
            logg.logger.error(me)

    def store_customer(self, customer_df):
        """儲存來客數預測"""
        try:
            for idx, row in customer_df.iterrows():
                customer_delete = "delete from SFI_F07_CUS_NBR_FORECAST where SDATE=%s"
                conn.delete(customer_delete, para=(str(row[0])[:10]))
                customer_insert = "insert into SFI_F07_CUS_NBR_FORECAST values(%s,%s,%s,%s,%s)"
                conn.insert(customer_insert, para=(str(row[0])[:10], int(row[1]), int(row[2]), row[3], row[4]))
                logg.logger.info(f'已儲存/更新 店號:{self.store_id},日期:{str(row[0])[:10]} 的來客數預測')
        except Exception as me:
            logg.logger.error(me)

    def store_data(self, eva, customer_df):
        """以R2判別是否儲存數據"""
        try:
            evaluate_query = "select * from SFI_F01_MODEL where STOREID=%s and GDATE=%s"
            cursor = conn.query(evaluate_query,  para=(eva[0], eva[8])).fetchone()
            if cursor is None:
                self.store_metrics(eva,delete=False)
                self.store_customer(customer_df)
            else:
                if cursor[4] <= eva[4]:
                    self.store_metrics(eva)
                    self.store_customer(customer_df)
                else:
                    logg.logger.info(f'已存在較佳 GBR_{self.store_id} 模型評估指標')
        except Exception as me:
            logg.logger.error(me)
