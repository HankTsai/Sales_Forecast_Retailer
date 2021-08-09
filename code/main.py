

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from configuration import DBConnect, CodeLogger
from configparser import ConfigParser
from data_get import DataGet
from data_transtorm import deal_dataframe
from model_predict import ModelProcess
from data_store import StoreData


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