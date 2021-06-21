
import sys
import argparse
from datetime import datetime
from configuration import DBConnect, CodeLogger
from data_get import DataGet
from data_transtorm import deal_dataframe
from model_predict import ModelProcess
from data_store import StoreData


def arguments_set():
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
    pass_day, run_day, date = arguments_set()
    logg = CodeLogger()
    conn = DBConnect()
    logg.store_logger()

    # 從DB導入數據
    gdata = DataGet(pass_day, run_day, date)
    T_query, P_query, store_ids = gdata.get_query()
    dataset = gdata.get_dataframe(T_query, P_query, store_ids)

    for key, value in dataset.items():
        # 處理數據
        train, verify, predict, date = deal_dataframe(value, key)

        if len(train)>1 and len(verify)>1 and len(predict)>1:
            # 模型訓練與預測
            MP = ModelProcess(train, verify, predict, key)
            model = MP.model_train()
            evaluate = MP.model_verify(model, len(train))
            customer_df = MP.model_predict(model, date)

            # 數據儲存
            store = StoreData(key)
            store.store_data(evaluate,customer_df)
        else: continue

    conn.commit()
    conn.close()
    print(f'已完成 {datetime.now().strftime("%Y/%m/%d")} 所有數據預測')


if __name__=='__main__':
    main()
