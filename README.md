銷售預測模組

## exe.py

### 操作步驟如下:
	1. 找到exec.cpython-38.pyc 這支編譯後的執行檔
	2. 在terminal開啟絕對路徑，輸入檔名以及欲預測時間參數
	3. 範例如下"C:\RTM\Salse_Forecast\code>exec.cpython-38.pyc"
	
### 日期參數說明:
	1. 日期區間參數為訓練集、驗證集與預測集的設定參數
	2. -p = pass :預計略過幾天後開始; 		int>=0 ;  default = 0
	3. -r = run  :預計預測的天數; 			int>=1;   default = 7
	4. -d = date :預計哪天開始往後計算; 	datetime; default = sysdate
	5. -m = model:預計使用哪種模型參數;		model; 	  default = 1
	6. 範例:"-p 0 -r 7 -d 2021/04/30 [-m 1]" 表示預測2021/05/01-2021/05/07之間的數值 
	
### 數據集日期計算公式:
	1. 訓練集:(sysdate-3year)<=train_set<(sysdate-3month)
	2. 驗證集:(sysdate-3month)<=verify_set<sysdate
	3. 預測集:(date+1+pass)<=predict_set<(date+pass+run)
	

## exe_sunday.py

### 操作步驟如下:
	與exec.py相同，但檔名請改成exe_sunday.py
	
### 日期參數說明:
	與exec.py相同，但無"-m"參數
	
### 數據集日期計算公式:
	與exec.py相同
	
	
## SQL表格
	1. exe_sunday.py 只會用到"SFI_F15_WEEKLY_BASE_QTY_DETAIL"，需要使用sql資料夾中的hypepara_table.sql
	2. exec.py 會使用到"SFI_F07_CUS_NBR_FORECAST",'SFI_F01_MODEL',"SFI_F17_FEATURE_INPORTANT"三張表格，sql資料夾中有"feature_table.sql"
	3. sql資料夾中的其餘三個txt檔"GetPredictData","GettrainData","ProductDate"都是程式運行時向DB索取數據集時會用到的文檔。
