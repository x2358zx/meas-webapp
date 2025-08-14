# 示波器格線校準器（內網版）

## 簡介
這是一個用來**手動校準示波器截圖格線**並進行**綠色波形平坦能階偵測與 mA 值計算**的網頁應用。  
前端可即時預覽格線、設定參數，後端則負責影像處理與批次輸出。

---

## 功能特色
- **格線手動調整**：拖曳四邊線或輸入座標，支援滑鼠、鍵盤微調。
- **0 mA 參考線設定**：可自訂 index (0~8) 與每大格 mA 值。
- **即時預覽**：快速檢查格線位置與參數是否正確。
- **批次處理**：
  - 支援多張圖片一次處理。
  - 自動輸出標註後的圖片與 `_summary.txt`（含狀態、序號、檔名與偵測值）。
- **檔名自動讀取 mA/div**：檔名含 `_IR?mA` 時，會用該值覆蓋設定檔參數。
- **調整顏色門檻**：修改 `config.json` 的 `hsv_detection` 可適應不同波形顏色。

---

## 啟動方式

### 1. Docker（推薦）
```bash
docker compose -f docker-compose.prod.yml up -d --build
# 開啟瀏覽器： http://<主機IP>:8000
```

### 2. 本機開發
```bash
pip install -r requirements.txt
uvicorn server:app --reload
# 預設 http://127.0.0.1:8000
```

---

## 使用流程

1. **選擇圖片並調整格線**
   - 點選「選檔」 → 拖曳邊界線或輸入數字微調。
   - 設定 0 mA index 與每大格 mA/div。
   - 可用滑鼠、方向鍵、Shift(5px)、Alt(10px) 微調。

2. **預覽格線**
   - 按「預覽格線」檢查效果。
   - 如果滿意設定，可「套用到設定檔」將參數寫入後端 `config.json`。

3. **批次處理**
   - 按「批次處理（可多選）」上傳多張圖片。
   - 後端處理完成後會自動下載 `results.zip`：
     ```
     results.zip
     └── DONE/
         ├── <原檔名>_I(<數值>mA).<ext>
         └── _summary.txt  # UTF-8 BOM，Excel 友善
     ```

4. **快速改參數再重跑**
   - 若前端版本支援「重載前一批」按鈕，可直接用上一次上傳的檔案重跑（免重新選檔）。

---

## API 介面

- `GET /api/config`  
  讀取 `config.json` 設定。

- `PUT /api/config`  
  更新設定檔內容。

- `POST /api/preview-grid`  
  上傳單張圖片與格線參數，回傳標註後的預覽 PNG。

- `POST /api/process`  
  上傳多張圖片進行批次處理，回傳 `results.zip`。

（若有「重載前一批」功能，則會多一個 `POST /api/process-reuse`。）

---

## 前端小技巧
- **拖曳吸附**：拖曳時按 Shift，可每 5px 吸附一次，方便對齊。
- **鍵盤移動**：
  - 單方向鍵：移動 1px
  - Shift + 方向鍵：移動 5px
  - Alt + 方向鍵：移動 10px
- **檔名帶參數**：例如 `test_IR2.5mA.png`，會用 2.5 mA/div 進行計算。
- **F5 重選檔案**：重新選檔時可快速刷新狀態。


---

## Ubuntu 常用指令小抄

### 1. 建立與啟動 Docker 服務
```bash
# 檢查 Docker 是否安裝
docker --version

# 從 docker-compose 啟動服務（背景執行）
docker compose -f docker-compose.prod.yml up -d --build

# 查看目前容器運行狀態
docker ps

# 查看特定容器日誌
docker logs <容器名稱或ID>

# 停止服務
docker compose -f docker-compose.prod.yml down
```

### 2. 本機開發環境
```bash
# 建立虛擬環境
python3 -m venv .venv

# 啟用虛擬環境
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 啟動 FastAPI 開發伺服器
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Git 常用指令
```bash
# 初始化 Git
git init

# 新增遠端 Repo
git remote add origin https://github.com/<帳號>/<repo>.git

# 檢查遠端設定
git remote -v

# 新增檔案並 Commit
git add .
git commit -m "Initial commit"

# 推送到 main 分支
git branch -M main
git push -u origin main

# 從遠端更新
git pull origin main
```

### 4. Ubuntu 系統操作
```bash
# 更新套件清單與升級
sudo apt update && sudo apt upgrade -y

# 安裝必要套件
sudo apt install python3 python3-venv python3-pip git -y

# 查看磁碟使用狀況
df -h

# 查看目前目錄位置
pwd

# 列出檔案與權限
ls -l
```
