
# 示波器格線校準器（內網版）

## 啟動（Docker 推薦）
```bash
docker compose up -d --build
# 打開 http://<主機IP>:8000
```

## 本機開發
```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

## 使用流程
1. 首頁選一張圖 → 拖曳四邊界線與 0mA index → 「預覽格線」。
2. 「套用到設定檔」會把 `manual_grid_settings` 寫入 `config.json`。
3. 「批次處理」可一次上傳多張圖片，下載 `results.zip`（含標註圖與 `_summary.txt`）。

## 說明
- 後端 API：`/api/preview-grid`、`/api/process`、`/api/config`。
- 使用 `opencv-numpy-scipy` 做綠色波形偵測與平坦能階找峰，並依 `ma_per_division` / `ref_0ma_index` 換算 mA。
- 需要調整顏色門檻時修改 `config.json` 的 `hsv_detection`。
