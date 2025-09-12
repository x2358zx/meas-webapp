import io, json, zipfile
from typing import List, Optional
import numpy as np, cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core import load_config, save_config, preview_grid_overlay, process_one_image


def format_ma_for_name(value: float, decimals: int = 1) -> str:
    s = f"{value:.{decimals}f}".rstrip("0").rstrip(".")  # 3.50 -> "3.5", 3.0 -> "3"
    return s.replace(".", "p") + "mA"                    # "3.5" -> "3p5mA"


app = FastAPI(title="Meas Web API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
def get_config():
    return load_config()

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # 直接回傳 static 目錄下的 ico
    return FileResponse("static/app_icon.ico", media_type="image/x-icon")


@app.put("/api/config")
async def put_config(cfg: dict):
    save_config(cfg); 
    return {"ok": True}

@app.post("/api/preview-grid")
async def api_preview_grid(
    file: UploadFile = File(...),
    grid: str = Form(...),
    ref_idx: int = Form(...),
    ma_div: Optional[float] = Form(None),
    ref_idx_y: Optional[int] = Form(None),
    ref_idx_m: Optional[int] = Form(None),
    label_clk: Optional[str] = Form(None),
    label_vdd: Optional[str] = Form(None),
    label_ivdd: Optional[str] = Form(None),
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    grid_dict = json.loads(grid)
    labels = {
        "clk":  label_clk or "CLK",
        "vdd":  label_vdd or "VDD",
        "ivdd": label_ivdd or "I(VDD)"
    }
    out = preview_grid_overlay(
        img, grid_dict, int(ref_idx), ma_div,
        ref_idx_yellow = (int(ref_idx_y) if ref_idx_y is not None else None),
        ref_idx_magenta= (int(ref_idx_m) if ref_idx_m is not None else None),
        labels=labels
    )
    ok, buf = cv2.imencode(".png", out)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.post("/api/process")
async def api_process(files: List[UploadFile] = File(...)):
    """
    多張圖片處理並打包 results.zip

    - 標註後圖片：沿用原始副檔名，檔名加上 _I(<數值>mA)
    - 摘要：DONE/_summary.txt（UTF-8 BOM）
      欄位：狀態,序號,檔名,偵測值 (mA)
      其中「檔名」= 實際輸出的檔名（與壓縮包內一致）
    """
    out_zip = io.BytesIO()
    lines = []   # 給 _summary.txt
    idx = 1

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
      for f in files:
        data = await f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        # 影像處理：得到標註圖與資訊
        annotated, info = process_one_image(img, f.filename)

        # 偵測狀態與 mA 值（僅取第一個）
        status = "成功" if info.get("levels_detected", 0) > 0 else "失敗"
        if info.get("values_mA"):
            raw = float(info["values_mA"][0])
            val_str      = f"{raw:.2f}mA"              # <-- 繼續給 _summary 用（不改）
            val_str_name = format_ma_for_name(raw)     # <-- 給檔名用（3p5mA）
        else:
            val_str = ""
            val_str_name = ""


        # 依原始副檔名輸出，檔名加上 _I(<mA>)
        if "." in f.filename:
            name, ext = f.filename.rsplit(".", 1)
            ext = ext.lower()
        else:
            name, ext = f.filename, "png"  # 無副檔名時預設 png

        out_name = f"{name}_{val_str_name}.{ext}" if val_str_name else f"{name}.{ext}"

        # 嘗試用原副檔名編碼；失敗則回退成 png，並同步更新檔名
        enc_ok, buf = cv2.imencode(f".{ext}", annotated)
        final_out_name = out_name
        if not enc_ok:
            final_out_name = f"{name}_{val_str_name}.png" if val_str_name else f"{name}.png"
            _, buf = cv2.imencode(".png", annotated)

        # 寫入圖片（使用最終實際檔名）
        z.writestr(f"DONE/{final_out_name}", buf.tobytes())

        # 摘要行使用「實際輸出的檔名」確保同步
        lines.append(f"{status},{idx},{final_out_name},{val_str}")
        idx += 1

      # 生成 _summary.txt（UTF-8 BOM，Excel 友善）
      header = "狀態,序號,檔名,偵測值 (mA)"
      summary_content = "\ufeff" + header + "\n" + "\n".join(lines)
      z.writestr("DONE/_summary.txt", summary_content.encode("utf-8"))

    out_zip.seek(0)
    headers = {"Content-Disposition": "attachment; filename=results.zip"}
    return StreamingResponse(out_zip, media_type="application/zip", headers=headers)
