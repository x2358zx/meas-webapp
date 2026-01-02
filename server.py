import io, json, zipfile, os, shutil
from typing import List, Optional, Generator
import numpy as np, cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core import load_config, save_config, preview_grid_overlay, process_one_image


def format_ma_for_name(value: float, decimals: int = 2) -> str: # 2026/1/2
    s = f"{value:.{decimals}f}"                         # 3.50 -> "3.50"
    return s.replace(".", "p") + "mA"                    # "3.50" -> "3p50mA"


app = FastAPI(title="Meas Web API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Static mounts
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure output directory exists
OUTPUT_DIR = "/app/OUTPUT"
DONE_DIR = os.path.join(OUTPUT_DIR, "DONE")

# Create directories immediately so StaticFiles mount works
os.makedirs(DONE_DIR, exist_ok=True)


# Also expose OUTPUT for viewing images
app.mount("/output", StaticFiles(directory="/app/OUTPUT"), name="output")


@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
def get_config():
    return load_config()

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/app_icon.ico", media_type="image/x-icon")


@app.put("/api/config")
async def put_config(cfg: dict):
    save_config(cfg)
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


# --- New Streaming Logic ---

def clean_output_dir():
    """Clear previous results"""
    if os.path.exists(DONE_DIR):
        shutil.rmtree(DONE_DIR)
    os.makedirs(DONE_DIR, exist_ok=True)

@app.post("/api/process_stream")
async def api_process_stream(files: List[UploadFile] = File(...)):
    """
    接收圖片，串流回傳處理結果 (SSE format)。
    同時將結果存檔至 /app/OUTPUT/DONE 供後續下載。
    """
    clean_output_dir()
    
    # Summary data collection
    lines = [] 
    idx = 1
    
    async def process_generator():
        nonlocal idx
        for f in files:
            try:
                data = await f.read()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                
                # Processing
                annotated, info = process_one_image(img, f.filename)
                
                # Status & Values
                status = "成功" if info.get("levels_detected", 0) > 0 else "失敗"
                vals = info.get("values_mA", [])
                
                val_str = ""
                val_str_name = ""
                
                if vals:
                    val_str = " / ".join([f"{v:.2f}mA" for v in vals])
                    raw_first = float(vals[0])
                    val_str_name = format_ma_for_name(raw_first)
                
                # Filename logic
                if "." in f.filename:
                    name, ext = f.filename.rsplit(".", 1)
                    ext = ext.lower()
                else:
                    name, ext = f.filename, "png"
                
                out_name = f"{name}_I({val_str_name}).{ext}" if val_str_name else f"{name}_I().{ext}"
                
                # Encoding & Saving
                enc_ok, buf = cv2.imencode(f".{ext}", annotated)
                final_out_name = out_name
                if not enc_ok:
                    final_out_name = f"{name}_I({val_str_name}).png" if val_str_name else f"{name}_I().png"
                    _, buf = cv2.imencode(".png", annotated)
                
                # Save to disk
                save_path = os.path.join(DONE_DIR, final_out_name)
                with open(save_path, "wb") as out_f:
                    out_f.write(buf.tobytes())
                
                # Record Summary
                lines.append(f"{status},{idx},{final_out_name},{val_str}")
                idx += 1
                
                # Prepare JSON response
                # img_url: points to static mount /output/DONE/filename
                resp_data = {
                    "id": idx, # Simple counter ID
                    "filename": final_out_name,
                    "status": status,
                    "values": vals,
                    "img_url": f"/output/DONE/{final_out_name}"
                }
                
                # Yield SSE data line
                yield f"data: {json.dumps(resp_data)}\n\n"
                
            except Exception as e:
                print(f"Error processing {f.filename}: {e}")
                err_data = {
                    "filename": f.filename,
                    "status": "Error",
                    "values": [],
                    "img_url": "" # Can use a placeholder error image if needed
                }
                yield f"data: {json.dumps(err_data)}\n\n"

        # Final Summary File
        try:
            header = "狀態,序號,檔名,偵測值 (mA)"
            summary_content = "\ufeff" + header + "\n" + "\n".join(lines)
            with open(os.path.join(DONE_DIR, "_summary.txt"), "w", encoding="utf-8") as sum_f:
                sum_f.write(summary_content)
        except Exception as e:
            print(f"Summary write error: {e}")

        # End of stream signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(process_generator(), media_type="text/event-stream")


@app.get("/api/results/zip")
def api_results_zip():
    """
    Pack /app/OUTPUT/DONE into a zip file for download.
    """
    if not os.path.exists(DONE_DIR):
        return {"error": "No results found"}
        
    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # Walk DONE_DIR
        for root, dirs, files in os.walk(DONE_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, OUTPUT_DIR) # rel to OUTPUT so it starts with DONE/
                z.write(abs_path, arcname=rel_path)
    
    out_zip.seek(0)
    headers = {"Content-Disposition": "attachment; filename=results.zip"}
    return StreamingResponse(out_zip, media_type="application/zip", headers=headers)



