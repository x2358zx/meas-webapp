import io, json, zipfile, os, shutil, time
from typing import List, Optional, Generator
import numpy as np, cv2
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core import load_config, save_config, preview_grid_overlay, process_one_image


def format_ma_for_name(value: float, decimals: int = 2) -> str: # 2026/1/2
    s = f"{value:.{decimals}f}"                         # 3.50 -> "3.50"
    return s.replace(".", "p") + "mA"                    # "3.50" -> "3p50mA"


app = FastAPI(title="Meas Web API", version="1.2")

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

def cleanup_old_sessions():
    """Remove session directories older than 1 hour to prevent disk fill-up."""
    try:
        now = time.time()
        cutoff = now - 3600  # 1 hour
        if not os.path.exists(DONE_DIR):
            return
            
        for item in os.listdir(DONE_DIR):
            path = os.path.join(DONE_DIR, item)
            if os.path.isdir(path):
                # Check mtime
                try:
                    mtime = os.path.getmtime(path)
                    if mtime < cutoff:
                        shutil.rmtree(path)
                except Exception:
                    pass
    except Exception as e:
        print(f"Cleanup error: {e}")

def ensure_session_dir(session_id: str):
    """Ensure {DONE_DIR}/{session_id} exists."""
    # Build path and ensure it's inside DONE_DIR to prevent traversal
    # Simple check: session_id shouldn't contain path separators
    safe_id = os.path.basename(session_id)
    path = os.path.join(DONE_DIR, safe_id)
    os.makedirs(path, exist_ok=True)
    return path

@app.post("/api/process_stream")
async def api_process_stream(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    grid_config: Optional[str] = Form(None)
):
    """
    接收圖片，串流回傳處理結果 (SSE format)。
    支援 Session ID 分離不同使用者的資料。
    """
    # 1. Cleanup old sessions occasionally
    cleanup_old_sessions()

    # 2. Ensure session directory
    session_dir = ensure_session_dir(session_id)
    
    # Sort files by ID (Index 7) to ensure consistency
    def get_sort_key(file):
        try:
            # Extract ID from filename (e.g. index 7)
            parts = file.filename.split('_')
            if len(parts) >= 8:
                raw_id = parts[7]
                if '.' in raw_id:
                     raw_id = raw_id.rsplit('.', 1)[0]
                return raw_id
        except:
            pass
        return file.filename
        
    files.sort(key=get_sort_key)
    
    # Parse override config if present
    override_cfg = None
    if grid_config:
        try:
            override_cfg = json.loads(grid_config)
        except Exception as e:
            print(f"Error parsing grid_config: {e}")

    # Summary data collection
    lines = [] 
    
    # Global Session Calibration (In-Memory)
    global SESSION_CALIBRATION
    if "SESSION_CALIBRATION" not in globals():
        SESSION_CALIBRATION = {}
    
    idx = 1
    
    async def process_generator():
        nonlocal idx
        for f in files:
            try:
                data = await f.read()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                
                # Check Calibration
                calib_thresh = None
                if session_id in SESSION_CALIBRATION:
                    calib_thresh = SESSION_CALIBRATION[session_id]
                
                # Processing
                annotated, info = process_one_image(img, f.filename, override_config=override_cfg, calibration_threshold=calib_thresh)
                
                # Update Calibration if First Image
                if calib_thresh is None and "calculated_threshold" in info:
                     t = info["calculated_threshold"]
                     if t is not None:
                         SESSION_CALIBRATION[session_id] = t
                         print(f"Session {session_id} Calibrated Thresh: {t}")
                
                # Status & Values
                if "status" in info:
                    status = info["status"]
                else:
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
                
                # Save to disk (Session Directory)
                save_path = os.path.join(session_dir, final_out_name)
                with open(save_path, "wb") as out_f:
                    out_f.write(buf.tobytes())
                
                # Record Summary (Only Success)
                if status.startswith("成功") or status == "Success":
                    lines.append(f"{status},{idx},{final_out_name},{val_str}")
                idx += 1
                
                # Prepare JSON response
                # img_url: points to static mount /output/DONE/{session_id}/{filename}
                # Warning: We need to use safe_id logic again or ensure session_id is safe
                safe_id = os.path.basename(session_id)
                resp_data = {
                    "id": idx, # Simple counter ID (per batch)
                    "filename": final_out_name,
                    "status": status,
                    "values": vals,
                    "img_url": f"/output/DONE/{safe_id}/{final_out_name}"
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

        # Final Summary File (Append Mode)
        try:
            summary_path = os.path.join(session_dir, "_summary.txt")
            is_new = not os.path.exists(summary_path)
            
            with open(summary_path, "a", encoding="utf-8") as sum_f:
                if is_new:
                    header = "狀態,序號,檔名,偵測值 (mA)"
                    sum_f.write("\ufeff" + header + "\n")
                if lines:
                    sum_f.write("\n".join(lines) + "\n")
                    
        except Exception as e:
            print(f"Summary write error: {e}")

        # End of stream signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(process_generator(), media_type="text/event-stream")


@app.get("/api/results/zip")
def api_results_zip(session_id: str = Query(...)):
    """
    Pack /app/OUTPUT/DONE/{session_id} into a zip file for download.
    """
    # Ensure safe session id access
    safe_id = os.path.basename(session_id)
    target_dir = os.path.join(DONE_DIR, safe_id)
    summary_path = os.path.join(target_dir, "_summary.txt")
    
    if not os.path.exists(target_dir):
        return {"error": "No results found for this session"}
        
    # Read valid files from summary
    valid_files = set()
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[2].endswith(('.png', '.jpg', '.jpeg')):
                        valid_files.add(parts[2])
        except:
            pass
            
    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Add Summary
        if os.path.exists(summary_path):
            z.write(summary_path, arcname="_summary.txt")
            
        # 2. Add Valid Images
        for fname in valid_files:
            fpath = os.path.join(target_dir, fname)
            if os.path.exists(fpath):
                z.write(fpath, arcname=fname)
                
    out_zip.seek(0)
    
    # Format: "current probe results_YYYYMMDDHHMM.zip" (UTC+8)
    from datetime import datetime, timedelta
    # Docker uses UTC by default, so we add 8 hours for Taiwan Time
    timestamp = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d%H%M")
    filename = f"current probe results_{timestamp}.zip"
    
    return StreamingResponse(out_zip, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={filename}"})
