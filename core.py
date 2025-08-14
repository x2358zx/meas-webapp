import os
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks
import re


BASE_DIR = Path(__file__).resolve().parent
CFG_PATH = Path(os.getenv("MEAS_CONFIG_PATH", str(BASE_DIR / "config.json")))

def load_config():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))

def save_config(cfg: dict):
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    
def extract_ma_div_from_filename(filename: str):
    """
    從檔名萃取 mA/div 設定，樣式：..._IR5mA 或 ..._IR2.5mA
    回傳 float 或 None（未匹配時）
    """
    if not filename:
        return None
    m = re.search(r'_IR(\d+(?:\.\d+)?)mA', filename, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _detect_grid_coords(cfg):
    g = cfg["manual_grid_settings"]["grid_coords"]
    top = float(g["top_y"]); bottom = float(g["bottom_y"]); left = float(g["left_x"]); right = float(g["right_x"])
    return top, bottom, left, right

def _compute_major_lines(top, bottom, left, right):
    y_step = (bottom - top) / 8.0
    x_step = (right - left) / 10.0
    major_h = [top + i*y_step for i in range(9)]
    major_v = [left + j*x_step for j in range(11)]
    return major_h, major_v, y_step, x_step

def preview_grid_overlay(img_bgr: np.ndarray, grid: dict, ref_idx: int, ma_div: float=None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    top = float(grid["top_y"]); bottom = float(grid["bottom_y"]); left = float(grid["left_x"]); right = float(grid["right_x"])
    img = img_bgr.copy()
    # outer box
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0,255,255), 2)
    # major grid
    y_step = (bottom - top) / 8.0
    x_step = (right - left) / 10.0
    for i in range(9):
        y = int(top + i*y_step)
        cv2.line(img, (int(left), y), (int(right), y), (0,255,255), 1)
    for j in range(11):
        x = int(left + j*x_step)
        cv2.line(img, (x, int(top)), (x, int(bottom)), (255,0,255), 1)
    # 0 mA line
    ref_y = int(top + ref_idx * y_step)
    cv2.line(img, (int(left), ref_y), (int(right), ref_y), (0,255,0), 2)
    cv2.putText(img, f"0mA idx={ref_idx}", (int(right)+5, ref_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    if ma_div is not None:
        cv2.putText(img, f"{ma_div:.2f} mA/div", (int(left), int(top)-10 if top-10>15 else int(top+20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return img

def _detect_green_mask(img_bgr, cfg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(cfg["hsv_detection"]["lower_green_hsv"], dtype=np.uint8)
    hi = np.array(cfg["hsv_detection"]["upper_green_hsv"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k_w, k_h = cfg["morphology"]["kernel_size"]
    it = int(cfg["morphology"]["iterations"])
    kernel = np.ones((int(k_h), int(k_w)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
    return mask

def _find_flat_levels(mask, img_w, img_h, cfg, top, bottom):
    x0 = int(img_w * cfg["flat_region_detection"]["x_start_factor"])
    x1 = int(img_w * cfg["flat_region_detection"]["x_end_factor"])
    coords = cv2.findNonZero(mask)
    if coords is None:
        return []
    # filter by x range and above bottom major line
    cond = (coords[:,0,0] >= x0) & (coords[:,0,0] <= x1) & (coords[:,0,1] < bottom)
    coords = coords[cond]
    if coords.size == 0:
        return []
    min_pixels = int(cfg["flat_region_detection"]["min_pixels_for_analysis"])
    if coords.shape[0] < min_pixels:
        return []
    y = coords[:,0,1]
    hist, _ = np.histogram(y, bins=np.arange(0, img_h+1))
    mh = cfg["peak_detection"]["min_height_factor"]
    md = max(1, int(img_h * cfg["peak_detection"]["min_distance_factor"]))
    height_thr = max(1, int(hist.max() * mh)) if hist.max()>0 else 1
    peaks, _ = find_peaks(hist, height=height_thr, distance=md)
    win = int(cfg["peak_detection"]["pixel_grouping_window"])
    min_pixels_peak = int(cfg["flat_region_detection"]["min_pixels_per_peak"])
    levels = []
    for py in peaks:
        m = (y >= py-win) & (y <= py+win)
        yy = y[m]
        if yy.size > min_pixels_peak:
            xx = coords[:,0,0][(coords[:,0,1] >= py-win) & (coords[:,0,1] <= py+win)]
            levels.append((float(xx.mean()), float(yy.mean())))
    levels.sort(key=lambda t: t[1])
    return levels

def _annotate(img_bgr, points_ma, cfg):
    out = img_bgr.copy()
    fs = float(cfg["annotation"]["font_scale"])
    th = int(cfg["annotation"]["thickness"])
    color = tuple(int(c) for c in cfg["annotation"]["text_color_bgr"])
    for x, y, mA in points_ma:
        txt = f"{mA:.2f} mA"
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(5, min(int(x - tw/2), out.shape[1]-tw-5))
        ty = max(thh+10, min(int(y)+thh+10, out.shape[0]-10))
        cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 5, (0,0,255), -1)
    return out

def process_one_image(img_bgr, filename=""):
    cfg = load_config()
    top, bottom, left, right = _detect_grid_coords(cfg)
    major_h, major_v, y_step, _ = _compute_major_lines(top, bottom, left, right)
    ref_idx = int(cfg["manual_grid_settings"]["ref_0ma_index"])
    ma_div = extract_ma_div_from_filename(filename) or float(cfg["manual_grid_settings"]["ma_per_division"])
    # mask & levels
    mask = _detect_green_mask(img_bgr, cfg)
    levels = _find_flat_levels(mask, img_bgr.shape[1], img_bgr.shape[0], cfg, top, bottom)
    # mA calculation
    px_per_div_v = (bottom - top) / 8.0
    ma_per_px = ma_div / px_per_div_v if px_per_div_v>1e-9 else 0.0
    y0 = major_h[ref_idx]
    points_ma = []
    for (x, y) in levels:
        current = (y0 - y) * ma_per_px
        points_ma.append((x, y, float(current)))
    annotated = _annotate(img_bgr, points_ma, cfg)
    return annotated, {
    "levels_detected": len(points_ma),
    "values_mA": [v for _,_,v in points_ma],
    "used_ma_per_div": ma_div
}


