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

def extract_v_from_filename(filename: str):
    """
    從檔名第 4 欄(以 '_' 切)解析電壓字串，格式如：1p100V → 1.1V
    例：S019_..._1p100V_... → 回傳 '1.1V'
    未匹配則回傳 None
    """
    if not filename:
        return None
    try:
        stem = Path(filename).stem  # 去副檔名
        parts = stem.split('_')
        if len(parts) >= 4:
            token = parts[3].strip()
            m = re.match(r'(?i)^(\d+)p(\d+)v$', token)
            if m:
                int_part, frac_part = m.group(1), m.group(2)
                # 以小數長度動態格式化，再去除多餘 0 與末尾小數點
                val = float(f"{int_part}.{frac_part}")
                decimals = len(frac_part)
                s = f"{val:.{decimals}f}".rstrip('0').rstrip('.')
                return f"{s}V"
    except Exception:
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

def _draw_reference_labels_only(img, cfg, vdd_value_from_filename: str=None):
    """在 img 上只畫三個參考標籤（CLK 黃、VDD 洋紅、I(VDD) 綠），不畫任何線。"""
    # 讀設定裡的格線座標與步距
    top, bottom, left, right = _detect_grid_coords(cfg)
    _, _, y_step, x_step = _compute_major_lines(top, bottom, left, right)

    # 三色 index（若黃/洋紅缺，回退用綠線 index）
    ref_g = int(cfg["manual_grid_settings"]["ref_0ma_index"])
    ref_y = int(cfg["manual_grid_settings"].get("ref_0ma_index_yellow",  ref_g))
    ref_m = int(cfg["manual_grid_settings"].get("ref_0ma_index_magenta", ref_g))

    # 三個標籤文字（回退預設）
    labels = cfg.get("overlay_labels") or {}
    lbl_clk  = labels.get("clk",  "CLK")
    lbl_vdd  = labels.get("vdd",  "VDD")
    lbl_ivdd = labels.get("ivdd", "I(VDD)")
    
    # ★ 若有從檔名解到電壓值，就**動態**把洋紅標籤改成「VDD=1.1V」這種，不寫回 config
    if vdd_value_from_filename:
        lbl_vdd = f"{lbl_vdd}={vdd_value_from_filename}"

    # 位置規則：與「第一格」(0 div) 距離 <= 1 div → 放第一格上方；否則放到各自 0 mA index 的 y
    x_first = int(left + 0.10 * x_step)     # 左側第一條大格線「內側一點」：讓字靠近左上角，不會壓線
    y_first = int(top  + 0 * y_step)
    def place_y(ref_idx_local: int) -> int:
        return y_first if abs(ref_idx_local - 0) <= 1 else int(top + ref_idx_local * y_step)

    # 只畫「文字」，不畫任何線
    fs, th = 0.8, 2
    cv2.putText(img, lbl_clk,  (x_first+4, place_y(ref_y)+22),  cv2.FONT_HERSHEY_SIMPLEX, fs, (  0,255,255), th, cv2.LINE_AA)  # 黃
    cv2.putText(img, lbl_vdd,  (x_first+4, place_y(ref_m)+22),  cv2.FONT_HERSHEY_SIMPLEX, fs, (255,  0,255), th, cv2.LINE_AA)  # 洋紅
    cv2.putText(img, lbl_ivdd, (x_first+4, place_y(ref_g)+22),  cv2.FONT_HERSHEY_SIMPLEX, fs, (  0,255,  0), th, cv2.LINE_AA)  # 綠


def preview_grid_overlay(
    img_bgr: np.ndarray, grid: dict, ref_idx: int, ma_div: float=None,
    ref_idx_yellow: int=None, ref_idx_magenta: int=None, labels: dict=None
) -> np.ndarray:
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
                    
    labels = labels or {"clk":"CLK","vdd":"VDD","ivdd":"I(VDD)"}
    top = float(grid["top_y"]); bottom = float(grid["bottom_y"]); left = float(grid["left_x"]); right = float(grid["right_x"])
    y_step = (bottom - top) / 8.0
    x_step = (right - left) / 10.0
    y_first = int(top + 0*y_step)
    x_first = int(left + 1*x_step)
    
    # 三色 0mA 線座標
    ref_y_g = int(top + ref_idx * y_step)
    ref_y_y = int(top + ((ref_idx_yellow if ref_idx_yellow is not None else ref_idx) * y_step))
    ref_y_m = int(top + ((ref_idx_magenta if ref_idx_magenta is not None else ref_idx) * y_step))
    
    # 線：綠 / 黃 / 洋紅
    cv2.line(img, (int(left), ref_y_g), (int(right), ref_y_g), (0,255,0),   2)     # 綠
    cv2.line(img, (int(left), ref_y_y), (int(right), ref_y_y), (0,255,255), 2)     # 黃
    cv2.line(img, (int(left), ref_y_m), (int(right), ref_y_m), (255,0,255), 2)     # 洋紅
    
    # 自動標籤：<= 1div 貼第一格上方；否則貼 0mA 線上
    fs = 0.6; th = 2
    def put(txt, x, y, bgr):
        # 往上提一點，避免壓在線上
        cv2.putText(img, txt, (x+4, y+22), cv2.FONT_HERSHEY_SIMPLEX, fs, bgr, th, cv2.LINE_AA)
    
    def place_y(ref_idx_local):
        return y_first if abs(ref_idx_local - 0) <= 1 else int(top + ref_idx_local * y_step)
    
    put(labels.get("clk","CLK"),  x_first, place_y(ref_idx_yellow if ref_idx_yellow is not None else ref_idx), (0,255,255))
    put(labels.get("vdd","VDD"),  x_first, place_y(ref_idx_magenta if ref_idx_magenta is not None else ref_idx), (255,0,255))
    put(labels.get("ivdd","I(VDD)"), x_first, place_y(ref_idx), (0,255,0))
    
    # 仍保留右側「0mA idx=」提示（綠）
    cv2.putText(img, f"0mA idx={ref_idx}", (int(right)+5, ref_y_g), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

                    
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
    
    # 只加三個標籤，不畫線；★把檔名解析到的電壓值帶進去（不動 overlay_labels）
    vdd_from_name = extract_v_from_filename(filename)
    _draw_reference_labels_only(annotated, cfg, vdd_value_from_filename=vdd_from_name)

    return annotated, {
    "levels_detected": len(points_ma),
    "values_mA": [v for _,_,v in points_ma],
    "used_ma_per_div": ma_div
}


