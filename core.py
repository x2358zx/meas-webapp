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

def _detect_yellow_mask(img_bgr, cfg, grid_limits=None): # 2026/1/1
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(cfg["hsv_detection"]["lower_yellow_hsv"], dtype=np.uint8)
    hi = np.array(cfg["hsv_detection"]["upper_yellow_hsv"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k_w, k_h = cfg["morphology"]["kernel_size"]
    it = int(cfg["morphology"]["iterations"])
    kernel = np.ones((int(k_h), int(k_w)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
    
    if grid_limits:
        top, bottom, left, right = grid_limits
        t, b = int(top)+5, int(bottom)-5
        l, r = int(left)+5, int(right)-5
        
        # Slicing is safer and guaranteed
        h, w = mask.shape
        t = max(0, min(t, h))
        b = max(0, min(b, h))
        l = max(0, min(l, w))
        r = max(0, min(r, w))
        
        # Clear outside
        if t > 0: mask[:t, :] = 0
        if b < h: mask[b:, :] = 0
        if l > 0: mask[:, :l] = 0
        if r < w: mask[:, r:] = 0
        
        # 2. Contour Filtering (Remove small arrows/noise)
        # Calculate x_step to determine a reasonable width threshold
        grid_w = right - left
        if grid_w > 0:
            x_step = grid_w / 10.0
            min_w = x_step * 0.3 # Filter anything narrower than 0.3 grid (Arrows are usually small)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(mask)
            for c in contours:
                x, y, w_c, h_c = cv2.boundingRect(c)
                if w_c > min_w:
                    cv2.drawContours(clean_mask, [c], -1, 255, -1)
            mask = clean_mask
        
    return mask

def _get_green_y_at_x(green_mask, target_x, window=5): # User req: smaller window (was 10)
    h, w = green_mask.shape
    x1 = max(0, int(target_x - window))
    x2 = min(w, int(target_x + window))
    strip = green_mask[:, x1:x2]
    coords = cv2.findNonZero(strip)
    if coords is None:
        return None
    ys = coords[:, 0, 1]
    if ys.size == 0:
        return None
    # User requested Peak Detection (Min Y) instead of Mean (Center) - 2026/01/02
    return float(np.min(ys))

def _analyze_clk_pulse(yellow_mask, x_step): # 2026/1/1
    # 先分析黃色像素的 Y 分佈，分離 High Level (Pulse) 與 Low Level (Baseline)
    coords = cv2.findNonZero(yellow_mask)
    if coords is None:
        return []
    
    ys = coords[:, 0, 1]
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    
    # 簡單閾值：取 min 與 max 的中間，小於閾值 (較高) 的視為 Pulse
    # 前提：CLK "lifted" 代表 upward pulse (Active High)
    y_thresh = (y_min + y_max) / 2.0
    
    # 建立 High Level Mask
    # 使用 numpy 操作比 cv2.inRange 快且方便
    mask_high = np.zeros_like(yellow_mask)
    # 這裡需要把符合條件的座標設為 255
    # coords 格式 (N, 1, 2) -> x, y
    high_indices = np.where(ys < y_thresh)[0]
    
    if len(high_indices) == 0:
        return []
        
    # 將 High pixels 畫回 mask
    # 為了效率，直接用 indices
    high_pts = coords[high_indices]
    for pt in high_pts:
        mask_high[pt[0,1], pt[0,0]] = 255

    # 找 High Level 的 Contours
    contours, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    # 篩選掉太小的 (寬度 < 1/2 格)
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > (x_step * 0.5):
            valid_contours.append(c)
    
    if not valid_contours:
         if contours:
             c = max(contours, key=cv2.contourArea)
             # 若面積太小則放棄
             if cv2.contourArea(c) < 20: return []
         else:
             return []
    else:
        c = max(valid_contours, key=lambda x: cv2.boundingRect(x)[2])

    x, y, w, h = cv2.boundingRect(c)
    
    start_x = x
    end_x = x + w
    
    points_x = []
    
    # 1. Start Point: 抬升後約 0.2 個格子 (User request 2026/01/02)
    p1_x = start_x + (x_step * 0.2)
    points_x.append(p1_x)
    
    # 2. 黃線約 2/5 的位置 (Reverted to previous logic)
    p2_x = start_x + w * 0.4
    points_x.append(p2_x)
    
    # 3. 黃線抬升的最後一格 (Reverted to previous logic)
    w_check_last = min(w, x_step)
    p3_x = end_x - (w_check_last / 2)
    if p3_x <= p2_x: p3_x = end_x - 1
    points_x.append(p3_x)
    
    return points_x

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
    
    for i, (x, y, mA) in enumerate(points_ma):
        txt = f"{mA:.2f} mA"
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        
        # 決定文字位置：下、上、下 交錯
        # Even points (0, 2...): 下 (Below)
        # Odd points (1...): 上 (Above)
        is_above = (i % 2 != 0)
        
        tx = max(5, min(int(x - tw/2), out.shape[1]-tw-5))
        
        if is_above:
            # 放在點的上方
            ty = max(thh+5, int(y) - 15)
        else:
            # 放在點的下方 (原邏輯)
            ty = max(thh+10, min(int(y)+thh+15, out.shape[0]-10))
            
        cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 5, (0,0,255), -1)
    return out

def process_one_image(img_bgr, filename=""): # 2026/1/1
    cfg = load_config()
    top, bottom, left, right = _detect_grid_coords(cfg)
    major_h, major_v, y_step, x_step = _compute_major_lines(top, bottom, left, right)
    ref_idx = int(cfg["manual_grid_settings"]["ref_0ma_index"])
    ma_div = extract_ma_div_from_filename(filename) or float(cfg["manual_grid_settings"]["ma_per_division"])
    
    # 1. 偵測綠色 mask
    mask_green = _detect_green_mask(img_bgr, cfg)
    
    # 2. 偵測黃色 mask (限制在 Grid 內) 與 三個關鍵點
    mask_yellow = _detect_yellow_mask(img_bgr, cfg, grid_limits=(top, bottom, left, right))
    target_xs = _analyze_clk_pulse(mask_yellow, x_step)
    
    # 計算電流參數
    px_per_div_v = (bottom - top) / 8.0
    ma_per_px = ma_div / px_per_div_v if px_per_div_v > 1e-9 else 0.0
    y0_ref = major_h[ref_idx]
    
    points_ma = []
    
    # 若有偵測到 CLK 三點，就只測這三點
    if target_xs:
        for tx in target_xs:
            gy = _get_green_y_at_x(mask_green, tx)
            if gy is not None:
                # 計算 mA
                # 注意：y 越小(上方) 電流越大? 
                # 示波器圖通常上面是正電流，下面是負? 
                # y0 是 0mA 線。
                # 公式：current = (y0 - y) * ma_per_px
                # 若 y < y0 (在上方) -> current > 0.
                val = (y0_ref - gy) * ma_per_px
                points_ma.append((tx, gy, float(val)))
            else:
                # 該點沒綠線，略過或補 0? 暫時略過
                pass
    else:
        # Fallback: 若沒黃線或失敗，是否維持原邏輯?
        # 用戶目標是「新增偵測CLK黃線...總共會有三次...」
        # 若失敗，這裡保持空列表，或者 fallback 到舊邏輯?
        # 為了安全，若沒偵測到黃線，回傳空比較好，避免誤導。
        # 不過舊邏輯 `_find_flat_levels` 是找所有綠色平坦段。
        # 暫時回傳空。
        pass
        
    annotated = _annotate(img_bgr, points_ma, cfg)
    
    # 標籤與電壓
    vdd_from_name = extract_v_from_filename(filename)
    _draw_reference_labels_only(annotated, cfg, vdd_value_from_filename=vdd_from_name)

    return annotated, {
        "levels_detected": len(points_ma),
        "values_mA": [v for _,_,v in points_ma],
        "used_ma_per_div": ma_div
    }


