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

def _check_rising_edge(mask, x1, x2):
    h, w = mask.shape
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    if x2 <= x1: return False
    
    ROI = mask[:, x1:x2]
    w_roi = x2 - x1
    
    # Check left 1/3 and right 1/3
    split_L = int(w_roi * 0.33)
    split_R = int(w_roi * 0.66)
    
    strip_L = ROI[:, :split_L]
    strip_R = ROI[:, split_R:]
    
    def get_mean_y(strip):
        c = cv2.findNonZero(strip)
        if c is None: return None
        return np.mean(c[:, 0, 1])
        
    y_L = get_mean_y(strip_L)
    y_R = get_mean_y(strip_R)
    
    if y_L is None or y_R is None: return False
    
    # Rising edge: Y should decrease (High Y -> Low Y)
    # y_L > y_R
    return (y_L - y_R) > 5 # Minimal threshold

def process_one_image(img_bgr, filename="", override_config=None, calibration_threshold=None): # 2026/1/9
    cfg = load_config()
    if override_config:
        if "manual_grid_settings" in override_config:
            cfg["manual_grid_settings"].update(override_config["manual_grid_settings"])
        if "overlay_labels" in override_config:
            cfg["overlay_labels"] = override_config["overlay_labels"]

    top, bottom, left, right = _detect_grid_coords(cfg)
    major_h, major_v, y_step, x_step = _compute_major_lines(top, bottom, left, right)
    ref_idx = int(cfg["manual_grid_settings"]["ref_0ma_index"])
    ma_div = extract_ma_div_from_filename(filename) or float(cfg["manual_grid_settings"]["ma_per_division"])
    
    mask_green = _detect_green_mask(img_bgr, cfg)
    
    # Check Manual Lines
    manual_clk = cfg["manual_grid_settings"].get("clk_lines")
    manual_meas = cfg["manual_grid_settings"].get("meas_lines")
    
    result_status = "成功"
    points_ma = []
    calculated_thresh = None
    
    if manual_clk and len(manual_clk) == 2 and manual_meas and len(manual_meas) == 3:
        # Manual Logic
        mask_yellow = _detect_yellow_mask(img_bgr, cfg, grid_limits=None) # Global mask
        is_rising = _check_rising_edge(mask_yellow, manual_clk[0], manual_clk[1])
        
        if not is_rising:
            result_status = "失敗: 無上升邊緣"
            # Draw FAIL indicator on image
            cv2.putText(img_bgr, "FAIL: No Rising Edge", (int(left)+20, int(top)+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # 2026/01/09: First Image Calibration Logic
            
            # 1. Determine Threshold
            if calibration_threshold is not None:
                # Locked Mode
                y_thresh = calibration_threshold
            else:
                # Calibration Mode (First Image)
                # Find Yellow Ref Line (0V)
                ref_idx_y = int(cfg["manual_grid_settings"].get("ref_0ma_index_yellow", 7))
                y_ref_yellow = major_h[ref_idx_y] if ref_idx_y < len(major_h) else bottom
                
                # Find Peak High (Min Y) from mask
                y_coords = cv2.findNonZero(mask_yellow)
                if y_coords is not None:
                    ys = y_coords[:, 0, 1]
                    y_min = np.min(ys)
                    # Thresh = (Peak + Baseline) / 2
                    y_thresh = (y_min + y_ref_yellow) / 2.0
                else:
                    y_thresh = y_ref_yellow # Fallback (should be rare if is_rising passed)
                
                calculated_thresh = y_thresh

            # Measure at manual points
            # Measure at manual points
            for tx in manual_meas:
                 # Check logic
                 in_yellow_range = (manual_clk[0] <= tx <= manual_clk[1])
                 if in_yellow_range:
                     # Check Yellow Level at tx
                     y_at_tx = _get_green_y_at_x(mask_yellow, tx, window=5) # Reuse helper for yellow mask
                     if y_at_tx is not None and y_at_tx > y_thresh:
                         # Yellow is Low (Y is large) -> FAIL
                         result_status = "失敗: 黃色訊號未抬升"
                         cv2.putText(img_bgr, "FAIL: Yellow Not Lifted", (int(left)+20, int(top)+150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                         points_ma = [] # Clear points
                         break

                 gy = _get_green_y_at_x(mask_green, tx)
                 if gy is not None:
                      y0 = major_h[ref_idx]
                      px_per_div_v = (bottom - top) / 8.0
                      ma_per_px = ma_div / px_per_div_v if px_per_div_v > 1e-9 else 0.0
                      val = (y0 - gy) * ma_per_px
                      points_ma.append((tx, gy, float(val)))
            
            # If FAIL, ensure points_ma is empty (redundant safety)
            if result_status != "成功":
                points_ma = []
            
            # If any point is missing (green line not found), treat as FAIL
            elif len(points_ma) < len(manual_meas):
                result_status = "失敗: 缺少測量點"
                cv2.putText(img_bgr, "FAIL: Missing Meas Point", (int(left)+20, int(top)+100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                points_ma = [] # Clear points on this fail too
    else:
        # Auto Logic (Legacy)
        mask_yellow = _detect_yellow_mask(img_bgr, cfg, grid_limits=(top, bottom, left, right))
        target_xs = _analyze_clk_pulse(mask_yellow, x_step)
        
        px_per_div_v = (bottom - top) / 8.0
        ma_per_px = ma_div / px_per_div_v if px_per_div_v > 1e-9 else 0.0
        y0_ref = major_h[ref_idx]
        
        if target_xs:
            for tx in target_xs:
                gy = _get_green_y_at_x(mask_green, tx)
                if gy is not None:
                    val = (y0_ref - gy) * ma_per_px
                    points_ma.append((tx, gy, float(val)))
    
    annotated = _annotate(img_bgr, points_ma, cfg)
    
    # 2026/01/09: Draw ID on Result
    # Get ID from filename (index 7) or fallback
    display_id_str = ""
    try:
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 8:
             raw_id = parts[7]
             # Remove extension if present (e.g. L0569.jpg -> L0569)
             if '.' in raw_id:
                 raw_id = os.path.splitext(raw_id)[0]
             display_id_str = f"{raw_id}"
    except:
        pass
        
    if display_id_str:
        # Get Pos from Config
        id_pos = cfg["manual_grid_settings"].get("id_label_pos")
        # Default if missing: Bottom Center
        if not id_pos:
            h, w = img_bgr.shape[:2]
            id_pos = {"x": w // 2, "y": h - 20}
            
        cx = int(id_pos["x"])
        cy = int(id_pos["y"])
        
        # Match frontend "bold 24px monospace"
        # OpenCV Simplex 1.0 is ~30px. 0.8 is ~24px.
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness_fg = 2  # Bold
        thickness_bg = 4  # Outline
        
        # Calculate size to center
        (text_w, text_h), baseline = cv2.getTextSize(display_id_str, font_face, font_scale, thickness_fg)
        
        origin_x = cx - text_w // 2
        origin_y = cy # Frontend y is baseline, OpenCV y is baseline. So Cy is correct.
        
        # Draw Outline (Black)
        cv2.putText(annotated, display_id_str, (origin_x, origin_y), 
                    font_face, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
                    
        # Draw Text (White)
        cv2.putText(annotated, display_id_str, (origin_x, origin_y), 
                    font_face, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
    
    # labels
    vdd_from_name = extract_v_from_filename(filename)
    _draw_reference_labels_only(annotated, cfg, vdd_value_from_filename=vdd_from_name)

    return annotated, {
        "levels_detected": len(points_ma),
        "values_mA": [v for _,_,v in points_ma],
        "used_ma_per_div": ma_div,
        "status": result_status,
        "calculated_threshold": calculated_thresh
    }


