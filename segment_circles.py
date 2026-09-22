import cv2
import numpy as np
import json
from pathlib import Path

import random

def segment_circles(
    image_path,
    d_min_mm,
    d_max_mm,
    hfw_metres,
    stage_x_m = 0.0,
    stage_y_m = 0.0,
    output_dir = None,
    nms_overlap      = 0.3,
    edge_margin_frac = 0.5,
    min_circularity  = 0.85,
    hough_param1     = 50,
    hough_param2     = 25,
    random_seed      = 42
):
    """
    Detect circular dots in a microscopy image.

    Parameters
    ----------
    image_path       : str | Path | np.ndarray
    d_min_mm         : float   minimum dot diameter in mm
    d_max_mm         : float   maximum dot diameter in mm
    hfw_metres       : float   Horizontal Field Width from phenom.GetHFW()
    stage_x_m        : float   stage X position when image was taken
    stage_y_m        : float   stage Y position when image was taken
    output_dir       : str | Path, optional
        Directory to save annotated image and JSON metadata.
    nms_overlap      : float
    edge_margin_frac : float
    min_circularity  : float
    hough_param1     : int
    hough_param2     : int

    Returns
    -------
    annotated : np.ndarray (H,W,3)  BGR image with circles drawn (circle number only)
    mask      : np.ndarray (H,W)    uint8, 255 inside each circle
    circles   : list[dict]          circle metadata with stage coordinates
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    cv2.setRNGSeed(random_seed)
    cv2.setNumThreads(0)  # Disable OpenCV multithreading for reproducibility
    # ── Load ──────────────────────────────────────────────────────────────────
    if isinstance(image_path, np.ndarray):
        img = image_path.copy()
        original_path = None
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError("Cannot read: {}".format(image_path))
        original_path = Path(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Scale ─────────────────────────────────────────────────────────────────
    px_per_mm = w / (hfw_metres * 1000)
    mm_per_px = hfw_metres * 1000 / w
    r_min     = int((d_min_mm / 2) * px_per_mm)
    r_max     = int((d_max_mm / 2) * px_per_mm)
    min_dist  = int(r_min * 1.8)
    print("px/mm={:.1f}  |  d {:.1f}-{:.1f} mm  ->  r {}-{}px".format(
        px_per_mm, d_min_mm, d_max_mm, r_min, r_max))
    print("Stage position: ({:.6f}, {:.6f}) m".format(stage_x_m, stage_y_m))

    # ── Plate mask ────────────────────────────────────────────────────────────
    _, bright = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    k = np.ones((30, 30), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate = np.zeros_like(gray)
    if cnts:
        cv2.drawContours(plate, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

    # ── Preprocess ────────────────────────────────────────────────────────────
    masked = cv2.bitwise_and(gray, gray, mask=plate)
    bil    = cv2.bilateralFilter(masked, 15, 60, 60)
    gauss  = cv2.GaussianBlur(bil, (0, 0), 3)
    sharp  = cv2.addWeighted(bil, 1.8, gauss, -0.8, 0)
    enh    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(sharp)

    # ── Hough ─────────────────────────────────────────────────────────────────
    raw = cv2.HoughCircles(
        enh, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_dist,
        param1=hough_param1, param2=hough_param2,
        minRadius=r_min, maxRadius=r_max,
    )

    if raw is None:
        print("No circles detected.")
        return img.copy(), np.zeros((h, w), np.uint8), []

    candidates = np.round(raw[0]).astype(int)

    in_plate = [(x, y, r) for x, y, r in candidates
                if plate[int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))] > 0]

    def circularity_score(x, y, r):
        pad = r + 5
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(w, x+pad), min(h, y+pad)
        roi = plate[y1:y2, x1:x2]
        circ_mask = np.zeros_like(roi)
        cv2.circle(circ_mask, (x-x1, y-y1), r, 255, -1)
        expected = circ_mask.sum() / 255
        if expected == 0:
            return 0.0
        return (cv2.bitwise_and(roi, circ_mask).sum() / 255) / expected

    passed_circ = [(x, y, r) for x, y, r in in_plate
                   if circularity_score(x, y, r) >= min_circularity]
    print("After circularity filter: {}/{}".format(len(passed_circ), len(in_plate)))

    def not_clipped(x, y, r):
        margin = int(r * edge_margin_frac)
        return (x - r + margin >= 0 and x + r - margin <= w and
                y - r + margin >= 0 and y + r - margin <= h)

    passed_edge = [(x, y, r) for x, y, r in passed_circ if not_clipped(x, y, r)]
    print("After edge filter: {}/{}".format(len(passed_edge), len(passed_circ)))

    # scored = sorted(passed_edge, key=lambda d: np.hypot(d[0]-w/2, d[1]-h/2))
    scored = sorted(passed_edge, key=lambda d: (
    round(np.hypot(d[0]-w/2, d[1]-h/2), 1),  # primary: distance to centre
    d[0],                                       # secondary: x position
    d[1]))                                        # tertiary: y))
    kept = []
    while scored:
        best = scored.pop(0)
        kept.append(best)
        scored = [d for d in scored
                  if np.hypot(d[0]-best[0], d[1]-best[1]) > nms_overlap*(best[2]+d[2])]
    print("After NMS: {}".format(len(kept)))

    # ── Draw & build outputs ──────────────────────────────────────────────────
    out     = img.copy()
    overlay = img.copy()
    mask    = np.zeros((h, w), np.uint8)
    circles = []

    for i, (x, y, r) in enumerate(kept):
        d_mm = round((r * 2) / px_per_mm, 2)
        
        # Convert pixel offset to stage coordinates
        offset_x_m = (x - w/2) * mm_per_px / 1000
        offset_y_m = (y - h/2) * mm_per_px / 1000
        circle_x_m = round(stage_x_m + offset_x_m, 6)
        circle_y_m = round(stage_y_m + offset_y_m, 6)
        
        circles.append({
            "id": i+1,
            "stage_coords_m": (circle_x_m, - circle_y_m),
        }) # Invert Y to match typical stage coordinates
        # Draw circle with number and centre dot
        cv2.circle(overlay, (x, y), r, (0, 200, 255), -1)
        cv2.circle(out,     (x, y), r, (0, 255, 0), 3)
        cv2.circle(out,     (x, y), 3, (0, 0, 255), -1)
        cv2.circle(mask,    (x, y), r, 255, -1)
        cv2.putText(out, str(i+1),
                    (x-10, y+7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    annotated = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)
    print("Detected: {} circles".format(len(circles)))

    # ── Save annotated image & JSON metadata ───────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine input file format
        if original_path:
            input_fmt = original_path.suffix.lower()
        else:
            input_fmt = '.jpg'

        # Save annotated image with original format
        img_out_path = output_dir / "navcam_annotated{}".format(input_fmt)
        cv2.imwrite(str(img_out_path), annotated)
        print("Saved: {}".format(img_out_path.name))

        # Save JSON with full metadata
        json_data = {
            "random_seed": random_seed,
            "image_width_px": int(w),
            "image_height_px": int(h),
            "hfw_metres": float(hfw_metres),
            "stage_position_m": [float(stage_x_m), float(stage_y_m)], # Invert Y to match typical stage coordinates
            "px_per_mm": float(px_per_mm),
            "circles": [
                {
                    "id": int(c["id"]),
                    "pixel_centre": [int(x), int(y)],
                    "radius_px": int(r),
                    "diameter_mm": float(d_mm),
                    "stage_centre_m": [float(c["stage_coords_m"][0]), float(c["stage_coords_m"][1])],
                }
                for c, (x, y, r), d_mm in zip(circles, kept, [(r*2)/px_per_mm for x,y,r in kept])
            ]
        }
        json_path = output_dir / "circles_metadata.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print("Saved: {}".format(json_path.name))

    return annotated, mask, circles, json_path