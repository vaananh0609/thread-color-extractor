"""Thread color + code extraction (one row of bundles)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage import color
def to_hex(rgb) -> str:
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def sat_val(a):
    maxc = a.max(axis=2)
    minc = a.min(axis=2)
    sat = np.where(maxc > 1e-5, (maxc - minc) / maxc, 0.0)
    return sat, maxc


def _s2l(c):
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def sv(pix):
    s, v = sat_val(pix.reshape(1, -1, 3))
    return s[0], v[0]


def _thread_mask(a, sat_levels, min_px=30):
    sat, val = sv(a)
    for s in sat_levels:
        t = (sat > s) & (val > 30)
        if t.sum() >= min_px:
            return t
    return None


def color_v1(pix, sat_percentile=75):
    sat, val = sv(pix)
    good = ~np.all(pix > 220, axis=1) & (val > 30) & (val < 245)
    if good.sum() < 10:
        good = (val > 30) & (val < 252)
    if good.sum() < 10:
        return None
    th = np.percentile(sat[good], sat_percentile)
    sel = good & (sat >= th)
    return tuple(int(x) for x in np.median(pix[sel], axis=0))


def color_v2(pix, wb_gain, sat_min=0.25, v_lo=25, v_hi=75):
    a = np.clip(pix * wb_gain, 0, 255)
    t = _thread_mask(a, (sat_min, 0.12, 0.05, 0.0))
    if t is None:
        return None
    _, val = sv(a)
    lo, hi = np.percentile(val[t], [v_lo, v_hi])
    sel = t & (val >= lo) & (val <= hi)
    if sel.sum() == 0:
        sel = t
    return tuple(int(x) for x in _l2s(_s2l(a[sel]).mean(axis=0)).round())


def color_v3(pix, wb_gain, sat_min=0.20, chroma_keep=50, L_lo=30, L_hi=85):
    a = np.clip(pix * wb_gain, 0, 255)
    t = _thread_mask(a, (sat_min, 0.10, 0.04, 0.0))
    if t is None:
        return None
    lab = color.rgb2lab((a / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    step1 = t & (chroma >= np.percentile(chroma[t], 100 - chroma_keep))
    lo, hi = np.percentile(lab[step1, 0], [L_lo, L_hi])
    sel = step1 & (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    if sel.sum() == 0:
        sel = step1
    med = np.median(lab[sel], axis=0)
    rgb = color.lab2rgb(med.reshape(1, 1, 3)).reshape(3) * 255
    return tuple(int(x) for x in np.clip(rgb, 0, 255).round())


VERSIONS = [
    ("V1", "v1", color_v1),
    ("V2", "v2", color_v2),
    ("V3", "v3", color_v3),
]


def _comb_score(a, b, profile, width, n_bundles):
    xs = a + b * np.arange(n_bundles)
    if xs[-1] >= width - 1:
        return -1e9
    teeth = np.interp(xs, np.arange(width), profile)
    gaps = np.interp(xs[:-1] + b / 2, np.arange(width), profile)
    return teeth.sum() - gaps.sum()


@dataclass
class BundleResult:
    code: str
    x: int
    disp_roi: np.ndarray
    rgb: dict
    hex: dict


@dataclass
class ImageExtractResult:
    filename: str
    start_code: int
    n_bundles: int
    wb_gain: np.ndarray
    pitch: float
    a0: float
    label_coef: np.ndarray
    overlay: np.ndarray
    bundles: list[BundleResult]


def process_image(
    arr: np.ndarray,
    start_code: int = 2168,
    n_bundles: int = 28,
    use_wb: bool = True,
) -> ImageExtractResult:
    h, w = arr.shape[:2]

    bg = np.median(arr[int(h * 0.4) : int(h * 0.9), :12].reshape(-1, 3), axis=0)
    wb_gain = (bg.max() / (bg + 1e-6)) if use_wb else np.ones(3)

    y1_band, y2_band = int(h * 0.42), int(h * 0.72)
    band = arr[y1_band:y2_band].astype(float)
    sat_b, _ = sat_val(band)
    gray = band.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    profile = (sat_b * 255 * 0.5 + gx).mean(axis=0)
    profile = uniform_filter1d(profile, size=5)
    profile = profile - np.median(profile)

    best = (-1e9, None, None)
    for b in np.linspace(0.029 * w, 0.036 * w, 80):
        for a in np.linspace(0.01 * w, 0.07 * w, 120):
            s = _comb_score(a, b, profile, w, n_bundles)
            if s > best[0]:
                best = (s, a, b)
    _, a0, pitch = best
    if a0 is None or pitch is None:
        raise ValueError("Không khớp được các bó chỉ trên ảnh.")

    xs0 = a0 + pitch * np.arange(n_bundles)
    peaks = []
    for x in xs0:
        lo = int(max(0, x - pitch * 0.35))
        hi = int(min(w, x + pitch * 0.35))
        seg = np.clip(profile[lo:hi], 0, None)
        cx = (np.arange(lo, hi) * seg).sum() / seg.sum() if seg.sum() > 0 else x
        peaks.append(int(round(cx)))
    peaks = np.array(peaks)
    codes = [str(start_code + i) for i in range(len(peaks))]

    gray_full = arr.mean(axis=2)
    top = gray_full[: int(h * 0.5)]
    dark = gray_full < (np.median(top) - 50)
    run = uniform_filter1d(dark.astype(float), size=70, axis=1)
    y_lo, y_hi = int(h * 0.12), int(h * 0.55)
    bx, by = [], []
    for x in range(0, w, 3):
        rows = np.where(run[y_lo:y_hi, x] > 0.8)[0]
        if len(rows):
            bx.append(x)
            by.append(rows.max() + y_lo)
    bx, by = np.array(bx), np.array(by)
    if len(bx) < 10:
        raise ValueError("Không tìm thấy viền nhãn phía trên hàng chỉ.")
    coef = np.polyfit(bx, by, 1)
    for _ in range(3):
        ok = np.abs(by - np.polyval(coef, bx)) < 4
        if ok.sum() < 10:
            break
        coef = np.polyfit(bx[ok], by[ok], 1)

    lab = color.rgb2lab(arr / 255.0)
    bg_lab = np.median(lab[int(h * 0.4) : int(h * 0.9), :12].reshape(-1, 3), axis=0)
    d_e = np.linalg.norm(lab - bg_lab, axis=2)

    d0, d1 = 10, int(0.30 * h)
    fg_pct = 30
    bundles: list[BundleResult] = []

    overlay = arr.copy()

    for cx, code in zip(peaks, codes):
        yb = int(np.polyval(coef, cx))
        center = float(cx)
        pys, pxs = [], []
        for d in range(d0, d1):
            y = yb + d
            if y >= h:
                break
            hw = min(0.30 * pitch, 6 + 0.10 * d)
            lo, hi = int(max(0, center - 1.5 * hw)), int(min(w, center + 1.5 * hw + 1))
            weights = d_e[y, lo:hi]
            if weights.sum() > 0:
                c = (np.arange(lo, hi) * weights).sum() / weights.sum()
                center = 0.8 * center + 0.2 * c
            l2, h2 = int(max(0, center - hw)), int(min(w, center + hw + 1))
            pys += [y] * (h2 - l2)
            pxs += list(range(l2, h2))

        py, px = np.array(pys), np.array(pxs)
        if len(py) == 0:
            continue

        de = d_e[py, px]
        fg = de >= np.percentile(de, fg_pct)
        pix = arr[py[fg], px[fg]].astype(float)
        if pix.size == 0:
            pix = arr[py, px].astype(float)

        cols_rgb = {}
        for key, _, fn in VERSIONS:
            if key == "V1":
                rgb = fn(pix)
            else:
                rgb = fn(pix, wb_gain)
            if rgb is None:
                rgb = tuple(int(v) for v in np.median(pix, axis=0))
            cols_rgb[key] = rgb

        y0, y1, x0, x1 = py.min(), py.max() + 1, px.min(), px.max() + 1
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[py - y0, px - x0] = True
        disp = arr[y0:y1, x0:x1].copy()
        disp[~m] = 255

        overlay[py[::4], px[::4]] = np.array([0, 220, 255], dtype=np.uint8)

        bundles.append(
            BundleResult(
                code=code,
                x=int(cx),
                disp_roi=disp,
                rgb=cols_rgb,
                hex={k: to_hex(v) for k, v in cols_rgb.items()},
            )
        )

    return ImageExtractResult(
        filename="",
        start_code=start_code,
        n_bundles=n_bundles,
        wb_gain=wb_gain,
        pitch=float(pitch),
        a0=float(a0),
        label_coef=coef,
        overlay=overlay,
        bundles=bundles,
    )

