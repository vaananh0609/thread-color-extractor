from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from extractor import VERSIONS, process_image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


def widget_key(*parts) -> str:
    raw = "-".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)


st.set_page_config(
    page_title="Thread Color Extractor",
    page_icon="🧵",
    layout="wide",
)

st.markdown(
    """
    <style>
      .swatch {
        height: 42px;
        border-radius: 8px;
        border: 1px solid #333;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.25);
      }
      div[data-testid="stDialog"] code { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(result, sel_idx=None) -> Image.Image:
    img = Image.fromarray(result.overlay)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    coef = result.label_coef
    pts = [(x, int(np.polyval(coef, x))) for x in range(0, w, 2)]
    draw.line(pts, fill=(220, 30, 30), width=2)
    font = _font(16)
    for i, b in enumerate(result.bundles):
        draw.line([(b.x, 0), (b.x, h)], fill=(50, 220, 80), width=1)
        label = b.code[-3:]
        tw = draw.textlength(label, font=font)
        x0 = b.x - tw / 2 - 3
        fill = (255, 140, 0) if i == sel_idx else (255, 230, 80)
        draw.rectangle([x0, 4, x0 + tw + 6, 22], fill=fill)
        draw.text((x0 + 3, 5), label, fill=(0, 0, 0), font=font)
    if sel_idx is not None:
        b = result.bundles[sel_idx]
        y_top = int(np.polyval(coef, b.x))
        draw.rectangle(
            [b.x - result.pitch / 2, y_top, b.x + result.pitch / 2, h - 2],
            outline=(255, 140, 0),
            width=3,
        )
    return img


def results_to_csv(all_results: list) -> bytes:
    rows = []
    for res in all_results:
        for b in res.bundles:
            row = {"Ảnh": res.filename, "Mã chỉ": b.code}
            for key, _, _ in VERSIONS:
                row[f"{key}_HEX"] = b.hex[key]
                row[f"{key}_R"], row[f"{key}_G"], row[f"{key}_B"] = b.rgb[key]
            rows.append(row)
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def find_bundle(jobs, selected):
    if not selected:
        return None, None
    for job in jobs:
        if not job["ok"]:
            continue
        res = job["result"]
        if res.filename == selected["file"]:
            idx = selected["idx"]
            if 0 <= idx < len(res.bundles):
                return res, res.bundles[idx]
    return None, None


def render_bundle_grid(res, cols_n: int = 14, thumb_w: int = 56) -> None:
    grid = st.columns(cols_n)
    for idx, bundle in enumerate(res.bundles):
        with grid[idx % cols_n]:
            if bundle.disp_roi.size > 0:
                st.image(bundle.disp_roi, width=thumb_w)
            selected = st.session_state.selected
            is_on = bool(
                selected
                and selected["file"] == res.filename
                and selected["idx"] == idx
            )
            if st.button(
                bundle.code[-3:] if not is_on else f"●{bundle.code[-3:]}",
                key=widget_key("pick", res.filename, idx, bundle.code),
                use_container_width=True,
                type="primary" if is_on else "secondary",
                help=bundle.code,
            ):
                st.session_state.selected = {"file": res.filename, "idx": idx}
                st.session_state.selected_color = None
                st.rerun()


def render_detail(res, bundle) -> None:
    head, close = st.columns([3, 1])
    head.markdown(f"### Mã bó chỉ: `{bundle.code}`")
    if close.button("Đóng", key=widget_key("close", res.filename), use_container_width=True):
        st.session_state.selected = None
        st.session_state.selected_color = None
        st.rerun()

    if bundle.disp_roi.size > 0:
        st.image(bundle.disp_roi, width=130)

    st.caption("Bấm một ô màu để xem mã")
    cols = st.columns(3)
    css = []
    for col, (key, _, _) in zip(cols, VERSIONS):
        hex_code = bundle.hex[key]
        rgb = bundle.rgb[key]
        btn_key = widget_key("color", res.filename, bundle.code, key)
        picked = st.session_state.selected_color and st.session_state.selected_color["key"] == key
        border = "#111" if picked else "#333"
        thick = "3px" if picked else "1px"
        css.append(
            f""".st-key-{btn_key} button {{
                background:{hex_code} !important;
                color: transparent !important;
                border:{thick} solid {border} !important;
                height:56px;
            }}"""
        )
        with col:
            if st.button("\u00a0", key=btn_key, use_container_width=True, help=hex_code):
                st.session_state.selected_color = {
                    "key": key,
                    "hex": hex_code,
                    "rgb": rgb,
                }
                st.rerun()
            st.caption(key)
    st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)

    info = st.session_state.selected_color
    if info:
        rgb = info["rgb"]
        st.markdown("---")
        st.markdown(
            f'<div class="swatch" style="background:{info["hex"]};height:56px;"></div>',
            unsafe_allow_html=True,
        )
        st.caption("HEX")
        st.code(info["hex"], language=None)
        st.caption("RGB")
        st.code(f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})", language=None)
        st.code(f"{rgb[0]}, {rgb[1]}, {rgb[2]}", language=None)
    else:
        st.info("Chọn một trong 3 ô màu bên trên.")


def click_image(img, key):
    """Ảnh bấm được, tự co vừa chiều rộng cột."""
    try:
        return streamlit_image_coordinates(img, key=key, use_column_width="always")
    except TypeError:  # bản thư viện cũ không có use_column_width
        return streamlit_image_coordinates(img, key=key, width=1000)


st.title("🧵 Thread Color Extractor")
st.caption(
    "Tải nhiều ảnh hàng chỉ, nhập mã đầu hàng cho từng ảnh. "
    "Bấm một bó chỉ trên ảnh để xem chi tiết bên phải; bấm ô màu để lấy HEX / RGB."
)

with st.sidebar:
    st.header("Cấu hình")
    n_bundles = st.number_input("Số bó mỗi hàng", min_value=1, max_value=80, value=28, step=1)
    use_wb = st.checkbox("Cân bằng trắng (WB)", value=True)
    show_grid = st.checkbox("Hiện lưới bó chỉ nhỏ", value=not HAS_CLICK)

uploaded = st.file_uploader(
    "Tải ảnh (có thể chọn nhiều file)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)

if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "selected" not in st.session_state:
    st.session_state.selected = None
if "selected_color" not in st.session_state:
    st.session_state.selected_color = None
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if "last_click" not in st.session_state:
    st.session_state.last_click = {}

if uploaded:
    st.subheader("Mã đầu hàng theo từng ảnh")
    h1, h2 = st.columns([3, 1])
    h1.caption("Tên file")
    h2.caption("Mã đầu hàng")
    configs = []
    for i, f in enumerate(uploaded):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{f.name}**")
        with c2:
            start = st.number_input(
                f"Mã đầu hàng — {f.name}",
                min_value=0,
                max_value=99999,
                value=2168,
                step=1,
                key=widget_key("start", f.name, i),
                label_visibility="collapsed",
            )
        configs.append((f, int(start)))

        processed_names = {
            j["result"].filename for j in st.session_state.jobs if j["ok"]
        }
        if f.name not in processed_names:
            _, mid_prev, _ = st.columns([1, 10, 1])
            with mid_prev:
                st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                         use_container_width=True)

    run = st.button("Xử lý ảnh", type="primary")
    if run:
        jobs = []
        progress = st.progress(0.0, text="Đang xử lý...")
        for i, (f, start) in enumerate(configs):
            progress.progress(i / max(len(configs), 1), text=f"Đang xử lý {f.name}...")
            try:
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                result = process_image(
                    arr, start_code=start, n_bundles=int(n_bundles), use_wb=use_wb
                )
                result.filename = f.name
                jobs.append({"ok": True, "result": result, "error": None})
            except Exception as exc:
                jobs.append({"ok": False, "result": None, "error": f"{f.name}: {exc}"})
        progress.progress(1.0, text="Xong")
        st.session_state.jobs = jobs
        st.session_state.run_id += 1
        st.session_state.last_click = {}
        st.session_state.selected = None
        st.session_state.selected_color = None
        st.rerun()

jobs = st.session_state.jobs
if jobs:
    ok_jobs = [j for j in jobs if j["ok"]]
    err_jobs = [j for j in jobs if not j["ok"]]
    for e in err_jobs:
        st.error(e["error"])

    if ok_jobs:
        csv_bytes = results_to_csv([j["result"] for j in ok_jobs])
        st.download_button(
            "Tải CSV kết quả",
            data=csv_bytes,
            file_name="ket_qua_ma_chi.csv",
            mime="text/csv",
        )

    def render_jobs(cols_n: int) -> None:
        sel = st.session_state.selected
        for job in ok_jobs:
            res = job["result"]
            st.markdown("---")
            st.subheader(f"{res.filename}  ·  mã đầu {res.start_code}")
            st.caption(
                f"pitch = {res.pitch:.1f}px · bó đầu x = {res.a0:.1f} · "
                f"WB_GAIN = {np.round(res.wb_gain, 3).tolist()} · {len(res.bundles)} bó"
            )
            sel_idx = sel["idx"] if sel and sel["file"] == res.filename else None
            img = draw_overlay(res, sel_idx)

            left, right = st.columns([1.7, 1], gap="large")
            with left:
                if HAS_CLICK:
                    st.caption("👆 Bấm vào một bó chỉ trên ảnh để xem chi tiết bên phải")
                    val = click_image(
                        img, widget_key("img", res.filename, st.session_state.run_id)
                    )
                    if val and val != st.session_state.last_click.get(res.filename):
                        st.session_state.last_click[res.filename] = val
                        x_orig = val["x"] * (img.width / val["width"])
                        idx = int(np.argmin([abs(b.x - x_orig) for b in res.bundles]))
                        st.session_state.selected = {"file": res.filename, "idx": idx}
                        st.session_state.selected_color = None
                        st.rerun()
                else:
                    st.image(img, use_container_width=True)

                if show_grid or not HAS_CLICK:
                    with st.expander("Lưới bó chỉ", expanded=not HAS_CLICK):
                        render_bundle_grid(res, cols_n=cols_n)

            with right:
                if sel_idx is not None:
                    render_detail(res, res.bundles[sel_idx])
                else:
                    st.info("Bấm vào một bó chỉ trên ảnh, thông tin sẽ hiện ở đây.")

    render_jobs(8)
elif not uploaded:
    st.info("Hãy tải một hoặc nhiều ảnh hàng chỉ để bắt đầu.")
