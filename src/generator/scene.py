from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class SceneSpec:
    width: int
    height: int


def _rng_np(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randint(0, 10_000_000))


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _noise_layer(width: int, height: int, rng: random.Random, blur_radius: float) -> np.ndarray:
    nprng = _rng_np(rng)
    arr = nprng.integers(0, 256, size=(height, width), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(img, dtype=np.float32)


def create_base_surface(spec: SceneSpec, rng: random.Random) -> Image.Image:
    h, w = spec.height, spec.width
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    gx = rng.uniform(-24.0, 24.0) * (xx - 0.5)
    gy = rng.uniform(-24.0, 24.0) * (yy - 0.5)
    gradient = gx + gy

    n1 = _noise_layer(w, h, rng, blur_radius=rng.uniform(0.8, 1.8))
    n2 = _noise_layer(w, h, rng, blur_radius=rng.uniform(2.0, 4.5))
    n3 = _noise_layer(w, h, rng, blur_radius=rng.uniform(6.0, 10.0))

    texture = (
        128.0
        + gradient
        + 18.0 * (_normalize(n1) - 0.5)
        + 24.0 * (_normalize(n2) - 0.5)
        + 12.0 * (_normalize(n3) - 0.5)
    )
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    return Image.fromarray(texture, mode="L")


def _draw_scratch_mask(spec: SceneSpec, rng: random.Random, size: float, blur: float, density: float, orientation: float) -> Image.Image:
    mask = Image.new("L", (spec.width, spec.height), 0)
    draw = ImageDraw.Draw(mask)

    lines = max(1, int(1 + density * 4))
    length = int(spec.width * (0.2 + size * 0.7))
    cx = rng.randint(0, spec.width - 1)
    cy = rng.randint(0, spec.height - 1)
    ang = math.radians(orientation)
    dx = int(math.cos(ang) * length * 0.5)
    dy = int(math.sin(ang) * length * 0.5)
    width = max(1, int(1 + size * 2))

    for _ in range(lines):
        jx = rng.randint(-10, 10)
        jy = rng.randint(-10, 10)
        x0, y0 = cx - dx + jx, cy - dy + jy
        x1, y1 = cx + dx + jx, cy + dy + jy
        draw.line((x0, y0, x1, y1), fill=255, width=width)

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur * 1.6))
    return mask


def _draw_pit_mask(spec: SceneSpec, rng: random.Random, size: float, blur: float, density: float) -> Image.Image:
    mask = Image.new("L", (spec.width, spec.height), 0)
    draw = ImageDraw.Draw(mask)

    count = max(4, int(15 + density * 70))
    max_r = max(2, int(2 + size * 9))
    for _ in range(count):
        cx = rng.randint(0, spec.width - 1)
        cy = rng.randint(0, spec.height - 1)
        r = rng.randint(1, max_r)
        shade = rng.randint(140, 255)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=shade)

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur * 1.2))
    return mask


def _draw_stain_mask(spec: SceneSpec, rng: random.Random, size: float, blur: float, density: float) -> Image.Image:
    mask = Image.new("L", (spec.width, spec.height), 0)
    draw = ImageDraw.Draw(mask)

    blobs = max(1, int(1 + density * 5))
    base_r = int(spec.width * (0.04 + 0.13 * size))
    for _ in range(blobs):
        cx = rng.randint(base_r, max(base_r, spec.width - base_r))
        cy = rng.randint(base_r, max(base_r, spec.height - base_r))
        for _ in range(rng.randint(4, 10)):
            rx = rng.randint(max(3, base_r // 4), max(4, base_r))
            ry = rng.randint(max(3, base_r // 4), max(4, base_r))
            ox = rng.randint(-base_r, base_r)
            oy = rng.randint(-base_r, base_r)
            shade = rng.randint(90, 220)
            draw.ellipse((cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry), fill=shade)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, blur * 3.0)))
    return mask


def _draw_crack_mask(spec: SceneSpec, rng: random.Random, size: float, blur: float, orientation: float) -> Image.Image:
    mask = Image.new("L", (spec.width, spec.height), 0)
    draw = ImageDraw.Draw(mask)

    steps = int(20 + size * 80)
    step_len = 2 + size * 5
    x = rng.randint(0, spec.width - 1)
    y = rng.randint(0, spec.height - 1)
    ang = math.radians(orientation)
    pts = [(x, y)]

    for _ in range(steps):
        ang += math.radians(rng.uniform(-18.0, 18.0))
        x += int(math.cos(ang) * step_len)
        y += int(math.sin(ang) * step_len)
        x = max(0, min(spec.width - 1, x))
        y = max(0, min(spec.height - 1, y))
        pts.append((x, y))
        if x in (0, spec.width - 1) or y in (0, spec.height - 1):
            break

    width = max(1, int(1 + size * 1.5))
    draw.line(pts, fill=255, width=width)

    if len(pts) > 10 and rng.random() < 0.7:
        branch_start = pts[rng.randint(5, len(pts) - 1)]
        bx, by = branch_start
        bang = ang + math.radians(rng.uniform(-70, 70))
        bpts = [(bx, by)]
        for _ in range(rng.randint(8, 20)):
            bang += math.radians(rng.uniform(-20.0, 20.0))
            bx += int(math.cos(bang) * step_len * 0.8)
            by += int(math.sin(bang) * step_len * 0.8)
            bx = max(0, min(spec.width - 1, bx))
            by = max(0, min(spec.height - 1, by))
            bpts.append((bx, by))
        draw.line(bpts, fill=220, width=1)

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur * 0.8))
    return mask


def sample_params(rng: random.Random) -> dict[str, float]:
    return {
        "size": rng.uniform(0.15, 1.0),
        "contrast": rng.uniform(0.20, 0.95),
        "blur": rng.uniform(0.05, 1.0),
        "density": rng.uniform(0.10, 1.0),
        "orientation": rng.uniform(0.0, 180.0),
    }


def apply_defect(
    base: Image.Image,
    defect_type: str,
    params: dict[str, float],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    spec = SceneSpec(width=base.width, height=base.height)
    none_mask = Image.new("L", (base.width, base.height), 0)
    if defect_type == "none":
        return base, none_mask

    size = params["size"]
    contrast = params["contrast"]
    blur = params["blur"]
    density = params["density"]
    orientation = params["orientation"]

    if defect_type == "scratch":
        mask = _draw_scratch_mask(spec, rng, size, blur, density, orientation)
        sign = 1.0 if rng.random() < 0.6 else -1.0
    elif defect_type == "pit_corrosion":
        mask = _draw_pit_mask(spec, rng, size, blur, density)
        sign = -1.0
    elif defect_type == "stain":
        mask = _draw_stain_mask(spec, rng, size, blur, density)
        sign = -1.0 if rng.random() < 0.75 else 1.0
    elif defect_type == "crack":
        mask = _draw_crack_mask(spec, rng, size, blur, orientation)
        sign = -1.0
    else:
        raise ValueError(f"Unsupported defect type: {defect_type}")

    arr = np.array(base, dtype=np.float32)
    marr = np.array(mask, dtype=np.float32) / 255.0
    delta = sign * contrast * 105.0 * marr
    out = np.clip(arr + delta, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, mode="L")

    if rng.random() < 0.45:
        out_img = out_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    return out_img, mask


def serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[k] = round(float(v), 4) if isinstance(v, (float, int)) else v
    return out
