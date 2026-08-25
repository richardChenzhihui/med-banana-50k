#!/usr/bin/env python3
"""Render the Med-Banana README hero banner.

Pure typography + vector shapes; no medical imagery is embedded.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "hero.png"

S = 3  # supersampling factor
W, H = 1600, 460

INK = (12, 16, 26)
INK_SOFT = (24, 30, 44)
CREAM = (247, 244, 236)
MUTED = (158, 166, 184)
BANANA = (255, 196, 61)
GREEN = (74, 205, 148)
RED = (255, 118, 108)
BLUE = (120, 176, 255)

FONTS = "/System/Library/Fonts/Supplemental"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONTS}/{name}", size * S)


def rr(d: ImageDraw.ImageDraw, box, r, **kw) -> None:
    d.rounded_rectangle([c * S for c in box], radius=r * S, **kw)


def text(d, xy, s, f, fill, anchor="la", spacing=None):
    x, y = xy
    if spacing is None:
        d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
        return
    # manual letter-spacing
    cx = x * S
    for ch in s:
        d.text((cx, y * S), ch, font=f, fill=fill, anchor=anchor)
        cx += d.textlength(ch, font=f) + spacing * S


def width(d, s, f) -> float:
    return d.textlength(s, font=f) / S


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), INK)
    d = ImageDraw.Draw(img)

    # --- background: soft diagonal wash + faint grid -------------------------
    for i in range(H):
        t = i / H
        c = tuple(round(INK[k] + (INK_SOFT[k] - INK[k]) * (t ** 0.7)) for k in range(3))
        d.rectangle([0, i * S, W * S, (i + 1) * S], fill=c)
    for gx in range(0, W, 40):
        d.line([(gx * S, 0), (gx * S, H * S)], fill=(20, 26, 38), width=1)
    for gy in range(0, H, 40):
        d.line([(0, gy * S), (W * S, gy * S)], fill=(20, 26, 38), width=1)

    # warm glow behind the wordmark, feathered so it has no visible edge
    mask = Image.new("L", (W * S // 4, H * S // 4), 0)
    ImageDraw.Draw(mask).ellipse(
        [-140 * S // 4, -200 * S // 4, 700 * S // 4, 300 * S // 4], fill=150
    )
    mask = mask.filter(ImageFilter.GaussianBlur(70 * S // 4)).resize((W * S, H * S), Image.BILINEAR)
    img = Image.composite(Image.new("RGB", img.size, (86, 68, 24)), img, mask)
    d = ImageDraw.Draw(img)

    f_pill = font("Arial Bold.ttf", 17)
    f_mark = font("Arial Bold.ttf", 92)
    f_sub = font("Arial.ttf", 25)
    f_chip = font("Arial Bold.ttf", 19)
    f_chip_l = font("Arial.ttf", 19)
    f_flow = font("Arial Bold.ttf", 21)
    f_foot = font("Arial.ttf", 18)

    PAD = 72
    y = 46

    # --- EMNLP 2026 pill -----------------------------------------------------
    label = "EMNLP 2026"
    pw = width(d, label, f_pill) + 34
    rr(d, (PAD, y, PAD + pw, y + 34), 17, fill=BANANA)
    text(d, (PAD + pw / 2, y + 17), label, f_pill, INK, anchor="mm")

    tag = "Main Conference"
    text(d, (PAD + pw + 16, y + 17), tag, f_pill, MUTED, anchor="lm")

    # --- wordmark ------------------------------------------------------------
    y += 52
    text(d, (PAD, y), "Med-Banana", f_mark, CREAM)

    y += 116
    text(d, (PAD, y), "Learning Quality-Controlled Medical Image Editing", f_sub, (206, 212, 226))
    text(d, (PAD, y + 34), "from Success-and-Failure Trajectories", f_sub, (206, 212, 226))

    # --- stat chips ----------------------------------------------------------
    y += 86
    chips = [
        ("50,635", "accepted", GREEN),
        ("37,822", "rejected", RED),
        ("114K", "images", BLUE),
        ("3 modalities", "23 diseases", MUTED),
        ("90K+", "HF downloads", BANANA),
    ]
    x = PAD
    for value, label, colour in chips:
        vw = width(d, value, f_chip)
        lw = width(d, label, f_chip_l)
        cw = vw + lw + 42
        rr(d, (x, y, x + cw, y + 40), 20, fill=(20, 25, 37), outline=(44, 52, 70), width=S)
        d.ellipse([(x + 15) * S, (y + 17) * S, (x + 21) * S, (y + 23) * S], fill=colour)
        text(d, (x + 29, y + 20), value, f_chip, CREAM, anchor="lm")
        text(d, (x + 29 + vw + 7, y + 20), label, f_chip_l, MUTED, anchor="lm")
        x += cw + 12

    # --- edit -> verify -> refine loop (right side) --------------------------
    cx, cy, r = W - 232, 196, 112
    d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
              outline=(52, 60, 80), width=2 * S)

    nodes = [("EDIT", BANANA, -90), ("VERIFY", GREEN, 30), ("REFINE", BLUE, 150)]
    pts = []
    for name, colour, ang in nodes:
        a = math.radians(ang)
        nx, ny = cx + r * math.cos(a), cy + r * math.sin(a)
        pts.append((nx, ny, name, colour))

    # arrowheads along the circle
    for i in range(len(pts)):
        a0 = math.radians(nodes[i][2]) + 0.30
        a1 = math.radians(nodes[(i + 1) % 3][2]) - 0.30
        if a1 < a0:
            a1 += 2 * math.pi
        steps = 26
        prev = None
        for k in range(steps + 1):
            a = a0 + (a1 - a0) * k / steps
            p = (cx + r * math.cos(a), cy + r * math.sin(a))
            if prev:
                d.line([prev[0] * S, prev[1] * S, p[0] * S, p[1] * S],
                       fill=nodes[i][1], width=2 * S)
            prev = p
        # arrow head at a1
        ah = 9
        tip = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        back = a1 - 0.055
        base = (cx + r * math.cos(back), cy + r * math.sin(back))
        nx_, ny_ = math.cos(back), math.sin(back)
        d.polygon([
            (tip[0] * S, tip[1] * S),
            ((base[0] - ny_ * ah / 2) * S, (base[1] + nx_ * ah / 2) * S),
            ((base[0] + ny_ * ah / 2) * S, (base[1] - nx_ * ah / 2) * S),
        ], fill=nodes[i][1])

    for nx, ny, name, colour in pts:
        tw = width(d, name, f_flow)
        bw = tw + 32
        rr(d, (nx - bw / 2, ny - 19, nx + bw / 2, ny + 19), 19, fill=INK, outline=colour, width=2 * S)
        text(d, (nx, ny), name, f_flow, colour, anchor="mm")

    text(d, (cx, cy - 12), "R = 10", font("Arial Bold.ttf", 26), CREAM, anchor="mm")
    text(d, (cx, cy + 16), "retry from source", font("Arial.ttf", 16), MUTED, anchor="mm")

    # --- footer --------------------------------------------------------------
    d.line([(PAD * S, (H - 58) * S), ((W - PAD) * S, (H - 58) * S)], fill=(40, 47, 64), width=S)
    text(d, (PAD, H - 38), "Med-Banana-80K  ·  88,457 success-and-failure editing trajectories", f_foot, (176, 184, 200))
    text(d, (W - PAD, H - 38), "arXiv:2511.00801", f_foot, (176, 184, 200), anchor="ra")

    img.resize((W, H), Image.LANCZOS).save(OUT, optimize=True)
    print("wrote", OUT, Image.open(OUT).size)


if __name__ == "__main__":
    main()
