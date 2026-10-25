"""Draw the Ferrostat logo: assets/logo.png (icon) and assets/logo-wordmark.png.

A rust-coloured tile (ferro: iron, rust, Rust) holding a histogram with a
fitted density curve (stat). Requires Pillow:  python3 make_logo.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SS = 4  # supersampling factor for anti-aliasing

RUST_LIGHT = (214, 98, 38)
RUST_DARK = (128, 42, 14)
CREAM = (255, 240, 222)
CURVE = (255, 255, 255)
INK = (118, 112, 110)  # mid grey: readable on light and dark pages
WORD_RUST = (200, 84, 30)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def tile(size):
    """The square icon at `size` pixels."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    # diagonal rust gradient, clipped to a rounded square
    grad = Image.new("RGBA", (s, s))
    gp = grad.load()
    for y in range(s):
        for x in range(s):
            gp[x, y] = lerp(RUST_LIGHT, RUST_DARK, (x + y) / (2 * s)) + (255,)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    # histogram of a normal sample: seven bars under the density
    left, right = s * 0.17, s * 0.83
    base = s * 0.78
    top = s * 0.24
    nbars = 7
    gap = s * 0.018
    width = (right - left) / nbars

    def density(u):  # u in [0, 1] across the plot
        z = (u - 0.5) * 5.2
        return math.exp(-z * z / 2)

    for i in range(nbars):
        u = (i + 0.5) / nbars
        h = (base - top) * density(u) * 0.92
        x0 = left + i * width + gap / 2
        d.rounded_rectangle([x0, base - h, x0 + width - gap, base], radius=int(s * 0.012), fill=CREAM + (225,))

    # the fitted density curve, drawn thick with round joins
    pts = []
    for k in range(201):
        u = k / 200
        pts.append((left + u * (right - left), base - (base - top) * density(u) * 1.04))
    lw = int(s * 0.035)
    # stamp a round brush along the curve: smooth edges and round caps
    r = lw / 2
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (r / 4)))
        for j in range(steps):
            x, y = x0 + (x1 - x0) * j / steps, y0 + (y1 - y0) * j / steps
            d.ellipse([x - r, y - r, x + r, y + r], fill=CURVE + (255,))
    # baseline
    d.rounded_rectangle([left - s * 0.02, base + s * 0.025, right + s * 0.02, base + s * 0.025 + lw * 0.8], radius=int(lw * 0.4), fill=CREAM + (255,))

    img = img.resize((size, size), Image.LANCZOS)
    return img


def shadowed(icon, pad):
    """Icon on a transparent canvas with a soft drop shadow."""
    w, h = icon.size
    canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = icon.split()[3].point(lambda a: a * 0.35)
    shadow.paste((60, 20, 5, 255), (pad, pad + pad // 3), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(pad / 2.5))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(icon, (pad, pad))
    return canvas


def wordmark(height):
    icon = tile(int(height * 0.8))
    font = ImageFont.truetype(FONT, int(height * 0.46))
    text = "Ferrostat"
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    gap = int(height * 0.18)
    margin = int(height * 0.1)
    w = margin + icon.width + gap + (r - l) + margin
    img = Image.new("RGBA", (w, height), (0, 0, 0, 0))
    img.alpha_composite(icon, (margin, (height - icon.height) // 2))
    d = ImageDraw.Draw(img)
    x = margin + icon.width + gap - l
    y = (height - (b - t)) // 2 - t
    # "Ferro" in rust, "stat" in ink
    d.text((x, y), "Ferro", font=font, fill=WORD_RUST + (255,))
    fx = x + probe.textlength("Ferro", font=font)
    d.text((fx, y), "stat", font=font, fill=INK + (255,))
    return img


if __name__ == "__main__":
    shadowed(tile(448), 32).save(os.path.join(HERE, "logo.png"))
    tile(64).save(os.path.join(HERE, "icon-64.png"))
    wordmark(240).save(os.path.join(HERE, "logo-wordmark.png"))
    print("wrote logo.png, icon-64.png, logo-wordmark.png")
