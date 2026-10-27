//! Colours by name for graph options (`mcolor()`, `lcolor()`, `fcolor()`,
//! `color()`, `palette color`): standard X11/CSS colour names, the gray
//! scale `gs0` (black) to `gs16` (white), `"R G B"` triplets and `#rrggbb`.

/// Named colours (X11/CSS values) and the Okabe-Ito palette under its own names.
pub const NAMED: &[(&str, (u8, u8, u8))] = &[
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("red", (255, 0, 0)),
    ("green", (0, 128, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)),
    ("pink", (255, 192, 203)),
    ("brown", (165, 42, 42)),
    ("gray", (128, 128, 128)),
    ("grey", (128, 128, 128)),
    ("navy", (0, 0, 128)),
    ("maroon", (128, 0, 0)),
    ("olive", (128, 128, 0)),
    ("teal", (0, 128, 128)),
    ("cyan", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("lime", (0, 255, 0)),
    ("gold", (255, 215, 0)),
    ("khaki", (240, 230, 140)),
    ("sienna", (160, 82, 45)),
    ("lavender", (230, 230, 250)),
    ("cranberry", (159, 0, 15)),
    ("crimson", (220, 20, 60)),
    ("coral", (255, 127, 80)),
    ("salmon", (250, 128, 114)),
    ("tomato", (255, 99, 71)),
    ("chocolate", (210, 105, 30)),
    ("tan", (210, 180, 140)),
    ("beige", (245, 245, 220)),
    ("ivory", (255, 255, 240)),
    ("indigo", (75, 0, 130)),
    ("violet", (238, 130, 238)),
    ("orchid", (218, 112, 214)),
    ("plum", (221, 160, 221)),
    ("turquoise", (64, 224, 208)),
    ("skyblue", (135, 206, 235)),
    ("steelblue", (70, 130, 180)),
    ("royalblue", (65, 105, 225)),
    ("forest_green", (34, 139, 34)),
    ("forestgreen", (34, 139, 34)),
    ("darkgreen", (0, 100, 0)),
    ("dkgreen", (0, 100, 0)),
    ("ltblue", (173, 216, 230)),
    ("lightblue", (173, 216, 230)),
    ("dkblue", (0, 0, 139)),
    ("darkblue", (0, 0, 139)),
    ("dkorange", (255, 140, 0)),
    ("darkorange", (255, 140, 0)),
    ("ltgreen", (144, 238, 144)),
    ("lightgreen", (144, 238, 144)),
    ("emerald", (0, 155, 119)),
    ("ebblue", (0, 139, 188)),
    ("eltblue", (0, 191, 255)),
    ("none", (255, 255, 255)),
    // the remaining documented colour names, with approximate values
    ("dknavy", (12, 36, 80)),
    ("edkblue", (21, 56, 102)),
    ("midblue", (34, 112, 194)),
    ("emidblue", (102, 153, 204)),
    ("bluishgray", (214, 224, 230)),
    ("ltbluishgray", (232, 238, 241)),
    ("dimgray", (105, 105, 105)),
    ("eggshell", (240, 234, 214)),
    ("eltgreen", (194, 230, 194)),
    ("midgreen", (52, 140, 60)),
    ("mint", (180, 228, 200)),
    ("erose", (240, 200, 196)),
    ("ltkhaki", (240, 232, 190)),
    ("olive_teal", (68, 128, 108)),
    ("orange_red", (255, 69, 0)),
    ("sand", (214, 200, 160)),
    ("sandb", (230, 210, 160)),
    ("stone", (228, 222, 204)),
    ("sea", (40, 120, 160)),
    ("sky", (110, 180, 230)),
    ("reddish", (200, 60, 50)),
    ("vermillion", (213, 94, 0)),
    // Okabe & Ito (2008)
    ("oi_blue", (0, 114, 178)),
    ("oi_vermillion", (213, 94, 0)),
    ("oi_green", (0, 158, 115)),
    ("oi_purple", (204, 121, 167)),
    ("oi_orange", (230, 159, 0)),
    ("oi_skyblue", (86, 180, 233)),
    ("oi_yellow", (240, 228, 66)),
];

/// Parse a colour specification; `None` if it is not a colour.
pub fn parse(spec: &str) -> Option<(u8, u8, u8)> {
    let s = spec.trim().trim_matches('"').trim().to_lowercase();
    // intensity/opacity suffixes such as `navy*0.5` or `red%50` lighten the colour
    let (base, factor) = match s.split_once(['*', '%']) {
        Some((b, f)) => {
            let f: f64 = f.trim().parse().ok()?;
            (b.trim().to_string(), if s.contains('%') { f / 100.0 } else { f })
        }
        None => (s.clone(), 1.0),
    };
    let rgb = if let Some(hex) = base.strip_prefix('#') {
        let v = u32::from_str_radix(hex, 16).ok()?;
        if hex.len() != 6 {
            return None;
        }
        ((v >> 16) as u8, (v >> 8) as u8, v as u8)
    } else if let Some(n) = base.strip_prefix("gs") {
        let k: u8 = n.parse().ok().filter(|k: &u8| *k <= 16)?;
        let g = (k as u32 * 255 / 16) as u8;
        (g, g, g)
    } else if base.split_whitespace().count() == 3 {
        let v: Vec<u8> = base.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if v.len() != 3 {
            return None;
        }
        (v[0], v[1], v[2])
    } else {
        NAMED.iter().find(|(n, _)| *n == base).map(|(_, c)| *c)?
    };
    if factor >= 1.0 {
        return Some(rgb);
    }
    // blend towards white
    let mix = |c: u8| (c as f64 * factor + 255.0 * (1.0 - factor)).round() as u8;
    Some((mix(rgb.0), mix(rgb.1), mix(rgb.2)))
}

/// Gray value (ITU-R BT.601 luma) of a colour, for grayscale printing.
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_specifications() {
        assert_eq!(parse("navy"), Some((0, 0, 128)));
        assert_eq!(parse("\"255 0 128\""), Some((255, 0, 128)));
        assert_eq!(parse("#0072B2"), Some((0, 114, 178)));
        assert_eq!(parse("gs0"), Some((0, 0, 0)));
        assert_eq!(parse("gs16"), Some((255, 255, 255)));
        assert_eq!(parse("gs8"), Some((127, 127, 127)));
        assert_eq!(parse("black*0.5"), Some((128, 128, 128)));
        assert_eq!(parse("nosuchcolour"), None);
        assert_eq!(luma(255, 255, 255), 255);
    }
}
