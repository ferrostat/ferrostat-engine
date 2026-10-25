//! Display formats: `%9.0g`, `%9.2f`, `%10.3e`, `%9.2fc`, `%-20s`, `%td` ...

use ferrostat_core::missing::{missing_code, missing_name};

#[derive(Debug, Clone, PartialEq)]
pub struct NumFormat {
    pub left: bool,
    pub zero_pad: bool,
    pub width: usize,
    pub decimals: usize,
    pub kind: char, // 'g', 'f', 'e', 's', 't'
    pub comma: bool,
    pub raw: String,
}

impl NumFormat {
    pub fn parse(s: &str) -> Option<NumFormat> {
        let raw = s.to_string();
        let mut t = s.strip_prefix('%')?;
        let mut left = false;
        let mut zero_pad = false;
        if let Some(r) = t.strip_prefix('-') {
            left = true;
            t = r;
        }
        if let Some(r) = t.strip_prefix('~') {
            t = r;
        }
        if t.starts_with('t') || t.starts_with('d') && !t[1..].starts_with(|c: char| c.is_ascii_digit()) {
            return Some(NumFormat { left, zero_pad, width: 0, decimals: 0, kind: 't', comma: false, raw });
        }
        if t.starts_with('0') && t.len() > 1 {
            zero_pad = true;
        }
        let wend = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
        let width: usize = t[..wend].parse().unwrap_or(0);
        t = &t[wend..];
        let mut decimals = 0;
        if let Some(r) = t.strip_prefix('.') {
            let dend = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            decimals = r[..dend].parse().unwrap_or(0);
            t = &r[dend..];
        }
        let mut ch = t.chars();
        let kind = ch.next()?;
        if !"gfes".contains(kind) {
            return None;
        }
        if kind == 't' {
            return None;
        }
        let comma = ch.next() == Some('c');
        Some(NumFormat { left, zero_pad, width, decimals, kind, comma, raw })
    }

    pub fn apply(&self, v: f64) -> String {
        let body = match self.kind {
            't' => crate::dates::format_date(v, &self.raw),
            _ => format_body(v, self),
        };
        pad(&body, self.width, self.left)
    }
}

fn pad(s: &str, w: usize, left: bool) -> String {
    let n = s.chars().count();
    if n >= w {
        s.to_string()
    } else if left {
        format!("{s}{}", " ".repeat(w - n))
    } else {
        format!("{}{s}", " ".repeat(w - n))
    }
}

/// Pad/align a string to a `%Ns` / `%-Ns` format.
pub fn fmt_str(s: &str, fmt: &str) -> String {
    match NumFormat::parse(fmt) {
        Some(f) if f.kind == 's' => pad(s, f.width, f.left),
        _ => s.to_string(),
    }
}

fn group_thousands(int_part: &str) -> String {
    let (sign, digits) = if let Some(d) = int_part.strip_prefix('-') { ("-", d) } else { ("", int_part) };
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{sign}{out}")
}

fn format_body(v: f64, f: &NumFormat) -> String {
    if let Some(code) = missing_code(v) {
        return missing_name(code);
    }
    match f.kind {
        'f' => {
            let s = format!("{:.*}", f.decimals, v);
            let s = if s.starts_with("-0") && s.trim_start_matches(['-', '0', '.']).is_empty() {
                s[1..].to_string()
            } else {
                s
            };
            let s = if f.comma {
                let (i, d) = s.split_once('.').map_or((s.as_str(), None), |(a, b)| (a, Some(b)));
                let g = group_thousands(i);
                match d {
                    Some(d) => format!("{g}.{d}"),
                    None => g,
                }
            } else {
                s
            };
            if f.zero_pad && s.len() < f.width {
                let (sign, rest) = if let Some(r) = s.strip_prefix('-') { ("-", r) } else { ("", s.as_str()) };
                format!("{sign}{}{rest}", "0".repeat(f.width - s.len()))
            } else {
                s
            }
        }
        'e' => fmt_e(v, f.decimals),
        _ => {
            let w = if f.width == 0 { 9 } else { f.width };
            if f.decimals > 0 {
                fmt_sig(v, f.decimals, w.max(f.decimals + 6))
            } else {
                let s = fmt_g(v, w.saturating_sub(1).max(1));
                if f.comma && v.abs() >= 1000.0 && !s.contains('e') {
                    let (i, d) = s.split_once('.').map_or((s.as_str(), None), |(a, b)| (a, Some(b)));
                    let g = group_thousands(i);
                    match d {
                        Some(d) => format!("{g}.{d}"),
                        None => g,
                    }
                } else {
                    s
                }
            }
        }
    }
}

/// Exponential notation: mantissa with `d` decimals, two-digit exponent.
pub fn fmt_e(v: f64, d: usize) -> String {
    let s = format!("{:.*e}", d, v);
    let (m, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0').trim_end_matches('.');
        t.to_string()
    } else {
        s.to_string()
    }
}

fn drop_leading_zero(s: String) -> String {
    if let Some(r) = s.strip_prefix("0.") {
        format!(".{r}")
    } else if let Some(r) = s.strip_prefix("-0.") {
        format!("-.{r}")
    } else {
        s
    }
}

/// General format with at most `sig` significant digits, as short as possible.
fn fmt_sig(v: f64, sig: usize, maxlen: usize) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let exp = v.abs().log10().floor() as i32;
    if exp >= -5 && exp < sig as i32 {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        let s = drop_leading_zero(strip_zeros(&format!("{:.*}", dec, v)));
        if s.len() <= maxlen {
            return s;
        }
    }
    let m = fmt_e(v, sig.saturating_sub(1));
    let (mant, e) = m.split_once('e').unwrap();
    format!("{}e{}", strip_zeros(mant), e)
}

/// `%w.0g`-like rendering in at most `width` characters.
pub fn fmt_g(v: f64, width: usize) -> String {
    if let Some(code) = missing_code(v) {
        return missing_name(code);
    }
    if v == 0.0 {
        return "0".into();
    }
    // the sign does not count against the digit budget (as in %w.0g columns)
    if v < 0.0 {
        return format!("-{}", fmt_g(-v, width));
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        let s = format!("{}", v as i64);
        if s.len() <= width {
            return s;
        }
    }
    // try fixed notation, as many digits as fit
    let exp = v.abs().log10().floor() as i32;
    if exp < width as i32 && exp >= -5 {
        for dec in (0..=16usize).rev() {
            let s = drop_leading_zero(strip_zeros(&format!("{:.*}", dec, v)));
            let sig_ok = {
                let digits = s.chars().filter(|c| c.is_ascii_digit()).count();
                digits <= 16
            };
            if s.len() <= width && sig_ok {
                if s == "0" || s == "-0" || s.is_empty() {
                    break;
                }
                return s;
            }
        }
    }
    // exponential with as many mantissa digits as fit
    for d in (0..=15usize).rev() {
        let s = fmt_e(v, d);
        let (mant, e) = s.split_once('e').unwrap();
        let s = format!("{}e{}", if d > 0 { mant.to_string() } else { mant.to_string() }, e);
        if s.len() <= width || d == 0 {
            return s;
        }
    }
    format!("{v}")
}

/// Default rendering used by `display` with no format (like `%10.0g`).
pub fn display_number(v: f64) -> String {
    fmt_g(v, 9)
}

/// Format a number using a format string; falls back to `%9.0g`.
pub fn fmt(v: f64, fmt: &str) -> String {
    match NumFormat::parse(fmt) {
        Some(f) => f.apply(v),
        None => pad(&fmt_g(v, 8), 9, false),
    }
}

/// Format for table cells, trimmed of padding.
pub fn fmt_trim(v: f64, f: &str) -> String {
    fmt(v, f).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general() {
        assert_eq!(display_number(1.0 / 3.0), ".33333333");
        assert_eq!(display_number(2.0 / 3.0), ".66666667");
        assert_eq!(display_number(12345678.0), "12345678");
        assert_eq!(display_number(1234567890.0), "1.235e+09");
        assert_eq!(display_number(-0.5), "-.5");
        assert_eq!(display_number(13.008333333), "13.008333");
        assert_eq!(fmt(0.5, "%9.2f"), "     0.50");
        assert_eq!(fmt(1234567.891, "%12.2fc"), "1,234,567.89");
        assert_eq!(fmt(3.14159, "%-8.3f"), "3.142   ");
        assert_eq!(fmt(12345.0, "%9.2e"), " 1.23e+04");
        assert_eq!(fmt(7.0, "%03.0f"), "007");
        assert_eq!(fmt(1.0 / 3.0, "%9.0g"), " .3333333");
        assert_eq!(fmt(14610.0, "%td"), "01jan2000");
        assert_eq!(fmt_str("ab", "%-5s"), "ab   ");
    }
}
