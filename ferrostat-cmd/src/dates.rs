//! Calendar arithmetic. Dates are days since 01jan1960 (`%td`), datetimes
//! milliseconds since 01jan1960 00:00:00 (`%tc`), and the period types count
//! months/quarters/weeks/half-years/years since 1960.

use ferrostat_core::missing::SYSMISS;

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const MONTHS_FULL: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
];
const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Days from civil date (proleptic Gregorian) relative to 1970-01-01.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

const EPOCH_1960: i64 = -3653; // days_from_civil(1960,1,1)

pub fn mdy(m: f64, d: f64, y: f64) -> f64 {
    if !(1.0..=12.0).contains(&m) || d < 1.0 || m.fract() != 0.0 || d.fract() != 0.0 || y.fract() != 0.0 {
        return SYSMISS;
    }
    let (m, d, y) = (m as i64, d as i64, y as i64);
    let days = days_from_civil(y, m, d);
    // validate day of month
    if civil_from_days(days) != (y, m, d) {
        return SYSMISS;
    }
    (days - EPOCH_1960) as f64
}

/// (year, month, day) of a %td date.
pub fn ymd(td: f64) -> Option<(i64, i64, i64)> {
    if !td.is_finite() || td >= SYSMISS {
        return None;
    }
    Some(civil_from_days(td.floor() as i64 + EPOCH_1960))
}

pub fn year(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |x| x.0 as f64)
}
pub fn month(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |x| x.1 as f64)
}
pub fn day(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |x| x.2 as f64)
}
/// Day of week, 0 = Sunday.
pub fn dow(td: f64) -> f64 {
    if td >= SYSMISS { SYSMISS } else { ((td.floor() as i64 + 5).rem_euclid(7)) as f64 }
}
pub fn doy(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |(y, _, _)| td.floor() - mdy(1.0, 1.0, y as f64) + 1.0)
}
pub fn quarter(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |x| ((x.1 - 1) / 3 + 1) as f64)
}
pub fn week(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |_| ((doy(td) - 1.0) / 7.0).floor().min(51.0) + 1.0)
}
pub fn halfyear(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |x| if x.1 <= 6 { 1.0 } else { 2.0 })
}

pub fn mofd(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |(y, m, _)| ((y - 1960) * 12 + m - 1) as f64)
}
pub fn qofd(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |(y, m, _)| ((y - 1960) * 4 + (m - 1) / 3) as f64)
}
pub fn yofd(td: f64) -> f64 {
    year(td)
}
pub fn wofd(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |(y, _, _)| ((y - 1960) * 52) as f64 + week(td) - 1.0)
}
pub fn hofd(td: f64) -> f64 {
    ymd(td).map_or(SYSMISS, |(y, m, _)| ((y - 1960) * 2 + if m <= 6 { 0 } else { 1 }) as f64)
}
pub fn dofm(tm: f64) -> f64 {
    if tm >= SYSMISS {
        return SYSMISS;
    }
    let t = tm.floor() as i64;
    mdy((t.rem_euclid(12) + 1) as f64, 1.0, (1960 + t.div_euclid(12)) as f64)
}
pub fn dofq(tq: f64) -> f64 {
    if tq >= SYSMISS {
        return SYSMISS;
    }
    let t = tq.floor() as i64;
    mdy((t.rem_euclid(4) * 3 + 1) as f64, 1.0, (1960 + t.div_euclid(4)) as f64)
}
pub fn dofy(ty: f64) -> f64 {
    if ty >= SYSMISS { SYSMISS } else { mdy(1.0, 1.0, ty.floor()) }
}
pub fn dofw(tw: f64) -> f64 {
    if tw >= SYSMISS {
        return SYSMISS;
    }
    let t = tw.floor() as i64;
    mdy(1.0, 1.0, (1960 + t.div_euclid(52)) as f64) + (t.rem_euclid(52) * 7) as f64
}
pub fn dofh(th: f64) -> f64 {
    if th >= SYSMISS {
        return SYSMISS;
    }
    let t = th.floor() as i64;
    mdy((t.rem_euclid(2) * 6 + 1) as f64, 1.0, (1960 + t.div_euclid(2)) as f64)
}
pub fn ym(y: f64, m: f64) -> f64 {
    (y - 1960.0) * 12.0 + m - 1.0
}
pub fn yq(y: f64, q: f64) -> f64 {
    (y - 1960.0) * 4.0 + q - 1.0
}
pub fn yh(y: f64, h: f64) -> f64 {
    (y - 1960.0) * 2.0 + h - 1.0
}
pub fn yw(y: f64, w: f64) -> f64 {
    (y - 1960.0) * 52.0 + w - 1.0
}

pub const MS_PER_DAY: f64 = 86_400_000.0;

pub fn dofc(tc: f64) -> f64 {
    if tc >= SYSMISS { SYSMISS } else { (tc / MS_PER_DAY).floor() }
}
pub fn cofd(td: f64) -> f64 {
    if td >= SYSMISS { SYSMISS } else { td * MS_PER_DAY }
}
pub fn mdyhms(m: f64, d: f64, y: f64, h: f64, mi: f64, s: f64) -> f64 {
    let td = mdy(m, d, y);
    if td >= SYSMISS { SYSMISS } else { td * MS_PER_DAY + ((h * 60.0 + mi) * 60.0 + s) * 1000.0 }
}
pub fn hh(tc: f64) -> f64 {
    if tc >= SYSMISS { SYSMISS } else { ((tc.rem_euclid(MS_PER_DAY)) / 3_600_000.0).floor() }
}
pub fn mm(tc: f64) -> f64 {
    if tc >= SYSMISS { SYSMISS } else { ((tc.rem_euclid(3_600_000.0)) / 60_000.0).floor() }
}
pub fn ss(tc: f64) -> f64 {
    if tc >= SYSMISS { SYSMISS } else { (tc.rem_euclid(60_000.0)) / 1000.0 }
}

fn month_from_word(w: &str) -> Option<i64> {
    let w = w.to_lowercase();
    if w.len() < 3 {
        return None;
    }
    MONTHS.iter().position(|m| w.starts_with(m)).map(|i| i as i64 + 1)
}

/// Parse a date string according to a mask made of D, M, Y (e.g. "DMY",
/// "YMD", "MDY"). Accepts numeric or named months and any separators.
/// A two-digit year needs a century prefix in the mask, e.g. "DM19Y"; or
/// a `topyear`.
pub fn date(s: &str, mask: &str, topyear: Option<f64>) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return SYSMISS;
    }
    // tokenise into numbers and words
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_digit = None;
    for c in s.chars() {
        let is_d = c.is_ascii_digit();
        let is_a = c.is_alphabetic();
        if !(is_d || is_a) {
            if !cur.is_empty() {
                toks.push(std::mem::take(&mut cur));
            }
            cur_digit = None;
            continue;
        }
        if cur_digit.is_some() && cur_digit != Some(is_d) {
            toks.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        cur_digit = Some(is_d);
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    // compact numeric form, e.g. 20000115 with mask YMD
    if toks.len() == 1 && toks[0].len() == 8 && toks[0].chars().all(|c| c.is_ascii_digit()) {
        let t = &toks[0];
        let order: Vec<char> = mask.chars().filter(|c| "DMY".contains(*c)).collect();
        let mut parts = Vec::new();
        let mut pos = 0;
        for c in &order {
            let w = if *c == 'Y' { 4 } else { 2 };
            parts.push(t[pos..pos + w].to_string());
            pos += w;
        }
        toks = parts;
    }
    let mut century: Option<i64> = None;
    let mut order = Vec::new();
    let mchars: Vec<char> = mask.chars().collect();
    let mut i = 0;
    while i < mchars.len() {
        let c = mchars[i];
        if c.is_ascii_digit() && i + 1 < mchars.len() && mchars[i + 1].is_ascii_digit() {
            century = Some(format!("{}{}", c, mchars[i + 1]).parse().unwrap());
            i += 2;
            continue;
        }
        if "DMY".contains(c) {
            order.push(c);
        }
        i += 1;
    }
    if order.len() != toks.len() {
        return SYSMISS;
    }
    let (mut d, mut m, mut y) = (None, None, None);
    for (c, t) in order.iter().zip(&toks) {
        match c {
            'D' => d = t.parse::<i64>().ok(),
            'M' => m = t.parse::<i64>().ok().or_else(|| month_from_word(t)),
            'Y' => {
                let v = t.parse::<i64>().ok();
                y = v.map(|v| {
                    if t.len() <= 2 {
                        if let Some(cn) = century {
                            cn * 100 + v
                        } else if let Some(top) = topyear {
                            let top = top as i64;
                            let mut yy = (top / 100) * 100 + v;
                            if yy > top {
                                yy -= 100;
                            }
                            yy
                        } else {
                            -1
                        }
                    } else {
                        v
                    }
                })
            }
            _ => {}
        }
    }
    match (d, m, y) {
        (Some(d), Some(m), Some(y)) if y >= 0 => mdy(m as f64, d as f64, y as f64),
        _ => SYSMISS,
    }
}

/// Parse a datetime string ("15jan2020 13:45:10", mask "DMYhms").
pub fn clock(s: &str, mask: &str) -> f64 {
    let (dpart, tpart) = match s.trim().rsplit_once(' ') {
        Some((a, b)) if b.contains(':') => (a, b),
        _ => (s.trim(), ""),
    };
    let dmask: String = mask.chars().filter(|c| "DMY0123456789".contains(*c)).collect();
    let td = date(dpart, &dmask, None);
    if td >= SYSMISS {
        return SYSMISS;
    }
    let mut hms = [0.0f64; 3];
    for (i, p) in tpart.split(':').enumerate().take(3) {
        if let Ok(v) = p.trim().parse::<f64>() {
            hms[i] = v;
        }
    }
    td * MS_PER_DAY + ((hms[0] * 60.0 + hms[1]) * 60.0 + hms[2]) * 1000.0
}

/// Format a value with a date/time format (`%td`, `%tdDD/NN/CCYY`, `%tc`, `%tm`, ...).
pub fn format_date(v: f64, fmt: &str) -> String {
    if v >= SYSMISS {
        return ".".into();
    }
    let f = fmt.trim_start_matches('%').trim_start_matches('-');
    let (kind, custom) = if let Some(rest) = f.strip_prefix("tc").or_else(|| f.strip_prefix("tC")) {
        ('c', rest)
    } else if let Some(rest) = f.strip_prefix('t') {
        let mut ch = rest.chars();
        let k = ch.next().unwrap_or('d');
        (k, ch.as_str())
    } else if let Some(rest) = f.strip_prefix('d') {
        ('d', rest)
    } else {
        ('d', "")
    };
    match kind {
        'd' => {
            if custom.is_empty() {
                let (y, m, d) = ymd(v).unwrap();
                format!("{:02}{}{}", d, MONTHS[(m - 1) as usize], y)
            } else {
                custom_date(v, 0.0, custom)
            }
        }
        'c' => {
            let td = dofc(v);
            if custom.is_empty() {
                let (y, m, d) = ymd(td).unwrap();
                format!(
                    "{:02}{}{} {:02}:{:02}:{:02}",
                    d,
                    MONTHS[(m - 1) as usize],
                    y,
                    hh(v) as i64,
                    mm(v) as i64,
                    ss(v).floor() as i64
                )
            } else {
                custom_date(td, v, custom)
            }
        }
        'm' => {
            let t = v.floor() as i64;
            format!("{}m{}", 1960 + t.div_euclid(12), t.rem_euclid(12) + 1)
        }
        'q' => {
            let t = v.floor() as i64;
            format!("{}q{}", 1960 + t.div_euclid(4), t.rem_euclid(4) + 1)
        }
        'h' => {
            let t = v.floor() as i64;
            format!("{}h{}", 1960 + t.div_euclid(2), t.rem_euclid(2) + 1)
        }
        'w' => {
            let t = v.floor() as i64;
            format!("{}w{}", 1960 + t.div_euclid(52), t.rem_euclid(52) + 1)
        }
        'y' => format!("{}", v.floor() as i64),
        _ => format!("{v}"),
    }
}

fn custom_date(td: f64, tc: f64, spec: &str) -> String {
    let (y, m, d) = match ymd(td) {
        Some(x) => x,
        None => return ".".into(),
    };
    let mut out = String::new();
    let c: Vec<char> = spec.chars().collect();
    let mut i = 0;
    while i < c.len() {
        let two = if i + 1 < c.len() { format!("{}{}", c[i], c[i + 1]) } else { String::new() };
        let (txt, adv) = match two.as_str() {
            "CC" => (format!("{:02}", y / 100), 2),
            "cc" => (format!("{}", y / 100), 2),
            "YY" => (format!("{:02}", y % 100), 2),
            "yy" => (format!("{}", y % 100), 2),
            "NN" => (format!("{:02}", m), 2),
            "nn" => (format!("{}", m), 2),
            "DD" => (format!("{:02}", d), 2),
            "dd" => (format!("{}", d), 2),
            "Mo" => (MONTHS_FULL[(m - 1) as usize][..3].to_string(), 2),
            "HH" => (format!("{:02}", hh(tc) as i64), 2),
            "MM" => (format!("{:02}", mm(tc) as i64), 2),
            "SS" => (format!("{:02}", ss(tc).floor() as i64), 2),
            _ => match c[i] {
                'M' => (MONTHS_FULL[(m - 1) as usize].to_string(), 1),
                'm' => (MONTHS[(m - 1) as usize].to_string(), 1),
                'D' | 'd' => (format!("{d}"), 1),
                'J' | 'j' => (format!("{:03}", doy(td) as i64), 1),
                'W' | 'w' => (format!("{}", week(td) as i64), 1),
                'L' | 'l' => (DAYS[dow(td) as usize].to_string(), 1),
                '_' => (" ".into(), 1),
                '!' if i + 1 < c.len() => (c[i + 1].to_string(), 2),
                ch => (ch.to_string(), 1),
            },
        };
        out.push_str(&txt);
        i += adv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_dates() {
        assert_eq!(mdy(1.0, 1.0, 1960.0), 0.0);
        assert_eq!(mdy(1.0, 1.0, 2000.0), 14610.0);
        assert_eq!(mdy(2.0, 30.0, 2000.0), SYSMISS);
        assert_eq!(date("2000-01-01", "YMD", None), 14610.0);
        assert_eq!(date("15jan2000", "DMY", None), 14624.0);
        assert_eq!(date("1/15/99", "MD19Y", None), mdy(1.0, 15.0, 1999.0));
        assert_eq!(dow(0.0), 5.0);
        assert_eq!(format_date(14610.0, "%td"), "01jan2000");
        assert_eq!(format_date(14610.0, "%tdCCYY-NN-DD"), "2000-01-01");
        assert_eq!(format_date(ym(2001.0, 3.0), "%tm"), "2001m3");
        assert_eq!(dofm(ym(2001.0, 3.0)), mdy(3.0, 1.0, 2001.0));
    }
}
