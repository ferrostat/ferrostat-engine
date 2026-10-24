//! Missing values.
//!
//! Every numeric type has 27 missing values: system missing `.` and the
//! extended missing values `.a` .. `.z`. They sort above every non-missing
//! number and in the order `. < .a < ... < .z`.
//!
//! In memory, all numeric values are handled as `f64` and missing values use
//! the same encoding as the on-disk `double` type: `.` is 2^1023 and each
//! extended value adds 2^40 to the bit pattern. Because these are the largest
//! finite doubles, ordinary comparisons and sorting already give the right
//! answer. NaN is also accepted as system missing for robustness.
//!
//! On-disk codes per storage type (from the published .dta format
//! description), with `i` = 0 for `.` and 1..=26 for `.a`..`.z`:
//!
//! | type   | `.`          | `.a`..`.z`            |
//! |--------|--------------|-----------------------|
//! | byte   | 101          | 102..127              |
//! | int    | 32741        | 32742..32767          |
//! | long   | 2147483621   | 2147483622..2147483647|
//! | float  | 0x7f000000   | + i * 0x800           |
//! | double | 0x7fe0000000000000 | + i * 0x10000000000 |

const DOUBLE_BASE: u64 = 0x7FE0_0000_0000_0000;
const DOUBLE_STEP: u64 = 0x0000_0100_0000_0000;
const FLOAT_BASE: u32 = 0x7F00_0000;
const FLOAT_STEP: u32 = 0x800;

/// System missing value `.`.
pub const SYSMISS: f64 = f64::from_bits(DOUBLE_BASE);

/// Largest non-missing values per storage type.
pub const MAX_BYTE: i8 = 100;
pub const MAX_INT: i16 = 32740;
pub const MAX_LONG: i32 = 2147483620;

/// The missing value with the given code (0 = `.`, 1 = `.a`, ..., 26 = `.z`).
pub fn missing(code: u8) -> f64 {
    f64::from_bits(DOUBLE_BASE + code.min(26) as u64 * DOUBLE_STEP)
}

/// Extended missing value for a letter `a`..`z`.
pub fn missing_letter(c: char) -> Option<f64> {
    c.is_ascii_lowercase().then(|| missing(c as u8 - b'a' + 1))
}

/// Is this in-memory numeric value missing?
#[inline]
pub fn is_missing(v: f64) -> bool {
    v.is_nan() || v >= SYSMISS
}

/// Missing code of a value (0 for `.`), or None if not missing.
pub fn missing_code(v: f64) -> Option<u8> {
    if v.is_nan() {
        return Some(0);
    }
    if v < SYSMISS {
        return None;
    }
    let off = (v.to_bits() - DOUBLE_BASE) / DOUBLE_STEP;
    Some(off.min(26) as u8)
}

/// Normalise a value: NaN becomes `.`, everything else unchanged.
#[inline]
pub fn normalize(v: f64) -> f64 {
    if v.is_nan() { SYSMISS } else { v }
}

/// Text representation: ".", ".a", ...
pub fn missing_name(code: u8) -> String {
    if code == 0 { ".".into() } else { format!(".{}", (b'a' + code - 1) as char) }
}

// ---- per-type encodings (`suffix`: None = `.`, Some(0) = `.a`, ..) ----------

fn idx(suffix: Option<u8>) -> u8 {
    suffix.map(|c| c.min(25) + 1).unwrap_or(0)
}

pub fn encode_missing_byte(suffix: Option<u8>) -> i8 {
    (101 + idx(suffix) as i16) as i8
}

pub fn encode_missing_int(suffix: Option<u8>) -> i16 {
    32741 + idx(suffix) as i16
}

pub fn encode_missing_long(suffix: Option<u8>) -> i32 {
    2147483621 + idx(suffix) as i32
}

pub fn encode_missing_float(suffix: Option<u8>) -> f32 {
    f32::from_bits(FLOAT_BASE + idx(suffix) as u32 * FLOAT_STEP)
}

pub fn encode_missing_double(suffix: Option<u8>) -> f64 {
    missing(idx(suffix))
}

pub fn is_missing_byte(v: i8) -> bool {
    v > MAX_BYTE
}

pub fn is_missing_int(v: i16) -> bool {
    v > MAX_INT
}

pub fn is_missing_long(v: i32) -> bool {
    v > MAX_LONG
}

pub fn is_missing_float(v: f32) -> bool {
    v.is_nan() || (v.is_sign_positive() && v.to_bits() >= FLOAT_BASE)
}

pub fn is_missing_double(v: f64) -> bool {
    is_missing(v)
}

/// Missing code (0..=26) of on-disk values.
pub fn code_byte(v: i8) -> u8 {
    (v as i16 - 101).clamp(0, 26) as u8
}
pub fn code_int(v: i16) -> u8 {
    (v as i32 - 32741).clamp(0, 26) as u8
}
pub fn code_long(v: i32) -> u8 {
    (v as i64 - 2147483621).clamp(0, 26) as u8
}
pub fn code_float(v: f32) -> u8 {
    if v.is_nan() { 0 } else { ((v.to_bits() - FLOAT_BASE) / FLOAT_STEP).min(26) as u8 }
}

/// A typed missing value enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissingValue {
    Byte(i8),
    Int(i16),
    Long(i32),
    Float(f32),
    Double(f64),
}

impl MissingValue {
    pub fn is_missing(&self) -> bool {
        match self {
            MissingValue::Byte(v) => is_missing_byte(*v),
            MissingValue::Int(v) => is_missing_int(*v),
            MissingValue::Long(v) => is_missing_long(*v),
            MissingValue::Float(v) => is_missing_float(*v),
            MissingValue::Double(v) => is_missing_double(*v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering() {
        assert!(SYSMISS > 1e300);
        assert!(missing(1) > SYSMISS);
        assert!(missing(26) > missing(25));
        assert_eq!(missing_code(missing(3)), Some(3));
        assert_eq!(missing_code(f64::NAN), Some(0));
        assert_eq!(missing_code(5.0), None);
        assert_eq!(missing_name(1), ".a");
    }

    #[test]
    fn disk_codes() {
        assert_eq!(code_byte(encode_missing_byte(Some(25))), 26);
        assert_eq!(code_int(encode_missing_int(None)), 0);
        assert_eq!(code_long(encode_missing_long(Some(0))), 1);
        assert_eq!(code_float(encode_missing_float(Some(1))), 2);
    }
}
