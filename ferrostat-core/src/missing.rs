/// Ferrostat missing values are encoded as special bit patterns.
///
/// For each numeric type there are 27 missing values:
///   .   (system missing)
///   .a  through .z
///
/// Byte missing:   0x64 + i  (i=0..26)  => 100..126 (int8: 0x64..0x7E)
/// Int missing:    0x7E00 + i => 32256..32282
/// Long missing:   0x7FFFFFE0 + i
/// Float missing:  0x7F000000 + i * 0x00800000  (bit 23..29 pattern)
/// Double missing: 0x7FE0000000000000 + i * 0x0020000000000000

/// Is a byte value a Ferrostat missing?
pub fn is_missing_byte(v: i8) -> bool {
    v >= 101 // 0x65 = .a, up to 126 = .z; note: 100 = . system missing
}

/// Is an i16 value a Ferrostat missing?
pub fn is_missing_int(v: i16) -> bool {
    (v as u16) >= 0x7E00 + 1 // .a starts at 0x7e01
}

/// Is an i32 value a Ferrostat missing?
pub fn is_missing_long(v: i32) -> bool {
    (v as u32) >= 0x7FFFFFE0 + 1
}

/// Is an f32 value a Ferrostat missing?
pub fn is_missing_float(v: f32) -> bool {
    if v.is_nan() {
        let bits = v.to_bits();
        // Ferrostat float missing values have mantissa = 0 in the top bits.
        // System missing: 0x7f000000
        // .a..z: 0x7f000000 + i * 0x00800000 for i=1..26
        (bits & 0x007FFFFF) == 0 && (bits >> 23) == 0xFE // exponent all 1s
    } else {
        false
    }
}

/// Is an f64 value a Ferrostat missing?
pub fn is_missing_double(v: f64) -> bool {
    if v.is_nan() {
        let bits = v.to_bits();
        (bits & 0x000FFFFFFFFFFFFF) == 0 && (bits >> 52) == 0x7FF
    } else {
        false
    }
}

/// Encode a missing value suffix (None for system missing, Some(0) for .a, etc.)
/// into the raw bit patterns.
pub fn encode_missing_byte(suffix: Option<u8>) -> i8 {
    let idx = suffix.map(|c| c + 1).unwrap_or(0);
    0x64 + idx as i8
}

pub fn encode_missing_int(suffix: Option<u8>) -> i16 {
    let idx = suffix.map(|c| c + 1).unwrap_or(0);
    0x7E00 + idx as i16
}

pub fn encode_missing_long(suffix: Option<u8>) -> i32 {
    let idx = suffix.map(|c| c + 1).unwrap_or(0);
    0x7FFFFFE0 + idx as i32
}

pub fn encode_missing_float(suffix: Option<u8>) -> f32 {
    let idx = suffix.map(|c| c + 1).unwrap_or(0);
    f32::from_bits(0x7F000000 + (idx as u32) * 0x00800000)
}

pub fn encode_missing_double(suffix: Option<u8>) -> f64 {
    let idx = suffix.map(|c| c + 1).unwrap_or(0);
    f64::from_bits(0x7FE0000000000000 + (idx as u64) * 0x0020000000000000)
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
