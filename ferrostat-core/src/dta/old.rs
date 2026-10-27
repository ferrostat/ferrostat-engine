//! Reading the early binary .dta releases 102 through 112.
//!
//! These share the binary layout of 113-115 but differ in field widths:
//!
//! | release  | names | formats | data label | var labels | time stamp | type codes |
//! |----------|-------|---------|------------|------------|------------|------------|
//! | 102-104  | 9     | 7       | 32         | 32         | none       | characters |
//! | 105      | 9     | 12      | 32         | 32         | 18         | characters |
//! | 108      | 9     | 12      | 81         | 81         | 18         | characters |
//! | 110      | 33    | 12      | 81         | 81         | 18         | characters |
//! | 111-112  | 33    | 12      | 81         | 81         | 18         | 251..255   |
//!
//! Character type codes are `b`, `i`, `l`, `f`, `d` for the numeric types
//! and `0x7f + n` for `str n`. Missing values predate the extended missing
//! codes: the largest value of each integer type is system missing.
//! Value labels before release 106 use a compact table of 8-byte labels.

use super::{bad, decode_str, read_label_table, Cursor, Result};
use crate::dataset::DataSet;
use crate::missing::{self, SYSMISS};
use crate::value_label::ValueLabel;
use crate::var::{Value, VarType, Variable};

pub(super) fn is_old_release(b: u8) -> bool {
    matches!(b, 102..=112)
}

struct OldLayout {
    name_w: usize,
    fmt_w: usize,
    datalbl_w: usize,
    varlbl_w: usize,
    timestamp: bool,
    char_types: bool,
    /// Expansion fields: None (absent), Some(2) or Some(4) bytes of length.
    exp_len: Option<usize>,
    old_labels: bool,
}

fn layout(rel: u8) -> OldLayout {
    OldLayout {
        name_w: if rel >= 110 { 33 } else { 9 },
        fmt_w: if rel >= 105 { 12 } else { 7 },
        datalbl_w: if rel > 105 { 81 } else { 32 },
        varlbl_w: if rel > 105 { 81 } else { 32 },
        timestamp: rel > 104,
        char_types: rel <= 110,
        exp_len: if rel <= 104 {
            None
        } else if rel <= 108 {
            Some(2)
        } else {
            Some(4)
        },
        old_labels: rel < 106,
    }
}

fn type_of(code: u8, chars: bool) -> Result<VarType> {
    if chars {
        return Ok(match code {
            b'b' => VarType::Byte,
            b'i' => VarType::Int,
            b'l' => VarType::Long,
            b'f' => VarType::Float,
            b'd' => VarType::Double,
            0x80..=0xff => VarType::Str((code - 0x7f) as usize),
            _ => return Err(bad(format!("unknown type code {code}"))),
        });
    }
    super::old_type(code)
}

/// One numeric cell under the pre-extended-missing rules.
fn read_old_num(c: &mut Cursor, ty: VarType) -> Result<f64> {
    Ok(match ty {
        VarType::Byte => {
            let v = c.u8()? as i8;
            if v == i8::MAX { SYSMISS } else { v as f64 }
        }
        VarType::Int => {
            let v = c.i16()?;
            if v == i16::MAX { SYSMISS } else { v as f64 }
        }
        VarType::Long => {
            let v = c.i32()?;
            if v == i32::MAX { SYSMISS } else { v as f64 }
        }
        VarType::Float => {
            let v = c.f32()?;
            if v.is_nan() || v as f64 >= 1.7014118346046923e38 { SYSMISS } else { v as f64 }
        }
        _ => {
            let v = c.f64()?;
            if v.is_nan() || v >= SYSMISS { SYSMISS } else { v }
        }
    })
}

/// Storage type able to hold every value of a column (old integer types
/// allow values that are missing codes in the current encoding).
fn fit_type(ty: VarType, vals: &[f64]) -> VarType {
    let max = vals.iter().copied().filter(|v| !missing::is_missing(*v)).fold(f64::NEG_INFINITY, f64::max);
    match ty {
        VarType::Byte if max > missing::MAX_BYTE as f64 => fit_type(VarType::Int, vals),
        VarType::Int if max > missing::MAX_INT as f64 => fit_type(VarType::Long, vals),
        VarType::Long if max > missing::MAX_LONG as f64 => VarType::Double,
        t => t,
    }
}

fn to_value(ty: VarType, v: f64) -> Value {
    if missing::is_missing(v) {
        return Value::Double(SYSMISS);
    }
    match ty {
        VarType::Byte => Value::Byte(v as i8),
        VarType::Int => Value::Int(v as i16),
        VarType::Long => Value::Long(v as i32),
        VarType::Float => Value::Float(v as f32),
        _ => Value::Double(v),
    }
}

/// Value labels of releases before 106: a 2-byte table length, the label
/// name, then 2-byte values followed by 8-byte texts.
fn read_old_labels(c: &mut Cursor, name_w: usize, ds: &mut DataSet) -> Result<()> {
    while c.pos + 2 + name_w + 1 <= c.buf.len() {
        let len = c.u16()? as usize;
        let name = c.fixed_str(name_w, false)?;
        c.take(1)?;
        let n = len / 10;
        let vals: Vec<i16> = (0..n).map(|_| c.i16()).collect::<Result<_>>()?;
        let mut vl = ValueLabel::new(name.clone());
        for v in vals {
            vl.add(v as f64, decode_str(c.take(8)?, false));
        }
        ds.value_labels.insert(name, vl);
    }
    Ok(())
}

pub(super) fn read_old(data: &[u8]) -> Result<DataSet> {
    let mut c = Cursor::new(data);
    let rel = c.u8()?;
    let lay = layout(rel);
    c.big = match c.u8()? {
        1 => true,
        2 => false,
        _ => return Err(bad("bad byteorder")),
    };
    c.take(2)?; // filetype, unused
    let k = c.u16()? as usize;
    let n = c.u32()? as usize;
    let datalabel = c.fixed_str(lay.datalbl_w, false)?;
    let timestamp = if lay.timestamp { c.fixed_str(18, false)? } else { String::new() };
    let types: Vec<VarType> = (0..k).map(|_| c.u8().and_then(|t| type_of(t, lay.char_types))).collect::<Result<_>>()?;
    let names: Vec<String> = (0..k).map(|_| c.fixed_str(lay.name_w, false)).collect::<Result<_>>()?;
    // the list ends at the first zero; bytes after it are unused
    let mut sortlist = Vec::new();
    let mut ended = false;
    for _ in 0..=k {
        let v = c.u16()? as usize;
        ended |= v == 0;
        if !ended && v <= k && sortlist.len() < k {
            sortlist.push(v);
        }
    }
    let formats: Vec<String> = (0..k).map(|_| c.fixed_str(lay.fmt_w, false)).collect::<Result<_>>()?;
    let lblnames: Vec<String> = (0..k).map(|_| c.fixed_str(lay.name_w, false)).collect::<Result<_>>()?;
    let varlabels: Vec<String> = (0..k).map(|_| c.fixed_str(lay.varlbl_w, false)).collect::<Result<_>>()?;
    if let Some(w) = lay.exp_len {
        loop {
            let t = c.u8()?;
            let len = if w == 2 { c.u16()? as usize } else { c.u32()? as usize };
            if t == 0 && len == 0 {
                break;
            }
            c.take(len)?;
        }
    }
    let mut strs: Vec<Vec<Value>> = vec![Vec::new(); k];
    let mut nums: Vec<Vec<f64>> = vec![Vec::new(); k];
    for _ in 0..n {
        for (vi, ty) in types.iter().enumerate() {
            match ty {
                VarType::Str(w) => strs[vi].push(Value::Str(c.fixed_str(*w, false)?)),
                t => nums[vi].push(read_old_num(&mut c, *t)?),
            }
        }
    }
    let mut ds = DataSet::new();
    ds.datalabel = datalabel;
    ds.time_stamp = timestamp;
    for i in 0..k {
        let ty = if types[i].is_string() { types[i] } else { fit_type(types[i], &nums[i]) };
        let mut v = Variable::new(names[i].clone(), ty);
        v.format = formats[i].clone();
        v.value_label_name = lblnames[i].clone();
        v.label = varlabels[i].clone();
        v.data = if ty.is_string() {
            std::mem::take(&mut strs[i])
        } else {
            nums[i].iter().map(|&x| to_value(ty, x)).collect()
        };
        ds.add_variable(v).map_err(bad)?;
    }
    ds.n_obs = n;
    ds.sort_by = sortlist.iter().filter_map(|&i| names.get(i - 1).cloned()).collect();
    // value labels are optional trailing data; a damaged table does not
    // prevent the data from loading
    if lay.old_labels {
        let _ = read_old_labels(&mut c, lay.name_w, &mut ds);
    } else {
        while c.pos + 4 <= data.len() {
            let Ok(len) = c.u32() else { break };
            let Ok(name) = c.fixed_str(lay.name_w, false) else { break };
            if c.take(3).is_err() {
                break;
            }
            let Ok(table) = c.take(len as usize) else { break };
            let mut tc = Cursor { buf: table, pos: 0, big: c.big };
            match read_label_table(&mut tc, &name, false) {
                Ok(vl) => {
                    ds.value_labels.insert(name, vl);
                }
                Err(_) => break,
            }
        }
    }
    Ok(ds)
}
