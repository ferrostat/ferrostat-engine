//! .dta file format: reading, writing and round-tripping through `use`/`save`.

use ferrostat_core::dta::{DtaReader, DtaWriter};
use ferrostat_core::missing::{
    encode_missing_byte, encode_missing_double, encode_missing_float, encode_missing_int,
    encode_missing_long, is_missing_byte, is_missing_double, is_missing_float, is_missing_int,
    is_missing_long,
};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_tests::*;

fn roundtrip(ds: &DataSet) -> DataSet {
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, ds).expect("write dta");
    DtaReader::from_bytes(buf).expect("read dta back")
}

#[test]
fn writer_emits_format_118_header() {
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, &fixture()).unwrap();
    let head = String::from_utf8_lossy(&buf[..80]);
    assert!(head.starts_with("<stata_dta><header><release>118</release>"), "{head}");
    assert!(buf.ends_with(b"</stata_dta>"));
}

#[test]
fn roundtrip_preserves_shape_names_and_order() {
    let ds = roundtrip(&fixture());
    assert_eq!(ds.n_obs, 12);
    assert_eq!(ds.varlist(), ["id", "name", "group", "x", "y", "z", "d", "cnt"]);
}

#[test]
fn roundtrip_preserves_storage_types() {
    let ds = roundtrip(&fixture());
    let ty = |n: &str| ds.get_var(n).unwrap().var_type.clone();
    assert_eq!(ty("id"), VarType::Int);
    assert_eq!(ty("group"), VarType::Byte);
    assert_eq!(ty("x"), VarType::Double);
    assert_eq!(ty("name"), VarType::Str(4));
}

#[test]
fn roundtrip_preserves_values() {
    let ds = roundtrip(&fixture());
    let y: Vec<f64> = ds.get_var("y").unwrap().data.iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(y, Y);
    let names: Vec<&str> = ds.get_var("name").unwrap().data.iter().map(|v| v.as_string().unwrap()).collect();
    assert_eq!(names, NAMES);
}

#[test]
#[ignore = "bug: value-label tables are lost when a file is written and read back"]
fn roundtrip_preserves_labels() {
    let ds = roundtrip(&fixture());
    assert_eq!(ds.datalabel, "Ferrostat test fixture");
    let d = ds.get_var("d").unwrap();
    assert_eq!(d.label, "Binary outcome");
    assert_eq!(d.value_label_name, "yesno");
    let vl = ds.value_labels.get("yesno").expect("value label table survives");
    assert_eq!(vl.lookup(0.0), Some("no"));
    assert_eq!(vl.lookup(1.0), Some("yes"));
}

#[test]
#[ignore = "bug: missing values in byte/int/long variables are written as 0 instead of the missing code"]
fn roundtrip_system_missing_in_every_numeric_type() {
    let mut ds = DataSet::new();
    for (name, ty) in [("b", VarType::Byte), ("i", VarType::Int), ("l", VarType::Long), ("f", VarType::Float), ("d", VarType::Double)] {
        let mut v = Variable::new(name, ty);
        v.data = vec![Value::Double(1.0), Value::Double(f64::NAN)];
        ds.add_variable(v).unwrap();
    }
    ds.n_obs = 2;
    let back = roundtrip(&ds);
    for name in ["b", "i", "l", "f", "d"] {
        let v = &back.get_var(name).unwrap().data;
        assert_eq!(v[0].as_f64(), Some(1.0), "{name}[1]");
        assert!(v[1].as_f64().map_or(true, f64::is_nan), "{name}[2] should be missing");
    }
}

#[test]
#[ignore = "missing: extended missing values (.a-.z) are collapsed to system missing on read"]
fn roundtrip_extended_missing_values() {
    let mut ds = DataSet::new();
    let mut v = Variable::new("v", VarType::Double);
    v.data = vec![Value::Double(encode_missing_double(Some(0))), Value::Double(encode_missing_double(Some(25)))];
    ds.add_variable(v).unwrap();
    ds.n_obs = 2;
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, &ds).unwrap();
    let back = DtaReader::from_bytes(buf).unwrap();
    // .a and .z must remain distinguishable from each other and from `.`
    let a = back.get_var("v").unwrap().data[0].clone();
    let z = back.get_var("v").unwrap().data[1].clone();
    assert_ne!(format!("{a:?}"), format!("{z:?}"));
}

#[test]
#[ignore = "bug: missing-value codes differ from the .dta spec (byte . is 101, int . is 32741, long . is 2147483621); float/double detection requires NaN"]
fn missing_value_encodings_follow_dta_spec() {
    // Values from the published .dta format description.
    assert_eq!(encode_missing_byte(None), 101);
    assert_eq!(encode_missing_int(None), 32741);
    assert_eq!(encode_missing_long(None), 2147483621);
    assert_eq!(encode_missing_byte(Some(0)), 102); // .a
    assert_eq!(encode_missing_byte(Some(25)), 127); // .z
    assert!(is_missing_byte(101) && !is_missing_byte(100));
    assert!(is_missing_int(32741) && !is_missing_int(32740));
    assert!(is_missing_long(2147483621) && !is_missing_long(2147483620));
    assert!(is_missing_float(encode_missing_float(None)));
    assert!(is_missing_double(encode_missing_double(Some(25))));
    assert!(!is_missing_double(1.0e300));
    assert!(!is_missing_float(1.0e38));
}

#[test]
fn roundtrip_long_strings_and_strl() {
    let mut ds = DataSet::new();
    let long = "x".repeat(2045);
    let mut v = Variable::new("s", VarType::Str(2045));
    v.data = vec![Value::Str(long.clone())];
    ds.add_variable(v).unwrap();
    ds.n_obs = 1;
    let back = roundtrip(&ds);
    assert_eq!(back.get_var("s").unwrap().data[0].as_string(), Some(long.as_str()));
}

#[test]
fn roundtrip_utf8_strings() {
    let mut ds = DataSet::new();
    let mut v = Variable::new("s", VarType::Str(12));
    v.data = vec![Value::Str("città".into()), Value::Str("Größe".into())];
    ds.add_variable(v).unwrap();
    ds.n_obs = 2;
    let back = roundtrip(&ds);
    let s: Vec<&str> = back.get_var("s").unwrap().data.iter().map(|v| v.as_string().unwrap()).collect();
    assert_eq!(s, ["città", "Größe"]);
}

#[test]
fn roundtrip_empty_dataset_with_variables() {
    let mut ds = DataSet::new();
    ds.add_variable(Variable::new("a", VarType::Double)).unwrap();
    let back = roundtrip(&ds);
    assert_eq!(back.n_obs, 0);
    assert_eq!(back.varlist(), ["a"]);
}

#[test]
fn rejects_garbage() {
    assert!(DtaReader::from_bytes(b"not a dta file".to_vec()).is_err());
    assert!(DtaReader::from_bytes(Vec::new()).is_err());
}

#[test]
fn rejects_unsupported_release() {
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, &fixture()).unwrap();
    let s = String::from_utf8_lossy(&buf).replacen("<release>118</release>", "<release>999</release>", 1);
    let mut patched = buf.clone();
    patched[..s.len().min(200)].copy_from_slice(&s.as_bytes()[..s.len().min(200)]);
    assert!(DtaReader::from_bytes(patched).is_err());
}

#[test]
fn reads_format_119_header() {
    // 119 differs from 118 only in the width of K; a file written as 118 and
    // relabelled must still be rejected or read consistently, never panic.
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, &fixture()).unwrap();
    let pos = buf.windows(3).position(|w| w == b"118").unwrap();
    buf[pos..pos + 3].copy_from_slice(b"119");
    let _ = DtaReader::from_bytes(buf);
}

#[test]
#[ignore = "missing: reading pre-117 binary formats (113-115, older DTA variants)"]
fn reads_format_115() {
    // Minimal 115 header: ds_format, byteorder (LOHI), filetype, unused, nvar, nobs
    let mut buf = vec![115u8, 2, 1, 0];
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 81 + 18]);
    let ds = DtaReader::from_bytes(buf).expect("format 115 should be readable");
    assert_eq!(ds.n_obs, 0);
}

// ---- through the command language ---------------------------------------

#[test]
fn save_and_use() {
    let dir = temp_dir("save_and_use");
    let path = dir.join("fx.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("save {}", path.display()));
    s.run("clear");
    assert_eq!(s.n(), 0);
    s.run(&format!("use {}", path.display()));
    assert_eq!(s.n(), 12);
    assert_eq!(s.nums("y"), Y);
}

#[test]
fn use_with_quoted_path_containing_spaces() {
    let dir = temp_dir("quoted path");
    let path = dir.join("my data.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("save \"{}\"", path.display()));
    s.run("clear");
    s.run(&format!("use \"{}\"", path.display()));
    assert_eq!(s.n(), 12);
}

#[test]
fn use_missing_file_fails() {
    let mut s = Session::new();
    s.fails("use /nonexistent/nothing.dta");
}

#[test]
#[ignore = "missing: `save` refuses to overwrite without explicit replace flag"]
fn save_requires_replace_to_overwrite() {
    let dir = temp_dir("save_replace");
    let path = dir.join("a.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("save {}", path.display()));
    s.fails(&format!("save {}", path.display()));
    s.run(&format!("save {}, replace", path.display()));
}

#[test]
#[ignore = "missing: `use` refuses to discard changed data without `clear`"]
fn use_requires_clear_when_data_in_memory() {
    let dir = temp_dir("use_clear");
    let path = dir.join("a.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("save {}", path.display()));
    s.run("generate w = 1");
    s.fails(&format!("use {}", path.display()));
    s.run(&format!("use {}, clear", path.display()));
}

#[test]
#[ignore = "missing: `use varlist if using`"]
fn use_subset_of_variables_and_observations() {
    let dir = temp_dir("use_subset");
    let path = dir.join("a.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("save {}", path.display()));
    s.run(&format!("use x y if group == 2 using {}, clear", path.display()));
    assert_eq!(s.varnames(), ["x", "y"]);
    assert_eq!(s.n(), 4);
}

#[test]
#[ignore = "missing: `saveold` / `save, version()` to write formats 117 and 119"]
fn saveold_writes_format_117() {
    let dir = temp_dir("saveold");
    let path = dir.join("old.dta");
    let mut s = Session::with_fixture();
    s.run(&format!("saveold {}, version(13)", path.display()));
    let bytes = std::fs::read(&path).unwrap();
    assert!(String::from_utf8_lossy(&bytes[..60]).contains("<release>117</release>"));
}

#[test]
fn sysuse_unknown_dataset_fails() {
    let mut s = Session::new();
    s.fails("sysuse definitely_not_a_dataset");
}
