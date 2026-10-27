//! Columnar and record formats: `import/export parquet`, `import/export arrow`
//! (Arrow IPC file and stream, Feather v2), `import/export json` (array of
//! records and JSON Lines), and `use`/`save` by file extension.

use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::types::Int32Type;
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use ferrostat_core::missing::missing_letter;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_tests::*;

fn p(dir: &Path, name: &str) -> String {
    dir.join(name).display().to_string()
}

/// A dataset with every storage type, missing values, a date, a date-time,
/// value labels, a variable label, a data label and a long string.
fn rich() -> DataSet {
    let mut ds = fixture();
    let mut add = |name: &str, ty: VarType, fmt: &str, data: Vec<Value>| {
        let mut v = Variable::new(name, ty);
        if !fmt.is_empty() {
            v.format = fmt.into();
        }
        v.data = data;
        ds.add_variable(v).unwrap();
    };
    add("big", VarType::Long, "%12.0g", (0..12).map(|i| Value::Long((-2_000_000_000i64 + i as i64 * 300_000_000) as i32)).collect());
    add("f", VarType::Float, "%9.3f", (0..12).map(|i| if i == 3 { Value::Double(ferrostat_core::SYSMISS) } else { Value::Float(i as f32 / 3.0) }).collect());
    add("when", VarType::Long, "%td", (0..12).map(|i| Value::Long(-400 + i * 1000)).collect());
    add("stamp", VarType::Double, "%tc", (0..12).map(|i| Value::Double(1_900_000_000_000.0 + i as f64 * 1234.0)).collect());
    add("ext", VarType::Double, "", (0..12).map(|i| Value::Double(if i == 2 { missing_letter('b').unwrap() } else { i as f64 * 0.1 })).collect());
    add("long_text", VarType::StrL, "%9s", (0..12).map(|i| Value::Str(if i == 0 { "é".repeat(1500) } else { format!("row {i}") })).collect());
    ds.get_var_mut("name").unwrap().data[5] = Value::Str(String::new());
    ds
}

/// Same variables, types, formats, labels and values (extended missing
/// values compare equal to `.`).
fn assert_same(a: &DataSet, b: &DataSet) {
    assert_eq!(a.varlist(), b.varlist());
    assert_eq!(a.n_obs, b.n_obs);
    assert_eq!(a.datalabel, b.datalabel);
    for (va, vb) in a.vars.iter().zip(&b.vars) {
        assert_eq!(va.var_type, vb.var_type, "type of {}", va.name);
        assert_eq!(va.format, vb.format, "format of {}", va.name);
        assert_eq!(va.label, vb.label, "label of {}", va.name);
        assert_eq!(va.value_label_name, vb.value_label_name, "value label of {}", va.name);
        for i in 0..a.n_obs {
            if va.var_type.is_string() {
                assert_eq!(va.str_at(i), vb.str_at(i), "{}[{}]", va.name, i + 1);
            } else {
                let (x, y) = (va.num(i), vb.num(i));
                assert!(x == y || (ferrostat_core::is_missing(x) && ferrostat_core::is_missing(y)), "{}[{}]: {x} vs {y}", va.name, i + 1);
            }
        }
    }
    for (name, vl) in &a.value_labels {
        let other = b.value_labels.get(name).unwrap_or_else(|| panic!("value label {name} lost"));
        assert_eq!(vl.iter().collect::<Vec<_>>(), other.iter().collect::<Vec<_>>());
    }
}

#[test]
fn parquet_round_trip_keeps_types_labels_dates_and_missing() {
    let dir = temp_dir("t22_pq_rt");
    let file = p(&dir, "rich.parquet");
    let mut s = Session::new();
    s.load(rich());
    let out = s.run(&format!("export parquet using \"{file}\""));
    assert!(out.contains("ext: 1 extended missing value(s) (.a-.z) written as null"), "{out}");
    assert!(out.contains("(14 vars, 12 obs)"), "{out}");
    // the file exists now
    s.fails(&format!("export parquet using \"{file}\""));
    let out = s.run(&format!("import parquet using \"{file}\", clear"));
    assert!(out.contains("(14 vars, 12 obs)"), "{out}");
    assert_same(&rich(), s.ds());
    assert_eq!(s.ds().get_var("long_text").unwrap().var_type, VarType::StrL);
    assert_eq!(s.strs("long_text")[0].chars().count(), 1500);
    assert_eq!(s.strs("name")[5], "");
    assert!(s.run("list d in 3").contains("yes"));
    // every codec
    for c in ["snappy", "gzip", "zstd", "lz4", "none"] {
        s.load(rich());
        s.run(&format!("export parquet using \"{file}\", replace compression({c})"));
        s.run(&format!("import parquet using \"{file}\", clear"));
        assert_same(&rich(), s.ds());
    }
    s.fails(&format!("export parquet using \"{file}\", replace compression(brotli)"));
}

#[test]
fn parquet_options_columns_rowrange_if_in_varlist() {
    let dir = temp_dir("t22_pq_opts");
    let file = p(&dir, "a.parquet");
    let mut s = Session::with_fixture();
    s.run(&format!("export parquet id name y using \"{file}\" if group != 2"));
    s.run(&format!("import parquet using \"{file}\", clear"));
    assert_eq!(s.varnames(), ["id", "name", "y"]);
    assert_eq!(s.nums("id"), [1., 2., 3., 4., 9., 10., 11., 12.]);
    s.run(&format!("import parquet using \"{file}\", clear columns(y id) rowrange(2:4)"));
    assert_eq!(s.varnames(), ["id", "y"]);
    assert_eq!(s.nums("id"), [2., 3., 4.]);
    assert_eq!(s.nums("y"), [3.9, 6.2, 7.8]);
    s.run(&format!("import parquet using \"{file}\", clear rowrange(7:)"));
    assert_eq!(s.nums("id"), [11., 12.]);
    s.fails(&format!("import parquet using \"{file}\", clear columns(nosuch)"));
    // varlist written after the file name, `in` range, no extension
    let stem = p(&dir, "b");
    s.load(fixture());
    s.run(&format!("export parquet using {stem}.parquet x name in 1/3"));
    s.run(&format!("import parquet using {stem}, clear"));
    assert_eq!(s.varnames(), ["x", "name"]);
    assert_eq!(s.strs("name"), ["ant", "bee", "cat"]);
    // data in memory has changed: clear required
    s.run("replace x = 0");
    s.fails(&format!("import parquet using {stem}.parquet"));
    s.fails(&format!("import parquet using \"{}\", clear", p(&dir, "missing.parquet")));
}

#[test]
fn arrow_ipc_round_trip_and_compression() {
    let dir = temp_dir("t22_arrow_rt");
    let mut s = Session::new();
    for (name, c) in [("a.arrow", "none"), ("b.feather", "lz4"), ("c.arrow", "zstd")] {
        let file = p(&dir, name);
        s.load(rich());
        let out = s.run(&format!("export arrow using \"{file}\", compression({c})"));
        assert!(out.contains("saved"), "{out}");
        let out = s.run(&format!("import arrow using \"{file}\", clear"));
        assert!(out.contains("(14 vars, 12 obs)"), "{out}");
        assert_same(&rich(), s.ds());
    }
    let file = p(&dir, "a.arrow");
    s.run(&format!("import arrow using \"{file}\", clear columns(stamp id) rowrange(11:12)"));
    assert_eq!(s.varnames(), ["id", "stamp"]);
    assert_eq!(s.nums("id"), [11., 12.]);
    assert_eq!(s.ds().get_var("stamp").unwrap().format, "%tc");
}

#[test]
fn use_and_save_by_extension() {
    let dir = temp_dir("t22_use_save");
    let mut s = Session::new();
    for ext in ["parquet", "arrow", "feather"] {
        let file = p(&dir, &format!("d.{ext}"));
        s.load(rich());
        assert!(s.run(&format!("save \"{file}\"")).contains("saved"));
        s.run(&format!("use \"{file}\", clear"));
        assert_same(&rich(), s.ds());
    }
    s.run(&format!("use id name using \"{}\" if id > 10, clear", p(&dir, "d.parquet")));
    assert_eq!(s.varnames(), ["id", "name"]);
    assert_eq!(s.strs("name"), ["koi", "lynx"]);
}

/// Parquet written with types the exporter never produces.
fn foreign_batch() -> RecordBatch {
    let mut dict = StringDictionaryBuilder::<Int32Type>::new();
    for w in ["low", "high", "low", "mid"] {
        dict.append_value(w);
    }
    dict.append_null();
    let dict = dict.finish();
    let fields = vec![
        Field::new("id64", DataType::Int64, true),
        Field::new("u8", DataType::UInt8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("ratio", DataType::Float64, true),
        Field::new("born", DataType::Date32, true),
        Field::new("seen at", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), true),
        Field::new("txt", DataType::LargeUtf8, true),
        Field::new("level", dict.data_type().clone(), true),
        Field::new("t", DataType::Time64(TimeUnit::Microsecond), true),
    ];
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![Some(1), Some(9_007_199_254_740_993), None, Some(-5), Some(42)])),
        Arc::new(UInt8Array::from(vec![Some(0), Some(200), Some(7), None, Some(100)])),
        Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true), Some(false)])),
        Arc::new(Float64Array::from(vec![Some(0.5), Some(f64::NAN), None, Some(-1.25), Some(1e300)])),
        // 1970-01-01, 1960-01-01, null, 2000-01-01, 1959-12-31
        Arc::new(Date32Array::from(vec![Some(0), Some(-3653), None, Some(10957), Some(-3654)])),
        // 1970-01-01T00:00:00.001 UTC, ...
        Arc::new(TimestampMicrosecondArray::from(vec![Some(1000), Some(0), None, Some(946_684_800_000_000), Some(-1)]).with_timezone("UTC")),
        Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("ccc"), Some("ünï"), Some("")])),
        Arc::new(dict),
        Arc::new(Time64MicrosecondArray::from(vec![Some(3_600_000_000), Some(0), None, Some(61_500_000), Some(1)])),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

fn check_foreign(s: &mut Session, out: &str) {
    assert!(out.contains("(9 vars, 5 obs)"), "{out}");
    assert!(out.contains("id64: 1 64-bit integer value(s) beyond 2^53"), "{out}");
    let ds = s.ds();
    let ty = |v: &str| ds.get_var(v).unwrap().var_type;
    assert_eq!(ty("id64"), VarType::Double);
    assert_eq!(ty("u8"), VarType::Int); // 200 does not fit byte
    assert_eq!(ty("flag"), VarType::Byte);
    assert_eq!(ty("ratio"), VarType::Double);
    assert_eq!(ty("born"), VarType::Long);
    assert_eq!(ty("seen_at"), VarType::Double);
    assert_eq!(ty("txt"), VarType::Str(5));
    assert_eq!(ty("level"), VarType::Long);
    assert_eq!(ds.get_var("seen_at").unwrap().label, "seen at");
    assert_eq!(ds.get_var("born").unwrap().format, "%td");
    assert_eq!(ds.get_var("seen_at").unwrap().format, "%tc");
    let n = s.nums("id64");
    assert_eq!(n[0], 1.0);
    assert!(n[2].is_nan());
    assert_eq!(s.nums("u8")[1], 200.0);
    assert!(s.nums("u8")[3].is_nan());
    let f = s.nums("flag");
    assert_eq!((f[0], f[1]), (1.0, 0.0));
    assert!(f[2].is_nan());
    let r = s.nums("ratio");
    assert!(r[1].is_nan() && r[2].is_nan());
    assert_eq!(r[4], 1e300);
    let b = s.nums("born");
    assert_eq!((b[0], b[1], b[3], b[4]), (3653.0, 0.0, 14610.0, -1.0));
    assert!(b[2].is_nan());
    let t = s.nums("seen_at");
    assert_eq!((t[0], t[1], t[3], t[4]), (315_619_200_001.0, 315_619_200_000.0, 1_262_304_000_000.0, 315_619_200_000.0));
    assert_eq!(s.strs("txt"), ["a", "", "ccc", "ünï", ""]);
    // dictionary strings become labelled codes
    let l = s.nums("level");
    assert_eq!((l[0], l[1], l[2], l[3]), (1.0, 2.0, 1.0, 3.0));
    assert!(l[4].is_nan());
    assert!(s.run("list level in 2").contains("high"));
    assert_eq!(s.nums("t")[0], 3_600_000.0);
    assert_close(s.scalar("year(dofc(seen_at[4]))"), 2000.0, 0.0);
}

#[test]
fn foreign_parquet_and_arrow_types() {
    let dir = temp_dir("t22_foreign");
    let batch = foreign_batch();
    // parquet, zstd
    let file = p(&dir, "f.parquet");
    let props = parquet::file::properties::WriterProperties::builder().set_compression(parquet::basic::Compression::ZSTD(Default::default())).build();
    let mut w = parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&file).unwrap(), batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let mut s = Session::new();
    let out = s.run(&format!("import parquet using \"{file}\""));
    check_foreign(&mut s, &out);
    // arrow file format, lz4
    let file = p(&dir, "f.arrow");
    let opts = arrow_ipc::writer::IpcWriteOptions::default().try_with_compression(Some(arrow_ipc::CompressionType::LZ4_FRAME)).unwrap();
    let mut w = arrow_ipc::writer::FileWriter::try_new_with_options(std::fs::File::create(&file).unwrap(), &batch.schema(), opts).unwrap();
    w.write(&batch).unwrap();
    w.finish().unwrap();
    let out = s.run(&format!("import arrow using \"{file}\", clear"));
    check_foreign(&mut s, &out);
    // arrow stream format, split over two batches
    let file = p(&dir, "f.arrows");
    let mut w = arrow_ipc::writer::StreamWriter::try_new(std::fs::File::create(&file).unwrap(), &batch.schema()).unwrap();
    w.write(&batch.slice(0, 2)).unwrap();
    w.write(&batch.slice(2, 3)).unwrap();
    w.finish().unwrap();
    let out = s.run(&format!("import arrow using \"{file}\", clear"));
    check_foreign(&mut s, &out);
    // round trip through our own writer keeps everything
    let file2 = p(&dir, "g.parquet");
    let before = s.ds().clone();
    s.run(&format!("export parquet using \"{file2}\""));
    s.run(&format!("import parquet using \"{file2}\", clear"));
    assert_same(&before, s.ds());
}

#[test]
fn json_records_and_lines() {
    let dir = temp_dir("t22_json");
    let arr = p(&dir, "a.json");
    std::fs::write(
        &arr,
        r#"[
  {"id": 1, "name": "Ann", "score": 9.5, "ok": true, "day": "2024-03-01", "at": "2024-03-01T12:30:00Z", "tags": ["x", "y"]},
  {"id": 2, "name": null, "score": 7, "ok": false, "day": null, "at": "2024-03-01T14:30:00+02:00", "extra": "late"},
  {"id": 300, "name": "Bob \"B\"", "ok": null, "day": "1960-01-01", "at": null, "mixed": 5},
  {"id": -4, "name": "Zoë", "score": -1e3, "mixed": "five"}
]"#,
    )
    .unwrap();
    let mut s = Session::new();
    let out = s.run(&format!("import json using \"{arr}\""));
    assert!(out.contains("(9 vars, 4 obs)"), "{out}");
    assert_eq!(s.varnames(), ["id", "name", "score", "ok", "day", "at", "tags", "extra", "mixed"]);
    let ds = s.ds();
    assert_eq!(ds.get_var("id").unwrap().var_type, VarType::Int);
    assert_eq!(ds.get_var("score").unwrap().var_type, VarType::Double);
    assert_eq!(ds.get_var("ok").unwrap().var_type, VarType::Byte);
    assert_eq!(ds.get_var("day").unwrap().format, "%td");
    assert_eq!(ds.get_var("at").unwrap().format, "%tc");
    assert!(ds.get_var("mixed").unwrap().var_type.is_string());
    assert_eq!(s.nums("id"), [1., 2., 300., -4.]);
    assert_eq!(s.strs("name"), ["Ann", "", "Bob \"B\"", "Zoë"]);
    assert!(s.nums("score")[2].is_nan());
    assert_eq!(s.nums("score")[3], -1000.0);
    assert!(s.nums("ok")[2].is_nan() && s.nums("ok")[3].is_nan());
    let day = s.nums("day");
    assert_eq!(day[2], 0.0);
    assert_close(s.scalar("day[1] - mdy(3,1,2024)"), 0.0, 0.0);
    // both instants are 12:30 UTC
    assert_eq!(s.nums("at")[0], s.nums("at")[1]);
    assert_close(s.scalar("at[1] - clock(\"2024-03-01 12:30:00\", \"YMDhms\")"), 0.0, 0.0);
    assert_eq!(s.strs("tags")[0], r#"["x","y"]"#);
    assert_eq!(s.strs("mixed"), ["", "", "5", "five"]);

    // JSON Lines, detected from the content and from the extension
    let lines = p(&dir, "b.jsonl");
    std::fs::write(&lines, "{\"a\": 1, \"b\": \"x\"}\n\n{\"b\": \"y\", \"a\": 2.5}\n").unwrap();
    s.run(&format!("import json using \"{lines}\", clear"));
    assert_eq!(s.varnames(), ["a", "b"]);
    assert_eq!(s.nums("a"), [1.0, 2.5]);
    let txt = p(&dir, "c.txt");
    std::fs::write(&txt, "{\"a\": 1}\n{\"a\": 2}\n").unwrap();
    s.run(&format!("import json using \"{txt}\", clear"));
    assert_eq!(s.nums("a"), [1.0, 2.0]);
    s.run(&format!("import json using \"{txt}\", clear lines rowrange(2:2)"));
    assert_eq!(s.nums("a"), [2.0]);
    let bad = p(&dir, "bad.json");
    std::fs::write(&bad, "[{\"a\": 1},").unwrap();
    s.fails(&format!("import json using \"{bad}\", clear"));
}

#[test]
fn json_export_round_trip_and_layouts() {
    let dir = temp_dir("t22_json_rt");
    let mut s = Session::new();
    let mut ds = rich();
    ds.drop_variable("long_text").unwrap();
    s.load(ds);
    let file = p(&dir, "r.json");
    let out = s.run(&format!("export json using \"{file}\""));
    assert!(out.contains("ext: 1 extended missing value(s) (.a-.z) written as null"), "{out}");
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.starts_with("[\n{\"id\":1,\"name\":\"ant\",\"group\":1,\"x\":1,\"y\":2.1"), "{text}");
    assert!(text.contains("\"f\":0.33333334"), "{text}");
    assert!(text.contains("\"when\":\"1958-11-27\""), "{text}");
    assert!(text.contains("\"stamp\":\"2020-03-16T"), "{text}");
    assert!(text.contains("\"ext\":null"), "{text}");
    s.run(&format!("import json using \"{file}\", clear"));
    for v in ["id", "group", "x", "y", "z", "d", "cnt", "big", "ext", "when", "stamp"] {
        let a: Vec<f64> = s.nums(v);
        let mut t = Session::new();
        let mut ds = rich();
        ds.drop_variable("long_text").unwrap();
        t.load(ds);
        let b = t.nums(v);
        for (x, y) in a.iter().zip(&b) {
            assert!(x == y || (x.is_nan() && y.is_nan()), "{v}: {x} vs {y}");
        }
    }
    assert_eq!(s.ds().get_var("when").unwrap().format, "%td");
    assert_eq!(s.ds().get_var("stamp").unwrap().format, "%tc");
    assert_close(s.scalar("f[2]"), 1.0 / 3.0, 1e-7);
    // lines + pretty + varlist + if
    let lf = p(&dir, "r.jsonl");
    s.run(&format!("export json id name using \"{lf}\" if id <= 2, lines pretty"));
    assert_eq!(std::fs::read_to_string(&lf).unwrap(), "{\"id\": 1, \"name\": \"ant\"}\n{\"id\": 2, \"name\": \"bee\"}\n");
    let pf = p(&dir, "p.json");
    s.run(&format!("export json id using \"{pf}\" in 1/2, pretty"));
    assert_eq!(std::fs::read_to_string(&pf).unwrap(), "[\n  {\n    \"id\": 1\n  },\n  {\n    \"id\": 2\n  }\n]\n");
    s.fails(&format!("export json id using \"{pf}\""));
    s.run(&format!("export json id using \"{pf}\", replace"));
}
