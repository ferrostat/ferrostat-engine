pub mod dataset;
pub mod dta;
pub mod missing;
pub mod value_label;
pub mod var;
pub mod vfs;

pub use dataset::DataSet;
pub use missing::{is_missing, is_missing_byte, is_missing_int, is_missing_long, is_missing_float, is_missing_double, MissingValue, SYSMISS};
pub use value_label::ValueLabel;
pub use var::{Variable, VarType, Value};

#[cfg(test)]
mod tests {
    use crate::dta::DtaReader;
    use crate::DataSet;
    use crate::Variable;
    use crate::VarType;
    use crate::Value;

    #[test]
    fn test_dta_roundtrip() {
        let mut ds = DataSet::new();
        let mut v = Variable::new("make", VarType::Str(8));
        v.data = vec![Value::Str("Alpha".into()), Value::Str("Beta".into())];
        ds.add_variable(v).unwrap();
        let mut p = Variable::new("price", VarType::Int);
        p.data = vec![Value::Int(4099), Value::Int(4749)];
        ds.add_variable(p).unwrap();
        ds.n_obs = 2;
        let mut buf = Vec::new();
        crate::dta::DtaWriter::write(&mut buf, &ds).unwrap();
        let back = DtaReader::from_bytes(buf).unwrap();
        assert_eq!(back.n_obs, 2);
        assert_eq!(back.get_var("make").unwrap().data[1].as_string(), Some("Beta"));
        assert_eq!(back.get_var("price").unwrap().data[0].as_f64(), Some(4099.0));
    }

    #[test]
    fn test_dataset_sort() {
        let mut ds = DataSet::new();
        ds.add_variable(Variable::new("x", VarType::Int)).unwrap();
        ds.add_variable(Variable::new("y", VarType::Int)).unwrap();
        for i in 0..5 {
            ds.vars[0].data.push(Value::Int((5 - i) as i16));
            ds.vars[1].data.push(Value::Int(i as i16));
        }
        ds.n_obs = 5;
        ds.sort(&vec!["x".to_string()], false).unwrap();
        let xs: Vec<i16> = ds.vars[0].data.iter().map(|v| match v {
            Value::Int(x) => *x,
            _ => 0,
        }).collect();
        assert_eq!(xs, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_dataset_keep_if() {
        let mut ds = DataSet::new();
        ds.add_variable(Variable::new("x", VarType::Int)).unwrap();
        for i in 0..5 {
            ds.vars[0].data.push(Value::Int(i as i16));
        }
        ds.n_obs = 5;
        ds.keep_if(&vec![false, true, false, true, false]).unwrap();
        assert_eq!(ds.n_obs, 2);
        let xs: Vec<i16> = ds.vars[0].data.iter().map(|v| match v {
            Value::Int(x) => *x,
            _ => 0,
        }).collect();
        assert_eq!(xs, vec![1, 3]);
    }
}
