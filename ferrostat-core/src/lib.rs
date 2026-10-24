pub mod dataset;
pub mod dta;
pub mod missing;
pub mod value_label;
pub mod var;

pub use dataset::DataSet;
pub use missing::{is_missing_byte, is_missing_int, is_missing_long, is_missing_float, is_missing_double, MissingValue};
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
    fn test_read_auto_dta() {
        let ds = DtaReader::open("/home/ubuntu/installer/auto.dta").expect("read auto.dta");
        assert_eq!(ds.n_obs, 74);
        assert_eq!(ds.n_vars(), 12);

        let names: Vec<String> = ds.vars.iter().map(|v| v.name.clone()).collect();
        assert_eq!(names, vec![
            "make", "price", "mpg", "rep78", "headroom", "trunk",
            "weight", "length", "turn", "displacement", "gear_ratio", "foreign"
        ]);

        let make = ds.get_var("make").unwrap();
        assert!(make.var_type.is_string());
        assert_eq!(make.data[0].as_string(), Some("AMC Concord"));

        let price = ds.get_var("price").unwrap();
        assert!(price.var_type.is_numeric());
        assert_eq!(price.data[0].as_f64(), Some(4099.0));

        let foreign = ds.get_var("foreign").unwrap();
        assert_eq!(foreign.value_label_name, "origin");
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
