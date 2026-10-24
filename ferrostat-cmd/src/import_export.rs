use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use ferrostat_core::{DataSet, Variable, VarType, Value};
use ferrostat_parser::parser::Command;
use crate::commands::CommandHandler;

pub struct ImportDelimited;

impl CommandHandler for ImportDelimited {
    fn name(&self) -> &'static str { "import_delimited" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let path = cmd.args.first()
            .cloned()
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        if path.is_empty() {
            return Err("import delimited requires a filename".into());
        }
        Self::import(&path, ds).map(|_| format!("imported {}", path))
    }
}

impl ImportDelimited {
    pub fn import(path: &str, ds: &mut DataSet) -> Result<(), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let headers: Vec<String> = rdr.headers()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut all_records: Vec<Vec<String>> = Vec::new();
        let mut types: Vec<VarType> = Vec::new();

        for result in rdr.records() {
            let record = result.map_err(|e| e.to_string())?;
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();

            if types.is_empty() {
                for field in &row {
                    let trimmed = field.trim();
                    if trimmed.is_empty() || trimmed == "." || trimmed == "NA" {
                        types.push(VarType::Str(32));
                    } else if trimmed.parse::<i64>().is_ok() {
                        types.push(VarType::Long);
                    } else if trimmed.parse::<f64>().is_ok() {
                        types.push(VarType::Double);
                    } else {
                        types.push(VarType::Str(32));
                    }
                }
            }
            all_records.push(row);
        }

        if types.is_empty() {
            for _ in &headers {
                types.push(VarType::Str(32));
            }
        }

        for (name, vtype) in headers.iter().zip(&types) {
            let var = Variable::new(name.clone(), *vtype);
            ds.add_variable(var).map_err(|e| e.to_string())?;
        }

        for row in &all_records {
            for (i, field) in row.iter().enumerate() {
                let val = parse_value(field, types.get(i).copied().unwrap_or(VarType::Str(32)));
                if let Some(var) = ds.vars.get_mut(i) {
                    var.data.push(val);
                }
            }
            ds.n_obs += 1;
        }

        Ok(())
    }
}

fn parse_value(s: &str, vtype: VarType) -> Value {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == "NA" {
        return if vtype.is_numeric() { Value::Double(f64::NAN) } else { Value::Str(String::new()) };
    }
    match vtype {
        VarType::Byte => trimmed.parse::<i8>().map(Value::Byte).unwrap_or(Value::Byte(0)),
        VarType::Int => trimmed.parse::<i16>().map(Value::Int).unwrap_or(Value::Int(0)),
        VarType::Long => trimmed.parse::<i32>().map(Value::Long).unwrap_or(Value::Long(0)),
        VarType::Float => trimmed.parse::<f32>().map(Value::Float).unwrap_or(Value::Float(f32::NAN)),
        VarType::Double => trimmed.parse::<f64>().map(Value::Double).unwrap_or(Value::Double(f64::NAN)),
        _ => Value::Str(trimmed.to_string()),
    }
}

pub struct ExportDelimited;

impl CommandHandler for ExportDelimited {
    fn name(&self) -> &'static str { "export_delimited" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let path = cmd.args.first()
            .cloned()
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        if path.is_empty() {
            return Err("export delimited requires a filename".into());
        }
        Self::export(&path, ds).map(|_| format!("exported {}", path))
    }
}

impl ExportDelimited {
    pub fn export(path: &str, ds: &DataSet) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut wtr = WriterBuilder::new().from_writer(file);

        let headers: Vec<String> = ds.vars.iter().map(|v| v.name.clone()).collect();
        wtr.write_record(&headers).map_err(|e| e.to_string())?;

        for i in 0..ds.n_obs {
            let mut row = Vec::new();
            for var in &ds.vars {
                let s = match &var.data[i] {
                    Value::Str(s) => s.clone(),
                    Value::Double(v) => if v.is_nan() { "".to_string() } else { format!("{}", v) },
                    Value::Float(v) => if v.is_nan() { "".to_string() } else { format!("{}", v) },
                    Value::Int(v) => v.to_string(),
                    Value::Long(v) => v.to_string(),
                    Value::Byte(v) => v.to_string(),
                    Value::StrL(_, _) => "(strL)".to_string(),
                    Value::Missing => "".to_string(),
                };
                row.push(s);
            }
            wtr.write_record(&row).map_err(|e| e.to_string())?;
        }

        wtr.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}
