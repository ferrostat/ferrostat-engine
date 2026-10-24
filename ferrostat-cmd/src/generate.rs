use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use ferrostat_parser::parser::Parser;
use crate::commands::CommandHandler;
use crate::eval::{eval_expr, EvalResult};

pub struct Generate;

impl CommandHandler for Generate {
    fn name(&self) -> &'static str { "generate" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("generate requires: generate newvar = expression".into());
        }

        let newvar = &cmd.args[0];
        if ds.get_var(newvar).is_some() {
            return Err(format!("variable {} already defined", newvar));
        }

        // Parse expression from raw string after '='
        let raw = &cmd.raw;
        let eq_pos = raw.find('=')
            .ok_or("generate syntax: generate newvar = expr")?;
        let expr_str = &raw[eq_pos + 1..];
        
        let mut parser = Parser::new(expr_str);
        let expr = parser.parse_expr()
            .map_err(|e| format!("cannot evaluate expression: {} - {}", expr_str.trim(), e.message))?;
        
        let mut data = Vec::with_capacity(ds.n_obs);
        for i in 0..ds.n_obs {
            match eval_expr(&expr, ds, i) {
                Ok(EvalResult::Number(n)) => data.push(Value::Double(n)),
                Ok(EvalResult::String(s)) => data.push(Value::Str(s)),
                Ok(EvalResult::Missing) => data.push(Value::Double(f64::NAN)),
                Err(e) => {
                    // If variable not found, maybe it's a constant expression
                    data.push(Value::Double(f64::NAN));
                }
            }
        }
        
        let mut var = Variable::new(newvar.clone(), VarType::Double);
        var.data = data;
        ds.add_variable(var)?;
        
        Ok(format!("generated variable: {}", newvar))
    }
}

pub struct Replace;

impl CommandHandler for Replace {
    fn name(&self) -> &'static str { "replace" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("replace requires: replace varname = expression".into());
        }

        let varname = &cmd.args[0];
        let var_idx = ds.vars.iter().position(|v| v.name == *varname)
            .ok_or_else(|| format!("variable {} not found", varname))?;

        let raw = &cmd.raw;
        let eq_pos = raw.find('=')
            .ok_or("replace syntax: replace varname = expr")?;
        let expr_str = &raw[eq_pos + 1..];
        
        let mut parser = Parser::new(expr_str);
        let expr = parser.parse_expr()
            .map_err(|e| format!("cannot evaluate expression: {} - {}", expr_str.trim(), e.message))?;
        
        for i in 0..ds.n_obs {
            // Apply if condition
            let should_replace = if let Some(ref if_e) = cmd.if_expr {
                match eval_expr(if_e, ds, i) {
                    Ok(v) => v.as_bool() && !v.is_missing(),
                    Err(_) => false,
                }
            } else {
                true
            };
            
            if should_replace {
                match eval_expr(&expr, ds, i) {
                    Ok(EvalResult::Number(n)) => ds.vars[var_idx].data[i] = Value::Double(n),
                    Ok(EvalResult::String(s)) => ds.vars[var_idx].data[i] = Value::Str(s),
                    Ok(EvalResult::Missing) => ds.vars[var_idx].data[i] = Value::Double(f64::NAN),
                    Err(_) => {}
                }
            }
        }
        
        Ok(format!("replaced values in variable: {}", varname))
    }
}
