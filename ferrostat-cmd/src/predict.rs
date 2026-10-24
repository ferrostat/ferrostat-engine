use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::Command;
use crate::commands::CommandHandler;
use crate::estimation::{get_results, ModelType};

/// Post-estimation predictions.
pub struct Predict;

impl CommandHandler for Predict {
    fn name(&self) -> &'static str { "predict" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let est = get_results().ok_or("no estimation results in memory; run a regression first")?;

        let newvar = if cmd.args.is_empty() {
            return Err("predict requires a new variable name".into());
        } else {
            &cmd.args[0]
        };

        if ds.get_var(newvar).is_some() {
            return Err(format!("variable {} already defined", newvar));
        }

        let mut options: Vec<String> = cmd.options.iter().map(|(k, _)| k.clone()).collect();
        if let Some(ref opts_str) = cmd.args.get(1) {
            options.push(opts_str.to_string());
        }

        let prediction_type = if options.contains(&"residuals".to_string()) || options.contains(&"resid".to_string()) {
            PredictionType::Residuals
        } else if options.contains(&"xb".to_string()) {
            PredictionType::Xb
        } else if options.contains(&"pr".to_string()) || options.contains(&"p".to_string()) {
            PredictionType::Probability
        } else {
            PredictionType::Xb // default: linear predictor
        };

        // Default option: if indepvars is empty (was anova), handle differently
        let has_xb = !est.indepvars.is_empty() || est.beta.len() > 1;

        let mut result = vec![Value::Missing; ds.n_obs];

        match (est.model_type, prediction_type) {
            // Linear models: predict = xb, residuals = y - xb
            (ModelType::Linear, PredictionType::Xb) | (ModelType::FixedEffects, PredictionType::Xb) => {
                for i in 0..ds.n_obs {
                    let mut xb = est.beta[0]; // intercept
                    for (j, name) in est.indepvars.iter().enumerate() {
                        if let Some(var) = ds.get_var(name) {
                            if let Some(val) = var.data[i].as_f64() {
                                xb += est.beta[j + 1] * val;
                            }
                        }
                    }
                    result[i] = Value::Double(xb);
                }
            }
            (ModelType::Linear, PredictionType::Residuals) | (ModelType::FixedEffects, PredictionType::Residuals) => {
                for i in 0..ds.n_obs {
                    let mut xb = est.beta[0];
                    for (j, name) in est.indepvars.iter().enumerate() {
                        if let Some(var) = ds.get_var(name) {
                            if let Some(val) = var.data[i].as_f64() {
                                xb += est.beta[j + 1] * val;
                            }
                        }
                    }
                    if let Some(dep) = ds.get_var(&est.depvar) {
                        if let Some(y) = dep.data[i].as_f64() {
                            result[i] = Value::Double(y - xb);
                        }
                    }
                }
            }
            // Logit: predict probability = 1/(1+exp(-xb))
            (ModelType::Logit, PredictionType::Probability) | (ModelType::Logit, PredictionType::Xb) => {
                for i in 0..ds.n_obs {
                    let mut xb = est.beta[0];
                    for (j, name) in est.indepvars.iter().enumerate() {
                        if let Some(var) = ds.get_var(name) {
                            if let Some(val) = var.data[i].as_f64() {
                                xb += est.beta[j + 1] * val;
                            }
                        }
                    }
                    let pr = 1.0 / (1.0 + (-xb).exp());
                    result[i] = Value::Double(pr);
                }
            }
            // Probit: predict probability = Phi(xb)
            (ModelType::Probit, PredictionType::Probability) | (ModelType::Probit, PredictionType::Xb) => {
                for i in 0..ds.n_obs {
                    let mut xb = est.beta[0];
                    for (j, name) in est.indepvars.iter().enumerate() {
                        if let Some(var) = ds.get_var(name) {
                            if let Some(val) = var.data[i].as_f64() {
                                xb += est.beta[j + 1] * val;
                            }
                        }
                    }
                    let pr = 0.5 * (1.0 + erf_approx(xb / 2.0f64.sqrt()));
                    result[i] = Value::Double(pr);
                }
            }
            // Poisson: predict count = exp(xb)
            (ModelType::Poisson, PredictionType::Xb) => {
                for i in 0..ds.n_obs {
                    let mut xb = est.beta[0];
                    for (j, name) in est.indepvars.iter().enumerate() {
                        if let Some(var) = ds.get_var(name) {
                            if let Some(val) = var.data[i].as_f64() {
                                xb += est.beta[j + 1] * val;
                            }
                        }
                    }
                    result[i] = Value::Double(xb.exp());
                }
            }
            _ => return Err(format!("prediction type {:?} not supported for model {:?}", prediction_type, est.model_type)),
        }

        let mut var = Variable::new(newvar.clone(), VarType::Double);
        var.data = result;
        ds.add_variable(var)?;

        Ok(format!("predicted values stored in variable: {}", newvar))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PredictionType {
    Xb,
    Residuals,
    Probability,
}

fn erf_approx(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
