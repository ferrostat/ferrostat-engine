use std::collections::HashMap;

use ferrostat_core::{DataSet, VarType, Variable, Value};
use ferrostat_parser::Command;
use ferrostat_cmd::commands::CommandHandler;
use ferrostat_cmd::summarize::Summarize;
use ferrostat_cmd::regress::Regress;
use ferrostat_cmd::generate::{Generate, Replace};
use ferrostat_cmd::list::ListCmd;
use ferrostat_cmd::describe_cmd::DescribeCmd;
use ferrostat_cmd::count::Count;
use ferrostat_cmd::contract::Contract;
use ferrostat_cmd::tabulate::Tabulate;
use ferrostat_cmd::correlate::Correlate;
use ferrostat_cmd::ttest::Ttest;
use ferrostat_cmd::logit::Logit;
use ferrostat_cmd::probit::Probit;
use ferrostat_cmd::poisson::Poisson;
use ferrostat_cmd::collapse::Collapse;
use ferrostat_cmd::expand_fillin::{Expand, Fillin};
use ferrostat_cmd::reshape::Reshape;
use ferrostat_cmd::anova::Anova;
use ferrostat_cmd::xtreg::Xtreg;
use ferrostat_cmd::kdensity::Kdensity;
use ferrostat_cmd::predict::Predict;
use ferrostat_cmd::import_export::{ImportDelimited, ExportDelimited};
use ferrostat_cmd::plot::{Scatter, Histogram, LinePlot, BarChart};
use ferrostat_cmd::egen::Egen;

use ferrostat_cmd::display::{DisplayCmd, LocalMacro, ByPrefix};
use ferrostat_cmd::merge_append::{Merge, Append};

/// Stored estimation results for post-estimation.
#[derive(Debug, Clone, Default)]
pub struct EstimationResults {
    pub cmd: String,
    pub depvar: String,
    pub indepvars: Vec<String>,
    pub beta: Vec<f64>,
    pub n_obs: usize,
    pub r2: Option<f64>,
    pub ll: Option<f64>,
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Linear,
    Logit,
    Probit,
    Poisson,
    FixedEffects,
    Other,
}

impl Default for ModelType {
    fn default() -> Self { ModelType::Other }
}

/// Execution engine: holds the active dataset and dispatches commands.
pub struct Engine {
    pub dataset: DataSet,
    pub handlers: HashMap<String, Box<dyn CommandHandler>>,
    pub macro_vars: HashMap<String, String>,
    pub version: f64,
    pub last_estimates: Option<EstimationResults>,
}


/// Extract a filesystem path from a raw command line after the verb.
fn extract_path(cmd: &Command) -> String {
    let raw = cmd.raw.trim();
    let verb_end = raw.find(|c: char| c.is_whitespace()).unwrap_or(raw.len());
    let after_verb = raw[verb_end..].trim();
    // Remove options after comma
    let without_opts = if let Some(comma_pos) = after_verb.find(',') {
        &after_verb[..comma_pos]
    } else {
        after_verb
    };
    // Remove if/in clauses
    let mut path = without_opts.to_string();
    for kw in [" if ", " in ", " ["] {
        if let Some(pos) = path.to_lowercase().find(kw) {
            path = path[..pos].to_string();
        }
    }
    path.trim().trim_matches('"').to_string()
}

impl Engine {
    pub fn new() -> Self {
        let mut handlers: HashMap<String, Box<dyn CommandHandler>> = HashMap::new();
        let s = Summarize;
        handlers.insert(s.name().to_string(), Box::new(s));
        let r = Regress;
        handlers.insert(r.name().to_string(), Box::new(r));
        let g = Generate;
        handlers.insert(g.name().to_string(), Box::new(g));
        let replace = Replace;
        handlers.insert(replace.name().to_string(), Box::new(replace));
        let l = ListCmd;
        handlers.insert(l.name().to_string(), Box::new(l));
        let d = DescribeCmd;
        handlers.insert(d.name().to_string(), Box::new(d));
        let c = Count;
        handlers.insert(c.name().to_string(), Box::new(c));
        let contract = Contract;
        handlers.insert(contract.name().to_string(), Box::new(contract));
        let tab = Tabulate;
        handlers.insert(tab.name().to_string(), Box::new(tab));
        let corr = Correlate;
        handlers.insert(corr.name().to_string(), Box::new(corr));
        let tt = Ttest;
        handlers.insert(tt.name().to_string(), Box::new(tt));
        let logit = Logit;
        handlers.insert(logit.name().to_string(), Box::new(logit));
        let probit = Probit;
        handlers.insert(probit.name().to_string(), Box::new(probit));
        let poisson = Poisson;
        handlers.insert(poisson.name().to_string(), Box::new(poisson));
        let collapse = Collapse;
        handlers.insert(collapse.name().to_string(), Box::new(collapse));
        let expand = Expand;
        handlers.insert(expand.name().to_string(), Box::new(expand));
        let fillin = Fillin;
        handlers.insert(fillin.name().to_string(), Box::new(fillin));
        let reshape = Reshape;
        handlers.insert(reshape.name().to_string(), Box::new(reshape));
        let anova = Anova;
        handlers.insert(anova.name().to_string(), Box::new(anova));
        let xtreg = Xtreg;
        handlers.insert(xtreg.name().to_string(), Box::new(xtreg));
        let kdensity = Kdensity;
        handlers.insert(kdensity.name().to_string(), Box::new(kdensity));
        let predict = Predict;
        handlers.insert(predict.name().to_string(), Box::new(predict));
        let import = ImportDelimited;
        handlers.insert("import_delimited".to_string(), Box::new(import));
        let export = ExportDelimited;
        handlers.insert("export_delimited".to_string(), Box::new(export));
        let scatter = Scatter;
        handlers.insert(scatter.name().to_string(), Box::new(scatter));
        let hist = Histogram;
        handlers.insert(hist.name().to_string(), Box::new(hist));
        let line = LinePlot;
        handlers.insert(line.name().to_string(), Box::new(line));
        let bar = BarChart;
        handlers.insert(bar.name().to_string(), Box::new(bar));
        let egen = Egen;
        handlers.insert(egen.name().to_string(), Box::new(egen));
        let by = ByPrefix;
        handlers.insert(by.name().to_string(), Box::new(by));
        let local = LocalMacro;
        handlers.insert(local.name().to_string(), Box::new(local));
        let display = DisplayCmd;
        handlers.insert(display.name().to_string(), Box::new(display));

        Self {
            dataset: DataSet::new(),
            handlers,
            macro_vars: HashMap::new(),
            version: 18.0,
            last_estimates: None,
        }
    }

    pub fn execute(&mut self, cmd: &Command) -> Result<String, String> {
        let verb = cmd.verb.to_lowercase();
        match verb.as_str() {
            "clear" => {
                self.dataset = DataSet::new();
                self.last_estimates = None;
                Ok("cleared".into())
            }
            "use" => {
                let path = cmd.using.clone()
                    .unwrap_or_else(|| extract_path(cmd));
                if path.is_empty() {
                    return Err("use requires a filename".into());
                }
                match ferrostat_core::dta::DtaReader::open(&path) {
                    Ok(ds) => {
                        self.dataset = ds;
                        Ok(format!("loaded dataset from {}", path))
                    }
                    Err(e) => Err(format!("could not load {}: {}", path, e)),
                }
            }
            "sysuse" => {
                let name = cmd.args.first()
                    .cloned()
                    .unwrap_or_default()
                    .trim_matches('"')
                    .to_string();
                if name.is_empty() {
                    return Err("sysuse requires a dataset name".into());
                }
                // Try standard locations
                let paths = vec![
                    format!("/home/ubuntu/installer/{}.dta", name),
                    format!("/usr/local/share/ferrostat/{}.dta", name),
                    format!("{}.dta", name),
                ];
                for path in &paths {
                    if let Ok(ds) = ferrostat_core::dta::DtaReader::open(path) {
                        self.dataset = ds;
                        return Ok(format!("loaded system dataset {}", name));
                    }
                }
                Err(format!("could not find system dataset {}", name))
            }
            "save" => {
                let path = cmd.using.clone()
                    .unwrap_or_else(|| extract_path(cmd));
                if path.is_empty() {
                    return Err("save requires a filename".into());
                }
                let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                match ferrostat_core::dta::DtaWriter::write(&mut file, &self.dataset) {
                    Ok(_) => Ok(format!("saved dataset to {}", path)),
                    Err(e) => Err(format!("could not save {}: {}", path, e)),
                }
            }
            "import" => {
                // import delimited, import excel, etc.
                let subcmd = cmd.args.first().map(|s| s.to_lowercase()).unwrap_or_default();
                match subcmd.as_str() {
                    "delimited" => {
                        let path = cmd.args.get(1)
                            .cloned()
                            .unwrap_or_default()
                            .trim_matches('"')
                            .to_string();
                        if path.is_empty() {
                            return Err("import delimited requires a filename".into());
                        }
                        match ImportDelimited::import(&path, &mut self.dataset) {
                            Ok(_) => Ok(format!("imported {}", path)),
                            Err(e) => Err(format!("import failed: {}", e)),
                        }
                    }
                    _ => Err(format!("unknown import type: {}", subcmd)),
                }
            }
            "export" => {
                let subcmd = cmd.args.first().map(|s| s.to_lowercase()).unwrap_or_default();
                match subcmd.as_str() {
                    "delimited" => {
                        let path = cmd.args.get(1)
                            .cloned()
                            .unwrap_or_default()
                            .trim_matches('"')
                            .to_string();
                        if path.is_empty() {
                            return Err("export delimited requires a filename".into());
                        }
                        match ExportDelimited::export(&path, &self.dataset) {
                            Ok(_) => Ok(format!("exported {}", path)),
                            Err(e) => Err(format!("export failed: {}", e)),
                        }
                    }
                    _ => Err(format!("unknown export type: {}", subcmd)),
                }
            }
            "drop" => self.handle_drop(cmd),
            "keep" => self.handle_keep(cmd),
            "sort" => self.handle_sort(cmd),
            "merge" => {
                let m = Merge;
                m.execute(&mut self.dataset, cmd)
            }
            "append" => {
                let a = Append;
                a.execute(&mut self.dataset, cmd)
            }
            "exit" | "quit" => Ok("__EXIT__".into()),
            _ => {
                // Check for regression commands that need to store estimates
                let is_estimation = matches!(verb.as_str(), "regress" | "logit" | "probit" | "poisson" | "xtreg" | "anova");
                
                if let Some(handler) = self.handlers.get_mut(&verb) {
                    let result = handler.execute(&mut self.dataset, cmd);
                    
                    if is_estimation && result.is_ok() {
                        self.store_estimates(&verb, cmd);
                    }
                    
                    result
                } else {
                    Err(format!("unknown command: {}", verb))
                }
            }
        }
    }

    fn store_estimates(&mut self, verb: &str, cmd: &Command) {
        if cmd.args.len() < 2 {
            return;
        }
        let depvar = cmd.args[0].clone();
        let indepvars = cmd.args[1..].to_vec();
        
        let model_type = match verb {
            "regress" => ModelType::Linear,
            "logit" => ModelType::Logit,
            "probit" => ModelType::Probit,
            "poisson" => ModelType::Poisson,
            "xtreg" => ModelType::FixedEffects,
            _ => ModelType::Other,
        };

        self.last_estimates = Some(EstimationResults {
            cmd: verb.to_string(),
            depvar,
            indepvars,
            beta: Vec::new(), // Will be populated by the command itself
            n_obs: self.dataset.n_obs,
            r2: None,
            ll: None,
            model_type,
        });
    }

    fn handle_drop(&mut self, cmd: &Command) -> Result<String, String> {
        if let Some(ref expr) = cmd.if_expr {
            let rows = ferrostat_cmd::summarize::apply_if(&self.dataset, expr)?;
            let n_drop = rows.len();
            let mut mask = vec![true; self.dataset.n_obs];
            for &i in &rows {
                mask[i] = false;
            }
            self.dataset.keep_if(&mask)?;
            return Ok(format!("dropped {} observations", n_drop));
        }
        if cmd.args.is_empty() {
            return Err("drop requires variable names or if".into());
        }
        for name in &cmd.args {
            if name.eq_ignore_ascii_case("_all") {
                self.dataset = DataSet::new();
                return Ok("all variables dropped".into());
            }
        }
        for name in &cmd.args {
            self.dataset.drop_variable(name)?;
        }
        Ok(format!("dropped: {}", cmd.args.join(" ")))
    }

    fn handle_keep(&mut self, cmd: &Command) -> Result<String, String> {
        if let Some(ref expr) = cmd.if_expr {
            let rows = ferrostat_cmd::summarize::apply_if(&self.dataset, expr)?;
            let mut mask = vec![false; self.dataset.n_obs];
            for &i in &rows {
                mask[i] = true;
            }
            self.dataset.keep_if(&mask)?;
            return Ok(format!("kept {} observations", self.dataset.n_obs));
        }
        if let Some(ref range) = cmd.in_range {
            let rows = ferrostat_cmd::summarize::apply_in(&self.dataset, range)?;
            let mut mask = vec![false; self.dataset.n_obs];
            for i in rows {
                mask[i] = true;
            }
            self.dataset.keep_if(&mask)?;
            return Ok(format!("kept {} observations", self.dataset.n_obs));
        }
        if cmd.args.is_empty() {
            return Err("keep requires variable names or if/in".into());
        }
        self.dataset.keep_variables(&cmd.args);
        Ok(format!("kept: {}", cmd.args.join(" ")))
    }

    fn handle_sort(&mut self, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("sort requires variable names".into());
        }
        self.dataset.sort(&cmd.args, false)?;
        Ok(format!("sorted by: {}", cmd.args.join(" ")))
    }
}
