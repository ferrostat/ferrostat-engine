//! `lasso`, `elasticnet` and `sqrtlasso`: fitting, the knot summary
//! display and the stored results.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::core::Family;
use super::fit::{self, Lasso, Sel};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub struct LassoCmd {
    pub verb: &'static str,
}

impl CommandHandler for LassoCmd {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            let (_, l) = fit::load_one(&st.e, cmd)?;
            return Ok(display(&l, self.verb));
        }
        // model word (sqrtlasso is linear only and has none)
        let (fam, rest) = if self.verb == "sqrtlasso" {
            (Family::Linear, main.to_string())
        } else {
            let (m, r) = crate::xmodels::first_word(main);
            let f = Family::parse(&m.to_lowercase()).ok_or_else(|| rc_err(198, format!("model {m} not allowed; specify linear, logit, probit or poisson")))?;
            (f, r)
        };
        let (dep, always, others) = fit::parse_main(&rest);
        if dep.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let dep = crate::xmodels::var_name(ds, &dep)?;
        let mut s = fit::settings(cmd, Sel::Cv)?;
        if self.verb == "sqrtlasso" {
            s.sqrt = true;
            if s.sel == Sel::Bic || s.sel == Sel::Adaptive {
                return Err(rc_err(198, "selection() must be cv, plugin or none with sqrtlasso"));
            }
        }
        if self.verb == "elasticnet" {
            if matches!(s.sel, Sel::Plugin | Sel::Adaptive) {
                return Err(rc_err(198, "selection() must be cv, bic or none with elasticnet"));
            }
            let a = match cmd.opt("alpha", 5) {
                Some(t) => crate::w7b::lasso::numlist(t)?,
                None => vec![0.5, 0.75, 1.0],
            };
            if a.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err(rc_err(198, "alpha() values must be between 0 and 1"));
            }
            let mut a = a;
            a.sort_by(|x, y| y.total_cmp(x));
            s.alphas = a;
        }
        if s.sel == Sel::Plugin && matches!(fam, Family::Probit) {
            // probit plug-in uses the logit-style formula with probit scores
        }
        let d = fit::build(ds, st, cmd, &dep, &always, &others, &[])?;
        let y = fit::outcome(fam, &d.y, &dep)?;
        let p = fit::problem(fam, &d, y, 1.0, s.standardize);
        fit::rseed(cmd, st)?;
        let l = {
            let mut rng = st.rng.borrow_mut();
            let mut u = || rng.uniform();
            fit::run(self.verb, &dep, &d.names, &d.always, &p, &s, &mut u)?
        };
        let text = display(&l, self.verb);
        post(st, self.verb, &l, &d.mask, &always, &others);
        Ok(text)
    }
}

/// Post a single lasso in e().
pub fn post(st: &mut Store, verb: &str, l: &Lasso, mask: &[bool], always: &str, others: &str) {
    let e = st.begin_e();
    e.set_str("cmd", verb);
    e.set_str("model", l.family.name());
    e.set_str("depvar", l.depvar.clone());
    e.set_str("title", title(l, verb));
    e.set_str("sel_method", l.selection.clone());
    e.set_str("allvars", l.names.join(" "));
    e.set_str("alwaysvars", l.names.iter().zip(&l.always).filter(|(_, a)| **a).map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" "));
    e.set_str("othervars", l.names.iter().zip(&l.always).filter(|(_, a)| !**a).map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" "));
    let _ = (always, others);
    e.set("N", l.n as f64);
    e.set("k_allvars", l.names.len() as f64);
    e.set("n_lambda", l.pts.len() as f64);
    if l.nfolds > 0 {
        e.set("n_fold", l.nfolds as f64);
    }
    e.sample = Some(mask.to_vec());
    fit::save(e, &[(l.depvar.clone(), l.clone())]);
    set_selected(e, l);
}

/// Results that depend on the selected point (also refreshed by lassoselect).
pub fn set_selected(e: &mut Results, l: &Lasso) {
    let lmax = l.pts.iter().map(|p| p.lambda).fold(f64::NAN, f64::max);
    let lmin = l.pts.iter().map(|p| p.lambda).fold(f64::NAN, f64::min);
    e.set("lambda_max", lmax);
    e.set("lambda_min", lmin);
    match l.selected() {
        Some(p) => {
            let k = l.sel.unwrap();
            e.set("lambda_sel", p.lambda);
            e.set("ID_sel", (k + 1) as f64);
            e.set("k_nonzero_sel", p.df as f64);
            if l.kind == "elasticnet" {
                e.set("alpha_sel", p.alpha);
            }
            if p.cvm.is_finite() {
                e.set("cvm_min", l.pts.iter().map(|q| q.cvm).fold(f64::NAN, f64::min));
                e.set("cvm_sel", p.cvm);
            }
            let nz = l.nonzero(k);
            let mut names: Vec<String> = nz.iter().map(|&j| l.names[j].clone()).collect();
            let mut b: Vec<f64> = nz.iter().map(|&j| p.b[j]).collect();
            e.set_str("post_sel_vars", names.join(" "));
            names.push("_cons".into());
            b.push(p.b0);
            e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names));
            e.set_str("lasso_selection", if l.how.is_empty() { l.selection.clone() } else { l.how.clone() });
        }
        None => {
            e.scalars.remove("lambda_sel");
            e.scalars.remove("ID_sel");
            e.matrices.remove("b");
            e.set_str("post_sel_vars", "");
        }
    }
}

pub fn title(l: &Lasso, verb: &str) -> String {
    let base = match verb {
        "elasticnet" => "Elastic net",
        "sqrtlasso" => "Square-root lasso",
        _ => "Lasso",
    };
    format!("{base} {} model", l.family.name())
}

/// Fit summary: header and the knot table around the selected lambda.
pub fn display(l: &Lasso, verb: &str) -> String {
    let mut out = vec![String::new()];
    let hr = |a: &str, k: &str, v: String| format!("{a:<44}{k:<18}= {v:>10}");
    out.push(hr(&title(l, verb), "No. of obs", util::commas(l.n)));
    out.push(hr("", "No. of covariates", util::commas(l.names.len())));
    let sl = format!("Selection: {}", fit::sel_label(l));
    if l.nfolds > 0 {
        out.push(hr(&sl, "No. of CV folds", l.nfolds.to_string()));
    } else {
        out.push(sl);
    }
    out.push(String::new());
    let elastic = verb == "elasticnet";
    let cv = l.pts.iter().any(|p| p.cvm.is_finite());
    let linear = l.family == Family::Linear;
    let bar = "-".repeat(if elastic { 85 } else { 74 });
    out.push(bar.clone());
    let pre = if elastic { "           " } else { "" };
    if cv {
        out.push(format!("{pre}         |                                No. of      Out-of-      CV mean"));
        out.push(format!("{pre}         |                               nonzero       sample   {}", if linear { "prediction" } else { "  deviance" }));
        out.push(format!("{}      ID |     Description      lambda     coef.    {}        {}", if elastic { "      alpha" } else { "" }, if linear { "R-squared" } else { "dev. ratio" }, if linear { "error" } else { "     " }));
    } else if l.selection == "bic" {
        out.push(format!("{pre}         |                                No. of    In-sample"));
        out.push(format!("{pre}         |                               nonzero    {}", if linear { "R-squared" } else { "dev. ratio" }));
        out.push(format!("{}      ID |     Description      lambda     coef.  {}          BIC", if elastic { "      alpha" } else { "" }, "         "));
    } else {
        out.push(format!("{pre}         |                                No. of    In-sample"));
        out.push(format!("{pre}         |                               nonzero    {}", if linear { "R-squared" } else { "dev. ratio" }));
        out.push(format!("{}      ID |     Description      lambda     coef.", if elastic { "      alpha" } else { "" }));
    }
    out.push(format!("{}---------+{}", if elastic { "-----------" } else { "" }, "-".repeat(64)));
    let n = l.pts.len();
    let mut ids: Vec<(usize, &str)> = Vec::new();
    match l.sel {
        Some(k) => {
            if l.selection == "plugin" {
                ids.push((k, "selected lambda"));
            } else {
                if k > 1 {
                    ids.push((0, "first lambda"));
                }
                if k >= 1 {
                    ids.push((k - 1, if k == 1 { "first lambda" } else { "lambda before" }));
                }
                ids.push((k, "selected lambda"));
                if k + 1 < n {
                    ids.push((k + 1, if k + 2 == n { "last lambda" } else { "lambda after" }));
                }
                if k + 2 < n {
                    ids.push((n - 1, "last lambda"));
                }
            }
        }
        None => {
            ids.push((0, "first lambda"));
            if n > 1 {
                ids.push((n - 1, "last lambda"));
            }
        }
    }
    for (k, desc) in ids {
        let p = &l.pts[k];
        let star = if Some(k) == l.sel { "*" } else { " " };
        let r2 = if cv { 1.0 - p.cvm / l.dev0 } else { 1.0 - p.dev / l.dev0 };
        let mut line = String::new();
        if elastic {
            line.push_str(&format!("{:>10} ", fmt_g(p.alpha, 6)));
        }
        line.push_str(&format!("{star}{:>7} | {:>15} {:>11} {:>9}", k + 1, desc, fmt_g(p.lambda, 7), p.df));
        if cv {
            line.push_str(&format!("   {:>10}   {:>10}", format!("{r2:.4}"), fmt_g(p.cvm, 7)));
        } else if l.selection == "bic" {
            line.push_str(&format!("   {:>10}   {:>10}", format!("{:.4}", 1.0 - p.dev / l.dev0), fmt_g(p.bic, 7)));
        } else {
            line.push_str(&format!("   {:>10}", format!("{r2:.4}")));
        }
        out.push(line);
    }
    out.push(bar);
    match (l.sel, l.how.as_str()) {
        (Some(_), "cv") => out.push("* lambda selected by cross-validation.".into()),
        (Some(_), "serule") => out.push("* lambda selected by the one-standard-error rule.".into()),
        (Some(_), "bic") => out.push("* lambda selected by Bayesian information criterion.".into()),
        (Some(_), "user") => out.push("* lambda selected by user.".into()),
        (Some(_), _) if l.selection == "plugin" => out.push("* lambda selected by plugin formula assuming heteroskedastic errors.".replace("heteroskedastic", if l.how == "homoskedastic" { "homoskedastic" } else { "heteroskedastic" })),
        (Some(_), _) => {}
        (None, _) => out.push("Note: no lambda selected; use lassoselect to choose one.".into()),
    }
    if l.selection == "adaptive" {
        out.push("  (adaptive lasso: final step shown)".into());
    }
    if let Some(k) = l.sel {
        if k + 1 == n && l.selection != "plugin" && l.selection != "none" {
            out.push("Note: selected lambda is the smallest on the grid; the minimum may not have been reached.".into());
        }
    }
    out.join("\n")
}
