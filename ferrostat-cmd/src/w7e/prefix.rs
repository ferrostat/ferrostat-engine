//! The `bayes:` prefix: translate a fitted estimator into a `bayesmh` model.
//!
//! The estimator runs first (maximum likelihood); its results give the
//! outcome, the starting values and the model family. Models with a
//! likelihood implemented here are sampled exactly: linear (normal),
//! logit, probit, cloglog, Poisson, negative binomial, ordered logit and
//! probit, multinomial logit, tobit, fractional and beta regression, GLMs,
//! and the multilevel versions of the linear, binary and count models with
//! random intercepts and slopes. Other estimators are sampled from the
//! posterior formed by the priors and a normal approximation of their
//! likelihood around the maximum (asymptotically equivalent).
//!
//! Default priors: normal(0, 10000) for coefficients, cutpoints and
//! log-scale parameters; igamma(0.01, 0.01) for variances.

use ferrostat_core::DataSet;

use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;

use super::text;

fn g17(v: f64) -> String {
    format!("{v:.17e}").parse::<f64>().map(|x| format!("{x}")).unwrap_or_else(|_| fmt_g(v, 17))
}

/// Split an estimator command line into (verb, varlist part, if/in text, options).
fn split_cmd(line: &str) -> (String, String, String, String) {
    let line = line.trim();
    let (main, opts) = match text::find_top(line, ',') {
        Some(i) => (&line[..i], line[i + 1..].trim().to_string()),
        None => (line, String::new()),
    };
    let ws = text::words(main);
    let verb = ws.first().cloned().unwrap_or_default();
    let mut vars = Vec::new();
    let mut k = 1;
    while k < ws.len() && ws[k] != "if" && ws[k] != "in" && !ws[k].starts_with('[') {
        vars.push(ws[k].clone());
        k += 1;
    }
    let ifin: Vec<String> = ws[k..].iter().filter(|w| !w.starts_with('[')).cloned().collect();
    (verb, vars.join(" "), ifin.join(" "), opts)
}

fn opt_val(opts: &str, full: &str, min: usize) -> Option<String> {
    text::options(opts).into_iter().find(|(n, _)| text::abbrev(n, full, min)).and_then(|(_, a)| a)
}

fn has_opt(opts: &str, full: &str, min: usize) -> bool {
    text::options(opts).iter().any(|(n, _)| text::abbrev(n, full, min))
}

pub struct Defaults {
    pub nvar: f64,
    pub ig: (f64, f64),
}

/// Translate the estimation results in `st.e` from `estline` into
/// `bayesmh` text; `bopts` are the prefix's options.
pub fn translate(ds: &DataSet, st: &mut Store, estline: &str, bopts: &str) -> Result<String, String> {
    let mut cmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "the estimation command did not store e(b)"))?;
    let v = st.e.matrices.get("V").cloned();
    // residual covariance of multivariate models (VAR) for the postestimation commands
    match st.e.matrices.get("Sigma").cloned() {
        Some(sg) => {
            st.matrices.insert("__bayes_Sigma".into(), sg);
        }
        None => {
            st.matrices.remove("__bayes_Sigma");
        }
    }
    let (verb, vars, ifin, eopts) = split_cmd(estline);
    let _ = verb;
    // default prior settings
    let mut d = Defaults { nvar: 10000.0, ig: (0.01, 0.01) };
    let mut pass = Vec::new();
    for (n, a) in text::options(bopts) {
        if text::abbrev(&n, "normalprior", 7) {
            if let Some(sd) = a.as_ref().and_then(|s| s.trim().parse::<f64>().ok()) {
                d.nvar = sd * sd;
            }
        } else if text::abbrev(&n, "igammaprior", 7) {
            if let Some(a) = a {
                let w: Vec<f64> = a.split(|c: char| c.is_whitespace() || c == ',').filter_map(|s| s.trim().parse().ok()).collect();
                if w.len() == 2 {
                    d.ig = (w[0], w[1]);
                }
            }
        } else {
            pass.push(match a {
                Some(a) => format!("{n}({a})"),
                None => n,
            });
        }
    }
    let np = |r: &str| format!("_dprior({r}, normal(0,{}))", fmt_g(d.nvar, 12));
    let ig = |r: &str| format!("_dprior({r}, igamma({},{}))", fmt_g(d.ig.0, 10), fmt_g(d.ig.1, 10));
    let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default().split_whitespace().next().unwrap_or("").to_string();
    let bval = |name: &str| b.col_index(name).map(|j| b.get(0, j));
    let mut inits: Vec<(String, f64)> = Vec::new();
    let mut add_coef_inits = |eq: &str, inits: &mut Vec<(String, f64)>| {
        for (j, c) in b.colnames.iter().enumerate() {
            if c.starts_with('/') || c.contains("/:") {
                continue;
            }
            let (e, n) = match c.split_once(':') {
                Some((e, n)) => (e, n),
                None => (eq, c.as_str()),
            };
            if e != eq || n.contains("b.") || n.starts_with("o.") || n.contains("o.") && n.split('.').next().is_some_and(|p| p.ends_with('o')) {
                continue;
            }
            inits.push((format!("{eq}:{n}"), b.get(0, j)));
        }
    };
    let nocons = if has_opt(&eopts, "noconstant", 4) { ", noconstant" } else { "" };
    let mut parts: Vec<String> = Vec::new();
    let mut title = String::new();
    let mut lik_eq_opts = String::new();
    let off = opt_val(&eopts, "offset", 3).map(|o| format!(" offset({o})")).unwrap_or_default();
    let expo = opt_val(&eopts, "exposure", 3).map(|o| format!(" exposure({o})")).unwrap_or_default();
    let sub = if off.is_empty() && expo.is_empty() { String::new() } else { format!(",{off}{expo}") };
    let me_split = |vars: &str| -> (String, Vec<(String, Vec<String>)>) {
        // "y x || id: z" -> fixed part and (group, slopes)
        let mut it = vars.split("||");
        let fixed = it.next().unwrap_or("").trim().to_string();
        let mut levels = Vec::new();
        for lv in it {
            let lv = lv.split(',').next().unwrap_or("");
            if let Some((g, s)) = lv.split_once(':') {
                let slopes: Vec<String> = s.split_whitespace().filter(|w| !w.starts_with('_')).map(String::from).collect();
                levels.push((g.trim().to_string(), slopes));
            }
        }
        (fixed, levels)
    };
    let fixed_family = |c: &str, eopts: &str| -> Option<&'static str> {
        Some(match c {
            "regress" | "mixed" | "xtreg" => "normal",
            "logit" | "logistic" | "melogit" | "xtlogit" => "logit",
            "probit" | "meprobit" | "xtprobit" => "probit",
            "cloglog" | "mecloglog" | "xtcloglog" => "cloglog",
            "poisson" | "mepoisson" | "xtpoisson" => "poisson",
            "meologit" | "xtologit" => "ologit",
            "meoprobit" | "xtoprobit" => "oprobit",
            "menbreg" => "_nbreg({lnalpha})",
            "glm" | "meglm" => {
                let fam = opt_val(eopts, "family", 3).unwrap_or_else(|| "gaussian".into()).to_lowercase();
                let link = opt_val(eopts, "link", 1).unwrap_or_default().to_lowercase();
                if fam.starts_with("gau") && (link.is_empty() || link.starts_with('i')) {
                    "normal"
                } else if fam.starts_with("bin") && (link.is_empty() || link.starts_with("logi")) && fam.trim() == "binomial" || fam.trim().starts_with("bern") {
                    "logit"
                } else if fam.starts_with("poi") && (link.is_empty() || link.starts_with("log")) {
                    "poisson"
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    };
    let fam = fixed_family(cmd.as_str(), &eopts);
    let multilevel = cmd.starts_with("me") || cmd == "mixed" || (cmd.starts_with("xt") && fam.is_some());
    if let (Some(fam), true) = (fam, multilevel) {
        // random intercepts (and slopes) by level
        let (fixed, mut levels) = me_split(&vars);
        if cmd.starts_with("xt") {
            if has_opt(&eopts, "fe", 2) || has_opt(&eopts, "pa", 2) {
                return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd);
            }
            let panel = st.tsset.as_ref().and_then(|t| t.0.clone()).ok_or_else(|| rc_err(459, "panel variable not set; use xtset"))?;
            levels.push((panel, Vec::new()));
        }
        if levels.is_empty() {
            return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd);
        }
        let mut terms = Vec::new();
        let mut re_priors = Vec::new();
        let mut k = 0;
        for (g, slopes) in &levels {
            let g = g.trim();
            if g.contains('>') || g == "_all" || g.contains('.') {
                return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd);
            }
            let nm = format!("U{k}");
            terms.push(format!("{nm}[{g}]"));
            re_priors.push(format!("_dprior({{{nm}}}, normal(0,{{{nm}:sigma2}}))"));
            re_priors.push(ig(&format!("{{{nm}:sigma2}}")));
            inits.push((format!("{nm}:sigma2"), 0.5));
            k += 1;
            for s in slopes {
                let nm = format!("U{k}");
                terms.push(format!("c.{s}#{nm}[{g}]"));
                re_priors.push(format!("_dprior({{{nm}}}, normal(0,{{{nm}:sigma2}}))"));
                re_priors.push(ig(&format!("{{{nm}:sigma2}}")));
                inits.push((format!("{nm}:sigma2"), 0.1));
                k += 1;
            }
        }
        let fw = text::words(&fixed);
        let dep = fw.first().cloned().unwrap_or_default();
        let eq_name = if b.colnames.iter().any(|c| c.starts_with(&format!("{dep}:"))) { dep.clone() } else { dep.clone() };
        add_coef_inits(&eq_name, &mut inits);
        let mut extra_priors = Vec::new();
        if fam == "ologit" || fam == "oprobit" {
            let ncut = b.colnames.iter().filter(|c| c.contains("cut")).count();
            for c in 1..=ncut {
                extra_priors.push(np(&format!("{{cut{c}}}")));
                if let Some(x) = bval(&format!("/:cut{c}")).or_else(|| bval(&format!("cut{c}:_cons"))) {
                    inits.push((format!("cut{c}"), x));
                }
            }
        }
        if fam.starts_with("_nbreg") {
            extra_priors.push(np("{lnalpha}"));
            if let Some(x) = bval("/:lnalpha").or_else(|| bval("lnalpha:_cons")) {
                inits.push(("lnalpha".into(), x.max(-5.0)));
            }
        }
        let lik = if fam == "normal" {
            inits.push(("e.".to_string() + &dep + ":sigma2", st.e.get("sigma_e").map(|s| s * s).unwrap_or(1.0)));
            format!("likelihood(normal({{e.{dep}:sigma2}}){sub})")
        } else {
            format!("likelihood({fam}{sub})")
        };
        parts.push(format!("{} {} {}{}", fixed, terms.join(" "), ifin, ""));
        parts.push(format!(", {lik}"));
        parts.push(np(&format!("{{{dep}:}}")));
        if fam == "normal" {
            parts.push(ig(&format!("{{e.{dep}:sigma2}}")));
        }
        parts.extend(re_priors);
        parts.extend(extra_priors);
        title = match fam {
            "normal" => "multilevel regression",
            "logit" => "multilevel logistic regression",
            "probit" => "multilevel probit regression",
            "poisson" => "multilevel Poisson regression",
            "ologit" => "multilevel ordered logistic regression",
            "oprobit" => "multilevel ordered probit regression",
            "_nbreg({lnalpha})" => "multilevel negative binomial regression",
            _ => "multilevel regression",
        }
        .into();
        if !nocons.is_empty() {
            parts.push("noconstant".into());
        }
    } else {
        let simple = |fam: &str, parts: &mut Vec<String>| {
            parts.push(format!("{vars} {ifin}, likelihood({fam}{sub}){}", if nocons.is_empty() { "" } else { " noconstant" }));
            parts.push(np(&format!("{{{depvar}:}}")));
        };
        match (cmd.as_str(), fam) {
            (_, Some("normal")) => {
                simple("normal({sigma2})", &mut parts);
                parts.push(ig("{sigma2}"));
                add_coef_inits(&depvar, &mut inits);
                let rmse = st.e.get("rmse").or_else(|| st.e.get("phi").map(|p| p.sqrt())).unwrap_or(1.0);
                inits.push(("sigma2".into(), rmse * rmse));
                title = "linear regression".into();
            }
            (_, Some(f)) => {
                simple(f, &mut parts);
                add_coef_inits(&depvar, &mut inits);
                title = match f {
                    "logit" => "logistic regression",
                    "probit" => "probit regression",
                    "cloglog" => "complementary log-log regression",
                    _ => "Poisson regression",
                }
                .into();
                if cmd == "glm" {
                    title = "generalized linear model".into();
                }
            }
            ("ologit" | "oprobit", _) => {
                let f = if cmd == "ologit" { "ologit" } else { "oprobit" };
                parts.push(format!("{vars} {ifin}, likelihood({f})"));
                parts.push(np(&format!("{{{depvar}:}}")));
                let ncut = b.colnames.iter().filter(|c| c.contains("cut")).count();
                for c in 1..=ncut {
                    parts.push(np(&format!("{{cut{c}}}")));
                    if let Some(x) = bval(&format!("/:cut{c}")).or_else(|| bval(&format!("cut{c}:_cons"))) {
                        inits.push((format!("cut{c}"), x));
                    }
                }
                add_coef_inits(&depvar, &mut inits);
                title = if f == "ologit" { "ordered logistic regression" } else { "ordered probit regression" }.into();
            }
            ("mlogit", _) => {
                // base outcome: equation whose coefficients are all zero
                let mut eqs: Vec<String> = Vec::new();
                for c in &b.colnames {
                    if let Some((e, _)) = c.split_once(':') {
                        if !eqs.contains(&e.to_string()) {
                            eqs.push(e.to_string());
                        }
                    }
                }
                let base = eqs.iter().find(|e| b.colnames.iter().enumerate().filter(|(_, c)| c.starts_with(&format!("{e}:"))).all(|(j, _)| b.get(0, j) == 0.0)).cloned();
                let bo = st.e.get("baseout").map(|x| fmt_g(x, 10)).or(base.clone());
                parts.push(format!("{vars} {ifin}, likelihood(mlogit{})", bo.as_ref().map(|b| format!(", baseoutcome({b})")).unwrap_or_default()));
                for e in &eqs {
                    if Some(e) != base.as_ref() {
                        parts.push(np(&format!("{{{e}:}}")));
                        for (j, c) in b.colnames.iter().enumerate() {
                            if let Some((ee, n)) = c.split_once(':') {
                                if ee == e && !n.contains("b.") && !n.starts_with("o.") {
                                    inits.push((c.clone(), b.get(0, j)));
                                }
                            }
                        }
                    }
                }
                title = "multinomial logistic regression".into();
            }
            ("nbreg", _) if opt_val(&eopts, "dispersion", 1).map_or(true, |x| x.starts_with('m')) => {
                parts.push(format!("{vars} {ifin}, likelihood(_nbreg({{lnalpha}}){sub})"));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(np("{lnalpha}"));
                add_coef_inits(&depvar, &mut inits);
                if let Some(x) = bval("/:lnalpha").or_else(|| bval("lnalpha:_cons")) {
                    inits.push(("lnalpha".into(), x.max(-5.0)));
                }
                title = "negative binomial regression".into();
            }
            ("tobit", _) => {
                let ll = opt_val(&eopts, "ll", 2).unwrap_or_else(|| if has_opt(&eopts, "ll", 2) { "min".into() } else { ".".into() });
                let ul = opt_val(&eopts, "ul", 2).unwrap_or_else(|| if has_opt(&eopts, "ul", 2) { "max".into() } else { ".".into() });
                let y = crate::util::col(ds, &depvar)?;
                let fix = |s: &str| -> String {
                    match s {
                        "min" => fmt_g(y.iter().copied().filter(|v| !ferrostat_core::missing::is_missing(*v)).fold(f64::INFINITY, f64::min), 17),
                        "max" => fmt_g(y.iter().copied().filter(|v| !ferrostat_core::missing::is_missing(*v)).fold(f64::NEG_INFINITY, f64::max), 17),
                        o => o.to_string(),
                    }
                };
                parts.push(format!("{vars} {ifin}, likelihood(_tobit({}, {}, {{sigma2}}))", fix(&ll), fix(&ul)));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(ig("{sigma2}"));
                add_coef_inits(&depvar, &mut inits);
                let s = bval("/:sigma").or_else(|| bval("sigma:_cons")).or_else(|| st.e.get("sigma")).unwrap_or(1.0);
                inits.push(("sigma2".into(), s * s));
                title = "tobit regression".into();
            }
            ("qreg", _) => {
                let q = st.e.get("q").or_else(|| opt_val(&eopts, "quantile", 1).and_then(|s| s.trim().parse::<f64>().ok()).map(|v| if v > 1.0 { v / 100.0 } else { v })).unwrap_or(0.5);
                parts.push(format!("{vars} {ifin}, likelihood(_ald({q}, {{sigma}})){}", if nocons.is_empty() { "" } else { " noconstant" }));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(ig("{sigma}"));
                add_coef_inits(&depvar, &mut inits);
                let n = st.e.get("N").unwrap_or(1.0).max(1.0);
                let s = st.e.get("sum_adev").map(|a| a / n).unwrap_or(1.0).max(1e-3);
                inits.push(("sigma".into(), s));
                title = "quantile regression".into();
            }
            ("intreg", _) => {
                let w: Vec<&str> = vars.split_whitespace().collect();
                if w.len() < 2 {
                    return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd);
                }
                let rest = w[2..].join(" ");
                parts.push(format!("({}: _ {rest}{}) {ifin}, likelihood(_interval({}, {}, {{sigma2}}))", w[0], if nocons.is_empty() { "" } else { ", noconstant" }, w[0], w[1]));
                parts.push(np(&format!("{{{}:}}", w[0])));
                parts.push(ig("{sigma2}"));
                add_coef_inits(w[0], &mut inits);
                if let Some(x) = bval("/:lnsigma") {
                    inits.push(("sigma2".into(), (2.0 * x).exp()));
                }
                title = "interval regression".into();
            }
            ("truncreg", _) => {
                let ll = opt_val(&eopts, "ll", 2).unwrap_or_else(|| ".".into());
                let ul = opt_val(&eopts, "ul", 2).unwrap_or_else(|| ".".into());
                parts.push(format!("{vars} {ifin}, likelihood(_trunc({ll}, {ul}, {{sigma2}})){}", if nocons.is_empty() { "" } else { " noconstant" }));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(ig("{sigma2}"));
                add_coef_inits(&depvar, &mut inits);
                if let Some(x) = bval("/:sigma") {
                    inits.push(("sigma2".into(), x * x));
                }
                title = "truncated regression".into();
            }
            ("tpoisson", _) => {
                let ll = opt_val(&eopts, "ll", 2).unwrap_or_else(|| "0".into());
                parts.push(format!("{vars} {ifin}, likelihood(_tpoisson({ll}))"));
                parts.push(np(&format!("{{{depvar}:}}")));
                add_coef_inits(&depvar, &mut inits);
                title = "truncated Poisson regression".into();
            }
            ("zip" | "zinb", _) => {
                let infl = opt_val(&eopts, "inflate", 3).unwrap_or_else(|| "_cons".into());
                let infl = infl.replace("_cons", "");
                let lik = if cmd == "zinb" { "_zinb({lnalpha})" } else { "_zip" };
                parts.push(format!("({depvar}: {vars}) (inflate: _ {infl}) {ifin}, likelihood({lik})"));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(np("{inflate:}"));
                add_coef_inits(&depvar, &mut inits);
                add_coef_inits("inflate", &mut inits);
                if cmd == "zinb" {
                    parts.push(np("{lnalpha}"));
                    if let Some(x) = bval("/:lnalpha") {
                        inits.push(("lnalpha".into(), x.max(-5.0)));
                    }
                }
                title = if cmd == "zinb" { "zero-inflated negative binomial regression" } else { "zero-inflated Poisson regression" }.into();
            }
            ("betareg", _) => {
                parts.push(format!("{vars} {ifin}, likelihood(_beta({{scale:_cons}}))"));
                parts.push(np(&format!("{{{depvar}:}}")));
                parts.push(np("{scale:_cons}"));
                add_coef_inits(&depvar, &mut inits);
                if let Some(x) = bval("scale:_cons") {
                    inits.push(("scale:_cons".into(), x));
                }
                title = "beta regression".into();
            }
            ("streg" | "weibull" | "ereg" | "exponential", _) if !has_opt(&eopts, "time", 4) && matches!(opt_val(&eopts, "distribution", 1).unwrap_or_default().to_lowercase().as_str(), d if d.starts_with("w") || d.starts_with("e")) && !has_opt(&eopts, "frailty", 4) => {
                let weib = opt_val(&eopts, "distribution", 1).unwrap_or_default().to_lowercase().starts_with('w');
                cmd = "streg".into();
                let lik = if weib { "_weibull({ln_p}, _d, _t0)" } else { "_exponential(_d, _t0)" };
                parts.push(format!("_t {vars} {ifin}, likelihood({lik}){}", if nocons.is_empty() { "" } else { " noconstant" }));
                parts.push(np("{_t:}"));
                add_coef_inits("_t", &mut inits);
                if weib {
                    parts.push(np("{ln_p}"));
                    if let Some(x) = bval("/:ln_p") {
                        inits.push(("ln_p".into(), x));
                    }
                }
                title = if weib { "Weibull PH regression" } else { "exponential PH regression" }.into();
            }
            ("mvreg" | "var", _) => {
                // multivariate normal outcomes with an inverse-Wishart covariance
                let mut eqs: Vec<String> = Vec::new();
                for c in &b.colnames {
                    if let Some((e, _)) = c.split_once(':') {
                        if !eqs.contains(&e.to_string()) {
                            eqs.push(e.to_string());
                        }
                    }
                }
                let k = eqs.len();
                if k < 2 {
                    return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd);
                }
                let mut eqtxt = String::new();
                for e in &eqs {
                    let terms: Vec<String> = b.colnames.iter().filter_map(|c| c.split_once(':')).filter(|(ee, n)| ee == e && *n != "_cons").map(|(_, n)| n.to_string()).collect();
                    let has_cons = b.colnames.iter().any(|c| c == &format!("{e}:_cons"));
                    eqtxt.push_str(&format!("({e} {}{}) ", terms.join(" "), if has_cons { "" } else { ", noconstant" }));
                    parts.push(String::new());
                    add_coef_inits(e, &mut inits);
                }
                parts.retain(|p| !p.is_empty());
                let ifc = if cmd == "var" { String::new() } else { ifin.clone() };
                parts.push(format!("{eqtxt}{ifc}, likelihood(mvnormal({{Sigma,m}}))"));
                for e in &eqs {
                    parts.push(np(&format!("{{{e}:}}")));
                }
                parts.push(format!("prior({{Sigma,m}}, iwishart({k}, {}, I({k})))", k + 1));
                let pd = st.e.matrices.get("Sigma").map_or(false, |sg| {
                    let dg: f64 = (0..k).map(|i| sg.get(i, i)).product();
                    sg.cholesky().is_ok() && sg.det() > 1e-8 * dg.abs()
                });
                if let (Some(sg), true) = (st.e.matrices.get("Sigma"), pd) {
                    for i in 0..k {
                        for j in 0..=i {
                            inits.push((format!("Sigma_{}_{}", i + 1, j + 1), sg.get(i, j)));
                        }
                    }
                }
                title = if cmd == "var" { "vector autoregression".into() } else { "multivariate regression".into() };
            }
            ("fracreg", _) => {
                let probit = vars.split_whitespace().next() == Some("probit");
                let rest: String = vars.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
                parts.push(format!("{rest} {ifin}, likelihood({})", if probit { "_fracprobit" } else { "_fraclogit" }));
                parts.push(np(&format!("{{{depvar}:}}")));
                add_coef_inits(&depvar, &mut inits);
                title = "fractional response regression".into();
            }
            _ => return quad(ds, st, &b, v.as_ref(), &depvar, &d, &pass, &cmd),
        }
    }
    let mut out = parts.join(" ");
    if !inits.is_empty() {
        out.push_str(&format!(" initial({})", inits.iter().filter(|(_, v)| v.is_finite()).map(|(n, v)| format!("{{{n}}} {}", g17(*v))).collect::<Vec<_>>().join(" ")));
    }
    for p in &pass {
        out.push(' ');
        out.push_str(p);
    }
    out.push_str(&format!(" _esample _prefix({cmd}) _title({title})"));
    Ok(out)
}

/// Normal approximation of the likelihood around the maximum-likelihood
/// fit, with default priors.
#[allow(clippy::too_many_arguments)]
fn quad(_ds: &DataSet, st: &mut Store, b: &Matrix, v: Option<&Matrix>, depvar: &str, d: &Defaults, pass: &[String], cmd: &str) -> Result<String, String> {
    let v = v.ok_or_else(|| rc_err(301, "the estimation command did not store e(V)"))?;
    let ll = st.e.get("ll").unwrap_or(0.0);
    let keep: Vec<usize> = (0..b.cols).filter(|&j| v.get(j, j) > 0.0 && v.get(j, j).is_finite()).collect();
    if keep.is_empty() {
        return Err(rc_err(504, "no estimable parameters"));
    }
    let names: Vec<String> = keep.iter().map(|&j| {
        let c = &b.colnames[j];
        let c = c.strip_prefix("/:").unwrap_or(c);
        let c = c.replace(' ', "");
        if c.contains(':') || c.starts_with('/') { c.trim_start_matches('/').to_string() } else if depvar.is_empty() { c } else { format!("{depvar}:{c}") }
    }).collect();
    // braces in names would break parameter references
    let names: Vec<String> = names.into_iter().map(|n| n.replace(['{', '}', ','], "_")).collect();
    let bb = Matrix::row_vector(&keep.iter().map(|&j| b.get(0, j)).collect::<Vec<_>>()).with_names(vec!["y1".into()], names.clone());
    let mut vv = Matrix::zeros(keep.len(), keep.len());
    for (a, &i) in keep.iter().enumerate() {
        for (c, &j) in keep.iter().enumerate() {
            vv[(a, c)] = v.get(i, j);
        }
    }
    let vv = vv.with_names(names.clone(), names.clone());
    st.matrices.insert("__bayes_qb".into(), bb.clone());
    st.matrices.insert("__bayes_qV".into(), vv);
    let mut out = format!(", likelihood(_quad(__bayes_qb, __bayes_qV, {}))", g17(ll));
    let positive = |n: &str| {
        let l = n.to_lowercase();
        (l.starts_with("var(") || l.starts_with("sigma") || l.contains(":sigma") || l.starts_with("sd(") || l.contains(":var(")) && !l.starts_with("ln")
    };
    let mut inits = Vec::new();
    for (k, n) in names.iter().enumerate() {
        let x = bb.get(0, k);
        if positive(n) && x > 0.0 {
            out.push_str(&format!(" _dprior({{{n}}}, igamma({},{}))", fmt_g(d.ig.0, 10), fmt_g(d.ig.1, 10)));
        } else {
            out.push_str(&format!(" _dprior({{{n}}}, normal(0,{}))", fmt_g(d.nvar, 12)));
        }
        inits.push(format!("{{{n}}} {}", g17(x)));
    }
    out.push_str(&format!(" initial({})", inits.join(" ")));
    for p in pass {
        out.push(' ');
        out.push_str(p);
    }
    let n = st.e.get("N").unwrap_or(0.0);
    out.push_str(&format!(" _nobs({}) _prefix({cmd}) _title({} (approximate likelihood))", n, cmd));
    Ok(out)
}

/// Estimators sampled without a preliminary maximum-likelihood fit:
/// `gnbreg depvar indepvars [, lnalpha(varlist) offset() exposure()]`.
pub fn direct(estline: &str, bopts: &str) -> Option<String> {
    let (verb, vars, ifin, eopts) = split_cmd(estline);
    if verb != "gnbreg" {
        return None;
    }
    let ws: Vec<&str> = vars.split_whitespace().collect();
    let depvar = ws.first()?.to_string();
    let la: Vec<String> = opt_val(&eopts, "lnalpha", 3).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let mut ex = "{lnalpha:_cons}".to_string();
    for v in &la {
        ex.push_str(&format!(" + {{lnalpha:{v}}}*{v}"));
    }
    let off = opt_val(&eopts, "offset", 3).map(|o| format!(", offset({o})")).or_else(|| opt_val(&eopts, "exposure", 3).map(|o| format!(", exposure({o})"))).unwrap_or_default();
    let mut nvar = 10000.0;
    let mut pass = Vec::new();
    for (n, a) in text::options(bopts) {
        if text::abbrev(&n, "normalprior", 7) {
            if let Some(sd) = a.as_ref().and_then(|s| s.trim().parse::<f64>().ok()) {
                nvar = sd * sd;
            }
        } else {
            pass.push(match a {
                Some(a) => format!("{n}({a})"),
                None => n,
            });
        }
    }
    let mut out = format!("{vars} {ifin}, likelihood(_gnbreg({ex}){off}) _dprior({{{depvar}:}}, normal(0,{nv})) _dprior({{lnalpha:}}, normal(0,{nv}))", nv = fmt_g(nvar, 12));
    for p in pass {
        out.push(' ');
        out.push_str(&p);
    }
    out.push_str(" _prefix(gnbreg) _title(generalized negative binomial regression)");
    Some(out)
}
