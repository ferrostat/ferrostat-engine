//! `estat gof` after poisson: deviance and Pearson goodness-of-fit tests.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_expression;

use crate::dist;
use crate::eval::{rc_err, Env};
use crate::store::{Results, Store};

pub fn estat_gof(ds: &DataSet, st: &mut Store) -> Result<String, String> {
    let cmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if cmd != "poisson" {
        return Err(rc_err(321, "estat gof not valid"));
    }
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let ye = parse_expression(&dep).map_err(|e| rc_err(198, e.message))?;
    let wexp = st.e.macros.get("wexp").cloned().filter(|w| !w.trim().is_empty());
    let fweight = st.e.macros.get("wtype").is_some_and(|w| w.starts_with("fw"));
    let we = match (&wexp, fweight) {
        (Some(w), true) => Some(parse_expression(w.trim_start_matches('=').trim()).map_err(|e| rc_err(198, e.message))?),
        _ => None,
    };
    let env = Env::with_store(ds, st);
    let (mut dev, mut pear, mut n) = (0.0, 0.0, 0.0);
    for i in 0..ds.n_obs.min(sample.len()) {
        if !sample[i] {
            continue;
        }
        let y = env.eval(&ye, i)?.num()?;
        let mu = crate::postest::default_prediction(st, &env, &b, i)?;
        if is_missing(y) || is_missing(mu) {
            continue;
        }
        let w = match &we {
            Some(e) => env.eval(e, i)?.num()?,
            None => 1.0,
        };
        dev += w * 2.0 * (if y > 0.0 { y * (y / mu).ln() } else { 0.0 } - (y - mu));
        pear += w * (y - mu) * (y - mu) / mu;
        n += w;
    }
    let k = st.e.get("rank").unwrap_or(b.cols as f64);
    let df = st.e.get("N").unwrap_or(n) - k;
    let (pd, pp) = (dist::chi2tail(df, dev), dist::chi2tail(df, pear));
    let r: &mut Results = st.begin_r();
    r.set("df", df);
    r.set("chi2_d", dev);
    r.set("p_d", pd);
    r.set("chi2_p", pear);
    r.set("p_p", pp);
    let dfs = format!("Prob > chi2({})", crate::format::fmt_g(df, 12));
    let f4 = |v: f64| if is_missing(v) || v.is_nan() { ".".to_string() } else { format!("{v:.4}") };
    Ok([
        String::new(),
        format!("         Deviance goodness-of-fit = {:>9}", f4(dev)),
        format!("         {dfs:<24} = {:>9}", f4(pd)),
        String::new(),
        format!("         Pearson goodness-of-fit  = {:>9}", f4(pear)),
        format!("         {dfs:<24} = {:>9}", f4(pp)),
    ]
    .join("\n"))
}
