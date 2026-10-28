//! Conditional logistic regression (`clogit`) with the exact recursive
//! denominator for groups with several positive outcomes, and the
//! alternative-specific conditional logit (`asclogit`) built on it.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::choice_mixed::{f2, f4};
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec, Likelihood, MlResult, Vce};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

/// Conditional logit likelihood over groups of observations.
pub struct CondLogit<'a> {
    pub x: Vec<&'a [f64]>,
    pub y: &'a [f64],
    pub off: &'a [f64],
    pub groups: &'a [Vec<usize>],
}

impl<'a> CondLogit<'a> {
    /// Log likelihood, gradient and Hessian of one group.
    fn group(&self, obs: &[usize], b: &[f64]) -> (f64, Vec<f64>, Matrix) {
        let p = self.x.len();
        let eta: Vec<f64> = obs.iter().map(|&t| self.off[t] + self.x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>()).collect();
        let kk = obs.iter().filter(|&&t| self.y[t] != 0.0).count();
        let c = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut f = vec![0.0; kk + 1];
        let mut gv = vec![vec![0.0; p]; kk + 1];
        let mut hv = vec![Matrix::zeros(p, p); kk + 1];
        f[0] = 1.0;
        for (k, &t) in obs.iter().enumerate() {
            let e = (eta[k] - c).exp();
            let xt: Vec<f64> = self.x.iter().map(|col| col[t]).collect();
            for j in (1..=kk.min(k + 1)).rev() {
                let fo = f[j - 1];
                let go = gv[j - 1].clone();
                {
                    let ho = hv[j - 1].clone();
                    let hj = &mut hv[j];
                    for a in 0..p {
                        for d in 0..p {
                            hj[(a, d)] += e * (ho[(a, d)] + go[a] * xt[d] + xt[a] * go[d] + fo * xt[a] * xt[d]);
                        }
                    }
                }
                for a in 0..p {
                    gv[j][a] += e * (go[a] + fo * xt[a]);
                }
                f[j] += fo * e;
            }
        }
        let fk = f[kk];
        let mut ll = -fk.ln() - kk as f64 * c;
        let mut g = vec![0.0; p];
        for (k, &t) in obs.iter().enumerate() {
            if self.y[t] != 0.0 {
                ll += eta[k];
                for a in 0..p {
                    g[a] += self.x[a][t];
                }
            }
        }
        let mut h = Matrix::zeros(p, p);
        for a in 0..p {
            g[a] -= gv[kk][a] / fk;
            for d in 0..p {
                h[(a, d)] = hv[kk][(a, d)] / fk - gv[kk][a] * gv[kk][d] / (fk * fk);
            }
        }
        (ll, g, h)
    }
}

impl<'a> Likelihood for CondLogit<'a> {
    fn k(&self) -> usize {
        self.x.len()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let p = self.k();
        let mut ll = 0.0;
        let mut g = vec![0.0; p];
        let mut h = Matrix::zeros(p, p);
        let mut sc = Vec::new();
        for obs in self.groups {
            let (l, gg, hh) = self.group(obs, b);
            ll += l;
            for a in 0..p {
                g[a] += gg[a];
                for d in 0..p {
                    h[(a, d)] += hh[(a, d)];
                }
            }
            if need_scores {
                sc.push(gg);
            }
        }
        if !ll.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; p], Matrix::identity(p), sc);
        }
        (ll, g, h, sc)
    }
}

/// Drop groups whose outcomes are all positive or all zero; returns the
/// kept rows flag and (groups dropped, observations dropped).
fn informative(y: &[f64], groups: &[Vec<usize>]) -> (Vec<bool>, usize, usize) {
    let mut keep = vec![true; y.len()];
    let (mut ng, mut no) = (0, 0);
    for g in groups {
        let pos = g.iter().filter(|&&t| y[t] != 0.0).count();
        if pos == 0 || pos == g.len() {
            ng += 1;
            no += g.len();
            for &t in g {
                keep[t] = false;
            }
        }
    }
    (keep, ng, no)
}

/// Fit by Newton-Raphson from zero.
fn fit(m: &CondLogit) -> Result<MlResult, String> {
    est::maximize(m, vec![0.0; m.k()], 300)
}

fn vce_of(res: &MlResult, vce: &Vce, cluster: Option<&[f64]>, ngroups: usize) -> (Matrix, Option<usize>) {
    let (hinv, _) = res.info.sym_inverse(1e-14);
    match vce {
        Vce::Robust | Vce::Cluster(_) => {
            // scores are per group: cluster on the group (or the given variable)
            let (v, g) = est::sandwich(&hinv, &res.scores, cluster, 1.0);
            let gf = g as f64;
            let _ = ngroups;
            (v.scale(gf / (gf - 1.0)), Some(g))
        }
        _ => (hinv, None),
    }
}

pub struct Clogit;

impl CommandHandler for Clogit {
    fn name(&self) -> &'static str {
        "clogit"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("clogit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let gopt = cmd.opt("group", 2).or_else(|| cmd.opt("strata", 2)).ok_or_else(|| rc_err(198, "group() required"))?;
        let gvar = crate::varlist::resolve_name(ds, gopt.trim())?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let level = util::level(cmd, st)?;
        let mut extra = vec![gvar.clone()];
        if let Vce::Cluster(c) = &vce {
            extra.push(crate::varlist::resolve_name(ds, c)?);
        }
        let offv = match cmd.opt("offset", 3) {
            Some(o) => Some(crate::varlist::resolve_name(ds, o.trim())?),
            None => None,
        };
        extra.extend(offv.iter().cloned());
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: true, weights: "", extra })?;
        let (_, groups) = crate::xtnl::panel_groups(ds, &gvar, &d.rows)?;
        let (keep, ngd, nod) = informative(&d.y, &groups);
        let mut notes = Vec::new();
        if ngd > 0 {
            notes.push(format!(
                "note: {} group{} ({} obs) dropped because of all positive or\n      all negative outcomes.",
                util::commas(ngd),
                if ngd == 1 { "" } else { "s" },
                util::commas(nod)
            ));
        }
        let mut d = crate::xtnl::subset_design(&d, &keep);
        if d.n() == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let (_, groups) = crate::xtnl::panel_groups(ds, &gvar, &d.rows)?;
        crate::xtnl::omit_within(&mut d, &groups, &mut notes);
        let multiple = groups.iter().any(|g| g.iter().filter(|&&t| d.y[t] != 0.0).count() > 1);
        if multiple {
            notes.insert(0, "note: multiple positive outcomes within groups encountered.".into());
        }
        let off: Vec<f64> = match &offv {
            Some(o) => {
                let c = util::col(ds, o)?;
                d.rows.iter().map(|&i| c[i]).collect()
            }
            None => vec![0.0; d.n()],
        };
        let act = d.active();
        let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let model = CondLogit { x: cols, y: &d.y, off: &off, groups: &groups };
        let res = fit(&model)?;
        let ll0 = res.iterations[0];
        // cluster ids per group (scores are per group)
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Cluster(c) => {
                let cv = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
                Some(groups.iter().map(|g| cv[d.rows[g[0]]]).collect())
            }
            _ => None,
        };
        let (vt, n_clust) = vce_of(&res, &vce, cl.as_deref(), groups.len());
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vt, &act, k);
        let robust = vce != Vce::Oim;
        let (chi2, df_m) = if robust { est::wald_all(&d, &b, &v) } else { (2.0 * (res.ll - ll0), act.len() as f64) };
        let pval = dist::chi2tail(df_m, chi2);
        let r2p = 1.0 - res.ll / ll0;
        let mut out = Vec::new();
        out.extend(notes.iter().cloned());
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        out.push("Conditional (fixed-effects) logistic regression".into());
        out.push(String::new());
        let hr = |l: &str, k: &str, v: String| format!("{:<56}{:<13} = {:>6}", l, k, v);
        let n = d.n();
        out.push(hr("", "Number of obs", util::commas(n)));
        out.push(hr("", &format!("{} chi2({})", if robust { "Wald" } else { "LR" }, fmt_g(df_m, 9)), f2(chi2)));
        out.push(hr("", "Prob > chi2", f4(pval)));
        out.push(hr(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(res.ll, 10)), "Pseudo R2", f4(r2p)));
        if let Some(g) = n_clust {
            out.push(String::new());
            let cvar = match &vce {
                Vce::Cluster(c) => c.clone(),
                _ => gvar.clone(),
            };
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), cvar)));
        }
        out.push(String::new());
        let eform = cmd.has_opt("or", 2);
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: if eform { Some("Odds ratio") } else { None }, coef_title: "Coefficient" }));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let ngroups = groups.len();
        est::post(st, "clogit", &d, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_0", ll0);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", pval);
            e.set("r2_p", r2p);
            e.set("N_drop", nod as f64);
            e.set("N_group_drop", ngd as f64);
            e.set("k", act.len() as f64);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            e.set("N_group", ngroups as f64);
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            e.set_str("group", gvar.clone());
            e.set_str("multiple", if multiple { "multiple" } else { "" });
            e.set_str("title", "Conditional (fixed-effects) logistic regression");
            e.set_str("chi2type", if robust { "Wald" } else { "LR" });
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("predict", "clogit_p");
            if let Some(o) = &offv {
                e.set_str("offset", o.clone());
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// asclogit
// ---------------------------------------------------------------------------

pub struct Asclogit;

impl CommandHandler for Asclogit {
    fn name(&self) -> &'static str {
        "asclogit"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("asclogit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let case = crate::varlist::resolve_name(ds, cmd.opt("case", 4).ok_or_else(|| rc_err(198, "case() required"))?.trim())?;
        let alt = crate::varlist::resolve_name(ds, cmd.opt("alternatives", 3).ok_or_else(|| rc_err(198, "alternatives() required"))?.trim())?;
        let casevars: Vec<String> = match cmd.opt("casevars", 5) {
            Some(c) => crate::varlist::expand(ds, c.trim())?,
            None => vec![],
        };
        let nocons = cmd.has_opt("noconstant", 3);
        let level = util::level(cmd, st)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let mut extra = vec![case.clone(), alt.clone()];
        extra.extend(casevars.iter().cloned());
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: true, weights: "", extra })?;
        let (_, groups) = crate::xtnl::panel_groups(ds, &case, &d.rows)?;
        // each case must have exactly one chosen alternative
        for g in &groups {
            let pos = g.iter().filter(|&&t| d.y[t] != 0.0).count();
            if pos != 1 {
                return Err(rc_err(459, format!("variable {dep} must have exactly one positive outcome per case")));
            }
        }
        let altc = util::col(ds, &alt)?;
        let altvar = ds.get_var(&alt).unwrap().clone();
        let mut alts: Vec<f64> = d.rows.iter().map(|&i| altc[i]).collect();
        alts.sort_by(f64::total_cmp);
        alts.dedup();
        let label_of = |a: f64| -> String {
            let i = d.rows.iter().position(|&r| altc[r] == a).unwrap();
            util::cell_text(ds, &altvar, d.rows[i], true)
        };
        // base: the most frequently chosen alternative, or basealternative()
        let base = match cmd.opt("basealternative", 4) {
            Some(bv) => {
                let bv = bv.trim();
                match bv.parse::<f64>() {
                    Ok(x) => x,
                    Err(_) => *alts.iter().find(|a| label_of(**a) == bv).ok_or_else(|| rc_err(198, format!("basealternative({bv}) not found")))?,
                }
            }
            None => {
                let mut best = (0usize, alts[0]);
                for &a in &alts {
                    let c = (0..d.n()).filter(|&t| d.y[t] != 0.0 && altc[d.rows[t]] == a).count();
                    if c > best.0 {
                        best = (c, a);
                    }
                }
                best.1
            }
        };
        let n = d.n();
        let mut names: Vec<String> = Vec::new();
        let mut cols: Vec<Vec<f64>> = Vec::new();
        let mut cons_idx: Vec<bool> = Vec::new();
        for (j, nm) in d.names.iter().enumerate() {
            if d.omitted[j] {
                continue;
            }
            names.push(format!("{alt}:{nm}"));
            cols.push(d.cols[j].clone());
            cons_idx.push(false);
        }
        let cvals: Vec<Vec<f64>> = casevars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mut altlabels = Vec::new();
        for &a in &alts {
            if a == base {
                continue;
            }
            let lab = label_of(a);
            altlabels.push(lab.clone());
            let dummy: Vec<f64> = d.rows.iter().map(|&r| if altc[r] == a { 1.0 } else { 0.0 }).collect();
            for (cv, cn) in cvals.iter().zip(&casevars) {
                names.push(format!("{lab}:{cn}"));
                cols.push(d.rows.iter().zip(&dummy).map(|(&r, dm)| dm * cv[r]).collect());
                cons_idx.push(false);
            }
            if !nocons {
                names.push(format!("{lab}:_cons"));
                cols.push(dummy);
                cons_idx.push(true);
            }
        }
        let off = vec![0.0; n];
        let cr: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let model = CondLogit { x: cr, y: &d.y, off: &off, groups: &groups };
        let res = fit(&model)?;
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Cluster(c) => {
                let cv = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
                Some(groups.iter().map(|g| cv[d.rows[g[0]]]).collect())
            }
            _ => None,
        };
        let (v, n_clust) = vce_of(&res, &vce, cl.as_deref(), groups.len());
        let p = names.len();
        let widx: Vec<usize> = (0..p).filter(|&j| !cons_idx[j]).collect();
        let (chi2, df_m) = crate::heckman::wald(&res.b, &v, &widx);
        let pval = dist::chi2tail(df_m, chi2);
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        let mut out = Vec::new();
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        let hr = |l: &str, k: &str, v: String| format!("{:<47}{:<18} = {:>10}", l, k, v);
        out.push(hr("Alternative-specific conditional logit", "Number of obs", util::commas(n)));
        out.push(hr(&format!("Case ID variable: {case}"), "Number of cases", util::commas(groups.len())));
        out.push(String::new());
        out.push(hr(&format!("Alternatives variable: {alt}"), "Alts per case: min", util::commas(*sizes.iter().min().unwrap_or(&0))));
        out.push(hr("", "               avg", format!("{:.1}", n as f64 / groups.len() as f64)));
        out.push(hr("", "               max", util::commas(*sizes.iter().max().unwrap_or(&0))));
        out.push(String::new());
        let robust = vce != Vce::Oim;
        out.push(format!("{:<50}{:<15} = {:>10}", "", format!("Wald chi2({})", fmt_g(df_m, 9)), f2(chi2)));
        out.push(format!("{:<50}{:<15} = {:>10}", format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(res.ll, 8)), "Prob > chi2", f4(pval)));
        if let Some(g) = n_clust {
            out.push(String::new());
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), match &vce { Vce::Cluster(c) => c.clone(), _ => case.clone() })));
        }
        out.push(String::new());
        let omitted = vec![false; p];
        let is_base = vec![false; p];
        let eform = cmd.has_opt("or", 2);
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &res.b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: if eform { Some("Odds ratio") } else { None }, coef_title: "Coefficient" });
        // insert the base-alternative marker after the alternative-specific block
        let lw = tbl.last().map_or(12, |b| b.len().saturating_sub(66));
        let base_lab = label_of(base);
        let first_alt = altlabels.first().cloned();
        let pos = first_alt.and_then(|fa| tbl.iter().position(|l| l.starts_with(&format!("{:<lw$} |", crate::summarize::abbrev(&fa, lw)))));
        let ins = vec![format!("{:<lw$} |  (base alternative)", crate::summarize::abbrev(&base_lab, lw)), crate::choice_mixed::sep(lw)];
        match pos {
            Some(pp) => {
                for (o, l) in ins.into_iter().enumerate() {
                    tbl.insert(pp + o, l);
                }
            }
            None => {
                let last = tbl.pop().unwrap_or_default();
                tbl.push(crate::choice_mixed::sep(lw));
                tbl.push(format!("{:<lw$} |  (base alternative)", crate::summarize::abbrev(&base_lab, lw)));
                tbl.push(last);
            }
        }
        out.extend(tbl);
        let text = out.join("\n");
        let mut dpost = d.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        dpost.cols = cols.clone();
        let nalt = alts.len();
        let altl: Vec<String> = alts.iter().map(|a| label_of(*a)).collect();
        est::post(st, "asclogit", &dpost, &res.b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", pval);
            e.set("N_case", groups.len() as f64);
            e.set("k_alt", nalt as f64);
            e.set("i_base", (alts.iter().position(|a| *a == base).unwrap_or(0) + 1) as f64);
            e.set("alt_min", *sizes.iter().min().unwrap_or(&0) as f64);
            e.set("alt_max", *sizes.iter().max().unwrap_or(&0) as f64);
            e.set("alt_avg", n as f64 / groups.len() as f64);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            e.set_mat("altvals", Matrix::row_vector(&alts));
            e.set_str("alteqs", altl.join(" "));
            e.set_str("casevars", casevars.join(" "));
            e.set_str("case", case.clone());
            e.set_str("altvar", alt.clone());
            e.set_str("title", "Alternative-specific conditional logit");
            e.set_str("chi2type", "Wald");
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("predict", "asclogit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// Linear index of a clogit/asclogit fit for observation `i`.
fn index_row(env: &Env, ds: &DataSet, st: &Store, b: &Matrix, i: usize) -> Result<f64, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd == "clogit" {
        return crate::postest::xb_row(env, b, None, i);
    }
    let alt = st.e.macros.get("altvar").cloned().unwrap_or_default();
    let altv = ds.get_var(&alt).ok_or_else(|| rc_err(111, format!("variable {alt} not found")))?;
    let lab = util::cell_text(ds, altv, i, true);
    let mut s = 0.0;
    for (j, name) in b.colnames.iter().enumerate() {
        let (eq, nm) = name.split_once(':').unwrap_or(("", name));
        if eq == alt || eq == lab {
            let x = crate::postest::coef_value(env, nm, i)?;
            if is_missing(x) {
                return Ok(SYSMISS);
            }
            s += b.get(0, j) * x;
        }
    }
    Ok(s)
}

/// `predict` after clogit / asclogit: pc1 (default), pu0, xb, stdp.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let known = ["pc1", "pu0", "xb", "stdp", "pr"];
    let stat = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).find(|n| known.contains(&n.as_str()));
    let default = if ecmd == "clogit" { "pc1" } else { "pr" };
    let msg = if stat.is_none() {
        if ecmd == "clogit" { "(option pc1 assumed; probability of success given one success within group)".to_string() } else { "(option pr assumed; Pr(alternative))".to_string() }
    } else {
        String::new()
    };
    let stat = stat.unwrap_or_else(|| default.to_string());
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let gvar = if ecmd == "clogit" { st.e.macros.get("group").cloned().unwrap_or_default() } else { st.e.macros.get("case").cloned().unwrap_or_default() };
    let off: Vec<f64> = match st.e.macros.get("offset") {
        Some(o) => util::col(ds, o)?,
        None => vec![0.0; n],
    };
    let env = Env::with_store(ds, st);
    let mut xb = vec![SYSMISS; n];
    for i in 0..n {
        if mask[i] {
            let x = index_row(&env, ds, st, &b, i)?;
            xb[i] = if is_missing(x) { SYSMISS } else { x + off[i] };
        }
    }
    let mut vals = vec![SYSMISS; n];
    match stat.as_str() {
        "xb" => vals = xb.clone(),
        "pu0" => {
            for i in 0..n {
                if !is_missing(xb[i]) {
                    vals[i] = dist::invlogit(xb[i]);
                }
            }
        }
        "stdp" => {
            for i in 0..n {
                if is_missing(xb[i]) {
                    continue;
                }
                let x: Vec<f64> = b.colnames.iter().map(|c| crate::postest::coef_value(&env, c, i)).collect::<Result<_, _>>()?;
                let k = x.len();
                vals[i] = (0..k).map(|a| (0..k).map(|c| x[a] * v[(a, c)] * x[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt();
            }
        }
        _ => {
            let gv = util::col(ds, &gvar)?;
            let mut mx: std::collections::HashMap<u64, f64> = Default::default();
            for i in 0..n {
                if !is_missing(xb[i]) {
                    let e = mx.entry(gv[i].to_bits()).or_insert(f64::NEG_INFINITY);
                    *e = e.max(xb[i]);
                }
            }
            let mut sums: std::collections::HashMap<u64, f64> = Default::default();
            for i in 0..n {
                if !is_missing(xb[i]) {
                    *sums.entry(gv[i].to_bits()).or_insert(0.0) += (xb[i] - mx[&gv[i].to_bits()]).exp();
                }
            }
            for i in 0..n {
                if !is_missing(xb[i]) {
                    vals[i] = (xb[i] - mx[&gv[i].to_bits()]).exp() / sums[&gv[i].to_bits()];
                }
            }
        }
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_denominator_matches_enumeration() {
        // a group of 5 with 2 positives: compare with brute-force subsets
        let x1 = [0.3, -1.2, 0.8, 2.0, -0.4];
        let x2 = [1.0, 0.5, -0.7, 0.1, 0.9];
        let y = [1.0, 0.0, 1.0, 0.0, 0.0];
        let off = [0.0; 5];
        let groups = vec![(0..5).collect::<Vec<usize>>()];
        let m = CondLogit { x: vec![&x1, &x2], y: &y, off: &off, groups: &groups };
        let b = [0.7, -0.3];
        let (ll, g, h, _) = m.eval(&b, false);
        let eta: Vec<f64> = (0..5).map(|t| x1[t] * b[0] + x2[t] * b[1]).collect();
        let mut den = 0.0;
        for a in 0..5 {
            for c in a + 1..5 {
                den += (eta[a] + eta[c]).exp();
            }
        }
        let want = eta[0] + eta[2] - den.ln();
        assert!((ll - want).abs() < 1e-12);
        // gradient and information against finite differences
        for j in 0..2 {
            let mut bp = b;
            bp[j] += 1e-6;
            let mut bm = b;
            bm[j] -= 1e-6;
            let (lp, gp, ..) = m.eval(&bp, false);
            let (lm, gm, ..) = m.eval(&bm, false);
            assert!(((lp - lm) / 2e-6 - g[j]).abs() < 1e-7);
            for i in 0..2 {
                assert!((-(gp[i] - gm[i]) / 2e-6 - h[(i, j)]).abs() < 1e-6);
            }
        }
    }
}
