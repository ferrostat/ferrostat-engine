//! Prefixes and post-estimation commands that run other commands:
//! `jackknife`, `statsby`, `rolling`, `nestreg`, `stepwise`,
//! `estimates for` and `linktest`.

use ferrostat_cmd::dist;
use ferrostat_cmd::est::{self, TableSpec};
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, split_options, split_top_level_words};

use crate::interpreter::{split_verb, AdoInterpreter, Flow};
use crate::parallel;
use crate::resample::{self, arg, has, num_arg, ExpItem, Kind, Opts};

/// Handle the prefixes of this module; None when `verb` is something else.
pub(crate) fn intercept(me: &mut AdoInterpreter, verb_raw: &str, rest: &str, _text: &str) -> Result<Option<Flow>, String> {
    let v = verb_raw.trim_end_matches(':').to_lowercase();
    if me.engine.store.programs.contains_key(verb_raw) {
        return Ok(None);
    }
    let is = |full: &str, min: usize| v.len() >= min && full.starts_with(v.as_str());
    if v == "jknife" || is("jackknife", 4) {
        jackknife(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "statsby" {
        statsby(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "rolling" {
        rolling(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "nestreg" {
        nestreg(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "sw" || v == "stepwise" {
        stepwise(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "linktest" {
        linktest(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if is("estimates", 3) {
        let r = rest.trim_start();
        let sub = r.split(|c: char| c.is_whitespace() || c == ',' || c == ':').next().unwrap_or("");
        if sub == "for" {
            estimates_for(me, r[3..].trim())?;
            return Ok(Some(Flow::Next));
        }
        return Ok(None);
    }
    if let Some(f) = crate::mlprog::intercept(me, &v, rest)? {
        return Ok(Some(f));
    }
    if let Some(f) = crate::wyoung::intercept(me, &v, rest)? {
        return Ok(Some(f));
    }
    if let Some(f) = crate::fracpoly::intercept(me, &v, rest)? {
        return Ok(Some(f));
    }
    Ok(None)
}

fn run_quiet(me: &mut AdoInterpreter, cmd: &str) -> Result<(), String> {
    resample::run_cmd(me, cmd, false)
}

fn run_shown(me: &mut AdoInterpreter, cmd: &str) -> Result<(), String> {
    me.exec_text(cmd, None).map(|_| ())
}

// ---------------------------------------------------------------------------
// jackknife
// ---------------------------------------------------------------------------

/// Sample used by an r-class command: its if/in restriction and the
/// observations with all variables of its varlist nonmissing.
fn rclass_sample(me: &AdoInterpreter, command: &str) -> Vec<bool> {
    let ds = &me.engine.dataset;
    let n = ds.n_obs;
    let Ok(c) = ferrostat_parser::parser::parse_line(command) else { return vec![true; n] };
    let mut mask = util::touse(ds, &me.engine.store, &c).unwrap_or_else(|_| vec![true; n]);
    let vars: Vec<String> = c.main.split_whitespace().filter_map(|w| varlist::expand(ds, w).ok()).flatten().collect();
    util::markout(ds, &mut mask, &vars);
    mask
}

fn jackknife(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    if find_top_level(rest, ':').is_none() {
        if me.engine.store.e.macros.get("prefix").map(String::as_str) != Some("jackknife") {
            return Err(rc_err(301, "last estimates not found"));
        }
        let opts = split_options(rest.trim().trim_start_matches(','));
        let level = resample::level_opt(me, &opts)?;
        let out = jk_display(me, level, &opts)?;
        me.emit(&out);
        return Ok(());
    }
    let p = resample::parse_prefix(rest, false)?;
    let opts = &p.opts;
    let level = resample::level_opt(me, opts)?;
    let nodots = has(opts, "nodots", 6);
    let noisily = has(opts, "noisily", 1);
    let mse = has(opts, "mse", 3);
    let force_r = has(opts, "rclass", 1);
    let cluster = match arg(opts, "cluster", 2) {
        Some(c) => varlist::expand(&me.engine.dataset, &c)?,
        None => Vec::new(),
    };
    let cmd_verb = split_verb(&p.command).0;
    let cmd_name = me.engine.resolve_verb(&cmd_verb.to_lowercase());
    let orig = me.engine.dataset.clone();
    let orig_changed = me.engine.store.data_changed;
    let old_e = std::mem::take(&mut me.engine.store.e);
    let restore = |me: &mut AdoInterpreter, orig: DataSet, old_e: Results| {
        me.engine.dataset = orig;
        me.engine.store.e = old_e;
    };
    if let Err(e) = run_quiet(me, &p.command) {
        restore(me, orig, old_e);
        return Err(e);
    }
    let eclass = !force_r && !me.engine.store.e.is_empty();
    let exps = if p.exps.is_empty() {
        if !eclass {
            restore(me, orig, old_e);
            return Err(rc_err(198, "expression list required"));
        }
        vec![ExpItem { name: None, kind: Kind::B }]
    } else {
        p.exps.clone()
    };
    let default_b = eclass && exps.len() == 1 && matches!(exps[0].kind, Kind::B);
    let (stats, parsed) = match resample::build_stats(me, &exps, "_jk") {
        Ok(x) => x,
        Err(e) => {
            restore(me, orig, old_e);
            return Err(e);
        }
    };
    let observed = resample::eval_stats(me, &stats, &parsed);
    let obs_e = me.engine.store.e.clone();
    let n0 = orig.n_obs;
    let sample: Vec<bool> = match &obs_e.sample {
        Some(s) if eclass && s.len() == n0 => s.clone(),
        _ => {
            me.engine.dataset = orig.clone();
            rclass_sample(me, &p.command)
        }
    };
    let mut base = orig.clone();
    if sample.iter().any(|x| !x) {
        base.keep_if(&sample)?;
    }
    me.emit(&format!("(running {cmd_name} on estimation sample)"));
    let u = resample::units(&base, &cluster, &[]);
    let nunits = u.rows.len();
    if nunits < 2 {
        restore(me, orig, old_e);
        return Err(rc_err(2001, "insufficient observations"));
    }
    let job = parallel::Job {
        shell: resample::empty_shell(&base),
        base,
        command: p.command.clone(),
        draw: parallel::Draw::LeaveOut { units: u.rows.clone() },
        stats: resample::stats_fn(&stats, &parsed),
        spec: resample::spec_of(&stats),
        seed: 0,
    };
    let rng = me.engine.store.rng.borrow().clone();
    let (results, marks) = resample::collect_reps(parallel::run_all(me, &job, 0, nunits, noisily), stats.len());
    *me.engine.store.rng.borrow_mut() = rng;
    me.engine.dataset = orig;
    me.engine.store.data_changed = orig_changed;
    if !nodots {
        me.emit(&format!("\n{}", resample::dots_line("Jackknife replications", nunits, &marks)));
    }
    let k = stats.len();
    let complete: Vec<&Vec<f64>> = results.iter().filter(|r| r.iter().all(|x| !is_missing(*x))).collect();
    let nc = complete.len();
    let ncf = nc as f64;
    let mut mean = vec![SYSMISS; k];
    let mut v = Matrix::zeros(k, k);
    if nc > 0 {
        for j in 0..k {
            mean[j] = complete.iter().map(|r| r[j]).sum::<f64>() / ncf;
        }
    }
    if nc > 1 {
        for a in 0..k {
            for b in 0..k {
                let (ca, cb) = if mse { (observed[a], observed[b]) } else { (mean[a], mean[b]) };
                let s: f64 = complete.iter().map(|r| (r[a] - ca) * (r[b] - cb)).sum();
                v[(a, b)] = s * (ncf - 1.0) / ncf;
            }
        }
    }
    if let Some(spec) = arg(opts, "saving", 3) {
        let ds = resample::results_dataset(&stats, &results, resample::saving_double(&spec))?;
        resample::save_results(me, &spec, ds)?;
    }
    // pseudovalues as variables (keep)
    if has(opts, "keep", 4) {
        let ds = &mut me.engine.dataset;
        let srows: Vec<usize> = util::rows(&sample);
        for (j, s) in stats.iter().enumerate() {
            let mut vals = vec![SYSMISS; ds.n_obs];
            for (ui, rows) in u.rows.iter().enumerate() {
                let th = results[ui][j];
                let pv = if is_missing(th) { SYSMISS } else { nunits as f64 * observed[j] - (nunits as f64 - 1.0) * th };
                for &r in rows {
                    vals[srows[r]] = pv;
                }
            }
            let _ = ds.drop_variable(&s.var);
            let mut var = Variable::new(s.var.clone(), VarType::Double);
            var.label = format!("pseudovalues: {}", s.label);
            var.data = vals.iter().map(|&x| util::typed_value(VarType::Double, x)).collect();
            ds.add_variable(var)?;
        }
    }
    let names: Vec<String> = stats.iter().map(|s| s.name.clone()).collect();
    let rowv = |vals: &[f64]| Matrix::row_vector(vals).with_names(vec!["y1".into()], names.clone());
    let mut e = if default_b { obs_e.clone() } else { Results::new() };
    if default_b {
        e.set_str("cmdname", e.macros.get("cmd").cloned().unwrap_or_default());
        for s in ["F", "chi2", "p", "ll", "ll_0", "r2_p"] {
            e.scalars.remove(s);
        }
    } else {
        e.set_str("cmd", "jackknife");
        e.set_str("cmdname", cmd_name.clone());
        e.set_str("title", "Jackknife results");
        let n = if eclass { obs_e.get("N").unwrap_or(sample.iter().filter(|x| **x).count() as f64) } else { sample.iter().filter(|x| **x).count() as f64 };
        e.set("N", n);
        e.sample = Some(sample.clone());
    }
    e.set_str("prefix", "jackknife");
    e.set_str("command", p.command.clone());
    e.set_str("vce", "jackknife");
    e.set_str("vcetype", "Jackknife");
    e.set_str("properties", "b V");
    e.set_mat("b", rowv(&observed));
    e.set_mat("V", v.clone().with_names(names.clone(), names.clone()));
    e.set_mat("b_jk", rowv(&mean));
    e.set("N_reps", nunits as f64);
    e.set("N_misreps", (nunits - nc) as f64);
    e.set("df_r", ncf - 1.0);
    e.set("k_exp", if default_b { 0.0 } else { k as f64 });
    e.set("k_eexp", if default_b { k as f64 } else { 0.0 });
    e.set("level", level);
    if mse {
        e.set_str("mse", "mse");
    }
    if !default_b {
        for (j, s) in stats.iter().enumerate() {
            e.set_str(&format!("exp{}", j + 1), s.label.clone());
        }
    }
    if !cluster.is_empty() {
        e.set("N_clust", nunits as f64);
        e.set_str("clustvar", cluster.join(" "));
    }
    if default_b {
        // model F test of the non-constant coefficients
        if let Some((w, q)) = resample::wald(&names, &observed, &v) {
            e.set("F", w / q as f64);
            e.set("df_m", q as f64);
        }
    }
    me.engine.store.e = e;
    if !(has(opts, "notable", 5) && has(opts, "noheader", 5)) {
        let out = jk_display(me, level, opts)?;
        me.emit(&format!("\n{out}"));
    }
    Ok(())
}

fn jk_display(me: &mut AdoInterpreter, level: f64, opts: &Opts) -> Result<String, String> {
    let e = me.engine.store.e.clone();
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = b.colnames.clone();
    let df = e.get("df_r").unwrap_or(SYSMISS);
    let mut out = Vec::new();
    if !has(opts, "noheader", 5) {
        let mut rows = vec![
            (String::new(), "Number of obs".to_string(), util::commas(e.get("N").unwrap_or(0.0) as usize)),
            (String::new(), "Replications".to_string(), util::commas(e.get("N_reps").unwrap_or(0.0) as usize)),
        ];
        if e.macros.get("cmd").map(String::as_str) != Some("jackknife") {
            if let (Some(f), Some(q)) = (e.get("F"), e.get("df_m")) {
                rows.push((String::new(), format!("F({}, {})", q, fmt_g(df, 8)), format!("{:.2}", f)));
                rows.push((String::new(), "Prob > F".into(), format!("{:.4}", dist::ftail(q, df, f))));
            }
            for (k, l) in [("r2", "R-squared"), ("r2_a", "Adj R-squared")] {
                if let Some(x) = e.get(k) {
                    rows.push((String::new(), l.into(), format!("{:.4}", x)));
                }
            }
            if let Some(r) = e.get("rmse") {
                rows.push((String::new(), "Root MSE".into(), ferrostat_cmd::format::fmt(r, "%10.5g").trim().to_string()));
            }
        }
        out.extend(resample::header_lines(&resample::title_for(&e).replace("Bootstrap results", "Jackknife results"), &rows));
        if let (Some(g), Some(cv)) = (e.get("N_clust"), e.macros.get("clustvar")) {
            out.push(String::new());
            out.push(format!("{:>78}", format!("(Replications based on {} clusters in {})", util::commas(g as usize), cv)));
        }
        if e.macros.get("cmd").map(String::as_str) == Some("jackknife") && !has(opts, "nolegend", 5) {
            out.push(String::new());
            out.push(format!("{:>13}: {}", "Command", e.macros.get("command").cloned().unwrap_or_default()));
            for (j, n) in names.iter().enumerate() {
                if let Some(x) = e.macros.get(&format!("exp{}", j + 1)) {
                    out.push(format!("{:>13}: {}", resample::summ_abbrev(n, 13), x));
                }
            }
        }
        out.push(String::new());
    }
    if !has(opts, "notable", 5) {
        let k = names.len();
        let is_base: Vec<bool> = names.iter().map(|c| resample::is_base_name(c)).collect();
        let omitted: Vec<bool> = names.iter().map(|c| resample::is_omitted_name(c)).collect();
        let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
        let eform = resample::eform_title(&e);
        let spec = TableSpec { depvar: &depvar, names: &names, b: &b.data[..k], v, omitted: &omitted, is_base: &is_base, df: Some(df), level, eform, coef_title: "Coefficient" };
        let mut t = est::coef_table(&spec);
        if t.len() > 1 {
            let bar = t[1].find('|').unwrap_or(13);
            t[1] = t[1].replacen("Std. err.", "std. err.", 1);
            t.insert(1, format!("{}|              Jackknife", " ".repeat(bar)));
        }
        out.extend(t);
        let mis = e.get("N_misreps").unwrap_or(0.0) as usize;
        if mis > 0 {
            out.push(format!("Note: One or more parameters could not be estimated in {mis} jackknife replicate{};", if mis == 1 { "" } else { "s" }));
            out.push("      standard-error estimates include only complete replications.".into());
        }
        if let Some((m, lvl)) = est::take_table() {
            let mut r = Results::new();
            r.set_mat("table", m);
            r.set("level", lvl);
            me.engine.store.r = r;
        }
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// statsby / rolling
// ---------------------------------------------------------------------------

/// Run `command` on each subset of rows and collect the statistics.
/// Returns (stats, values per subset) — failed runs give missing values.
fn run_subsets(me: &mut AdoInterpreter, p: &resample::Prefix, subsets: &[Vec<usize>], noisily: bool, stem: &str) -> Result<(Vec<resample::Stat>, Vec<Vec<f64>>, Vec<char>), String> {
    let orig = me.engine.dataset.clone();
    let shell = resample::empty_shell(&orig);
    let take = |rows: &[usize]| -> DataSet {
        let mut d = shell.clone();
        for (v, src) in d.vars.iter_mut().zip(&orig.vars) {
            v.data = rows.iter().map(|&r| src.data[r].clone()).collect();
        }
        d.n_obs = rows.len();
        d
    };
    let mut stats: Option<(Vec<resample::Stat>, Vec<ferrostat_parser::Expr>)> = None;
    let mut vals: Vec<Option<Vec<f64>>> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    let mut err = None;
    for (gi, rows) in subsets.iter().enumerate() {
        me.engine.dataset = take(rows);
        me.engine.store.e = Results::new();
        match resample::run_cmd(me, &p.command, noisily) {
            Ok(()) => {
                if stats.is_none() {
                    let exps = if p.exps.is_empty() {
                        if me.engine.store.e.matrices.get("b").is_none() {
                            err = Some(rc_err(198, "expression list required"));
                            break;
                        }
                        vec![ExpItem { name: None, kind: Kind::B }]
                    } else {
                        p.exps.clone()
                    };
                    match resample::build_stats(me, &exps, stem) {
                        Ok(s) => stats = Some(s),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                let (s, parsed) = stats.as_ref().unwrap();
                vals.push(Some(resample::eval_stats(me, s, parsed)));
            }
            Err(_) => {
                pending.push(gi);
                vals.push(None);
            }
        }
    }
    me.engine.dataset = orig;
    if let Some(e) = err {
        return Err(e);
    }
    let (stats, _) = stats.ok_or_else(|| rc_err(2000, "the command failed in every group"))?;
    let k = stats.len();
    let marks: Vec<char> = vals.iter().map(|v| match v {
        None => 'x',
        Some(x) if x.iter().any(|y| is_missing(*y)) => 'x',
        _ => '.',
    }).collect();
    let values = vals.into_iter().map(|v| v.unwrap_or_else(|| vec![SYSMISS; k])).collect();
    Ok((stats, values, marks))
}

fn dots_block(title: &str, marks: &[char]) -> String {
    let mut s = format!("{title}\n----+--- 1 ---+--- 2 ---+--- 3 ---+--- 4 ---+--- 5 \n");
    for (i, c) in marks.iter().enumerate() {
        s.push(*c);
        if (i + 1) % 50 == 0 {
            s.push_str(&format!("{:>6}\n", i + 1));
        }
    }
    s.trim_end().to_string()
}

fn finish_dataset(me: &mut AdoInterpreter, opts: &Opts, ds: DataSet) -> Result<(), String> {
    match arg(opts, "saving", 3) {
        Some(spec) => resample::save_results(me, &spec, ds),
        None => {
            me.engine.dataset = ds;
            me.engine.store.data_changed = true;
            Ok(())
        }
    }
}

fn add_stat_vars(ds: &mut DataSet, stats: &[resample::Stat], values: &[Vec<f64>], double: bool) -> Result<(), String> {
    let ty = if double { VarType::Double } else { VarType::Float };
    for (j, s) in stats.iter().enumerate() {
        let mut v = Variable::new(s.var.clone(), ty);
        v.label = s.label.clone();
        v.data = values.iter().map(|r| util::typed_value(ty, r[j])).collect();
        ds.add_variable(v)?;
    }
    ds.n_obs = values.len();
    Ok(())
}

fn statsby(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let p = resample::parse_prefix(rest, false)?;
    let opts = &p.opts;
    let saving = arg(opts, "saving", 3);
    if saving.is_none() && !has(opts, "clear", 5) && me.engine.store.data_changed {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved\n(use option clear to replace the data in memory)"));
    }
    let by = match arg(opts, "by", 2) {
        Some(b) => varlist::expand(&me.engine.dataset, b.split(',').next().unwrap_or(""))?,
        None => Vec::new(),
    };
    let missing_ok = arg(opts, "by", 2).is_some_and(|b| b.contains("missing"));
    let noisily = has(opts, "noisily", 1);
    let total = has(opts, "total", 3);
    let ds = &me.engine.dataset;
    // groups in sorted order of the by variables
    let n = ds.n_obs;
    let cols: Vec<&Variable> = by.iter().map(|b| ds.get_var(b).unwrap()).collect();
    let mut idx: Vec<usize> = (0..n)
        .filter(|&i| missing_ok || cols.iter().all(|c| if c.var_type.is_string() { true } else { !is_missing(c.num(i)) }))
        .collect();
    let cmp = |a: &usize, b: &usize| -> std::cmp::Ordering {
        for c in &cols {
            let o = if c.var_type.is_string() { c.str_at(*a).cmp(c.str_at(*b)) } else { c.num(*a).total_cmp(&c.num(*b)) };
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        a.cmp(b)
    };
    idx.sort_by(cmp);
    let same = |a: usize, b: usize| cols.iter().all(|c| if c.var_type.is_string() { c.str_at(a) == c.str_at(b) } else { c.num(a).to_bits() == c.num(b).to_bits() });
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    let mut firsts: Vec<Option<usize>> = Vec::new();
    for &i in &idx {
        match subsets.last_mut() {
            Some(g) if same(g[0], i) => g.push(i),
            _ => {
                subsets.push(vec![i]);
                firsts.push(Some(i));
            }
        }
    }
    if total {
        subsets.push((0..n).collect());
        firsts.push(None);
    }
    let cmd_name = me.engine.resolve_verb(&split_verb(&p.command).0.to_lowercase());
    me.emit(&format!("(running {cmd_name} on estimation sample)"));
    let (stats, values, marks) = run_subsets(me, &p, &subsets, noisily, "_stat")?;
    let mut legend = vec![String::new(), format!("{:>13}: {}", "Command", p.command)];
    for s in &stats {
        if !matches!(s.src, resample::Src::B(_) | resample::Src::Se(_)) {
            legend.push(format!("{:>13}: {}", resample::summ_abbrev(&s.var, 13), s.label));
        }
    }
    if !by.is_empty() {
        legend.push(format!("{:>13}: {}", "By", by.join(" ")));
    }
    me.emit(&legend.join("\n"));
    if !has(opts, "nodots", 6) {
        me.emit(&format!("\n{}", dots_block("Statsby groups", &marks)));
    }
    // result dataset: the by variables then the statistics
    let src = &me.engine.dataset;
    let mut out = DataSet::new();
    for b in &by {
        let v = src.get_var(b).unwrap();
        let mut nv = v.clone();
        nv.data = firsts.iter().map(|f| match f {
            Some(i) => v.data[*i].clone(),
            None => if v.var_type.is_string() { ferrostat_core::Value::Str(String::new()) } else { util::typed_value(v.var_type, SYSMISS) },
        }).collect();
        out.add_variable(nv)?;
    }
    for (name, vl) in &src.value_labels {
        if by.iter().any(|b| src.get_var(b).is_some_and(|v| &v.value_label_name == name)) {
            out.value_labels.insert(name.clone(), vl.clone());
        }
    }
    let double = has(opts, "double", 6) || arg(opts, "saving", 3).is_some_and(|s| resample::saving_double(&s));
    add_stat_vars(&mut out, &stats, &values, double)?;
    finish_dataset(me, opts, out)
}

fn rolling(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let p = resample::parse_prefix(rest, false)?;
    let opts = &p.opts;
    let saving = arg(opts, "saving", 3);
    if saving.is_none() && !has(opts, "clear", 5) && me.engine.store.data_changed {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved\n(use option clear to replace the data in memory)"));
    }
    let (panel, time) = me.engine.store.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset varname ..."))?;
    if panel.is_some() {
        return Err(rc_err(198, "rolling with panel data is not supported"));
    }
    let window = num_arg(opts, "window", 1)?.ok_or_else(|| rc_err(198, "option window() required"))?;
    let step = num_arg(opts, "stepsize", 4)?.unwrap_or(1.0).max(1.0);
    let recursive = has(opts, "recursive", 3);
    let rrecursive = has(opts, "rrecursive", 3);
    let ds = &me.engine.dataset;
    let t = util::col(ds, &time)?;
    let tv: Vec<f64> = t.iter().copied().filter(|x| !is_missing(*x)).collect();
    if tv.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let tmin = num_arg(opts, "start", 5)?.unwrap_or(tv.iter().cloned().fold(f64::INFINITY, f64::min));
    let tmax = num_arg(opts, "end", 3)?.unwrap_or(tv.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let mut windows: Vec<(f64, f64)> = Vec::new();
    let mut s = tmin;
    while s + window - 1.0 <= tmax + 1e-9 {
        let e = s + window - 1.0;
        windows.push(if recursive {
            (tmin, e)
        } else if rrecursive {
            (s, tmax)
        } else {
            (s, e)
        });
        s += step;
    }
    let subsets: Vec<Vec<usize>> = windows.iter().map(|(a, b)| (0..ds.n_obs).filter(|&i| !is_missing(t[i]) && t[i] >= *a - 1e-9 && t[i] <= *b + 1e-9).collect()).collect();
    let cmd_name = me.engine.resolve_verb(&split_verb(&p.command).0.to_lowercase());
    me.emit(&format!("(running {cmd_name} on estimation sample)"));
    let tsset = me.engine.store.tsset.clone();
    let (stats, values, marks) = run_subsets(me, &p, &subsets, has(opts, "noisily", 1), "_stat")?;
    me.engine.store.tsset = tsset;
    if !has(opts, "nodots", 6) {
        me.emit(&format!("\n{}", resample::dots_line("Rolling replications", subsets.len(), &marks)));
    }
    let tvar = me.engine.dataset.get_var(&time).cloned();
    let mut out = DataSet::new();
    for (k, name) in ["start", "end"].iter().enumerate() {
        let mut v = Variable::new(name.to_string(), VarType::Double);
        if let Some(tv) = &tvar {
            v.format = tv.format.clone();
        }
        v.data = windows.iter().map(|w| util::typed_value(VarType::Double, if k == 0 { w.0 } else { w.1 })).collect();
        out.add_variable(v)?;
    }
    let double = arg(opts, "saving", 3).is_some_and(|s| resample::saving_double(&s));
    add_stat_vars(&mut out, &stats, &values, double)?;
    if saving.is_none() {
        me.engine.store.tsset = None;
    }
    finish_dataset(me, opts, out)
}

// ---------------------------------------------------------------------------
// nestreg / stepwise
// ---------------------------------------------------------------------------

/// A model command split into verb, dependent variable, terms and the rest
/// (if, in, weights, options).
struct ModelCmd {
    verb: String,
    depvar: String,
    terms: Vec<Vec<String>>,
    tail: String,
}

impl ModelCmd {
    fn parse(me: &AdoInterpreter, text: &str) -> Result<ModelCmd, String> {
        let (verb, rest) = split_verb(text);
        let rest = rest.trim();
        // the term list ends at if / in / [weight] / using / the options
        let words = split_top_level_words(rest);
        let mut used = 0usize;
        let mut items = Vec::new();
        for w in &words {
            let lw = w.to_lowercase();
            if lw == "if" || lw == "in" || w.starts_with('[') || w.starts_with(',') || lw == "using" {
                break;
            }
            if let Some(i) = w.find(',') {
                if !w.starts_with('(') {
                    items.push(w[..i].to_string());
                    break;
                }
            }
            items.push(w.clone());
            used += 1;
        }
        let pos = {
            // byte offset of the first unused word in `rest`
            let mut off = 0usize;
            for w in words.iter().take(used) {
                off = rest[off..].find(w.as_str()).map_or(off, |i| off + i + w.len());
            }
            off
        };
        let tail = rest[pos.min(rest.len())..].trim().to_string();
        let tail = if let Some(stripped) = items.last().filter(|it| !words.contains(it)) {
            // `x,opts` glued: drop the varname part from the tail
            tail.strip_prefix(stripped.as_str()).unwrap_or(&tail).to_string()
        } else {
            tail
        };
        if items.is_empty() {
            return Err(rc_err(102, "too few variables specified"));
        }
        let depvar = items.remove(0);
        let ds = &me.engine.dataset;
        let mut terms = Vec::new();
        for it in items {
            if it.starts_with('(') && it.ends_with(')') {
                let inner = &it[1..it.len() - 1];
                let vars: Vec<String> = inner.split_whitespace().map(|w| if w.contains('.') { Ok(vec![w.to_string()]) } else { varlist::expand(ds, w) }).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
                terms.push(vars);
            } else if it.contains('.') {
                terms.push(vec![it]);
            } else {
                for v in varlist::expand(ds, &it)? {
                    terms.push(vec![v]);
                }
            }
        }
        Ok(ModelCmd { verb, depvar, terms, tail })
    }

    fn line(&self, include: &[usize]) -> String {
        let vars: Vec<String> = include.iter().flat_map(|&i| self.terms[i].iter().cloned()).collect();
        format!("{} {} {} {}", self.verb, self.depvar, vars.join(" "), self.tail).split_whitespace().collect::<Vec<_>>().join(" ").replace(" ,", ",")
    }
}

/// Wald test of `vars` in the current estimates: (statistic, df, df_r, p, is_F).
fn wald_test(me: &mut AdoInterpreter, vars: &[String]) -> Result<(f64, f64, Option<f64>, f64, bool), String> {
    run_quiet(me, &format!("test {}", vars.join(" ")))?;
    let r = &me.engine.store.r;
    let p = r.get("p").unwrap_or(SYSMISS);
    let df = r.get("df").unwrap_or(SYSMISS);
    match r.get("F") {
        Some(f) => Ok((f, df, r.get("df_r"), p, true)),
        None => Ok((r.get("chi2").unwrap_or(SYSMISS), df, None, p, false)),
    }
}

/// Keep only the observations of the full model's estimation sample.
fn common_sample(me: &mut AdoInterpreter, line: &str) -> Result<Option<DataSet>, String> {
    run_quiet(me, line)?;
    let sample = me.engine.store.e.sample.clone();
    match sample {
        Some(s) if s.len() == me.engine.dataset.n_obs && s.iter().any(|x| !x) => {
            let orig = me.engine.dataset.clone();
            me.engine.dataset.keep_if(&s)?;
            Ok(Some(orig))
        }
        _ => Ok(None),
    }
}

fn nestreg(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "nestreg: command required"))?;
    let head = rest[..colon].trim().trim_start_matches(',');
    let opts = split_options(head);
    let mc = ModelCmd::parse(me, rest[colon + 1..].trim())?;
    let lr = has(&opts, "lr", 2);
    let quiet_models = has(&opts, "quietly", 1);
    let store = arg(&opts, "store", 2).map(|s| s.trim().to_string());
    let nb = mc.terms.len();
    let all: Vec<usize> = (0..nb).collect();
    let orig = common_sample(me, &mc.line(&all))?;
    let res = (|| -> Result<(Vec<Vec<f64>>, bool, bool), String> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut prev_ll = None;
        let mut prev_r2 = None;
        let mut is_f = false;
        let mut has_r2 = false;
        if lr {
            // constant-only model for the first block's LR test
            run_quiet(me, &mc.line(&[]))?;
            prev_ll = me.engine.store.e.get("ll");
        }
        for k in 0..nb {
            let inc: Vec<usize> = (0..=k).collect();
            let line = mc.line(&inc);
            me.emit(&format!("\nBlock {:>2}: {}", k + 1, mc.terms[k].join(" ")));
            if quiet_models {
                run_quiet(me, &line)?;
            } else {
                run_shown(me, &line)?;
            }
            if let Some(stub) = &store {
                run_quiet(me, &format!("estimates store {stub}{}", k + 1))?;
            }
            let e = me.engine.store.e.clone();
            if lr {
                let ll = e.get("ll").ok_or_else(|| rc_err(498, "lr requires a log likelihood"))?;
                let df_k = mc.terms[k].len() as f64;
                let lrs = 2.0 * (ll - prev_ll.unwrap_or(ll));
                let p = dist::chi2tail(df_k, lrs);
                let rank = e.get("rank").unwrap_or(0.0);
                let n = e.get("N").unwrap_or(0.0);
                rows.push(vec![ll, lrs, df_k, p, -2.0 * ll + 2.0 * rank, -2.0 * ll + rank * n.ln()]);
                prev_ll = Some(ll);
            } else {
                let terms = mc.terms[k].clone();
                let saved_r = me.engine.store.r.clone();
                let (stat, df, dfr, p, f) = wald_test(me, &terms)?;
                me.engine.store.r = saved_r;
                is_f = f;
                let r2 = e.get("r2");
                has_r2 = r2.is_some();
                let ch = match (r2, prev_r2) {
                    (Some(a), Some(b)) => a - b,
                    _ => SYSMISS,
                };
                rows.push(vec![stat, df, dfr.unwrap_or(SYSMISS), p, r2.unwrap_or(SYSMISS), ch]);
                prev_r2 = r2;
            }
        }
        Ok((rows, is_f, has_r2))
    })();
    if let Some(o) = orig {
        me.engine.dataset = o;
    }
    let (rows, is_f, has_r2) = res?;
    let mut out = vec![String::new()];
    let mut mat = Matrix::zeros(rows.len(), rows.first().map_or(0, |r| r.len()));
    for (i, r) in rows.iter().enumerate() {
        for (j, x) in r.iter().enumerate() {
            mat[(i, j)] = *x;
        }
    }
    let rn: Vec<String> = (1..=rows.len()).map(|i| i.to_string()).collect();
    let f4 = |x: f64| if is_missing(x) { String::new() } else { format!("{x:.4}") };
    let mut r = Results::new();
    if lr {
        out.push("  +----------------------------------------------------------------+".into());
        out.push("  | Block |        LL       LR     df  Pr > LR       AIC       BIC |".into());
        out.push("  |-------+--------------------------------------------------------|".into());
        for (i, x) in rows.iter().enumerate() {
            out.push(format!("  | {:>5} | {:>9} {:>8.2} {:>6}  {:>7.4} {:>9} {:>9} |", i + 1, fmt_g(x[0], 9), x[1], x[2], x[3], fmt_g(x[4], 9), fmt_g(x[5], 9)));
        }
        out.push("  +----------------------------------------------------------------+".into());
        r.set_mat("lr", mat.with_names(rn, ["ll", "chi2", "df", "p", "aic", "bic"].iter().map(|s| s.to_string()).collect()));
    } else if is_f && has_r2 {
        out.push("  +-------------------------------------------------------------+".into());
        out.push("  |       |          Block  Residual                     Change |".into());
        out.push("  | Block |       F     df        df   Pr > F       R2    in R2 |".into());
        out.push("  |-------+-----------------------------------------------------|".into());
        for (i, x) in rows.iter().enumerate() {
            out.push(format!("  | {:>5} | {:>7.2} {:>6} {:>9} {:>8} {:>8} {:>8} |", i + 1, x[0], x[1], fmt_g(x[2], 9), f4(x[3]), f4(x[4]), f4(x[5])));
        }
        out.push("  +-------------------------------------------------------------+".into());
        r.set_mat("wald", mat.with_names(rn, ["F", "df", "df_r", "p", "r2", "dr2"].iter().map(|s| s.to_string()).collect()));
    } else {
        out.push("  +----------------------------------------+".into());
        out.push("  |       |          Block                 |".into());
        out.push(format!("  | Block | {:>10}     df   {:>10} |", if is_f { "F" } else { "chi2" }, if is_f { "Pr > F" } else { "Pr > chi2" }));
        out.push("  |-------+--------------------------------|".into());
        for (i, x) in rows.iter().enumerate() {
            out.push(format!("  | {:>5} | {:>10.2} {:>6}   {:>10} |", i + 1, x[0], x[1], f4(x[3])));
        }
        out.push("  +----------------------------------------+".into());
        let m2 = Matrix::from_rows(&rows.iter().map(|x| vec![x[0], x[1], x[3]]).collect::<Vec<_>>());
        r.set_mat("wald", m2.with_names(rn, [if is_f { "F" } else { "chi2" }, "df", "p"].iter().map(|s| s.to_string()).collect()));
    }
    me.emit(&out.join("\n"));
    me.engine.store.r = r;
    Ok(())
}

fn stepwise(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "stepwise: command required"))?;
    let head = rest[..colon].trim().trim_start_matches(',');
    let opts = split_options(head);
    let mc = ModelCmd::parse(me, rest[colon + 1..].trim())?;
    let pr = num_arg(&opts, "pr", 2)?;
    let pe = num_arg(&opts, "pe", 2)?;
    if pr.is_none() && pe.is_none() {
        return Err(rc_err(198, "must specify pr() or pe()"));
    }
    if let (Some(r), Some(e)) = (pr, pe) {
        if r <= e {
            return Err(rc_err(198, "pr() must be greater than pe()"));
        }
    }
    let lr = has(&opts, "lr", 2);
    let hier = has(&opts, "hierarchical", 3);
    let lock1 = has(&opts, "lockterm1", 5);
    let forward = has(&opts, "forward", 3);
    let nt = mc.terms.len();
    let all: Vec<usize> = (0..nt).collect();
    let orig = common_sample(me, &mc.line(&all))?;
    let label = |t: usize| mc.terms[t].join(" ");
    // p-value of term t within the model `inc` (which contains t)
    let pval = |me: &mut AdoInterpreter, inc: &[usize], t: usize| -> Result<f64, String> {
        if lr {
            run_quiet(me, &mc.line(inc))?;
            let ll1 = me.engine.store.e.get("ll").ok_or_else(|| rc_err(498, "lr requires a log likelihood"))?;
            let without: Vec<usize> = inc.iter().copied().filter(|&x| x != t).collect();
            run_quiet(me, &mc.line(&without))?;
            let ll0 = me.engine.store.e.get("ll").unwrap_or(SYSMISS);
            Ok(dist::chi2tail(mc.terms[t].len() as f64, 2.0 * (ll1 - ll0)))
        } else {
            run_quiet(me, &mc.line(inc))?;
            Ok(wald_test(me, &mc.terms[t])?.3)
        }
    };
    let res = (|| -> Result<(Vec<usize>, Vec<String>), String> {
        let mut log = Vec::new();
        let start_full = pr.is_some() && !(forward || pr.is_none());
        let mut inc: Vec<usize> = if start_full { all.clone() } else if lock1 { vec![0] } else { Vec::new() };
        log.push(format!("{:>22}begin with {} model", "", if start_full { "full" } else { "empty" }));
        let removable = |inc: &[usize]| -> Vec<usize> {
            let c: Vec<usize> = inc.iter().copied().filter(|&t| !(lock1 && t == 0)).collect();
            if hier { c.last().copied().into_iter().collect() } else { c }
        };
        let addable = |inc: &[usize]| -> Vec<usize> {
            let c: Vec<usize> = (0..nt).filter(|t| !inc.contains(t)).collect();
            if hier { c.first().copied().into_iter().collect() } else { c }
        };
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 4 * nt + 10 {
                break;
            }
            let mut changed = false;
            // removal step
            if let Some(prv) = pr {
                let cands = removable(&inc);
                let mut worst: Option<(usize, f64)> = None;
                for &t in &cands {
                    let p = pval(me, &inc, t)?;
                    if worst.map_or(true, |(_, wp)| p > wp) {
                        worst = Some((t, p));
                    }
                }
                if let Some((t, p)) = worst {
                    if p >= prv {
                        log.push(format!("p = {:.4} >= {:.4}  removing {}", p, prv, label(t)));
                        inc.retain(|&x| x != t);
                        changed = true;
                    }
                }
            }
            if changed && pe.is_none() {
                continue;
            }
            // addition step
            if !changed {
                if let Some(pev) = pe {
                    let mut best: Option<(usize, f64)> = None;
                    for t in addable(&inc) {
                        let mut with = inc.clone();
                        with.push(t);
                        with.sort();
                        let p = pval(me, &with, t)?;
                        if best.map_or(true, |(_, bp)| p < bp) {
                            best = Some((t, p));
                        }
                    }
                    if let Some((t, p)) = best {
                        if p < pev {
                            log.push(format!("p = {:.4} <  {:.4}  adding   {}", p, pev, label(t)));
                            inc.push(t);
                            inc.sort();
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Ok((inc, log))
    })();
    let (inc, log) = match res {
        Ok(x) => x,
        Err(e) => {
            if let Some(o) = orig {
                me.engine.dataset = o;
            }
            return Err(e);
        }
    };
    let mut log = log;
    if log.len() == 1 {
        match (pr, pe) {
            (Some(p), _) if inc.len() == nt => log.push(format!("p < {:.4}  for all terms in model", p)),
            (_, Some(p)) => log.push(format!("p >= {:.4}  for all terms not in model", p)),
            _ => {}
        }
    }
    me.emit(&log.join("\n"));
    let line = mc.line(&inc);
    let r = run_shown(me, &line);
    if let Some(o) = orig {
        me.engine.dataset = o;
    }
    r
}

// ---------------------------------------------------------------------------
// estimates for, linktest
// ---------------------------------------------------------------------------

fn estimates_for(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "estimates for namelist: command"))?;
    let head = rest[..colon].trim();
    let command = rest[colon + 1..].trim().to_string();
    let (list, opts) = match find_top_level(head, ',') {
        Some(i) => (head[..i].trim(), split_options(&head[i + 1..])),
        None => (head, Vec::new()),
    };
    let mut names: Vec<String> = Vec::new();
    for w in list.split_whitespace() {
        if w == "_all" || w == "*" {
            names.extend(me.engine.store.estimates_order.iter().filter(|n| me.engine.store.estimates.contains_key(*n)).cloned());
        } else if w.contains('*') || w.contains('?') {
            let pat = w.to_string();
            names.extend(me.engine.store.estimates.keys().filter(|k| glob(&pat, k)).cloned());
        } else {
            names.push(w.to_string());
        }
    }
    let noheader = has(&opts, "noheader", 5);
    let nostop = has(&opts, "nostop", 6);
    let saved = me.engine.store.e.clone();
    let mut res = Ok(());
    for n in &names {
        let e = match me.engine.store.estimates.get(n) {
            Some(e) => e.clone(),
            None => {
                res = Err(rc_err(111, format!("estimation result {n} not found")));
                break;
            }
        };
        me.engine.store.e = e;
        if !noheader {
            me.emit(&format!("\n{}\nModel {n}\n{}", "-".repeat(78), "-".repeat(78)));
        }
        if let Err(err) = me.exec_text(&command, None) {
            if !nostop {
                res = Err(err);
                break;
            }
        }
        // results changed by the command are kept with the model
        let cur = me.engine.store.e.clone();
        if !cur.is_empty() {
            me.engine.store.estimates.insert(n.clone(), cur);
        }
    }
    me.engine.store.e = saved;
    res
}

fn glob(pat: &str, s: &str) -> bool {
    fn rec(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => rec(&p[1..], s) || (!s.is_empty() && rec(p, &s[1..])),
            (Some(b'?'), Some(_)) => rec(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => rec(&p[1..], &s[1..]),
            _ => false,
        }
    }
    rec(pat.as_bytes(), s.as_bytes())
}

fn linktest(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let e = me.engine.store.e.clone();
    let ecmd = e.macros.get("cmd").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
    let depvar = depvar.split_whitespace().next().unwrap_or("").to_string();
    if depvar.is_empty() || !e.matrices.contains_key("b") {
        return Err(rc_err(301, "last estimates not found"));
    }
    let sample = e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
    // options of the original command (weights and options are kept)
    let cmdline = e.macros.get("cmdline").cloned().unwrap_or_default();
    let (_, crest) = split_verb(&cmdline);
    let (opts_part, weight) = {
        let comma = find_top_level(&crest, ',');
        let opts = comma.map(|i| crest[i + 1..].trim().to_string()).unwrap_or_default();
        let w = crest.find('[').and_then(|i| crest[i..].find(']').map(|j| crest[i..=i + j].to_string())).unwrap_or_default();
        (opts, w)
    };
    let user_opts = rest.trim().trim_start_matches(',').trim().to_string();
    let ds = &mut me.engine.dataset;
    let _ = ds.drop_variable("_hat");
    let _ = ds.drop_variable("_hatsq");
    run_quiet(me, "predict double _hat, xb")?;
    let ds = &mut me.engine.dataset;
    let hat = util::col(ds, "_hat")?;
    let mut hv = Vec::new();
    let mut sq = Variable::new("_hatsq".to_string(), VarType::Float);
    sq.data = hat.iter().enumerate().map(|(i, &h)| {
        let ok = sample.get(i).copied().unwrap_or(false) && !is_missing(h);
        hv.push(ok);
        util::typed_value(VarType::Float, if ok { h * h } else { SYSMISS })
    }).collect();
    if let Some(v) = ds.get_var_mut("_hat") {
        v.var_type = VarType::Float;
        v.data = hat.iter().zip(&hv).map(|(&h, &ok)| util::typed_value(VarType::Float, if ok { h } else { SYSMISS })).collect();
        v.label = "Linear prediction".into();
    }
    ds.add_variable(sq)?;
    me.engine.store.data_changed = true;
    let mut opts = opts_part;
    if !user_opts.is_empty() {
        opts = format!("{opts} {user_opts}");
    }
    let line = format!("{ecmd} {depvar} _hat _hatsq {weight}{}", if opts.trim().is_empty() { String::new() } else { format!(", {}", opts.trim()) });
    run_shown(me, &line)
}
