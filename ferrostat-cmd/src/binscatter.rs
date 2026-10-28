//! `binscatter`: binned scatter plots. The x variable is cut into
//! quantile bins (or one bin per distinct value), the mean (or median) of
//! y is plotted against the mean of x within each bin, and a linear or
//! quadratic fit of the underlying data is overlaid. Controls and an
//! absorbed fixed effect are partialled out of both variables first (the
//! means added back), as in the Frisch-Waugh-Lovell representation used by
//! Cattaneo, Crump, Farrell & Feng (2024).

use ferrostat_core::{vfs, DataSet};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

pub struct Binscatter;

fn wsum(w: &[f64]) -> f64 {
    w.iter().sum()
}

fn wmean(v: &[f64], w: &[f64]) -> f64 {
    v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / wsum(w)
}

fn wmedian(v: &[f64], w: &[f64]) -> f64 {
    let mut p: Vec<(f64, f64)> = v.iter().copied().zip(w.iter().copied()).collect();
    util::pctile_weighted(&mut p, 50.0)
}

/// Weighted least squares; returns (coefficients, standard errors, residuals).
fn ols(y: &[f64], x: &[Vec<f64>], w: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = x.len();
    let n = y.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for a in 0..k {
        for i in 0..n {
            xty[a] += x[a][i] * w[i] * y[i];
        }
        for c in 0..=a {
            let v: f64 = (0..n).map(|i| x[a][i] * w[i] * x[c][i]).sum();
            xtx[(a, c)] = v;
            xtx[(c, a)] = v;
        }
    }
    let (inv, _) = xtx.sym_inverse(1e-12);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    let resid: Vec<f64> = (0..n).map(|i| y[i] - (0..k).map(|a| b[a] * x[a][i]).sum::<f64>()).collect();
    let rss: f64 = resid.iter().zip(w).map(|(r, wi)| r * r * wi).sum();
    let dof = (wsum(w) - k as f64).max(1.0);
    let se: Vec<f64> = (0..k).map(|a| (inv[(a, a)] * rss / dof).sqrt()).collect();
    (b, se, resid)
}

/// Residualise `v` on controls and (optionally) one absorbed grouping.
fn residualize(v: &[f64], controls: &[Vec<f64>], groups: Option<&[usize]>, w: &[f64]) -> Vec<f64> {
    let demean = |x: &[f64]| -> Vec<f64> {
        match groups {
            None => x.to_vec(),
            Some(g) => {
                let ng = g.iter().max().map_or(0, |m| m + 1);
                let mut s = vec![0.0; ng];
                let mut sw = vec![0.0; ng];
                for i in 0..x.len() {
                    s[g[i]] += w[i] * x[i];
                    sw[g[i]] += w[i];
                }
                (0..x.len()).map(|i| x[i] - s[g[i]] / sw[g[i]]).collect()
            }
        }
    };
    let vt = demean(v);
    let mut xs: Vec<Vec<f64>> = controls.iter().map(|c| demean(c)).collect();
    if groups.is_none() {
        xs.push(vec![1.0; v.len()]);
    }
    if xs.is_empty() {
        return vt;
    }
    ols(&vt, &xs, w).2
}

#[derive(Clone, Copy, PartialEq)]
enum LineType {
    Lfit,
    Qfit,
    Connect,
    None,
}

impl CommandHandler for Binscatter {
    fn name(&self) -> &'static str {
        "binscatter"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let xname = vars[vars.len() - 1].clone();
        let ynames: Vec<String> = vars[..vars.len() - 1].to_vec();
        let opt_vars = |name: &str, min: usize| -> Result<Vec<String>, String> {
            match cmd.opt(name, min) {
                Some(t) if !t.trim().is_empty() => crate::varlist::expand(ds, t),
                _ => Ok(Vec::new()),
            }
        };
        let controls = opt_vars("controls", 4)?;
        let absorb = opt_vars("absorb", 3)?;
        if absorb.len() > 1 {
            return Err(rc_err(103, "absorb() takes a single variable"));
        }
        let by = opt_vars("by", 2)?;
        if by.len() > 1 {
            return Err(rc_err(103, "by() takes a single variable"));
        }
        let nq = util::opt_num(cmd, "nquantiles", 2)?.unwrap_or(20.0);
        if nq < 1.0 || nq.fract() != 0.0 {
            return Err(rc_err(198, "nquantiles() must be a positive integer"));
        }
        let nq = nq as usize;
        let discrete = cmd.has_opt("discrete", 4);
        let medians = cmd.has_opt("medians", 3);
        let addmean = !cmd.has_opt("noaddmean", 5);
        let linetype = match cmd.opt("linetype", 5).map(|s| s.trim().to_lowercase()) {
            None => LineType::Lfit,
            Some(s) => match s.as_str() {
                "lfit" => LineType::Lfit,
                "qfit" => LineType::Qfit,
                "connect" => LineType::Connect,
                "none" => LineType::None,
                _ => return Err(rc_err(198, "linetype() must be lfit, qfit, connect or none")),
            },
        };
        let rd: Vec<f64> = match cmd.opt("rd", 2) {
            Some(t) => t.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "rd() must be a numlist"))).collect::<Result<_, _>>()?,
            None => Vec::new(),
        };
        let reportreg = cmd.has_opt("reportreg", 4);
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut mv = vars.clone();
        mv.extend(controls.iter().cloned());
        mv.extend(absorb.iter().cloned());
        mv.extend(by.iter().cloned());
        util::markout(ds, &mut mask, &mv);
        let wts = util::weights(ds, st, cmd, &mut mask, "fa")?;
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let n = rows.len();
        let w: Vec<f64> = match &wts {
            Some(wt) => rows.iter().map(|&i| wt.w[i]).collect(),
            None => vec![1.0; n],
        };
        let get = |name: &str| -> Result<Vec<f64>, String> {
            let c = util::col(ds, name)?;
            Ok(rows.iter().map(|&i| c[i]).collect())
        };
        let ctrl: Vec<Vec<f64>> = controls.iter().map(|c| get(c)).collect::<Result<_, _>>()?;
        let groups: Option<Vec<usize>> = match absorb.first() {
            Some(a) => {
                let (codes, _) = {
                    let var = ds.get_var(a).unwrap();
                    if var.var_type.is_string() {
                        let keys: Vec<&str> = rows.iter().map(|&i| var.str_at(i)).collect();
                        crate::hdfe::codes_str(&keys)
                    } else {
                        let keys: Vec<f64> = rows.iter().map(|&i| var.data[i].num()).collect();
                        crate::hdfe::codes_f64(&keys)
                    }
                };
                Some(codes.into_iter().map(|c| c as usize).collect())
            }
            None => None,
        };
        let adjust = !ctrl.is_empty() || groups.is_some();
        let prep = |v: Vec<f64>| -> Vec<f64> {
            if !adjust {
                return v;
            }
            let m = wmean(&v, &w);
            let r = residualize(&v, &ctrl, groups.as_deref(), &w);
            if addmean { r.iter().map(|x| x + m).collect() } else { r }
        };
        let x = prep(get(&xname)?);
        let ys: Vec<Vec<f64>> = ynames.iter().map(|y| get(y).map(prep)).collect::<Result<_, _>>()?;
        // by groups
        let (by_ids, by_labels): (Vec<usize>, Vec<String>) = match by.first() {
            None => (vec![0; n], vec![String::new()]),
            Some(b) => {
                let var = ds.get_var(b).unwrap();
                let mut labels: Vec<(String, f64)> = Vec::new();
                let mut ids = Vec::with_capacity(n);
                let keyv: Vec<(String, f64)> = rows.iter().map(|&i| (util::cell_text(ds, var, i, true), if var.var_type.is_string() { 0.0 } else { var.data[i].num() })).collect();
                let mut uniq: Vec<(String, f64)> = keyv.clone();
                uniq.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                uniq.dedup();
                for k in &keyv {
                    ids.push(uniq.iter().position(|u| u == k).unwrap());
                }
                labels.extend(uniq);
                (ids, labels.into_iter().map(|(s, _)| format!("{b}={s}")).collect())
            }
        };
        let ngroups = by_labels.len();
        let mut layers: Vec<Layer> = Vec::new();
        let mut csv = String::new();
        let mut header = vec![];
        if !by.is_empty() {
            header.push(by[0].clone());
        }
        header.push(xname.clone());
        header.extend(ynames.iter().cloned());
        csv.push_str(&header.join(","));
        csv.push('\n');
        let mut report: Vec<String> = Vec::new();
        let mut coefs: Vec<Vec<f64>> = Vec::new();
        let mut nbins_total = 0usize;
        for g in 0..ngroups {
            let idx: Vec<usize> = (0..n).filter(|&i| by_ids[i] == g).collect();
            if idx.is_empty() {
                continue;
            }
            let gx: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
            let gw: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
            // bins
            let bin: Vec<usize> = if discrete {
                let mut u = gx.clone();
                u.sort_by(|a, b| a.total_cmp(b));
                u.dedup();
                gx.iter().map(|v| u.iter().position(|q| q == v).unwrap()).collect()
            } else {
                let mut pairs: Vec<(f64, f64)> = gx.iter().copied().zip(gw.iter().copied()).collect();
                let cuts: Vec<f64> = (1..nq).map(|k| util::pctile_weighted(&mut pairs, 100.0 * k as f64 / nq as f64)).collect();
                gx.iter().map(|v| cuts.iter().filter(|c| **c < *v).count()).collect()
            };
            let nb = bin.iter().max().map_or(0, |m| m + 1);
            let mut bx = Vec::new();
            let mut by_vals: Vec<Vec<f64>> = vec![Vec::new(); ys.len()];
            for b in 0..nb {
                let members: Vec<usize> = (0..idx.len()).filter(|&i| bin[i] == b).collect();
                if members.is_empty() {
                    continue;
                }
                let mx: Vec<f64> = members.iter().map(|&i| gx[i]).collect();
                let mw: Vec<f64> = members.iter().map(|&i| gw[i]).collect();
                bx.push(if medians { wmedian(&mx, &mw) } else { wmean(&mx, &mw) });
                for (k, yv) in ys.iter().enumerate() {
                    let my: Vec<f64> = members.iter().map(|&i| yv[idx[i]]).collect();
                    by_vals[k].push(if medians { wmedian(&my, &mw) } else { wmean(&my, &mw) });
                }
            }
            nbins_total += bx.len();
            for j in 0..bx.len() {
                let mut row = Vec::new();
                if !by.is_empty() {
                    row.push(by_labels[g].split_once('=').map(|(_, v)| v.to_string()).unwrap_or_default());
                }
                row.push(crate::format::fmt_g(bx[j], 12).trim().to_string());
                for yv in &by_vals {
                    row.push(crate::format::fmt_g(yv[j], 12).trim().to_string());
                }
                csv.push_str(&row.join(","));
                csv.push('\n');
            }
            for (k, yname) in ynames.iter().enumerate() {
                let series = k * ngroups + g;
                let label = match (ngroups > 1, ynames.len() > 1) {
                    (true, true) => format!("{yname}, {}", by_labels[g]),
                    (true, false) => by_labels[g].clone(),
                    (false, true) => yname.clone(),
                    _ => String::new(),
                };
                let pts: Vec<(f64, f64)> = bx.iter().copied().zip(by_vals[k].iter().copied()).collect();
                let color = graph::palette(&st.scheme).get(series % graph::palette(&st.scheme).len().max(1)).map(|c| (c.0, c.1, c.2));
                layers.push(Layer { kind: if linetype == LineType::Connect { LayerKind::Connected } else { LayerKind::Scatter }, points: pts, label: label.clone(), color, ..Default::default() });
                if matches!(linetype, LineType::Lfit | LineType::Qfit) {
                    // separate fits between rd cutpoints
                    let mut edges = vec![f64::NEG_INFINITY];
                    edges.extend(rd.iter().copied());
                    edges.push(f64::INFINITY);
                    for seg in edges.windows(2) {
                        let si: Vec<usize> = (0..idx.len()).filter(|&i| gx[i] >= seg[0] && gx[i] < seg[1]).collect();
                        if si.len() < 3 {
                            continue;
                        }
                        let sy: Vec<f64> = si.iter().map(|&i| ys[k][idx[i]]).collect();
                        let sx: Vec<f64> = si.iter().map(|&i| gx[i]).collect();
                        let sw: Vec<f64> = si.iter().map(|&i| gw[i]).collect();
                        let mut xs = vec![vec![1.0; sx.len()], sx.clone()];
                        if linetype == LineType::Qfit {
                            xs.push(sx.iter().map(|v| v * v).collect());
                        }
                        let (b, se, _) = ols(&sy, &xs, &sw);
                        coefs.push(b.clone());
                        if reportreg {
                            report.push(String::new());
                            let segtxt = if rd.is_empty() { String::new() } else { format!(" ({} <= {xname} < {})", crate::format::fmt_g(seg[0], 9).trim(), crate::format::fmt_g(seg[1], 9).trim()) };
                            let grp = if ngroups > 1 { format!(", {}", by_labels[g]) } else { String::new() };
                            report.push(format!("Fit of {yname} on {xname}{grp}{segtxt}: N = {}", sx.len()));
                            let mut names = vec!["_cons".to_string(), xname.clone()];
                            if linetype == LineType::Qfit {
                                names.push(format!("{xname}^2"));
                            }
                            report.push(format!("{:>14} | {:>12} {:>12}", "", "Coefficient", "Std. err."));
                            for (nm, (bv, sv)) in names.iter().zip(b.iter().zip(&se)) {
                                report.push(format!("{:>14} | {:>12} {:>12}", nm, crate::format::fmt_g(*bv, 10).trim(), crate::format::fmt_g(*sv, 10).trim()));
                            }
                        }
                        // the line spans the binned points of the segment
                        let bxs: Vec<f64> = bx.iter().copied().filter(|v| *v >= seg[0] && *v < seg[1]).collect();
                        if bxs.is_empty() {
                            continue;
                        }
                        let lo = bxs.iter().cloned().fold(f64::INFINITY, f64::min);
                        let hi = bxs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let line: Vec<(f64, f64)> = (0..=100)
                            .map(|t| {
                                let xv = lo + (hi - lo) * t as f64 / 100.0;
                                (xv, b[0] + b[1] * xv + if b.len() > 2 { b[2] * xv * xv } else { 0.0 })
                            })
                            .collect();
                        layers.push(Layer { kind: LayerKind::Line, points: line, color, ..Default::default() });
                    }
                }
            }
        }
        let lab = |v: &str| ds.get_var(v).map(|x| if x.label.is_empty() { v.to_string() } else { x.label.clone() }).unwrap_or_else(|| v.to_string());
        let mut g = Graph {
            xtitle: lab(&xname),
            ytitle: if ynames.len() == 1 { lab(&ynames[0]) } else { String::new() },
            layers,
            xlines: rd.clone(),
            legend: ngroups > 1 || ynames.len() > 1,
            ..Default::default()
        };
        let mut msg = graph::finish(&mut g, cmd, st)?;
        if !rd.is_empty() && g.xlines.is_empty() {
            g.xlines = rd.clone();
            st.graphs.insert(g.name.clone(), g.clone());
            st.graph = Some(g.clone());
        }
        if let Some(path) = cmd.opt("savedata", 5) {
            let path = util::unquote(path.split(',').next().unwrap_or(path));
            let csvp = format!("{path}.csv");
            let dop = format!("{path}.do");
            let replace = cmd.has_opt("replace", 7);
            if !replace && (vfs::exists(&csvp) || vfs::exists(&dop)) {
                return Err(rc_err(602, format!("file {csvp} already exists")));
            }
            vfs::write(&csvp, csv.as_bytes()).map_err(|e| rc_err(603, format!("{csvp}: {e}")))?;
            let mut dofile = format!("// binned scatter data\nimport delimited using \"{csvp}\", clear\n");
            let ylist = ynames.join(" ");
            dofile.push_str(&format!("twoway (scatter {ylist} {xname}), xtitle(\"{}\")\n", lab(&xname)));
            vfs::write(&dop, dofile.as_bytes()).map_err(|e| rc_err(603, format!("{dop}: {e}")))?;
            if !msg.is_empty() {
                msg.push('\n');
            }
            msg.push_str(&format!("(file {csvp} written)\n(file {dop} written)"));
        }
        if let Some(path) = cmd.opt("savegraph", 5) {
            let path = util::unquote(path.split(',').next().unwrap_or(path));
            if path.ends_with(".gph") {
                graph::save_gph(&g, &path)?;
            } else {
                graph::render(&g, &path, 800, 600)?;
            }
            if !msg.is_empty() {
                msg.push('\n');
            }
            msg.push_str(&format!("(file {path} written)"));
        }
        let r = st.begin_r();
        r.set("N", if wts.as_ref().is_some_and(|w| w.kind == util::WeightKind::F) { wsum(&w) } else { n as f64 });
        r.set("nbins", nbins_total as f64);
        r.set_str("graphcmd", format!("binscatter {}", cmd.raw.trim().strip_prefix("binscatter").unwrap_or(cmd.raw.trim()).trim()));
        if !coefs.is_empty() {
            let k = coefs[0].len();
            let mut m = Matrix::zeros(coefs.len(), k);
            for (i, c) in coefs.iter().enumerate() {
                for j in 0..k.min(c.len()) {
                    m[(i, j)] = c[j];
                }
            }
            let mut cn = vec!["_cons".to_string(), xname.clone()];
            if k > 2 {
                cn.push(format!("{xname}2"));
            }
            r.set_mat("coefs", m.with_names((1..=coefs.len()).map(|i| format!("fit{i}")).collect(), cn));
        }
        let mut out = report;
        if !msg.is_empty() {
            out.push(msg);
        }
        Ok(out.join("\n").trim_start_matches('\n').to_string())
    }
}
