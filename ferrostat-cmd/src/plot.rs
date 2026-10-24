use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

use plotters::prelude::*;


/// Scatter plot to PNG.
pub struct Scatter;

impl CommandHandler for Scatter {
    fn name(&self) -> &'static str { "scatter" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("scatter requires yvar xvar".into());
        }
        let yvar = &cmd.args[0];
        let xvar = &cmd.args[1];

        let y = ds.get_var(yvar).ok_or_else(|| format!("variable {} not found", yvar))?
            .as_f64_vec().ok_or("y variable must be numeric")?;
        let x = ds.get_var(xvar).ok_or_else(|| format!("variable {} not found", xvar))?
            .as_f64_vec().ok_or("x variable must be numeric")?;

        let pairs: Vec<(f64, f64)> = (0..y.len().min(x.len()))
            .filter_map(|i| {
                let xv = x.get(i)?;
                let yv = y.get(i)?;
                if xv.is_nan() || yv.is_nan() { None } else { Some((*xv, *yv)) }
            })
            .collect();

        if pairs.is_empty() {
            return Err("no valid observations to plot".into());
        }

        let xmin = pairs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let xmax = pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let ymin = pairs.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let ymax = pairs.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let outpath = format!("scatter_{}_{}.png", yvar, xvar);
        let root = BitMapBackend::new(&outpath, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| e.to_string())?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Scatter: {} vs {}", yvar, xvar), ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (xmin - (xmax-xmin)*0.1)..(xmax + (xmax-xmin)*0.1),
                (ymin - (ymax-ymin)*0.1)..(ymax + (ymax-ymin)*0.1),
            )
            .map_err(|e| e.to_string())?;

        chart.configure_mesh().draw().map_err(|e| e.to_string())?;

        chart.draw_series(
            pairs.iter().map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled()))
        ).map_err(|e| e.to_string())?;

        root.present().map_err(|e| e.to_string())?;
        Ok(format!("scatter plot saved to {}", outpath))
    }
}

/// Histogram plot to PNG.
pub struct Histogram;

impl CommandHandler for Histogram {
    fn name(&self) -> &'static str { "histogram" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("histogram requires a variable".into());
        }
        let var = &cmd.args[0];
        let vals: Vec<f64> = ds.get_var(var)
            .ok_or_else(|| format!("variable {} not found", var))?
            .as_f64_vec().ok_or("variable must be numeric")?
            .into_iter().filter(|v| !v.is_nan()).collect();

        if vals.is_empty() {
            return Err("no valid observations".into());
        }

        let bins = 20;
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max - min) / bins as f64;

        let mut counts = vec![0usize; bins];
        for v in &vals {
            let idx = ((v - min) / bin_width).min(bins as f64 - 1.0) as usize;
            counts[idx] += 1;
        }

        let max_count = *counts.iter().max().unwrap_or(&1);

        let outpath = format!("histogram_{}.png", var);
        let root = BitMapBackend::new(&outpath, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| e.to_string())?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Histogram of {}", var), ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (min - bin_width)..(max + bin_width),
                0.0..((max_count + max_count / 10) as f64),
            )
            .map_err(|e| e.to_string())?;

        chart.configure_mesh().draw().map_err(|e| e.to_string())?;

        let data: Vec<(f64, f64)> = counts.iter().enumerate()
            .map(|(i, &c)| (min + i as f64 * bin_width, c as f64))
            .collect();

        chart.draw_series(
            data.iter().map(|(x, c)| {
                Rectangle::new(
                    [(*x, 0.0), (*x + bin_width, *c)],
                    BLUE.filled(),
                )
            })
        ).map_err(|e| e.to_string())?;

        root.present().map_err(|e| e.to_string())?;
        Ok(format!("histogram saved to {}", outpath))
    }
}

/// Line plot to PNG.
pub struct LinePlot;

impl CommandHandler for LinePlot {
    fn name(&self) -> &'static str { "line" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("line requires yvar xvar".into());
        }
        let yvar = &cmd.args[0];
        let xvar = &cmd.args[1];

        let mut pairs: Vec<(f64, f64)> = Vec::new();
        let y = ds.get_var(yvar).ok_or_else(|| format!("variable {} not found", yvar))?;
        let x = ds.get_var(xvar).ok_or_else(|| format!("variable {} not found", xvar))?;

        for i in 0..ds.n_obs {
            if let (Some(yv), Some(xv)) = (y.data[i].as_f64(), x.data[i].as_f64()) {
                if !yv.is_nan() && !xv.is_nan() {
                    pairs.push((xv, yv));
                }
            }
        }

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if pairs.is_empty() {
            return Err("no valid observations".into());
        }

        let xmin = pairs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let xmax = pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let ymin = pairs.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let ymax = pairs.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

        let outpath = format!("line_{}_{}.png", yvar, xvar);
        let root = BitMapBackend::new(&outpath, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| e.to_string())?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Line: {} vs {}", yvar, xvar), ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (xmin - (xmax-xmin)*0.1)..(xmax + (xmax-xmin)*0.1),
                (ymin - (ymax-ymin)*0.1)..(ymax + (ymax-ymin)*0.1),
            )
            .map_err(|e| e.to_string())?;

        chart.configure_mesh().draw().map_err(|e| e.to_string())?;

        chart.draw_series(
            LineSeries::new(pairs.into_iter(), BLUE)
        ).map_err(|e| e.to_string())?;

        root.present().map_err(|e| e.to_string())?;
        Ok(format!("line plot saved to {}", outpath))
    }
}

/// Bar chart to PNG.
pub struct BarChart;

impl CommandHandler for BarChart {
    fn name(&self) -> &'static str { "bar" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("bar requires a categorical variable".into());
        }
        let var = &cmd.args[0];
        let v = ds.get_var(var).ok_or_else(|| format!("variable {} not found", var))?;

        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for i in 0..ds.n_obs {
            let s = match &v.data[i] {
                Value::Str(s) => s.clone(),
                Value::Double(v) => format!("{:.0}", v),
                Value::Float(v) => format!("{:.0}", v),
                Value::Int(v) => v.to_string(),
                Value::Long(v) => v.to_string(),
                Value::Byte(v) => v.to_string(),
                _ => continue,
            };
            *counts.entry(s).or_insert(0) += 1;
        }

        let mut items: Vec<_> = counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        let max_count = items.iter().map(|i| i.1).max().unwrap_or(1);
        let labels: Vec<String> = items.iter().map(|(l, _)| l.clone()).collect();

        let outpath = format!("bar_{}.png", var);
        let root = BitMapBackend::new(&outpath, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| e.to_string())?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Bar chart of {}", var), ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0..labels.len(),
                0..(max_count + max_count / 10),
            )
            .map_err(|e| e.to_string())?;

        chart.configure_mesh()
            .x_labels(labels.len())
            .x_label_formatter(&|x| labels.get(*x).cloned().unwrap_or_default())
            .draw().map_err(|e| e.to_string())?;

        let data: Vec<(usize, usize)> = items.iter().enumerate().map(|(i, (_, c))| (i, *c)).collect();
        chart.draw_series(
            data.iter().map(|(i, c)| {
                let mut bar = Rectangle::new(
                    [(*i, 0), (*i + 1, *c)],
                    BLUE.filled(),
                );
                bar.set_margin(5, 5, 0, 0);
                bar
            })
        ).map_err(|e| e.to_string())?;

        root.present().map_err(|e| e.to_string())?;
        Ok(format!("bar chart saved to {}", outpath))
    }
}
