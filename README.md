# Ferrostat

A statistical computing engine written in Rust. Ferrostat provides a command-line environment for data manipulation, statistical estimation, and graphics, driven by a do-file command language compatible with the widely used `.do`/`.dta` ecosystem.

[![CI](https://img.shields.io/badge/CI-linux%20%7C%20windows%20%7C%20macOS-blue.svg)](.github/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Command Reference](#command-reference)
- [Architecture](#architecture)
- [Data Formats](#data-formats)
- [Graphics](#graphics)
- [Testing and Validation](#testing-and-validation)
- [Known Limitations](#known-limitations)
- [Provenance](#provenance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Ferrostat runs existing analysis scripts unchanged: it reads and writes `.dta` files, executes `.do` files (macros, loops, programs), fits the common econometric models, and exports tables (Excel, tab-delimited), logs (SMCL/text) and graphs (PNG/SVG/PDF). It is designed for:

- **Reproducible research** — batch scripts with deterministic execution on Linux, Windows and macOS
- **Data archaeology** — reading legacy `.dta` files (formats 113–119) on any platform
- **Statistical education** — every algorithm is readable Rust
- **Embedding** — use the engine and its datasets from Rust applications

---

## Features

### Language
- Macros: `` `local' ``, `$global`, `` `=exp' ``, `` `:extended function' ``, `` `++i' ``, nested macros, compound quotes
- Control flow: `foreach` (in/of local/global/varlist/newlist/numlist), `forvalues`, `while`, `if`/`else if`/`else`, `continue`, `break`
- Prefixes: `quietly`, `noisily`, `capture` (sets `_rc`), `by`/`bysort` (group-aware `_n`, `_N`, subscripts)
- Programs: `program define … end` with `syntax`, `args`, `tokenize`, `gettoken`, `marksample`, `tempvar`/`tempname`/`tempfile`, `return`/`ereturn`
- Stored results `r()`, `e()`, `c()`, `_b[]`, `_se[]`, scalars and a matrix language (`J()`, `I()`, `diag()`, `inv()`, joins, subscripts, `accum`, `symeigen`, `mkmat`/`svmat`)
- Command abbreviations (`gen`, `su`, `reg`, `tab`, …), varlist wildcards and ranges, factor variables (`i.`, `ib#.`, `c.`, `#`, `##`) and time-series operators (`L.`, `F.`, `D.`, `S.`)
- `#delimit ;`, `///` continuation, `//` and `/* */` comments, `do`/`run`/`include`, `log using`, `set rmsg on`

### Expressions
Missing values sort above all numbers (`.` < `.a` < … < `.z`), arithmetic with missing yields missing, and logical tests follow the usual rules (`x > 5` is true for missing `x`). Around 200 functions: mathematics, distributions (`normal`, `invnormal`, `t`/`ttail`, `chi2`/`chi2tail`, `F`/`Ftail`, `binomial`, `poisson`, `gammap`, `ibeta`, inverses), random numbers (`runiform`, `rnormal`, … with `set seed`), strings (`substr`, `strpos`, `subinstr`, `regexm`, `ustrregexra`, `word`, `strtrim`, …), dates (`mdy`, `date`, `clock`, `year`, `dofm`, `%td`/`%tc`/`%tm`/`%tq` formats), programming (`cond`, `inlist`, `inrange`, `missing`, running `sum()`).

### Data management
`generate`/`replace` (storage types, sequential evaluation), `egen` (~35 functions), `drop`/`keep`, `rename` (groups, wildcards, `lower`/`upper`), `order`, `sort`/`gsort`, `label` (define/values/variable/data/list), `encode`/`decode`, `destring`/`tostring`, `recode`, `mvencode`/`mvdecode`, `split`, `format`, `compress`, `recast`, `collapse` (all statistics), `contract`, `expand`, `fillin`, `reshape long|wide` (numeric and string `j`), `merge 1:1|m:1|1:m|m:m` (with `keep()`, `keepusing()`, `update`, `replace`, `assert()`), `append`, `joinby`, `cross`, `xpose`, `sample`, `duplicates`, `isid`, `levelsof`, `preserve`/`restore`, `input`, `set obs`, `tsset`/`xtset`.

### Statistics
| Area | Commands |
|------|----------|
| Descriptive | `summarize` (incl. `detail`, weights), `tabulate` (one/two-way, `chi2`, `lrchi2`, `exact`, `V`, `gamma`, `taub`, `generate()`), `tab1`, `tab2`, `tabstat`, `correlate`, `pwcorr`, `spearman`, `centile`, `mean`, `ci`, `codebook`, `misstable` |
| Tests | `ttest` (one-sample, paired, by-group, unequal/Welch), `prtest`, `sdtest`, `ranksum`, `signrank`, `signtest`, `kwallis`, `oneway`, `ksmirnov`, `swilk`, `hausman` |
| Linear models | `regress` (robust, cluster, `fw`/`aw`/`pw`/`iw`), `anova`, `areg`, `qreg`, `ivregress 2sls` |
| Panel | `xtreg` (`fe`, `re`, `be`, cluster), `xtivreg`/`xtivreg2` (`fe`, cluster, Hansen J, first-stage and endogeneity tests), `xtpmg` (PMG, MG; Pesaran CD test) |
| Nonlinear | `logit`, `logistic`, `probit`, `cloglog`, `poisson`, `nbreg`, `ologit`, `oprobit`, `mlogit`, `tobit`, `glm` (6 families, 7 links) |
| Post-estimation | `predict` (xb, residuals, stdp/stdf/stdr, leverage, rstandard/rstudent, cooksd, pr, n, ir, mu, u/e/xbu, scores), `test`, `testparm`, `lincom`, `margins` (predictive margins, `dydx()`, `at()`, `atmeans`; delta-method SEs), `estimates store/restore/table/stats`, `estat ic/vce/hettest/ovtest/vif/summarize/firststage` |
| Spatial | `shp2dta` (ESRI shapefile + dBase import), `spmat contiguity` (queen/rook, row/minmax/spectral normalisation), `spmat lag`, `spmap` (choropleths) |
| Other | `kdensity` (8 kernels), `outreg2`-style regression tables, `putexcel` (.xlsx) |

---

## Quick Start

```bash
cargo build --release

# Interactive session
./target/release/ferrostat

# Run a do-file (commands are echoed, as in batch mode)
./target/release/ferrostat --do analysis.do
```

### Example Session

```
. set obs 200
. generate id = ceil(_n / 10)
. generate region = mod(id, 3) + 1
. generate double x1 = sin(_n) * 2 + id / 10
. generate double y = 0.5 * x1 + region + cos(_n * 3)
. regress y x1 i.region, vce(cluster id)
Linear regression                                   Number of obs   =        200
                                                    F(3, 19)        =    3102.65
                                                    Prob > F        =     0.0000
                                                    R-squared       =     0.7073
                                                    Root MSE        =     .71481

                                    (Std. err. adjusted for 20 clusters in id)

------------------------------------------------------------------------------
           y | Coefficient  Std. err.       t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .5001318   .0078421    63.78   0.000     .4837182    .5165454
             |
      region |
           2 |   .9955737   .0262567    37.92   0.000     .9406177     1.05053
           3 |   1.996566   .0259355    76.98   0.000     1.942282     2.05085
             |
       _cons |    .997623   .0205855    48.46   0.000     .9545371    1.040709
------------------------------------------------------------------------------
. scatter y x1
. graph export "fit.png", replace
file fit.png written in PNG format
```

---

## Installation

### Pre-built binaries

Tagged releases publish binaries for Linux (x86_64), Windows (x86_64) and macOS (Apple Silicon and Intel); see `.github/workflows/release.yml`.

### From Source

Requires a stable Rust toolchain.

```bash
git clone <repo-url>
cd ferrostat
cargo build --release
# Linux/macOS
sudo cp target/release/ferrostat /usr/local/bin/
# Windows: target\release\ferrostat.exe
```

### System Dependencies

```bash
# Ubuntu/Debian (fonts for graph rendering)
sudo apt-get install libfontconfig1-dev
```
Windows and macOS need no extra packages.

---

## Command Reference

### Syntax

```
[prefix:] command [varlist | = exp] [if exp] [in range] [weight] [using filename] [, options]
```

Commands, prefixes and options follow the familiar conventions of do-file syntax, including minimal abbreviations, `[fw=]`/`[aw=]`/`[pw=]`/`[iw=]` weights and `in f/l` ranges. Error messages end with the usual return code (`r(111);`), which `capture` stores in `_rc`; in batch mode the process exit status is the return code of the failing command.

Community packages that Ferrostat implements natively (`outreg2`, `xtivreg2`, `xtpmg`, `spmap`, `shp2dta`, `spmat`) are reported by `which` and accepted by `ssc install`, which installs nothing. Other community packages are not downloaded.

---

## Architecture

```
ferrostat-repl        -- terminal REPL and batch mode (--do)
ferrostat-ado         -- interpreter: statements, blocks, macros, programs, logs, putexcel, outreg2
ferrostat-engine      -- command registry, abbreviations, by-groups, data-level commands
ferrostat-cmd         -- commands, expression evaluator, estimation framework, graphics
ferrostat-parser      -- lexer, expression parser, command-line splitter
ferrostat-core        -- DataSet, Variable, Value, missing values, .dta read/write
ferrostat-tests       -- behavioural test suite and independent reference values
```

### Parser and evaluator

1. **Lexer** — tokens for identifiers, numbers, strings (including compound quotes), operators, extended missing values and factor-variable names.
2. **Parser** — precedence from loosest to tightest: `|`, `&`, relational, `+ -`, `* /`, unary minus, `^` (right-associative), `!`, postfix calls and subscripts.
3. **Command splitter** — separates varlist, `if`, `in`, weights, `using` and options (options keep their raw text, so `by(a b)` and `title("…")` work).
4. **Evaluator** — evaluates per observation with access to stored results, by-groups and time-series structure.

### Estimation framework
Design matrices are built with factor variables, interactions and time-series operators; collinear columns are detected and reported as omitted. Variance estimators include OLS, heteroskedasticity-robust (HC1) and cluster-robust sandwiches; likelihood models are fitted by Newton–Raphson with step halving.

---

## Data Formats

### .dta (formats 113–119)

- Reads formats 113, 114, 115, 117, 118 and 119 (both byte orders); writes 115, 117, 118 and 119 (`save`, `saveold, version()`)
- All storage types: `byte`, `int`, `long`, `float`, `double`, `str1`–`str2045`, `strL`
- Variable labels, value labels, display formats, characteristics, sort order and dataset label are preserved
- The 27 missing values per type (`.`, `.a`–`.z`) are preserved exactly

### Text and spreadsheets

- `import delimited` / `export delimited` (delimiters, `varnames()`, `case()`, `stringcols()`, locale-aware numbers via `parselocale()`, `decimalseparator()`, `groupseparator()`)
- `putexcel` writes `.xlsx` workbooks (cells, expressions, matrices with names); `outreg2` writes tab-delimited regression tables

---

## Graphics

Graph commands build the graph in memory; nothing is written until `graph export` (or a `saving()` option):

- `twoway` with `scatter`, `line`, `connected`, `area`, `bar`, `lfit` layers; `histogram`, `graph bar`, `graph box`, `kdensity`, `spmap` choropleths, `graph combine` grids
- `title()`, `xtitle()`, `ytitle()`, `yline()`, `xline()`, `legend()`, `name()`
- Export to **PNG**, **SVG** and **PDF**; `graph save`/`graph use` store graphs in Ferrostat's own text format

Rendering uses the open-source [`plotters`](https://crates.io/crates/plotters) crate.

---

## Testing and Validation

```bash
cargo test --workspace
```

- `ferrostat-tests` drives the interpreter exactly as a user would and checks results against **independent reference values** computed by `ferrostat-tests/reference/reference.py`, a standard-library-only Python implementation of the textbook formulas (OLS, robust SEs, logit/probit/poisson MLE, t tests, chi-square, ANOVA, fixed effects, kernel density).
- CI builds and tests on Linux, Windows and macOS (`.github/workflows/ci.yml`).
- Real-world validation: two research do-files (panel regressions, dynamic factor analysis, PMG/MG panel ARDL) were run unmodified apart from their machine-specific `cd`. Their results were compared with the logs of the original runs: all 686 coefficient rows of the regression tasks match (coefficient and standard error), as do the factor count and PMG/MG estimates and the Pesaran CD statistic.

---

## Known Limitations

- Mata and Python blocks are not supported.
- Community packages are not downloaded; only the natively implemented ones above are available.
- `xtivreg`: the header Wald χ² can differ from the original software (coefficients and standard errors match).
- `xtivreg2` diagnostics use robust first-stage F and two-step-GMM Hansen J; the Kleibergen–Paap rank statistics are not implemented.
- Random-number streams are reproducible within Ferrostat but differ from other software for the same seed.
- `graph combine` lays out graphs in a grid; graph schemes and most styling options are accepted but ignored.

---

## Provenance

- The `.dta` format is implemented from publicly available documentation.
- All statistical algorithms are textbook implementations (Wooldridge, Greene, Cameron & Trivedi, Pesaran–Shin–Smith) or standard numerical methods.
- Every parser rule, AST node, and evaluation function is original code.
- Expected test values come from an independent reference implementation, not from running other software.
- Graphics are produced via the open-source `plotters` crate.

This project exists to ensure long-term accessibility of statistical datasets and workflows without licensing dependencies.

---

## Contributing

Contributions are welcome! Priority areas:

- [ ] Survival analysis: `stset`, `stcox`, `streg`
- [ ] Time series: `arima`, `var`, `newey`, `prais`
- [ ] Survey data: `svyset`, `svy:`
- [ ] Multiple imputation: `mi`
- [ ] `heckman`, `clogit`, `xtlogit`, `mixed`
- [ ] A Mata-compatible matrix programming language
- [ ] Graph schemes and the remaining styling options

Please open an issue before large refactors.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- The `csv`, `plotters`, `chrono`, `regex` and `statrs` crates for foundational functionality.
- The statistical computing tradition that inspired the user interface.
- The Rust community for zero-cost abstractions that make numerical computing ergonomic.
