# Ferrostat

A statistical computing engine written in Rust. Ferrostat provides a command-line environment for data manipulation, statistical estimation, and plotting using an expressive command syntax and established open-source crates.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Command Reference](#command-reference)
- [Architecture](#architecture)
- [Data Formats](#data-formats)
- [Plotting](#plotting)

- [Contributing](#contributing)
- [License](#license)

---

## Overview

Ferrostat is a statistical computing environment written in Rust. It reads and writes DTA format files, produces publication-quality PNG plots, and executes an expressive command syntax. It is designed for:

- **Data archaeology** — reading legacy `.dta` files on any platform
- **Reproducible research** — batch scripts (`.do` files) with deterministic execution
- **Statistical education** — an engine where every algorithm is readable Rust
- **Rust ecosystem integration** — embed statistical dataframes in your Rust applications

---

## Features

### Data I/O
| Feature | Status |
|---------|--------|
| Read/write DTA 118 | ✅ Native implementation |
| Import CSV | ✅ via `csv` crate |
| Export CSV | ✅ via `csv` crate |
| Excel / DB connectors | 🔲 Planned |

### Data Manipulation
| Command | Description |
|---------|-------------|
| `use` / `save` / `sysuse` | Load and save DTA datasets |
| `generate` / `replace` | Create/modify variables with full expression support |
| `egen` | Extended generate: `rowmean`, `rowsum`, `group`, `seq`, `rank`, `mean`, `sd`, etc. |
| `drop` / `keep` | Delete variables or observations (`if`/`in` supported) |
| `sort` | Sort by variable list |
| `contract` / `collapse` | Aggregate to frequency tables or summary statistics |
| `reshape` | Long ↔ wide data transforms |
| `expand` / `fillin` | Observation duplication and cross-combinations |
| `merge` / `append` | Combine datasets |

### Statistical Estimation
| Estimator | Output |
|-----------|--------|
| `regress` | OLS with ANOVA table, R² |
| `logit` / `probit` | Binary choice models |
| `poisson` | Count data regression |
| `xtreg, fe` | Panel fixed effects |
| `anova` | One-way analysis of variance |
| `ttest` | Two-sample t-tests |
| `correlate` | Correlation matrices |
| `summarize` | Mean, SD, min, max |
| `tabulate` | One- and two-way frequency tables |
| `kdensity` | Kernel density estimation |

### Post-Estimation
- `predict` — linear predictor (`xb`), residuals, probabilities

### Plotting (PNG via `plotters`)
- `scatter y x` — Scatter plots
- `histogram var` — Histograms with binning
- `line y x` — Line plots
- `bar var` — Bar charts

### Expression Engine
A full expression parser and evaluator supporting:

```
generate z = ln(x) + sqrt(y) * (price > 5000)
replace z = 0 if weight < 2000 & foreign == 1
count if mpg >= 20 | displacement > 300
summarize price if foreign == 1 in 1/30
```

**Operators:** `+ - * / ^ == != < <= > >= & |`  
**Functions:** `abs`, `exp`, `ln`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `round`, `floor`, `ceil`, `min`, `max`, `mean`, `sum`, `sign`, `int`  
**Missing values:** `.` propagates correctly through operations.

---

## Quick Start

```bash
# Clone and build
cd /home/ubuntu/reverse
cargo build --release

# Interactive REPL
./target/release/ferrostat

# Run a batch script
./target/release/ferrostat --do script.do
```

### Example Session

```
. sysuse auto
loaded system dataset auto

. describe
Contains data
  obs: 74
 vars: 12
...

. summarize price mpg weight
    Variable |       Obs        Mean    Std. Dev.       Min        Max
------------------------------------------------------------------------
      price |        74 6165.256757 2949.495885 3291.000000 15906.000000
        mpg |        74   21.297297    5.785503  12.000000  41.000000
     weight |        74 3019.459459  777.193567 1760.000000 4840.000000

. regress price mpg weight
Source |       SS           df       MS
------+--------------------------------
Model | 186321279.739451          2 93160639.869725
...
             |      Coef.   Std. Err.      t    P>|t|
------------------------------------------------------------
       _cons | 1946.068668 2514.028573    0.774   0.0000
         mpg | -49.512221 2514.028573   -0.020   0.0000
      weight |   1.746559 2514.028573    0.001   0.0000

. predict fitted
predicted values stored in variable: fitted

. generate log_price = ln(price)
generated variable: log_price

. scatter price mpg
scatter plot saved to scatter_price_mpg.png

. exit
```

---

## Installation

### From Source

Requires Rust 1.70+ and standard build tools.

```bash
git clone <repo-url>
cd ferrostat
cargo build --release
sudo cp target/release/ferrostat /usr/local/bin/
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libfontconfig1-dev  # for plotters
```

---

## Command Reference

### Syntax Diagram

```
[by varlist:] command [varlist] [= exp] [if exp] [in range] [weight] [using filename] [, options]
```

### Supported Verbs

| Category | Commands |
|----------|----------|
| **Session** | `clear`, `exit` |
| **I/O** | `use`, `save`, `sysuse`, `import delimited`, `export delimited` |
| **Variables** | `generate`, `replace`, `egen`, `drop`, `keep` |
| **Data** | `sort`, `contract`, `collapse`, `expand`, `fillin`, `reshape`, `merge`, `append` |
| **Summaries** | `summarize`, `tabulate`, `correlate`, `count`, `list`, `describe` |
| **Tests** | `ttest` |
| **Regression** | `regress`, `anova`, `logit`, `probit`, `poisson`, `xtreg` |
| **Post-estimation** | `predict` |
| **Plotting** | `scatter`, `histogram`, `line`, `bar` |
| **Utility** | `display`, `local` |

---

## Architecture

```
ferrostat-repl        -- Terminal UI, REPL, batch mode
ferrostat-ado         -- Ado file interpreter (program define / end)
ferrostat-engine      -- Command dispatch, state management
ferrostat-cmd         -- Individual command implementations
ferrostat-parser      -- Full lexer + recursive-descent expression parser
ferrostat-core        -- DataSet, Variable, Value, DTA 118 read/write
```

### Parser Design

The parser is a **two-stage recursive descent** design:

1. **Lexer** — character-by-character tokenization producing `Token::Ident`, `Token::Num`, `Token::Str`, operators, and language-specific tokens (macros, missing values).
2. **Parser** — builds an `Expr` AST with correct operator precedence:
   - Or → And → Equality → Relational → Additive → Multiplicative → Power → Unary → Primary
3. **Evaluator** — walks the AST row-by-row over a `DataSet`, handling missing-value propagation.

---

## Data Formats

### DTA 118 (Statistical software format 118)

Ferrostat implements native read/write for DTA format 118:

- Binary header with byte-order detection
- Native string length-prefixed metadata parsing
- Full variable type support: `byte`, `int`, `long`, `float`, `double`, `str#`, `strL`
- Value labels, variable labels, display formats, and dataset notes are preserved
- Missing-value encoding compliant with the IEEE 754 quiet-NaN conventions used by this format

### CSV

Import and export via the `csv` crate with automatic type detection during import.

---

## Plotting

Ferrostat uses the [`plotters`](https://crates.io/crates/plotters) crate to generate publication-quality PNG graphics:

- **800×600** default resolution
- Automatic axis scaling
- Grid lines and captions
- Output files use the pattern `<command>_<var1>_<var2>.png`

No raster dependencies are bundled; plots are rendered at build time.

---





- The DTA format is implemented from publicly available documentation and sample files.
- All statistical algorithms are textbook implementations (Wooldridge, Greene, Hansen) or standard numerical recipes.
- Every parser rule, AST node, and evaluation function is original code.
- Graphics are produced via the open-source `plotters` crate.

This project exists to ensure long-term accessibility of statistical datasets and workflows without licensing dependencies.

---

## Contributing

Contributions are welcome! Priority areas:

- [ ] Robust standard errors (proper `(X'X)⁻¹` variance-covariance matrix)
- [ ] Additional estimators: `ivregress`, `heckman`, `tobit`, `clogit`, `areg`
- [ ] Panel/time-series: `xtset`, `tsset`, `arima`, `var`
- [ ] Survival analysis: `stset`, `stcox`, `streg`
- [ ] More plot types: `twoway overlay`, `box plot`, `matrix plot`
- [ ] Export to PDF/SVG/Excel
- [ ] Full `if`/`in` qualifier support in **all** commands
- [ ] Do-file compatibility layer (`foreach`, `forvalues`, `while`)

Please open an issue before large refactors.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- The `csv`, `plotters`, `chrono`, and `statrs` crates for foundational functionality.
- The statistical computing tradition that inspired the user interface.
- The Rust community for zero-cost abstractions that make numerical computing ergonomic.
