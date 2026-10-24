# Ferrostat

An open-source statistical computing engine, implemented in Rust, that reads and writes Stata® `.dta` files and accepts a compatible command syntax.

> Ferrostat is an independent project. It is not affiliated with, endorsed by, or sponsored by StataCorp LLC. Stata is a registered trademark of StataCorp LLC; the name is used here only to describe file-format and syntax compatibility.

## Architecture

- **`ferrostat-core`**: Core data structures, DTA file I/O (formats 117-119), missing values, value labels.
- **`ferrostat-parser`**: Lexer and parser for the Stata-compatible command language.
- **`ferrostat-engine`**: Execution engine with dataset state and command dispatch.
- **`ferrostat-cmd`**: Native Rust implementations of statistical procedures (clean-room reconstructions).
- **`ferrostat-ado`**: Ado-file interpreter supporting `program define` / `end`.
- **`ferrostat-repl`**: Interactive CLI binary.

## Building

```bash
cargo build --release
```

The binary is at `target/release/ferrostat`.

## Usage

Interactive:
```bash
./target/release/ferrostat
```

Batch mode (`do` file):
```bash
./target/release/ferrostat --do script.do
```

## Supported Commands

### Data I/O
- `use <file>` — Load a .dta file
- `save <file>` — Save dataset to .dta
- `clear` — Clear dataset

### Descriptive Statistics
- `describe` — Describe dataset structure
- `summarize [varlist] [if] [in]` — Summary statistics (mean, sd, min, max)
- `list [varlist] [if] [in]` — List observations
- `count [if] [in]` — Count observations
- `tabulate var1 [var2]` — One-way and two-way frequency tables with chi-square test
- `correlate varlist` — Pearson correlation matrix
- `kdensity var` — Kernel density estimation (Gaussian kernel, Silverman's bandwidth)

### Variable Creation
- `generate newvar = expr` — Create variables (arithmetic: `+ - * / ^`)
- `sort varlist` — Sort dataset

### Data Manipulation
- `drop varlist` — Drop variables
- `drop if expr` — Drop observations conditionally
- `keep varlist` — Keep variables
- `keep if expr` — Keep observations conditionally
- `contract varlist` — Contract to frequency table
- `collapse (stat) vars, by(var)` — Aggregate statistics by group
- `expand N` — Duplicate observations
- `expand var` — Expand by variable values

### Regression & Estimation
- `regress depvar indepvars` — OLS linear regression (normal equations / Gaussian elimination)
- `anova depvar factor` — One-way ANOVA
- `logit depvar indepvars` — Logistic regression (IRLS / Newton-Raphson)
- `probit depvar indepvars` — Probit regression (IRLS with normal CDF)
- `poisson depvar indepvars` — Poisson regression (log link IRLS)
- `xtreg panel depvar indepvars, fe` — Fixed-effects panel regression (within transformation)

### Tests
- `ttest var == #` — One-sample t-test
- `ttest var1 var2` — Two-sample t-test (pooled variance)

### Ado Programs
- `program define name` / `end` — Define reusable ado programs
- `do file.do` — Execute do-files

## DTA Format Support

Reads DTA formats **117**, **118** and **119**; writes format **118**. Support includes:
- XML-tagged header with release, byteorder, K, N
- 14-entry section map
- Variable type codes (numeric: byte/int/long/float/double, string: str1-str2045)
- Fixed-width metadata: 129-byte varnames, 57-byte formats, 129-byte value label names, 321-byte variable labels
- strL long-string table
- Value label dictionaries
- Missing values (`.`, `.a`–`.z`) for all numeric types

## Clean-room Reconstruction

All statistical procedures are clean-room implementations from first principles, using well-known algorithms documented in statistical literature. No proprietary Stata code has been copied or transliterated.

Key algorithms implemented:
- **OLS**: Normal equations solved via Gaussian elimination with partial pivoting
- **Logistic/Probit/Poisson**: Iteratively Reweighted Least Squares (IRLS) using the Fisher scoring / Newton-Raphson approach
- **Fixed Effects**: Within-transformation (subtracting group means)
- **ANOVA**: Between/within sum of squares decomposition
- **Kernel Density**: Gaussian kernel with Silverman's rule-of-thumb bandwidth
- **Correlation**: Pearson product-moment correlation from covariance matrix

## Testing

```bash
cargo test --workspace
```
