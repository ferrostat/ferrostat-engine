//! The graph gallery of the original documentation, reproduced as tests: every
//! example's commands, run on a synthetic dataset with the same variable
//! names, types, labels and time settings as the example dataset (whose values
//! are not redistributed). Each test checks that all commands succeed and that
//! the resulting graph exports to SVG and PNG.
//!
//! Examples that need estimators Ferrostat does not implement yet are
//! `#[ignore]`d with the reason; run them with `cargo test -- --ignored`.

use ferrostat_tests::*;

const AIR2: &str = r#"
clear
set obs 144
generate int t = tm(1949m1) + _n - 1
format t %tm
tsset t
generate int air = round(100 + 2*_n + 30*sin(2*_pi*_n/12) + 5*mod(_n*7, 11))
label variable air "Airline passengers (1949-1960)"
generate time = _n
"#;

const AUTO: &str = r#"
clear
set obs 74
set seed 1978
generate str18 make = "Car " + string(_n)
generate int price = round(4000 + 6000*runiform())
generate int weight = round(1800 + 3000*runiform())
generate int mpg = round(45 - weight/150 + 3*rnormal())
generate float headroom = 1.5 + 0.5*floor(6*runiform())
generate int turn = round(30 + weight/250 + 2*rnormal())
generate int displacement = round(70 + weight/15 + 20*rnormal())
generate byte foreign = runiform() < 0.3
label define origin 0 "Domestic" 1 "Foreign"
label values foreign origin
label variable make "Make and model"
label variable price "Price"
label variable mpg "Mileage (mpg)"
label variable weight "Weight (lbs.)"
label variable headroom "Headroom (in.)"
label variable turn "Turn circle (ft.)"
label variable displacement "Displacement (cu. in.)"
label variable foreign "Car origin"
"#;

const BPLONG: &str = r#"
clear
set obs 240
set seed 3
generate int patient = ceil(_n/2)
generate byte sex = mod(patient, 2)
generate byte agegrp = mod(patient, 3) + 1
label define agegrp 1 "30-45" 2 "46-59" 3 "60+"
label values agegrp agegrp
generate byte when = mod(_n-1, 2) + 1
label define when 1 "Before" 2 "After"
label values when when
generate int bp = round(150 + 5*agegrp - 4*when + 10*rnormal())
label variable bp "Blood pressure"
"#;

const BPWIDE: &str = r#"
clear
set obs 120
set seed 3
generate int patient = _n
generate byte sex = mod(patient, 2)
generate byte agegrp = mod(patient, 3) + 1
label define agegrp 1 "30-45" 2 "46-59" 3 "60+"
label values agegrp agegrp
generate int bp_before = round(155 + 5*agegrp + 10*rnormal())
generate int bp_after = round(150 + 5*agegrp + 10*rnormal())
label variable bp_before "Blood pressure before"
label variable bp_after "Blood pressure after"
"#;

const CANCER: &str = r#"
clear
set obs 48
set seed 4
generate int studytime = 1 + floor(39*runiform())
generate byte died = runiform() < 0.65
generate byte drug = mod(_n-1, 3) + 1
generate byte age = 47 + floor(20*runiform())
"#;

const CENSUS: &str = r#"
clear
set obs 50
set seed 1980
generate byte region = mod(_n-1, 4) + 1
label define cenreg 1 "NE" 2 "N Cntrl" 3 "South" 4 "West"
label values region cenreg
generate long pop = round(500000 + 10000000*runiform())
generate long pop65p = round(pop*(0.08 + 0.06*runiform()))
generate float medage = 26 + 8*runiform()
generate long death = round(pop*(0.007 + 0.003*runiform()))
label variable pop "Population"
label variable medage "Median age"
label variable death "Number of deaths"
"#;

const CITYTEMP: &str = r#"
clear
set obs 120
set seed 7
generate byte region = mod(_n-1, 4) + 1
label define cenreg 1 "NE" 2 "N Cntrl" 3 "South" 4 "West"
label values region cenreg
generate float tempjan = 20 + 8*region + 5*rnormal()
generate float tempjuly = 70 + 3*region + 4*rnormal()
generate int heatdd = round(7000 - 900*region + 400*rnormal())
generate int cooldd = round(500 + 400*region + 200*abs(rnormal()))
label variable tempjan "Average January temperature"
label variable tempjuly "Average July temperature"
label variable heatdd "Heating degree days"
label variable cooldd "Cooling degree days"
"#;

const COS4: &str = r#"
clear
set obs 512
generate int time = _n
tsset time
generate double sumfc = cos(2*_pi*_n/16) + cos(2*_pi*_n/4) + 0.5*cos(2*_pi*_n/32) + cos(2*_pi*_n/64)
"#;

const CT: &str = r#"
clear
set obs 112
set seed 13
generate byte status = runiform() < 0.5
generate byte mod1 = min(5, max(1, round(2 + 2*status + rnormal())))
generate byte mod2 = min(5, max(1, round(2 + 1.5*status + rnormal())))
generate byte mod3 = min(5, max(1, round(2 + status + rnormal())))
"#;

const CT2ROCGOLD: &str = r#"
clear
set obs 112
set seed 13
generate byte status = runiform() < 0.5
generate byte mod1 = min(5, max(1, round(2 + 2*status + rnormal())))
generate byte mod2 = min(5, max(1, round(2 + 1.5*status + rnormal())))
generate byte mod3 = min(5, max(1, round(2 + status + rnormal())))
"#;

const DRUG2: &str = r#"
clear
set obs 48
set seed 6
generate int studytime = 1 + floor(39*runiform())
generate byte died = runiform() < 0.65
generate byte drug = mod(_n, 2)
generate byte age = 47 + floor(20*runiform())
stset studytime, failure(died)
"#;

const FURNACE: &str = r#"
clear
set obs 296
set seed 9
generate int time = _n
tsset time
generate double input = sin(_n/10) + 0.2*rnormal()
generate double output = 0
replace output = 0.5*output[_n-1] + input[_n-5] + 0.1*rnormal() if _n > 5
"#;

const GNP96: &str = r#"
clear
set obs 160
generate int date = tq(1967q1) + _n - 1
format date %tq
generate double gnp96 = 3000 + 40*_n + 30*sin(_n/4)
label variable gnp96 "Real GNP"
tsset date
"#;

const HANLEY: &str = r#"
clear
set obs 109
set seed 12
generate byte disease = runiform() < 0.47
generate byte rating = min(5, max(1, round(2 + 2*disease + rnormal())))
"#;

const LORENZ: &str = r#"
clear
set obs 20
generate byte class = _n
generate byte disease = mod(_n, 2)
generate int pop = 100 + 10*_n
"#;

const LUTKEPOHL: &str = r#"
clear
set obs 92
set seed 14
generate int qtr = tq(1960q1) + _n - 1
format qtr %tq
tsset qtr
generate double linvestment = 5 + 0.02*_n + 0.05*rnormal()
generate double lincome = 6 + 0.02*_n + 0.02*rnormal()
generate double lconsumption = 5.8 + 0.02*_n + 0.02*rnormal()
generate double dlinvestment = D.linvestment
generate double dlincome = D.lincome
generate double dlconsumption = D.lconsumption
"#;

const NLSW88: &str = r#"
clear
set obs 400
set seed 88
generate byte occupation = mod(_n-1, 13) + 1
label define occlbl 1 "Professional/technical" 2 "Managers/admin" 3 "Sales" 4 "Clerical/unskilled" 5 "Craftsmen" 6 "Operatives" 7 "Transport" 8 "Laborers" 9 "Farmers" 10 "Farm laborers" 11 "Service" 12 "Household workers" 13 "Other"
label values occupation occlbl
generate float wage = 4 + occupation/2 + 3*abs(rnormal())
generate byte hours = round(30 + 10*runiform())
generate float ttl_exp = 5 + 15*runiform()
generate byte age = 34 + floor(12*runiform())
label variable wage "Hourly wage"
label variable hours "Usual hours worked"
label variable ttl_exp "Total work experience"
label variable age "Age in current year"
label variable occupation "Occupation"
"#;

const POP2000: &str = r#"
clear
set obs 17
generate byte agegrp = _n
label define agelbl 1 "Under 5" 2 "5 to 9" 3 "10 to 14" 4 "15 to 19" 5 "20 to 24" 6 "25 to 29" 7 "30 to 34" 8 "35 to 39" 9 "40 to 44" 10 "45 to 49" 11 "50 to 54" 12 "55 to 59" 13 "60 to 64" 14 "65 to 69" 15 "70 to 74" 16 "75 to 79" 17 "80 and over"
label values agegrp agelbl
generate long maletotal = round(10000000 - 400000*_n)
generate long femtotal = round(9800000 - 350000*_n)
label variable agegrp "Age category"
label variable maletotal "Male total"
label variable femtotal "Female total"
"#;

const SP500: &str = r#"
clear
set obs 248
set seed 500
generate int date = mdy(1,2,2001) + floor((_n-1)/5)*7 + mod(_n-1,5)
format date %d
label variable date "Date"
generate float open = 1300 + 40*sin(_n/9) + 5*rnormal()
generate float close = open + 10*rnormal()
generate float high = max(open, close) + 8*runiform()
generate float low = min(open, close) - 8*runiform()
generate double volume = 12000 + 3000*runiform()
format volume %12.0gc
generate float change = close - close[_n-1]
label variable open "Opening price"
label variable high "High price"
label variable low "Low price"
label variable close "Closing price"
label variable volume "Volume (thousands)"
label variable change "Closing price change"
"#;

const STAN3: &str = r#"
clear
set obs 172
set seed 5
generate int id = ceil(_n/2)
generate double t1 = 10 + 400*runiform()
generate byte died = runiform() < 0.6
generate byte posttran = mod(_n, 2)
generate float age = 30 + 25*runiform()
stset t1, failure(died)
"#;

const TSAPPEND1: &str = r#"
clear
set obs 476
set seed 11
generate int t = tm(1960m1) + _n - 1
format t %tm
tsset t
generate double e = rnormal()
generate double x = rnormal()
generate double y = 0
replace y = 0.6*y[_n-1] + e if _n > 1
"#;

const TUMOR: &str = r#"
clear
set obs 80
set seed 8
generate float t = 1 + 60*runiform()
generate byte d = runiform() < 0.7
"#;

const TXHPRICE: &str = r#"
clear
set obs 168
set seed 15
generate int t = tm(1990m1) + _n - 1
format t %tm
tsset t
generate double austin = 11 + 0.01*_n + 0.05*rnormal()
generate double dallas = 11.2 + 0.01*_n + 0.05*rnormal()
generate double houston = 11.1 + 0.01*_n + 0.05*rnormal()
generate double sa = 11 + 0.01*_n + 0.05*rnormal()
"#;

const USLIFEEXP: &str = r#"
clear
set obs 100
generate int year = 1899 + _n
generate float le = 47 + 0.3*(_n-1) - 0.0008*(_n-1)^2
generate float le_male = le - 3
generate float le_female = le + 3
generate float le_w = le + 1
generate float le_b = le - 8 + 0.05*_n
generate float le_wm = le_w - 3
generate float le_bm = le_b - 3
label variable le "Life expectancy"
label variable le_male "Life expectancy, males"
label variable le_female "Life expectancy, females"
label variable le_w "Life expectancy, whites"
label variable le_b "Life expectancy, blacks"
label variable le_wm "Life expectancy, white males"
label variable le_bm "Life expectancy, black males"
"#;

/// Load a synthetic dataset, run the example's commands, export the graph.
fn example(test: &str, dataset: &str, commands: &[&str]) {
    let mut s = Session::new();
    s.script(dataset);
    // files written by the examples (IRF files) go to the temporary directory
    let dir = temp_dir(test);
    s.run(&format!("cd \"{}\"", dir.display()));
    for c in commands {
        s.run(c);
    }
    for ext in ["svg", "png"] {
        let f = dir.join(format!("graph.{ext}"));
        s.run(&format!("graph export \"{}\", replace", f.display()));
        let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        assert!(size > 1000, "{test}: {ext} export too small ({size} bytes)");
    }
}

/// Area plot
#[test]
fn area_plot() {
    example(
        "area_plot",
        GNP96,
        &[
        r#"twoway area d.gnp96 date"#,
        ],
    );
}

/// Bar chart
#[test]
fn bar_chart() {
    example(
        "bar_chart",
        USLIFEEXP,
        &[
        r#"graph bar le le_w le_b"#,
        ],
    );
}

/// Bar chart by values of categorical variable
#[test]
fn bar_chart_by_values_of_categorical_variable() {
    example(
        "bar_chart_by_values_of_categorical_variable",
        CITYTEMP,
        &[
        r#"graph bar tempjan, over(region)"#,
        ],
    );
}

/// Bar chart with bar labels
#[test]
fn bar_chart_with_bar_labels() {
    example(
        "bar_chart_with_bar_labels",
        CITYTEMP,
        &[
        r#"graph bar heatdd cooldd, over(region) blabel(total)"#,
        ],
    );
}

/// Bar chart with multiple bars graphed over another variable
#[test]
fn bar_chart_with_multiple_bars_graphed_over_another_variable() {
    example(
        "bar_chart_with_multiple_bars_graphed_over_another_variable",
        CITYTEMP,
        &[
        r#"graph bar tempjan tempjuly, over(region)"#,
        ],
    );
}

/// Bartlett's periodogram-based test for white noise
#[test]
fn bartletts_periodogram_based_test_for_white_noise() {
    example(
        "bartletts_periodogram_based_test_for_white_noise",
        TSAPPEND1,
        &[
        r#"wntestb x in 1/100"#,
        ],
    );
}

/// Box plot by values of categorical variable
#[test]
fn box_plot_by_values_of_categorical_variable() {
    example(
        "box_plot_by_values_of_categorical_variable",
        BPLONG,
        &[
        r#"graph box bp, over(agegrp)"#,
        ],
    );
}

/// Box plot of two variables
#[test]
fn box_plot_of_two_variables() {
    example(
        "box_plot_of_two_variables",
        BPWIDE,
        &[
        r#"graph box bp_before bp_after"#,
        ],
    );
}

/// Box plot of two variables by values of categorical variable
#[test]
fn box_plot_of_two_variables_by_categorical_variable() {
    example(
        "box_plot_of_two_variables_by_categorical_variable",
        BPWIDE,
        &[
        r#"graph box bp_before bp_after, over(agegrp)"#,
        ],
    );
}

/// Box plot of two variables with symbol as median
#[test]
fn box_plot_of_two_variables_with_symbol_as_median() {
    example(
        "box_plot_of_two_variables_with_symbol_as_median",
        BPWIDE,
        &[
        r#"graph box bp_before bp_after, medtype(marker) medmarker(msymbol(diamond))"#,
        ],
    );
}

/// Chi-squared probability plot
#[test]
fn chi_squared_probability_plot() {
    example(
        "chi_squared_probability_plot",
        AUTO,
        &[
        r#"egen c1 = std(price)"#,
        r#"egen c2 = std(mpg)"#,
        r#"generate ch = c1^2 + c2^2"#,
        r#"pchi ch, df(2)"#,
        ],
    );
}

/// Chi-squared quantile plot
#[test]
fn chi_squared_quantile_plot() {
    example(
        "chi_squared_quantile_plot",
        AUTO,
        &[
        r#"egen c1 = std(price)"#,
        r#"egen c2 = std(mpg)"#,
        r#"generate ch = c1^2 + c2^2"#,
        r#"qchi ch, df(2)"#,
        ],
    );
}

/// Comparison of ROC areas
#[test]
fn comparison_of_roc_areas() {
    example(
        "comparison_of_roc_areas",
        CT,
        &[
        r#"roccomp status mod1 mod2, graph"#,
        ],
    );
}

/// Comparison of ROC areas with gold standard
#[test]
fn comparison_of_roc_areas_with_gold_standard() {
    example(
        "comparison_of_roc_areas_with_gold_standard",
        CT2ROCGOLD,
        &[
        r#"rocgold status mod1 mod2 mod3, graph"#,
        ],
    );
}

/// Connected estimated failure rate with capped CIs
#[test]
fn connected_estimated_failure_rate_with_capped_cis() {
    example(
        "connected_estimated_failure_rate_with_capped_cis",
        CANCER,
        &[
        r#"stset studytime, failure(died)"#,
        r#"strate drug, graph recast(connected) ciopts(recast(rcap))"#,
        ],
    );
}

/// Connected line graph
#[test]
fn connected_line_graph() {
    example(
        "connected_line_graph",
        SP500,
        &[
        r#"twoway connected close date in 1/15"#,
        ],
    );
}

/// Correlogram with confidence intervals
#[test]
fn correlogram_with_confidence_intervals() {
    example(
        "correlogram_with_confidence_intervals",
        AIR2,
        &[
        r#"ac air, lags(20)"#,
        ],
    );
}

/// Cross-correlogram for bivariate time series
#[test]
fn cross_correlogram_for_bivariate_time_series() {
    example(
        "cross_correlogram_for_bivariate_time_series",
        FURNACE,
        &[
        r#"xcorr input output, xline(5) lags(40)"#,
        ],
    );
}

/// Cumulative spectral distribution
#[test]
fn cumulative_spectral_distribution() {
    example(
        "cumulative_spectral_distribution",
        COS4,
        &[
        r#"cumsp sumfc"#,
        ],
    );
}

/// Cumulative spectral distribution recast as area
#[test]
fn cumulative_spectral_distribution_recast_as_area() {
    example(
        "cumulative_spectral_distribution_recast_as_area",
        COS4,
        &[
        r#"cumsp sumfc, recast(area)"#,
        ],
    );
}

/// Cumulative spectral distribution with x-line
#[test]
fn cumulative_spectral_distribution_with_x_line() {
    example(
        "cumulative_spectral_distribution_with_x_line",
        AIR2,
        &[
        r#"cumsp air, xline(.08333)"#,
        ],
    );
}

/// Dot chart
#[test]
fn dot_chart() {
    example(
        "dot_chart",
        NLSW88,
        &[
        r#"graph dot wage, over(occ)"#,
        ],
    );
}

/// Dot chart of three variables by categorical variable
#[test]
fn dot_chart_of_three_variables_by_categorical_variable() {
    example(
        "dot_chart_of_three_variables_by_categorical_variable",
        NLSW88,
        &[
        r#"graph dot wage hours ttl_exp, over(occ)"#,
        ],
    );
}

/// Dot chart of two variables by categorical variable
#[test]
fn dot_chart_of_two_variables_by_categorical_variable() {
    example(
        "dot_chart_of_two_variables_by_categorical_variable",
        NLSW88,
        &[
        r#"graph dot wage hours, over(occ)"#,
        ],
    );
}

/// Estimated failure rate with CIs
#[test]
fn estimated_failure_rate_with_cis() {
    example(
        "estimated_failure_rate_with_cis",
        CANCER,
        &[
        r#"stset studytime, failure(died)"#,
        r#"strate drug, graph"#,
        ],
    );
}

/// Estimated failure rate without CIs
#[test]
fn estimated_failure_rate_without_cis() {
    example(
        "estimated_failure_rate_without_cis",
        CANCER,
        &[
        r#"stset studytime, failure(died)"#,
        r#"strate drug, graph nowhisker"#,
        ],
    );
}

/// FEVDs after irf
#[test]
fn fevds_after_irf() {
    example(
        "fevds_after_irf",
        LUTKEPOHL,
        &[
        r#"var dlinvestment dlincome dlconsumption, lags(1/2) dfk"#,
        r#"irf create order1, step(10) set(myirf1, replace)"#,
        r#"irf graph fevd sfevd, impulse(dlincome) response(dlconsumption) lstep(1) legend(order(1 "95% CI for fevd" 2 "95% CI for sfevd" 3 "fraction of mse" 4 "(structural) fraction of mse"))"#,
        ],
    );
}

/// Forecasts of dependent variables after var
#[test]
fn forecasts_of_dependent_variables_after_var() {
    example(
        "forecasts_of_dependent_variables_after_var",
        LUTKEPOHL,
        &[
        r#"qui var d.linvestment d.lincome d.lconsumption if qtr<=q(1978q4), dfk"#,
        r#"varfcast compute, step(8)"#,
        r#"varfcast graph d.linvestment, observed"#,
        ],
    );
}

/// Fractional-polynomial prediction plot with CIs
#[test]
fn fractional_polynomial_prediction_plot_with_cis() {
    example(
        "fractional_polynomial_prediction_plot_with_cis",
        AUTO,
        &[
        r#"twoway fpfitci mpg weight"#,
        ],
    );
}

/// Fractional-polynomial prediction plot with CIs with overlaid scatterplot
#[test]
fn fractional_polynomial_prediction_plot_with_cis_with_overlaid_scatterplot() {
    example(
        "fractional_polynomial_prediction_plot_with_cis_with_overlaid_scatterplot",
        AUTO,
        &[
        r#"twoway fpfitci mpg weight || scatter mpg weight"#,
        ],
    );
}

/// Histogram of continuous variable as density
#[test]
fn histogram_of_continuous_variable_as_density() {
    example(
        "histogram_of_continuous_variable_as_density",
        SP500,
        &[
        r#"histogram open"#,
        ],
    );
}

/// Histogram of continuous variable with frequencies and overlaid kernel density estimate
#[test]
fn histogram_of_continuous_variable_with_frequencies_and_overlaid_kernel_density_estimate() {
    example(
        "histogram_of_continuous_variable_with_frequencies_and_overlaid_kernel_density_estimate",
        SP500,
        &[
        r#"histogram open, frequency kdensity"#,
        ],
    );
}

/// Histogram of continuous variable with frequencies and overlaid normal density curve
#[test]
fn histogram_of_continuous_variable_with_frequencies_and_overlaid_normal_density_curve() {
    example(
        "histogram_of_continuous_variable_with_frequencies_and_overlaid_normal_density_curve",
        SP500,
        &[
        r#"histogram open, frequency normal"#,
        ],
    );
}

/// Histograms of transforms according to ladder of powers
#[test]
fn histograms_of_transforms_according_to_ladder_of_powers() {
    example(
        "histograms_of_transforms_according_to_ladder_of_powers",
        AUTO,
        &[
        r#"gladder mpg, fraction ylabel(, labelminlen(3))"#,
        ],
    );
}

/// Horizontal bar chart with multiple bars graphed over another variable
#[test]
fn horizontal_bar_chart_with_multiple_bars_graphed_over_another_variable() {
    example(
        "horizontal_bar_chart_with_multiple_bars_graphed_over_another_variable",
        CITYTEMP,
        &[
        r#"graph hbar tempjan tempjuly, over(region)"#,
        ],
    );
}

/// Horizontal box plot of two variables by values of categorical variable
#[test]
fn horizontal_box_plot_of_two_variables_by_values_of_categorical_variable() {
    example(
        "horizontal_box_plot_of_two_variables_by_values_of_categorical_variable",
        BPWIDE,
        &[
        r#"graph box bp_before bp_after, over(agegrp) horizontal"#,
        ],
    );
}

/// Horizontal box plot of variable by values of categorical variable
#[test]
fn horizontal_box_plot_of_variable_by_values_of_categorical_variable() {
    example(
        "horizontal_box_plot_of_variable_by_values_of_categorical_variable",
        BPLONG,
        &[
        r#"graph box bp, over(agegrp) horizontal"#,
        ],
    );
}

/// Horizontal overlaid twoway bar plots
#[test]
fn horizontal_overlaid_twoway_bar_plots() {
    example(
        "horizontal_overlaid_twoway_bar_plots",
        POP2000,
        &[
        r#"replace maletotal = -maletotal"#,
        r#"twoway bar maletotal agegrp, horizontal || bar femtotal agegrp, horizontal"#,
        ],
    );
}

/// Horizontal twoway dropped-line plot
#[test]
fn horizontal_twoway_dropped_line_plot() {
    example(
        "horizontal_twoway_dropped_line_plot",
        GNP96,
        &[
        r#"twoway dropline d.gnp96 date in 1/71, horizontal"#,
        ],
    );
}

/// Impulse–response functions
#[test]
fn impulse_response_functions() {
    example(
        "impulse_response_functions",
        LUTKEPOHL,
        &[
        r#"mat a = (.,0,0\0,.,0\.,.,.)"#,
        r#"mat b = I(3)"#,
        r#"svar dlinvestment dlincome dlconsumption, aeq(a) beq(b)"#,
        r#"irf create modela, set(results3) step(8)"#,
        r#"irf graph oirf sirf, impulse(dlincome) response(dlconsumption)"#,
        ],
    );
}

/// Kaplan–Meier survival function
#[test]
fn kaplan_meier_survival_function() {
    example(
        "kaplan_meier_survival_function",
        STAN3,
        &[
        r#"sts graph"#,
        ],
    );
}

/// Kaplan–Meier survival function by variable
#[test]
fn kaplan_meier_survival_function_by_variable() {
    example(
        "kaplan_meier_survival_function_by_variable",
        STAN3,
        &[
        r#"sts graph, by(posttran)"#,
        ],
    );
}

/// Kaplan–Meier survival function with number at-risk displayed
#[test]
fn kaplan_meier_survival_function_with_number_at_risk_displayed() {
    example(
        "kaplan_meier_survival_function_with_number_at_risk_displayed",
        DRUG2,
        &[
        r#"sts graph, atrisk"#,
        ],
    );
}

/// Life table with survival function and CIs
#[test]
fn life_table_with_survival_function_and_cis() {
    example(
        "life_table_with_survival_function_and_cis",
        TUMOR,
        &[
        r#"ltable t d, survival graph"#,
        ],
    );
}

/// Life table with survival function by variable
#[test]
fn life_table_with_survival_function_by_variable() {
    example(
        "life_table_with_survival_function_by_variable",
        CANCER,
        &[
        r#"ltable studytime died, by(drug) noconf graph overlay"#,
        ],
    );
}

/// Life table with survival function without CIs
#[test]
fn life_table_with_survival_function_without_cis() {
    example(
        "life_table_with_survival_function_without_cis",
        TUMOR,
        &[
        r#"ltable t d, survival graph noconf"#,
        ],
    );
}

/// Line graph
#[test]
fn line_graph() {
    example(
        "line_graph",
        USLIFEEXP,
        &[
        r#"line le year"#,
        ],
    );
}

/// Linear prediction plot with CIs with overlaid scatterplot
#[test]
fn linear_prediction_plot_with_cis_with_overlaid_scatterplot() {
    example(
        "linear_prediction_plot_with_cis_with_overlaid_scatterplot",
        AUTO,
        &[
        r#"twoway lfitci mpg weight || scatter mpg weight"#,
        ],
    );
}

/// Multiple overlaid connected line graphs
#[test]
fn multiple_overlaid_connected_line_graph() {
    example(
        "multiple_overlaid_connected_line_graph",
        SP500,
        &[
        r#"twoway connected high low close date in 1/15"#,
        ],
    );
}

/// Multiple overlaid line graphs
#[test]
fn multiple_overlaid_line_graphs() {
    example(
        "multiple_overlaid_line_graphs",
        USLIFEEXP,
        &[
        r#"line le le_male le_female year, legend(size(medsmall))"#,
        ],
    );
}

/// Multiple overlaid scatterplots
#[test]
fn multiple_overlaid_scatterplots() {
    example(
        "multiple_overlaid_scatterplots",
        AUTO,
        &[
        r#"scatter mpg headroom turn weight"#,
        ],
    );
}

/// Nelson–Aalen cumulative hazard estimate
#[test]
fn nelson_aalen_cumulative_hazard_estimate() {
    example(
        "nelson_aalen_cumulative_hazard_estimate",
        STAN3,
        &[
        r#"sts graph, na"#,
        ],
    );
}

/// Nelson–Aalen cumulative hazard estimate by variable
#[test]
fn nelson_aalen_cumulative_hazard_estimate_by_variable() {
    example(
        "nelson_aalen_cumulative_hazard_estimate_by_variable",
        STAN3,
        &[
        r#"sts graph, na by(posttran)"#,
        ],
    );
}

/// Nelson–Aalen cumulative hazard estimate with number at-risk displayed
#[test]
fn nelson_aalen_cumulative_hazard_estimate_with_number_at_risk_displayed() {
    example(
        "nelson_aalen_cumulative_hazard_estimate_with_number_at_risk_displayed",
        STAN3,
        &[
        r#"sts graph, na atrisk"#,
        ],
    );
}

/// Nonparametric ROC curve
#[test]
fn nonparametric_roc_curve() {
    example(
        "nonparametric_roc_curve",
        HANLEY,
        &[
        r#"roctab disease rating, graph"#,
        ],
    );
}

/// Overlaid area plots
#[test]
fn overlaid_area_plots() {
    example(
        "overlaid_area_plots",
        SP500,
        &[
        r#"twoway area high close low date in 1/15"#,
        ],
    );
}

/// Overlaid connected line graph
#[test]
fn overlaid_connected_line_graph() {
    example(
        "overlaid_connected_line_graph",
        SP500,
        &[
        r#"twoway connected high low date in 1/15"#,
        ],
    );
}

/// Overlaid horizontal area plots
#[test]
fn overlaid_horizontal_area_plots() {
    example(
        "overlaid_horizontal_area_plots",
        POP2000,
        &[
        r#"replace maletotal = -maletotal"#,
        r#"twoway area maletotal agegrp, horizontal || area femtotal agegrp, horizontal"#,
        ],
    );
}

/// Overlaid impulse–response functions
#[test]
fn overlaid_impulse_response_function() {
    example(
        "overlaid_impulse_response_function",
        LUTKEPOHL,
        &[
        r#"var dlinvestment dlincome dlconsumption if qtr<=tq(1978q4), lags(1/2) dfk"#,
        r#"varirf create order1, step(10) set(myirf1, replace)"#,
        r#"varirf create order2, step(10) order(dlincome dlinvestment dlconsumption)"#,
        r#"varirf ograph (order1 dlincome dlconsumption oirf) (order2 dlincome dlconsumption oirf)"#,
        ],
    );
}

/// Overlaid line graph
#[test]
fn overlaid_line_graph() {
    example(
        "overlaid_line_graph",
        USLIFEEXP,
        &[
        r#"line le_wm le_bm year, legend(subtitle("Life expectancy") order(1 "White males" 2 "Black males"))"#,
        ],
    );
}

/// Overlaid scatterplot
#[test]
fn overlaid_scatterplot() {
    example(
        "overlaid_scatterplot",
        AUTO,
        &[
        r#"scatter mpg headroom weight"#,
        ],
    );
}

/// Overlaid twoway bar plots
#[test]
fn overlaid_twoway_bar_plots() {
    example(
        "overlaid_twoway_bar_plots",
        SP500,
        &[
        r#"twoway bar high low date in 1/15"#,
        ],
    );
}

/// Parametric ROC curve
#[test]
fn parametric_roc_curve() {
    example(
        "parametric_roc_curve",
        HANLEY,
        &[
        r#"rocfit disease rating"#,
        ],
    );
}

/// Partial correlogram with CIs and standardized variances
#[test]
fn partial_correlogram_with_cis_and_standardized_variances() {
    example(
        "partial_correlogram_with_cis_and_standardized_variances",
        AIR2,
        &[
        r#"pac DS12.air, lags(20)"#,
        ],
    );
}

/// Partial correlogram with confidence intervals
#[test]
fn partial_correlogram_with_confidence_intervals() {
    example(
        "partial_correlogram_with_confidence_intervals",
        AIR2,
        &[
        r#"pac air, lags(20)"#,
        ],
    );
}

/// Periodogram
#[test]
fn periodogram() {
    example(
        "periodogram",
        AIR2,
        &[
        r#"pergram air"#,
        ],
    );
}

/// Periodogram recast as area
#[test]
fn periodogram_recast_as_area() {
    example(
        "periodogram_recast_as_area",
        AIR2,
        &[
        r#"pergram air, recast(area)"#,
        ],
    );
}

/// Pie chart
#[test]
fn pie_chart() {
    example(
        "pie_chart",
        CENSUS,
        &[
        r#"graph pie pop, over(region)"#,
        ],
    );
}

/// Pie chart with exploded piece and percentage label
#[test]
fn pie_chart_with_exploded_piece_and_percentage_label() {
    example(
        "pie_chart_with_exploded_piece_and_percentage_label",
        CENSUS,
        &[
        r#"graph pie pop, over(region) pie(2,explode) plabel(2 percent)"#,
        ],
    );
}

/// Pie chart with exploded pieces
#[test]
fn pie_chart_with_exploded_pieces() {
    example(
        "pie_chart_with_exploded_pieces",
        CENSUS,
        &[
        r#"graph pie pop, over(region) pie(1,explode) pie(2,explode) pie(3,explode) pie(4,explode)"#,
        ],
    );
}

/// Pie chart with labels
#[test]
fn pie_chart_with_labels() {
    example(
        "pie_chart_with_labels",
        CENSUS,
        &[
        r#"graph pie pop, over(region) plabel(_all name)"#,
        ],
    );
}

/// Quadratic prediction over scatterplot with CIs with overlaid scatterplot
#[test]
fn quadratic_prediction_over_scatterplot_with_cis_with_overlaid_scatterplot() {
    example(
        "quadratic_prediction_over_scatterplot_with_cis_with_overlaid_scatterplot",
        AUTO,
        &[
        r#"twoway qfitci mpg weight || scatter mpg weight"#,
        ],
    );
}

/// Quadratic prediction plot with CIs
#[test]
fn quadratic_prediction_plot_with_cis() {
    example(
        "quadratic_prediction_plot_with_cis",
        AUTO,
        &[
        r#"twoway qfitci mpg weight"#,
        ],
    );
}

/// Quantile–normal plot
#[test]
fn quantile_normal_plot() {
    example(
        "quantile_normal_plot",
        AUTO,
        &[
        r#"qnorm price"#,
        ],
    );
}

/// Quantile plot
#[test]
fn quantile_plot() {
    example(
        "quantile_plot",
        AUTO,
        &[
        r#"quantile price"#,
        ],
    );
}

/// Quantile–quantile plot
#[test]
fn quantile_quantile_plot() {
    example(
        "quantile_quantile_plot",
        AUTO,
        &[
        r#"generate weightd = weight if !foreign"#,
        r#"generate weightf = weight if foreign"#,
        r#"qqplot weightd weightf"#,
        ],
    );
}

/// Quantiles of transforms according to ladder of powers
#[test]
fn quantiles_of_transforms_according_to_ladder_of_powers() {
    example(
        "quantiles_of_transforms_according_to_ladder_of_powers",
        CITYTEMP,
        &[
        r#"qladder heatdd, ylabel(,labelminlen(10)) xlabel(,format(%9.1f))"#,
        ],
    );
}

/// Range plot with area shading
#[test]
fn range_plot_with_area_shading() {
    example(
        "range_plot_with_area_shading",
        SP500,
        &[
        r#"twoway rarea high low date in 1/15"#,
        ],
    );
}

/// Range plot with bars
#[test]
fn range_plot_with_bars() {
    example(
        "range_plot_with_bars",
        SP500,
        &[
        r#"twoway rbar high low date in 1/15"#,
        ],
    );
}

/// Range plot with capped spikes
#[test]
fn range_plot_with_capped_spikes() {
    example(
        "range_plot_with_capped_spikes",
        SP500,
        &[
        r#"twoway rcap high low date in 1/15"#,
        ],
    );
}

/// Range plot with connected lines
#[test]
fn range_plot_with_connected_lines() {
    example(
        "range_plot_with_connected_lines",
        SP500,
        &[
        r#"twoway rconnected high low date in 1/15"#,
        ],
    );
}

/// Range plot with spikes
#[test]
fn range_plot_with_spikes() {
    example(
        "range_plot_with_spikes",
        SP500,
        &[
        r#"twoway rspike high low date in 1/15"#,
        ],
    );
}

/// Range plot with spikes capped by marker symbols
#[test]
fn range_plot_with_spikes_capped_by_marker_symbols() {
    example(
        "range_plot_with_spikes_capped_by_marker_symbols",
        SP500,
        &[
        r#"twoway rcapsym high low date in 1/15"#,
        ],
    );
}

/// ROC analysis with Lorenz-like curve
#[test]
fn roc_analysis_with_lorenz_like_curve() {
    example(
        "roc_analysis_with_lorenz_like_curve",
        LORENZ,
        &[
        r#"roctab disease class [fweight=pop], lorenz graph"#,
        ],
    );
}

/// Scatterplot
#[test]
fn scatterplot() {
    example(
        "scatterplot",
        AUTO,
        &[
        r#"scatter mpg weight"#,
        ],
    );
}

/// Scatterplot with marker labels
#[test]
fn scatterplot_with_marker_labels() {
    example(
        "scatterplot_with_marker_labels",
        AUTO,
        &[
        r#"scatter mpg weight in 1/15, mlabel(make)"#,
        ],
    );
}

/// Twoway linear prediction plot
#[test]
fn scatterplot_with_overlaid_fractional_polynomial_prediction_plot() {
    example(
        "scatterplot_with_overlaid_fractional_polynomial_prediction_plot",
        AUTO,
        &[
        r#"twoway scatter mpg weight || fpfit mpg weight"#,
        ],
    );
}

/// Scatterplot with overlaid fractional-polynomial prediction plot by variable
#[test]
fn scatterplot_with_overlaid_fractional_polynomial_prediction_plot_by_variable() {
    example(
        "scatterplot_with_overlaid_fractional_polynomial_prediction_plot_by_variable",
        AUTO,
        &[
        r#"twoway scatter mpg weight || fpfit mpg weight ||, by(foreign, total row(1))"#,
        ],
    );
}

/// Scatterplot with overlaid linear prediction plot
#[test]
fn scatterplot_with_overlaid_linear_prediction_plot() {
    example(
        "scatterplot_with_overlaid_linear_prediction_plot",
        AUTO,
        &[
        r#"twoway scatter mpg weight || lfit mpg weight"#,
        ],
    );
}

/// Scatterplot with overlaid linear prediction plot by variable
#[test]
fn scatterplot_with_overlaid_linear_prediction_plot_by_variable() {
    example(
        "scatterplot_with_overlaid_linear_prediction_plot_by_variable",
        AUTO,
        &[
        r#"twoway scatter mpg weight || lfit mpg weight ||, by(foreign, total row(1) style(altleg))"#,
        ],
    );
}

/// Scatterplot with overlaid quadratic prediction plot
#[test]
fn scatterplot_with_overlaid_quadratic_prediction_plot() {
    example(
        "scatterplot_with_overlaid_quadratic_prediction_plot",
        AUTO,
        &[
        r#"twoway scatter mpg weight || qfit mpg weight"#,
        ],
    );
}

/// Scatterplot with overlaid quadratic prediction plot by variable
#[test]
fn scatterplot_with_overlaid_quadratic_prediction_plot_by_variable() {
    example(
        "scatterplot_with_overlaid_quadratic_prediction_plot_by_variable",
        AUTO,
        &[
        r#"twoway scatter mpg weight || qfit mpg weight ||, by(foreign, total row(1) style(altleg))"#,
        ],
    );
}

/// Scatterplot with weighted markers
#[test]
fn scatterplot_with_weighted_markers() {
    example(
        "scatterplot_with_weighted_markers",
        CENSUS,
        &[
        r#"scatter death medage [w=pop65p], msymbol(circle_hollow)"#,
        ],
    );
}

/// Spike plot
#[test]
fn spike_plot() {
    example(
        "spike_plot",
        NLSW88,
        &[
        r#"spikeplot age"#,
        ],
    );
}

/// Stability condition of VECM estimates
#[test]
fn stability_condition_of_vecm_estimates() {
    example(
        "stability_condition_of_vecm_estimates",
        TXHPRICE,
        &[
        r#"vec austin dallas houston sa, lags(3)"#,
        r#"vecstable, graph subtitle("") note("")"#,
        ],
    );
}

/// Stacked bar chart
#[test]
fn stacked_bar_chart() {
    example(
        "stacked_bar_chart",
        CITYTEMP,
        &[
        r#"graph bar tempjan tempjuly, over(region) stack"#,
        ],
    );
}

/// Standardized normal probability plot
#[test]
fn standardized_normal_probability_plot() {
    example(
        "standardized_normal_probability_plot",
        AUTO,
        &[
        r#"pnorm price"#,
        ],
    );
}

/// Sunflower plot
#[test]
fn sunflower_plot() {
    example(
        "sunflower_plot",
        AUTO,
        &[
        r#"sunflower mpg displ"#,
        ],
    );
}

/// Symmetry plot
#[test]
fn symmetry_plot() {
    example(
        "symmetry_plot",
        AUTO,
        &[
        r#"symplot price"#,
        ],
    );
}

/// Twoway bar plot
#[test]
fn twoway_bar_plot() {
    example(
        "twoway_bar_plot",
        SP500,
        &[
        r#"twoway bar change date in 1/27"#,
        ],
    );
}

/// Twoway dropped-line plot
#[test]
fn twoway_dropped_line_plot() {
    example(
        "twoway_dropped_line_plot",
        SP500,
        &[
        r#"twoway dropline change date in 1/27"#,
        ],
    );
}

/// Twoway fractional-polynomial prediction plot
#[test]
fn twoway_fractional_polynomial_prediction_plot() {
    example(
        "twoway_fractional_polynomial_prediction_plot",
        AUTO,
        &[
        r#"twoway fpfit mpg weight"#,
        ],
    );
}

/// Twoway linear prediction plot
#[test]
fn twoway_linear_prediction_plot() {
    example(
        "twoway_linear_prediction_plot",
        AUTO,
        &[
        r#"twoway lfit mpg weight"#,
        ],
    );
}

/// Twoway linear prediction plot with CIs
#[test]
fn twoway_linear_prediction_plot_with_cis() {
    example(
        "twoway_linear_prediction_plot_with_cis",
        AUTO,
        &[
        r#"twoway lfitci mpg weight"#,
        ],
    );
}

/// Twoway quadratic prediction plot
#[test]
fn twoway_quadratic_prediction_plot() {
    example(
        "twoway_quadratic_prediction_plot",
        AUTO,
        &[
        r#"twoway qfit mpg weight"#,
        ],
    );
}

/// Twoway spike plot
#[test]
fn twoway_spike_plot() {
    example(
        "twoway_spike_plot",
        SP500,
        &[
        r#"twoway spike change date in 1/27"#,
        ],
    );
}

/// Twoway time-series line plot
#[test]
fn twoway_time_series_line_plot() {
    example(
        "twoway_time_series_line_plot",
        TSAPPEND1,
        &[
        r#"tsline y"#,
        ],
    );
}

/// Twoway time-series line plot with connected points
#[test]
fn twoway_time_series_line_plot_with_connected_points() {
    example(
        "twoway_time_series_line_plot_with_connected_points",
        TSAPPEND1,
        &[
        r#"tsline y, recast(connected)  plotregion(margin(r+1))"#,
        ],
    );
}

/// Twoway time-series range plot with lines
#[test]
fn twoway_time_series_range_plot_with_lines() {
    example(
        "twoway_time_series_range_plot_with_lines",
        TSAPPEND1,
        &[
        r#"tsrline y e, plotregion(margin(r+1))"#,
        ],
    );
}
