//! Command dispatch: resolves abbreviations, runs built-in commands and the
//! `by` prefix, and owns the data in memory and interpreter state.

use std::collections::HashMap;

use ferrostat_cmd::commands::CommandHandler;
use ferrostat_cmd::eval::{rc_err, Groups};
use ferrostat_cmd::store::Store;
use ferrostat_cmd::{
    correlate, data, egen, distplot, graph, io, kdensity, labels, listing, matcmd, models, postest, restructure, roc, summarize, survival, tabulate, tests, tsplot,
    util, varlist,
};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

/// Kept for API compatibility with earlier versions.
pub use ferrostat_cmd::store::Results as EstimationResults;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ModelType {
    Linear,
    Logit,
    Probit,
    Poisson,
    FixedEffects,
    #[default]
    Other,
}

/// Aliases accepted for command names.
const ALIASES: &[(&str, &str)] = &[
    ("gen", "generate"),
    ("g", "generate"),
    ("su", "summarize"),
    ("sum", "summarize"),
    ("tab", "tabulate"),
    ("ta", "tabulate"),
    ("reg", "regress"),
    ("d", "describe"),
    ("des", "describe"),
    ("desc", "describe"),
    ("l", "list"),
    ("li", "list"),
    ("di", "display"),
    ("dis", "display"),
    ("cor", "correlate"),
    ("corr", "correlate"),
    ("tw", "twoway"),
    ("sc", "scatter"),
    ("hist", "histogram"),
    ("mat", "matrix"),
    ("est", "estimates"),
    ("qui", "quietly"),
    ("noi", "noisily"),
    ("cap", "capture"),
    ("forv", "forvalues"),
    ("ren", "rename"),
    ("lab", "label"),
    ("br", "browse"),
    ("ed", "edit"),
    ("u", "use"),
    ("cou", "count"),
    ("xi", "xi"),
    ("tab1", "tab1"),
    ("tab2", "tab2"),
    ("bys", "bysort"),
    ("ivreg", "ivregress"),
    ("erase", "erase"),
    ("rm", "erase"),
    ("ls", "dir"),
    ("cat", "type"),
    ("logistic", "logistic"),
    ("outreg", "outreg2"),
];

/// (full name, minimal abbreviation length)
const ABBREV: &[(&str, usize)] = &[
    ("generate", 1),
    ("summarize", 2),
    ("tabulate", 2),
    ("regress", 3),
    ("describe", 1),
    ("list", 1),
    ("display", 2),
    ("correlate", 3),
    ("count", 3),
    ("rename", 3),
    ("label", 2),
    ("use", 1),
    ("save", 4),
    ("sort", 2),
    ("predict", 4),
    ("probit", 4),
    ("logit", 5),
    ("local", 3),
    ("global", 2),
    ("scalar", 3),
    ("matrix", 3),
    ("quietly", 3),
    ("noisily", 1),
    ("macro", 2),
    ("capture", 3),
    ("forvalues", 4),
    ("foreach", 7),
    ("estimates", 3),
    ("format", 3),
    ("merge", 5),
    ("append", 2),
    ("insheet", 3),
    ("infile", 3),
    ("outsheet", 4),
    ("histogram", 4),
    ("scatter", 2),
    ("twoway", 2),
    ("graph", 2),
    ("ttest", 5),
    ("collapse", 4),
    ("contract", 4),
    ("compress", 4),
    ("duplicates", 3),
    ("keep", 4),
    ("drop", 4),
    ("replace", 7),
    ("order", 5),
    ("set", 3),
    ("program", 3),
    ("tsset", 5),
    ("xtset", 5),
    ("margins", 6),
    ("lincom", 6),
    ("testparm", 8),
    ("test", 4),
    ("anova", 5),
    ("recode", 5),
    ("encode", 6),
    ("decode", 6),
    ("destring", 8),
    ("tostring", 8),
    ("ereturn", 4),
    ("return", 3),
    ("matlist", 7),
    ("sample", 4),
    ("tabstat", 7),
    ("pwcorr", 6),
    ("spearman", 5),
    ("kdensity", 5),
    ("poisson", 7),
    ("nbreg", 5),
    ("ologit", 6),
    ("oprobit", 7),
    ("mlogit", 6),
    ("xtreg", 5),
    ("ivregress", 9),
    ("logistic", 8),
    ("assert", 6),
    ("confirm", 4),
    ("codebook", 8),
    ("inspect", 4),
    ("browse", 2),
    ("edit", 2),
    ("clonevar", 8),
];

/// Commands that run once with group-aware evaluation under `by`.
const GROUPWISE: &[&str] = &["generate", "replace", "egen", "assert", "drop", "keep", "gegen", "fegen", "carryforward"];

pub struct Engine {
    pub dataset: DataSet,
    pub store: Store,
    handlers: HashMap<&'static str, Box<dyn CommandHandler>>,
    pub version: f64,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let mut list: Vec<(&'static str, Box<dyn CommandHandler>)> = vec![
            ("generate", Box::new(data::Generate)),
            ("replace", Box::new(data::Replace)),
            ("drop", Box::new(data::Drop)),
            ("keep", Box::new(data::Keep)),
            ("rename", Box::new(data::Rename)),
            ("order", Box::new(data::Order)),
            ("move", Box::new(data::Move)),
            ("sort", Box::new(data::Sort)),
            ("gsort", Box::new(data::Gsort)),
            ("format", Box::new(data::Format)),
            ("compress", Box::new(data::Compress)),
            ("assert", Box::new(data::Assert)),
            ("insobs", Box::new(data::Insobs)),
            ("recast", Box::new(data::Recast)),
            ("tsset", Box::new(data::Tsset)),
            ("xtset", Box::new(data::Tsset)),
            ("label", Box::new(labels::Label)),
            ("clonevar", Box::new(ferrostat_cmd::clonevar::Clonevar)),
            ("browse", Box::new(ferrostat_cmd::clonevar::Browse)),
            ("edit", Box::new(ferrostat_cmd::clonevar::Browse)),
            ("encode", Box::new(labels::Encode)),
            ("decode", Box::new(labels::Decode)),
            ("destring", Box::new(labels::Destring)),
            ("tostring", Box::new(labels::Tostring)),
            ("recode", Box::new(labels::Recode)),
            ("mvencode", Box::new(labels::Mvencode)),
            ("mvdecode", Box::new(labels::Mvdecode)),
            ("split", Box::new(labels::Split)),
            ("describe", Box::new(listing::Describe)),
            ("count", Box::new(listing::Count)),
            ("list", Box::new(listing::List)),
            ("duplicates", Box::new(listing::Duplicates)),
            ("isid", Box::new(listing::Isid)),
            ("levelsof", Box::new(listing::Levelsof)),
            ("frame", Box::new(ferrostat_cmd::frames::FrameCmd)),
            ("frames", Box::new(ferrostat_cmd::frames::FrameCmd)),
            ("cwf", Box::new(ferrostat_cmd::frames::Cwf)),
            ("pwf", Box::new(ferrostat_cmd::frames::Pwf)),
            ("frlink", Box::new(ferrostat_cmd::frames::Frlink)),
            ("frget", Box::new(ferrostat_cmd::frames::Frget)),
            ("unique", Box::new(ferrostat_cmd::unique::Unique)),
            ("codebook", Box::new(listing::Codebook)),
            ("misstable", Box::new(listing::Misstable)),
            ("expand", Box::new(restructure::Expand)),
            ("fillin", Box::new(restructure::Fillin)),
            ("contract", Box::new(restructure::Contract)),
            ("collapse", Box::new(restructure::Collapse)),
            ("reshape", Box::new(restructure::Reshape)),
            ("merge", Box::new(restructure::Merge)),
            ("append", Box::new(restructure::Append)),
            ("joinby", Box::new(restructure::Joinby)),
            ("cross", Box::new(restructure::Cross)),
            ("xpose", Box::new(restructure::Xpose)),
            ("sample", Box::new(restructure::Sample)),
            ("egen", Box::new(egen::Egen)),
            ("summarize", Box::new(summarize::Summarize)),
            ("tabstat", Box::new(summarize::Tabstat)),
            ("centile", Box::new(summarize::Centile)),
            ("mean", Box::new(summarize::MeanCmd)),
            ("ci", Box::new(summarize::Ci)),
            ("table", Box::new(ferrostat_cmd::table::Table)),
            ("total", Box::new(ferrostat_cmd::table::Total)),
            ("proportion", Box::new(ferrostat_cmd::proportion::Proportion)),
            ("svyset", Box::new(ferrostat_cmd::svy::Svyset)),
            ("svydescribe", Box::new(ferrostat_cmd::svy::Svydescribe)),
            ("svymarkout", Box::new(ferrostat_cmd::svy::Svymarkout)),
            ("teffects", Box::new(ferrostat_cmd::teffects::Teffects)),
            ("tebalance", Box::new(ferrostat_cmd::te_match::Tebalance)),
            ("teoverlap", Box::new(ferrostat_cmd::te_match::Teoverlap)),
            ("psmatch2", Box::new(ferrostat_cmd::psmatch2::Psmatch2)),
            ("pstest", Box::new(ferrostat_cmd::psmatch2::Pstest)),
            ("psgraph", Box::new(ferrostat_cmd::psmatch2::Psgraph)),
            ("cem", Box::new(ferrostat_cmd::cem::Cem)),
            ("imb", Box::new(ferrostat_cmd::cem::Imb)),
            ("ebalance", Box::new(ferrostat_cmd::ebalance::Ebalance)),
            ("synth", Box::new(ferrostat_cmd::synth::Synth)),
            ("boottest", Box::new(ferrostat_cmd::boottest::Boottest)),
            ("distinct", Box::new(ferrostat_cmd::community::Distinct)),
            ("mdesc", Box::new(ferrostat_cmd::community::Mdesc)),
            ("winsor2", Box::new(ferrostat_cmd::community::Winsor2)),
            ("tabulate", Box::new(tabulate::Tabulate)),
            ("tab1", Box::new(tabulate::Tab1)),
            ("tab2", Box::new(tabulate::Tab2)),
            ("correlate", Box::new(correlate::Correlate)),
            ("pwcorr", Box::new(correlate::Pwcorr)),
            ("spearman", Box::new(correlate::Spearman)),
            ("ttest", Box::new(tests::Ttest)),
            ("prtest", Box::new(tests::Prtest)),
            ("sdtest", Box::new(tests::Sdtest)),
            ("ranksum", Box::new(tests::Ranksum)),
            ("signrank", Box::new(tests::Signrank)),
            ("signtest", Box::new(tests::Signtest)),
            ("kwallis", Box::new(tests::Kwallis)),
            ("oneway", Box::new(tests::Oneway)),
            ("ksmirnov", Box::new(tests::Ksmirnov)),
            ("swilk", Box::new(tests::Swilk)),
            ("kdensity", Box::new(kdensity::Kdensity)),
            ("regress", Box::new(models::Regress)),
            ("logit", Box::new(models::BinaryModel { verb: "logit", link: models::Link::Logit })),
            ("logistic", Box::new(models::BinaryModel { verb: "logistic", link: models::Link::Logit })),
            ("probit", Box::new(models::BinaryModel { verb: "probit", link: models::Link::Probit })),
            ("cloglog", Box::new(models::BinaryModel { verb: "cloglog", link: models::Link::Cloglog })),
            ("poisson", Box::new(models::BinaryModel { verb: "poisson", link: models::Link::Poisson })),
            ("nbreg", Box::new(models::Nbreg)),
            ("ologit", Box::new(models::Ordered { verb: "ologit", probit: false })),
            ("oprobit", Box::new(models::Ordered { verb: "oprobit", probit: true })),
            ("mlogit", Box::new(models::Mlogit)),
            ("tobit", Box::new(models::Tobit)),
            ("anova", Box::new(models::Anova)),
            ("xtreg", Box::new(models::Xtreg)),
            ("areg", Box::new(models::Areg)),
            ("reghdfe", Box::new(ferrostat_cmd::reghdfe::Reghdfe)),
            ("ivreghdfe", Box::new(ferrostat_cmd::reghdfe::Ivreghdfe)),
            ("rdrobust", Box::new(ferrostat_cmd::rdrobust::Rdrobust)),
            ("rdbwselect", Box::new(ferrostat_cmd::rdrobust::Rdbwselect)),
            ("rdplot", Box::new(ferrostat_cmd::rdplot::Rdplot)),
            ("rddensity", Box::new(ferrostat_cmd::rddensity::Rddensity)),
            ("rdbwdensity", Box::new(ferrostat_cmd::rddensity::Rdbwdensity)),
            ("ppmlhdfe", Box::new(ferrostat_cmd::ppmlhdfe::Ppmlhdfe)),
            ("binscatter", Box::new(ferrostat_cmd::binscatter::Binscatter)),
            ("ivregress", Box::new(models::Ivregress)),
            ("xtivreg", Box::new(models::Xtivreg { ivreg2: false })),
            ("xtivreg2", Box::new(models::Xtivreg { ivreg2: true })),
            ("xtpmg", Box::new(ferrostat_cmd::panel_ardl::Xtpmg)),
            ("hausman", Box::new(ferrostat_cmd::panel_ardl::Hausman)),
            ("shp2dta", Box::new(ferrostat_cmd::spatial::Shp2dta)),
            ("spmat", Box::new(ferrostat_cmd::spatial::Spmat)),
            ("spmap", Box::new(ferrostat_cmd::spatial::Spmap)),
            ("spset", Box::new(ferrostat_cmd::sp_data::Spset)),
            ("spshape2dta", Box::new(ferrostat_cmd::sp_data::Spshape2dta)),
            ("spbalance", Box::new(ferrostat_cmd::sp_data::Spbalance)),
            ("spcompress", Box::new(ferrostat_cmd::sp_data::Spcompress)),
            ("spdistance", Box::new(ferrostat_cmd::sp_data::Spdistance)),
            ("spgenerate", Box::new(ferrostat_cmd::sp_data::Spgenerate)),
            ("spmatrix", Box::new(ferrostat_cmd::sp_data::Spmatrix)),
            ("grmap", Box::new(ferrostat_cmd::sp_data::Grmap)),
            ("spregress", Box::new(ferrostat_cmd::sp_reg::Spregress { iv: false })),
            ("spivregress", Box::new(ferrostat_cmd::sp_reg::Spregress { iv: true })),
            ("spxtregress", Box::new(ferrostat_cmd::sp_reg::Spxtregress)),
            ("glm", Box::new(models::Glm)),
            ("qreg", Box::new(models::Qreg)),
            ("predict", Box::new(postest::Predict)),
            ("test", Box::new(postest::Test)),
            ("testparm", Box::new(postest::Testparm)),
            ("lincom", Box::new(postest::Lincom)),
            ("estimates", Box::new(postest::Estimates)),
            ("margins", Box::new(ferrostat_cmd::margins::Margins)),
            ("marginsplot", Box::new(ferrostat_cmd::margins::Marginsplot)),
            ("contrast", Box::new(ferrostat_cmd::contrast::Contrast)),
            ("estat", Box::new(postest::Estat)),
            ("separate", Box::new(ferrostat_cmd::separate::Separate)),
            ("melogit", Box::new(ferrostat_cmd::melogit::Melogit { verb: "melogit" })),
            ("meqrlogit", Box::new(ferrostat_cmd::melogit::Melogit { verb: "meqrlogit" })),
            ("manova", Box::new(ferrostat_cmd::manova::Manova)),
            // multivariate analysis
            ("pca", Box::new(ferrostat_cmd::mv_pca::Pca { mat: false })),
            ("pcamat", Box::new(ferrostat_cmd::mv_pca::Pca { mat: true })),
            ("factor", Box::new(ferrostat_cmd::mv_pca::Factor { mat: false })),
            ("factormat", Box::new(ferrostat_cmd::mv_pca::Factor { mat: true })),
            ("rotate", Box::new(ferrostat_cmd::mv_pca::Rotate)),
            ("rotatemat", Box::new(ferrostat_cmd::mv_rotate::Rotatemat)),
            ("screeplot", Box::new(ferrostat_cmd::mv_pca::Screeplot)),
            ("scree", Box::new(ferrostat_cmd::mv_pca::Screeplot)),
            ("loadingplot", Box::new(ferrostat_cmd::mv_pca::Loadingplot { score: false })),
            ("scoreplot", Box::new(ferrostat_cmd::mv_pca::Loadingplot { score: true })),
            ("cluster", Box::new(ferrostat_cmd::mv_cluster::Cluster { mat: false })),
            ("clustermat", Box::new(ferrostat_cmd::mv_cluster::Cluster { mat: true })),
            ("mds", Box::new(ferrostat_cmd::mv_mds::Mds)),
            ("mdsmat", Box::new(ferrostat_cmd::mv_mds::Mdsmat)),
            ("mdslong", Box::new(ferrostat_cmd::mv_mds::Mdslong)),
            ("mdsconfig", Box::new(ferrostat_cmd::mv_mds::Mdsconfig)),
            ("ca", Box::new(ferrostat_cmd::mv_mds::Ca)),
            ("camat", Box::new(ferrostat_cmd::mv_mds::Camat)),
            ("cabiplot", Box::new(ferrostat_cmd::mv_mds::Cabiplot)),
            ("mca", Box::new(ferrostat_cmd::mv_mds::Mca)),
            ("biplot", Box::new(ferrostat_cmd::mv_mds::Biplot)),
            ("discrim", Box::new(ferrostat_cmd::mv_discrim::Discrim { candisc: false })),
            ("candisc", Box::new(ferrostat_cmd::mv_discrim::Discrim { candisc: true })),
            ("canon", Box::new(ferrostat_cmd::mv_misc::Canon)),
            ("alpha", Box::new(ferrostat_cmd::mv_misc::Alpha)),
            ("hotelling", Box::new(ferrostat_cmd::mv_misc::Hotelling)),
            ("mvtest", Box::new(ferrostat_cmd::mv_misc::Mvtest)),
            ("procrustes", Box::new(ferrostat_cmd::mv_misc::Procrustes)),
            ("manovatest", Box::new(ferrostat_cmd::manova::Manovatest)),
            ("mvreg", Box::new(ferrostat_cmd::manova::Mvreg)),
            ("rreg", Box::new(ferrostat_cmd::rreg::Rreg)),
            ("zip", Box::new(ferrostat_cmd::zip::Zip { nb: false })),
            ("zinb", Box::new(ferrostat_cmd::zip::Zip { nb: true })),
            ("matrix", Box::new(matcmd::MatrixCmd)),
            ("mkmat", Box::new(matcmd::Mkmat)),
            ("svmat", Box::new(matcmd::Svmat)),
            ("matlist", Box::new(matcmd::Matlist)),
            ("use", Box::new(io::Use)),
            ("save", Box::new(io::Save)),
            ("saveold", Box::new(io::Save)),
            ("import", Box::new(io::Import)),
            ("export", Box::new(io::Export)),
            ("insheet", Box::new(io::Insheet)),
            ("infile", Box::new(ferrostat_cmd::infile::Infile)),
            ("infix", Box::new(ferrostat_cmd::infile::Infix)),
            ("outsheet", Box::new(io::Outsheet)),
            ("twoway", Box::new(graph::Twoway)),
            ("scatter", Box::new(graph::Twoway)),
            ("line", Box::new(graph::Twoway)),
            ("connected", Box::new(graph::Twoway)),
            ("tsline", Box::new(graph::TsLine)),
            ("palette", Box::new(graph::Palette)),
            ("qnorm", Box::new(distplot::DistPlot)),
            ("pnorm", Box::new(distplot::DistPlot)),
            ("qchi", Box::new(distplot::DistPlot)),
            ("pchi", Box::new(distplot::DistPlot)),
            ("quantile", Box::new(distplot::DistPlot)),
            ("qqplot", Box::new(distplot::DistPlot)),
            ("symplot", Box::new(distplot::DistPlot)),
            ("gladder", Box::new(distplot::DistPlot)),
            ("qladder", Box::new(distplot::DistPlot)),
            ("spikeplot", Box::new(distplot::DistPlot)),
            ("sunflower", Box::new(distplot::DistPlot)),
            ("ac", Box::new(tsplot::TsPlot)),
            ("stset", Box::new(survival::Stset)),
            ("roctab", Box::new(roc::Roc)),
            ("roccomp", Box::new(roc::Roc)),
            ("rocgold", Box::new(roc::Roc)),
            ("rocfit", Box::new(roc::Roc)),
            ("sts", Box::new(survival::Sts)),
            ("strate", Box::new(survival::Strate)),
            ("ltable", Box::new(survival::Ltable)),
            ("pac", Box::new(tsplot::TsPlot)),
            ("xcorr", Box::new(tsplot::TsPlot)),
            ("pergram", Box::new(tsplot::TsPlot)),
            ("cumsp", Box::new(tsplot::TsPlot)),
            ("wntestb", Box::new(tsplot::TsPlot)),
            ("tsrline", Box::new(graph::TsLine)),
            ("histogram", Box::new(graph::Histogram)),
            ("rvfplot", Box::new(ferrostat_cmd::regplot::RegPlot)),
            ("rvpplot", Box::new(ferrostat_cmd::regplot::RegPlot)),
            ("avplot", Box::new(ferrostat_cmd::regplot::RegPlot)),
            ("avplots", Box::new(ferrostat_cmd::regplot::RegPlot)),
            ("lvr2plot", Box::new(ferrostat_cmd::regplot::RegPlot)),
            ("graph", Box::new(graph::GraphCmd)),
            ("coefplot", Box::new(ferrostat_cmd::coefplot::Coefplot)),
            // time series (tsreg, arima, tstests, tssmooth)
            ("newey", Box::new(ferrostat_cmd::tsreg::Newey)),
            ("prais", Box::new(ferrostat_cmd::tsreg::Prais)),
            ("arima", Box::new(ferrostat_cmd::arima::Arima)),
            ("dfuller", Box::new(ferrostat_cmd::tstests::Dfuller)),
            ("pperron", Box::new(ferrostat_cmd::tstests::Pperron)),
            ("dfgls", Box::new(ferrostat_cmd::tstests::Dfgls)),
            ("corrgram", Box::new(ferrostat_cmd::tstests::Corrgram)),
            ("wntestq", Box::new(ferrostat_cmd::tstests::Wntestq)),
            ("tssmooth", Box::new(ferrostat_cmd::tssmooth::Tssmooth)),
            ("tsfilter", Box::new(ferrostat_cmd::tssmooth::Tsfilter)),
        ];
        list.extend(ferrostat_cmd::community_dm2::handlers());
        list.extend(ferrostat_cmd::tsvar::handlers());
        list.extend(ferrostat_cmd::ts_more::handlers());
        list.extend(ferrostat_cmd::xtnl::handlers());
        list.extend(ferrostat_cmd::stcox::handlers());
        list.extend(ferrostat_cmd::choice_mixed::handlers());
        list.extend(ferrostat_cmd::xmodels::handlers());
        list.extend(ferrostat_cmd::r_rest::handlers());
        list.extend(ferrostat_cmd::basetools::handlers());
        list.extend(ferrostat_cmd::data_utils::handlers());
        list.extend(ferrostat_cmd::did::handlers());
        let mut handlers = HashMap::new();
        for (n, h) in list {
            handlers.insert(n, h);
        }
        Engine { dataset: DataSet::new(), store: Store::new(), handlers, version: 18.0 }
    }

    /// Resolve an abbreviated or aliased command name.
    pub fn resolve_verb(&self, verb: &str) -> String {
        let v = verb.to_lowercase();
        let alias = ALIASES.iter().find(|(a, _)| *a == v).map(|(_, full)| *full);
        if let Some(a) = alias {
            return a.to_string();
        }
        if self.handlers.contains_key(v.as_str()) {
            return v;
        }
        for (full, min) in ABBREV {
            if v.len() >= *min && full.starts_with(&v) {
                return full.to_string();
            }
        }
        v
    }

    /// Every word accepted as a command name: registered commands, aliases
    /// and all valid abbreviations (used for syntax highlighting).
    pub fn command_words(&self) -> Vec<String> {
        let mut words: Vec<String> = self.handlers.keys().map(|k| k.to_string()).collect();
        words.extend(ALIASES.iter().map(|(a, _)| a.to_string()));
        for (full, min) in ABBREV {
            for len in *min..=full.len() {
                words.push(full[..len].to_string());
            }
        }
        words.sort();
        words.dedup();
        words
    }

    pub fn is_command(&self, verb: &str) -> bool {
        self.handlers.contains_key(self.resolve_verb(verb).as_str())
    }

    /// Execute one parsed command (no prefixes).
    pub fn execute(&mut self, cmd: &Command) -> Result<String, String> {
        let verb = self.resolve_verb(&cmd.verb);
        let mut cmd = cmd.clone();
        cmd.verb = verb.clone();
        match verb.as_str() {
            "clear" => {
                let what = cmd.main.trim();
                match what {
                    "" | "all" | "*" => {
                        self.dataset = DataSet::new();
                        self.store.data_changed = false;
                        self.store.tsset = None;
                        if what == "all" || what == "*" {
                            ferrostat_cmd::frames::reset(&mut self.store);
                            self.store.matrices.clear();
                            self.store.scalars.clear();
                            self.store.e = Default::default();
                            self.store.r = Default::default();
                            self.store.estimates.clear();
                            self.store.programs.clear();
                            self.store.graphs.clear();
                            self.store.graph = None;
                        }
                    }
                    "results" => {
                        self.store.r = Default::default();
                        self.store.e = Default::default();
                    }
                    "matrix" => self.store.matrices.clear(),
                    "programs" => self.store.programs.clear(),
                    "mata" | "ado" => {}
                    other => return Err(rc_err(198, format!("clear {other} invalid"))),
                }
                Ok(String::new())
            }
            "preserve" => {
                self.store.preserved.push(self.dataset.clone());
                Ok(String::new())
            }
            "restore" => {
                if cmd.has_opt("not", 3) {
                    self.store.preserved.pop().ok_or_else(|| rc_err(622, "nothing to restore"))?;
                    return Ok(String::new());
                }
                let ds = if cmd.has_opt("preserve", 3) {
                    self.store.preserved.last().cloned()
                } else {
                    self.store.preserved.pop()
                };
                self.dataset = ds.ok_or_else(|| rc_err(622, "nothing to restore"))?;
                Ok(String::new())
            }
            "set" => self.set(&cmd),
            "sysuse" => {
                let name = cmd.main.trim().trim_end_matches(".dta");
                if name == "dir" {
                    return Ok(String::new());
                }
                for dir in ["/usr/local/share/ferrostat", "."] {
                    let p = format!("{dir}/{name}.dta");
                    if let Ok(ds) = ferrostat_core::dta::DtaReader::open(&p) {
                        self.dataset = ds;
                        self.store.data_changed = false;
                        self.store.filename = p;
                        return Ok(String::new());
                    }
                }
                Err(rc_err(601, format!("file \"{name}.dta\" not found")))
            }
            "describe" if cmd.main.trim().is_empty() && self.dataset.n_vars() == 0 && cmd.using.is_none() => {
                Ok("Contains data\n Observations:             0\n    Variables:             0".into())
            }
            _ => {
                let h = self.handlers.get(verb.as_str()).ok_or_else(|| rc_err(199, format!("command {} is unrecognized", cmd.verb)))?;
                ferrostat_cmd::est::take_table();
                let before = self.store.e.macros.clone();
                let res = h.execute(&mut self.dataset, &mut self.store, &cmd);
                // new estimation results remember the command line (e(cmdline))
                if res.is_ok() && self.store.e.macros != before && self.store.e.matrices.contains_key("b") && !self.store.e.macros.contains_key("cmdline") {
                    self.store.e.set_str("cmdline", cmd.raw.trim());
                }
                // commands that display a coefficient table return it in r(table)
                if let (Ok(_), Some((m, level))) = (&res, ferrostat_cmd::est::take_table()) {
                    let mut r = ferrostat_cmd::store::Results::new();
                    r.set_mat("table", m);
                    r.set("level", level);
                    self.store.r = r;
                }
                res
            }
        }
    }

    fn set(&mut self, cmd: &Command) -> Result<String, String> {
        let mut w = cmd.main.split_whitespace();
        let what = w.next().unwrap_or("").to_lowercase();
        let val: Vec<&str> = w.collect();
        let v = val.join(" ");
        match what.as_str() {
            "obs" => {
                let n: f64 = v.trim().parse().map_err(|_| rc_err(198, "invalid number of observations"))?;
                data::set_obs(&mut self.dataset, n as usize).map(|_| String::new())
            }
            "seed" => {
                let s: u64 = v.trim().parse().map_err(|_| rc_err(198, "invalid seed"))?;
                self.store.rng.borrow_mut().reseed(s);
                Ok(String::new())
            }
            "type" => {
                self.store.default_type = match v.trim() {
                    "double" => ferrostat_core::VarType::Double,
                    "float" => ferrostat_core::VarType::Float,
                    o => return Err(rc_err(198, format!("{o} invalid type"))),
                };
                Ok(String::new())
            }
            "level" => {
                self.store.level = v.trim().parse().map_err(|_| rc_err(198, "invalid level"))?;
                self.store.globals.insert("S_level".into(), v.trim().to_string());
                Ok(String::new())
            }
            "rmsg" => {
                self.store.rmsg = v.trim() == "on";
                Ok(String::new())
            }
            "varabbrev" => {
                self.store.varabbrev = v.trim() == "on";
                Ok(String::new())
            }
            "graphics" => {
                self.store.graphics_off = v.trim() == "off";
                Ok(String::new())
            }
            "printcolor" => {
                let p = v.split(',').next().unwrap_or("").trim().to_lowercase();
                if !matches!(p.as_str(), "automatic" | "asis" | "gs1" | "gs2" | "gs3") {
                    return Err(rc_err(198, format!("printcolor {p} not allowed")));
                }
                self.store.printcolor = p;
                Ok(String::new())
            }
            "scheme" => {
                // unknown scheme names (from other software) keep the default palette
                let name = v.split(',').next().unwrap_or("").trim().to_lowercase();
                self.store.scheme = if ferrostat_cmd::graph::is_scheme(&name) { name } else { String::new() };
                Ok(String::new())
            }
            "sortmethod" | "sortrngstate" | "trace" if !matches!(v.split(',').next().unwrap_or("").trim(), "" | "default" | "fsort" | "qsort" | "on" | "off" | "random") && v.trim().parse::<u64>().is_err() => {
                Err(rc_err(198, format!("set {what} {} not allowed", v.trim())))
            }
            // accepted for compatibility; no effect
            "sortrngstate" | "traceexpand" | "tracesep" | "traceindent" | "tracenumber" | "tracehilite" | "more" | "maxvar" | "matsize" | "linesize" | "pagesize" | "trace" | "tracedepth" | "logtype"
            | "output" | "processors" | "niceness" | "dp" | "emptycells" | "fvlabel" | "showbaselevels" | "cformat" | "pformat"
            | "sformat" | "maxiter" | "checksum" | "autotabgraphs" | "segmentsize" | "max_memory" | "min_memory" | "copycolor" | "locale_functions"
            | "locale_ui" | "charset" | "httpproxy" | "update_query" | "timeout1" | "timeout2" | "haverdir" | "odbcmgr" | "notifyuser"
            | "include_bitmap" | "doublebuffer" | "reventries" | "varkeyboard" | "showomitted" | "showemptycells" | "lstretch" | "coeftabresults"
            | "fvwrap" | "fvwrapon" | "collect_label" | "collect_style" | "table_style" | "dots" | "iterlog" | "python_exec" | "rng" | "rngstream"
            | "parallel_transport" | "sortseed" | "sortmethod" | "matacache" | "matafavor" | "matastrict" | "matalnum" | "mataoptimize" | "matalibs" | "matamofirst" => {
                let v = v.split(',').next().unwrap_or("").trim();
                if !v.is_empty() {
                    self.store.settings.insert(what.clone(), v.to_string());
                }
                Ok(String::new())
            }
            "" => Ok(String::new()),
            other => Err(rc_err(198, format!("set {other} not allowed"))),
        }
    }

    /// Run a command under `by varlist:`; `run` executes the inner command.
    pub fn by_groups(&mut self, byvars: &[String], sortvars: &[String], sort: bool) -> Result<Groups, String> {
        let mut all = byvars.to_vec();
        all.extend(sortvars.iter().cloned());
        let vars = varlist::expand(&self.dataset, &all.join(" "))?;
        if sort {
            self.dataset.sort(&vars, true)?;
        } else if !util::is_sorted_by(&self.dataset, &vars) {
            return Err(rc_err(5, "not sorted"));
        }
        let bv = varlist::expand(&self.dataset, &byvars.join(" "))?;
        Ok(util::groups_by(&self.dataset, &bv))
    }

    pub fn is_groupwise(&self, verb: &str) -> bool {
        GROUPWISE.contains(&self.resolve_verb(verb).as_str())
    }
}
