//! Wave 7b: penalized regression (lasso, elastic net, square-root lasso,
//! inference after selection), discrete-choice models over choice sets and
//! the remaining treatment-effect estimators. Synthetic data only; checks
//! are closed forms or agreement with other estimators.

#[path = "t53/causal.rs"]
mod causal;
#[path = "t53/causal2.rs"]
mod causal2;
#[path = "t53/cm.rs"]
mod cm;
#[path = "t53/lasso.rs"]
mod lasso;
