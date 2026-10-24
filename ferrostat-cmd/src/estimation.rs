use std::sync::{Arc, Mutex};

/// Stored estimation results for post-estimation.
#[derive(Debug, Clone, Default)]
pub struct EstimationResults {
    pub cmd: String,
    pub depvar: String,
    pub indepvars: Vec<String>,
    pub beta: Vec<f64>,
    pub n_obs: usize,
    pub r2: Option<f64>,
    pub ll: Option<f64>,
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Linear,
    Logit,
    Probit,
    Poisson,
    FixedEffects,
    Other,
}

impl Default for ModelType {
    fn default() -> Self { ModelType::Other }
}

lazy_static::lazy_static! {
    static ref EST_STORE: Arc<Mutex<Option<EstimationResults>>> = Arc::new(Mutex::new(None));
}

pub fn store_results(results: EstimationResults) {
    if let Ok(mut guard) = EST_STORE.lock() {
        *guard = Some(results);
    }
}

pub fn get_results() -> Option<EstimationResults> {
    if let Ok(guard) = EST_STORE.lock() {
        guard.clone()
    } else {
        None
    }
}

pub fn clear_results() {
    if let Ok(mut guard) = EST_STORE.lock() {
        *guard = None;
    }
}
