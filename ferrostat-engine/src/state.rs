use ferrostat_core::DataSet;

/// Persistent state across commands (results, scalars, matrices, etc.)
#[derive(Debug, Clone, Default)]
pub struct State {
    pub dataset: DataSet,
    pub scalars: Vec<(String, f64)>,
    pub matrices: Vec<(String, Vec<Vec<f64>>)>,
    pub r_results: Vec<(String, String)>,
    pub e_results: Vec<(String, String)>,
    pub s_results: Vec<(String, String)>,
}
