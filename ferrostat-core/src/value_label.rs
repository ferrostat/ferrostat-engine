use std::collections::HashMap;

/// Wrapper for f64 that implements Eq and Hash using bit representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatKey(pub f64);

impl Eq for FloatKey {}

impl std::hash::Hash for FloatKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Ferrostat value label: maps numeric values to string labels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValueLabel {
    pub name: String,
    pub mapping: HashMap<FloatKey, String>,
}

impl ValueLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mapping: HashMap::new(),
        }
    }

    pub fn add(&mut self, value: f64, label: impl Into<String>) {
        self.mapping.insert(FloatKey(value), label.into());
    }

    pub fn lookup(&self, value: f64) -> Option<&str> {
        self.mapping.get(&FloatKey(value)).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Iterate over (value, label) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (f64, &str)> + '_ {
        self.mapping.iter().map(|(k, v)| (k.0, v.as_str()))
    }
}
