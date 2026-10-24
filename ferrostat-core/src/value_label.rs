use std::collections::BTreeMap;

/// Wrapper for f64 that implements a total order so it can key a map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatKey(pub f64);

impl Eq for FloatKey {}

impl PartialOrd for FloatKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl std::hash::Hash for FloatKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Value label: maps numeric values to string labels, kept in value order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValueLabel {
    pub name: String,
    pub mapping: BTreeMap<FloatKey, String>,
}

impl ValueLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), mapping: BTreeMap::new() }
    }

    pub fn add(&mut self, value: f64, label: impl Into<String>) {
        self.mapping.insert(FloatKey(value), label.into());
    }

    pub fn remove(&mut self, value: f64) {
        self.mapping.remove(&FloatKey(value));
    }

    pub fn lookup(&self, value: f64) -> Option<&str> {
        self.mapping.get(&FloatKey(value)).map(|s| s.as_str())
    }

    /// Reverse lookup: the value carrying `label`.
    pub fn value_of(&self, label: &str) -> Option<f64> {
        self.mapping.iter().find(|(_, l)| l.as_str() == label).map(|(k, _)| k.0)
    }

    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Iterate over (value, label) pairs in ascending value order.
    pub fn iter(&self) -> impl Iterator<Item = (f64, &str)> + '_ {
        self.mapping.iter().map(|(k, v)| (k.0, v.as_str()))
    }
}
