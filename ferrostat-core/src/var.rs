use std::cmp::Ordering;
use std::f64;

/// Ferrostat variable type codes and metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarType {
    /// 1-byte signed integer (-127..100)
    Byte,
    /// 2-byte signed integer (-32,767..32,740)
    Int,
    /// 4-byte signed integer (-2,147,483,647..2,147,483,620)
    Long,
    /// 4-byte IEEE float
    Float,
    /// 8-byte IEEE double
    Double,
    /// Fixed-length string, N = 1..2045 bytes
    Str(usize),
    /// Long string reference (v, o) stored as 8 bytes in data area
    StrL,
}

impl VarType {
    /// Width in bytes for storage in the data matrix.
    pub fn storage_width(&self) -> usize {
        match self {
            VarType::Byte => 1,
            VarType::Int => 2,
            VarType::Long => 4,
            VarType::Float => 4,
            VarType::Double => 8,
            VarType::Str(n) => *n,
            VarType::StrL => 8,
        }
    }

    /// True for numeric types (not strings).
    pub fn is_numeric(&self) -> bool {
        matches!(self, VarType::Byte | VarType::Int | VarType::Long | VarType::Float | VarType::Double)
    }

    /// True for string types.
    pub fn is_string(&self) -> bool {
        matches!(self, VarType::Str(_) | VarType::StrL)
    }

    /// Ferrostat type code as used in DTA format 118+.
    pub fn type_code(&self) -> u16 {
        match self {
            VarType::Byte => 65530,
            VarType::Int => 65529,
            VarType::Long => 65528,
            VarType::Float => 65527,
            VarType::Double => 65526,
            VarType::Str(n) => *n as u16,
            VarType::StrL => 32768,
        }
    }

    /// Parse from a Ferrostat type string like "byte", "str18", "float".
    pub fn from_type_str(s: &str) -> Option<Self> {
        match s {
            "byte" => Some(VarType::Byte),
            "int" => Some(VarType::Int),
            "long" => Some(VarType::Long),
            "float" => Some(VarType::Float),
            "double" => Some(VarType::Double),
            "strL" => Some(VarType::StrL),
            _ if s.starts_with("str") => {
                s[3..].parse::<usize>().ok().and_then(|n| {
                    if n >= 1 && n <= 2045 { Some(VarType::Str(n)) } else { None }
                })
            }
            _ => None,
        }
    }

    /// Type string as Ferrostat displays it.
    pub fn type_str(&self) -> String {
        match self {
            VarType::Byte => "byte".to_string(),
            VarType::Int => "int".to_string(),
            VarType::Long => "long".to_string(),
            VarType::Float => "float".to_string(),
            VarType::Double => "double".to_string(),
            VarType::Str(n) => format!("str{n}"),
            VarType::StrL => "strL".to_string(),
        }
    }

    /// Convert from DTA type code.
    pub fn from_type_code(code: u16) -> Option<Self> {
        match code {
            65530 => Some(VarType::Byte),
            65529 => Some(VarType::Int),
            65528 => Some(VarType::Long),
            65527 => Some(VarType::Float),
            65526 => Some(VarType::Double),
            32768 => Some(VarType::StrL),
            n if n >= 1 && n <= 2045 => Some(VarType::Str(n as usize)),
            _ => None,
        }
    }
}

impl std::fmt::Display for VarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.type_str())
    }
}

/// A single cell value in a Ferrostat dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Missing,
    Byte(i8),
    Int(i16),
    Long(i32),
    Float(f32),
    Double(f64),
    Str(String),
    StrL(u32, u32),     // (v, o) reference
}

impl Value {
    pub fn is_missing(&self) -> bool {
        match self {
            Value::Missing => true,
            Value::Float(v) => crate::missing::is_missing(*v as f64),
            Value::Double(v) => crate::missing::is_missing(*v),
            _ => false,
        }
    }

    /// Numeric value with missing normalised to the in-memory missing codes.
    /// Strings give system missing.
    pub fn num(&self) -> f64 {
        match self {
            Value::Byte(v) => *v as f64,
            Value::Int(v) => *v as f64,
            Value::Long(v) => *v as f64,
            Value::Float(v) => {
                if v.is_nan() || *v as f64 >= 1.7014118346046923e38 { crate::missing::SYSMISS } else { *v as f64 }
            }
            Value::Double(v) => crate::missing::normalize(*v),
            _ => crate::missing::SYSMISS,
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::Str(_) | Value::StrL(_, _))
    }

    pub fn is_numeric(&self) -> bool {
        !self.is_string()
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Byte(v) => Some(*v as f64),
            Value::Int(v) => Some(*v as f64),
            Value::Long(v) => Some(*v as f64),
            Value::Float(v) => Some(*v as f64),
            Value::Double(v) => Some(*v),
            Value::Missing => Some(crate::missing::SYSMISS),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Str(a), Value::Str(b)) => a.partial_cmp(b),
            (Value::StrL(..), _) | (_, Value::StrL(..)) => None,
            (a, b) if !a.is_string() && !b.is_string() => a.num().partial_cmp(&b.num()),
            _ => None,
        }
    }
}

/// Variable metadata and data column.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub var_type: VarType,
    pub format: String,
    pub label: String,
    pub value_label_name: String,
    pub data: Vec<Value>,
}

impl Variable {
    pub fn new(name: impl Into<String>, var_type: VarType) -> Self {
        Self {
            name: name.into(),
            var_type,
            format: "%9.0g".to_string(),
            label: String::new(),
            value_label_name: String::new(),
            data: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, value: Value) {
        self.data.push(value);
    }

    pub fn push_missing(&mut self) {
        if self.var_type.is_numeric() {
            self.data.push(Value::Double(crate::missing::SYSMISS));
        } else {
            self.data.push(Value::Str(String::new()));
        }
    }

    pub fn remove(&mut self, idx: usize) {
        self.data.remove(idx);
    }

    pub fn get(&self, idx: usize) -> Option<&Value> {
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Value> {
        self.data.get_mut(idx)
    }

    pub fn set(&mut self, idx: usize, value: Value) {
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }

    /// Numeric values as f64 (missing values keep their missing codes).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        if self.var_type.is_string() {
            return None;
        }
        Some(self.data.iter().map(|v| v.num()).collect())
    }

    /// Numeric value of one observation.
    pub fn num(&self, i: usize) -> f64 {
        self.data.get(i).map_or(crate::missing::SYSMISS, |v| v.num())
    }

    /// String value of one observation ("" for numeric/out of range).
    pub fn str_at(&self, i: usize) -> &str {
        match self.data.get(i) {
            Some(Value::Str(s)) => s,
            _ => "",
        }
    }
}
