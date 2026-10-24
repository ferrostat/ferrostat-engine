use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::dataset::DataSet;
use crate::missing;
use crate::value_label::ValueLabel;
use crate::var::{Value, VarType, Variable};

/// Error type for DTA I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum DtaError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid DTA file: {0}")]
    InvalidFormat(String),
    #[error("Unsupported DTA release: {0}")]
    UnsupportedRelease(u16),
    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

/// DTA file reader that supports formats 117, 118, and 119.
pub struct DtaReader {
    data: Vec<u8>,
    release: u16,
    byteorder: ByteOrder,
    k: u16,          // number of variables
    n: u64,          // number of observations
    label: String,
    timestamp: String,
    map_offsets: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ByteOrder {
    Lsf, // little-endian
    Msf, // big-endian
}

impl ByteOrder {
    fn read_u16(&self, buf: &[u8]) -> u16 {
        match self {
            ByteOrder::Lsf => u16::from_le_bytes([buf[0], buf[1]]),
            ByteOrder::Msf => u16::from_be_bytes([buf[0], buf[1]]),
        }
    }

    fn read_u32(&self, buf: &[u8]) -> u32 {
        match self {
            ByteOrder::Lsf => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            ByteOrder::Msf => u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
        }
    }

    fn read_u64(&self, buf: &[u8]) -> u64 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(buf);
        match self {
            ByteOrder::Lsf => u64::from_le_bytes(arr),
            ByteOrder::Msf => u64::from_be_bytes(arr),
        }
    }

    fn read_i8(&self, buf: &[u8]) -> i8 {
        buf[0] as i8
    }

    fn read_i16(&self, buf: &[u8]) -> i16 {
        match self {
            ByteOrder::Lsf => i16::from_le_bytes([buf[0], buf[1]]),
            ByteOrder::Msf => i16::from_be_bytes([buf[0], buf[1]]),
        }
    }

    fn read_i32(&self, buf: &[u8]) -> i32 {
        match self {
            ByteOrder::Lsf => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            ByteOrder::Msf => i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
        }
    }

    fn read_f32(&self, buf: &[u8]) -> f32 {
        let bits = self.read_u32(buf);
        f32::from_bits(bits)
    }

    fn read_f64(&self, buf: &[u8]) -> f64 {
        let bits = self.read_u64(buf);
        f64::from_bits(bits)
    }
}

impl DtaReader {
    /// Read a .dta file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<DataSet, DtaError> {
        let mut f = File::open(path)?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;
        Self::from_bytes(data)
    }

    /// Parse a .dta from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<DataSet, DtaError> {
        let mut reader = DtaReader {
            data,
            release: 0,
            byteorder: ByteOrder::Lsf,
            k: 0,
            n: 0,
            label: String::new(),
            timestamp: String::new(),
            map_offsets: Vec::new(),
        };
        reader.parse_header()?;
        reader.read_map()?;
        reader.parse_data()
    }

    fn parse_header(&mut self) -> Result<(), DtaError> {
        // Ensure it starts with <stata_dta>
        if !self.data.starts_with(b"<stata_dta>") {
            return Err(DtaError::InvalidFormat("Missing <stata_dta> tag".into()));
        }

        // Find header bounds
        let header_start = Self::find_tag(&self.data, b"<header>").ok_or_else(||
            DtaError::InvalidFormat("Missing <header> tag".into()))?;
        let header_end = Self::find_tag(&self.data, b"</header>").ok_or_else(||
            DtaError::InvalidFormat("Missing </header> tag".into()))?;
        let header = &self.data[header_start..header_end + 9];

        // Release
        let rel_start = Self::find_tag(header, b"<release>").ok_or_else(||
            DtaError::InvalidFormat("Missing <release>".into()))?;
        let rel_end = Self::find_tag(header, b"</release>").ok_or_else(||
            DtaError::InvalidFormat("Missing </release>".into()))?;
        let rel_str = std::str::from_utf8(&header[rel_start + 9..rel_end])?;
        self.release = rel_str.parse().map_err(|_|
            DtaError::InvalidFormat(format!("Invalid release: {}", rel_str)))?;

        if self.release != 117 && self.release != 118 && self.release != 119 {
            return Err(DtaError::UnsupportedRelease(self.release));
        }

        // Byte order
        let bo_start = Self::find_tag(header, b"<byteorder>").ok_or_else(||
            DtaError::InvalidFormat("Missing <byteorder>".into()))?;
        let bo_end = Self::find_tag(header, b"</byteorder>").ok_or_else(||
            DtaError::InvalidFormat("Missing </byteorder>".into()))?;
        let bo_str = std::str::from_utf8(&header[bo_start + 11..bo_end])?;
        self.byteorder = match bo_str {
            "LSF" => ByteOrder::Lsf,
            "MSF" => ByteOrder::Msf,
            _ => return Err(DtaError::InvalidFormat(format!("Unknown byteorder: {}", bo_str))),
        };

        // K (number of variables) - 2 bytes
        let k_pos = Self::find_tag(header, b"<K>").ok_or_else(||
            DtaError::InvalidFormat("Missing <K>".into()))?;
        let k_end = Self::find_tag(header, b"</K>").ok_or_else(||
            DtaError::InvalidFormat("Missing </K>".into()))?;
        let k_bytes = &header[k_pos + 3..k_end];
        if k_bytes.len() != 2 {
            return Err(DtaError::InvalidFormat("Invalid K length".into()));
        }
        self.k = self.byteorder.read_u16(k_bytes);

        // N (number of observations) - 8 bytes
        let n_pos = Self::find_tag(header, b"<N>").ok_or_else(||
            DtaError::InvalidFormat("Missing <N>".into()))?;
        let n_end = Self::find_tag(header, b"</N>").ok_or_else(||
            DtaError::InvalidFormat("Missing </N>".into()))?;
        let n_bytes = &header[n_pos + 3..n_end];
        if n_bytes.len() != 8 {
            return Err(DtaError::InvalidFormat("Invalid N length".into()));
        }
        self.n = self.byteorder.read_u64(n_bytes);

        // Label (2-byte length + text)
        let label_pos = Self::find_tag(header, b"<label>");
        if let Some(lp) = label_pos {
            let label_end = Self::find_tag(header, b"</label>").unwrap_or(lp + 7);
            let label_data = &header[lp + 7..label_end];
            if label_data.len() >= 2 {
                let len = self.byteorder.read_u16(&label_data[..2]) as usize;
                if len > 0 && label_data.len() >= 2 + len {
                    self.label = String::from_utf8_lossy(&label_data[2..2 + len]).into_owned();
                }
            }
        }

        // Timestamp (1-byte length + text)
        let ts_pos = Self::find_tag(header, b"<timestamp>");
        if let Some(tp) = ts_pos {
            let ts_end = Self::find_tag(header, b"</timestamp>").unwrap_or(tp + 11);
            let ts_data = &header[tp + 11..ts_end];
            if !ts_data.is_empty() {
                let len = ts_data[0] as usize;
                if len > 0 && ts_data.len() >= 1 + len {
                    self.timestamp = String::from_utf8_lossy(&ts_data[1..1 + len]).into_owned();
                }
            }
        }

        Ok(())
    }

    fn read_map(&mut self) -> Result<(), DtaError> {
        let map_tag = Self::find_tag(&self.data, b"<map>").ok_or_else(||
            DtaError::InvalidFormat("Missing <map>".into()))?;
        let map_end = Self::find_tag(&self.data, b"</map>").ok_or_else(||
            DtaError::InvalidFormat("Missing </map>".into()))?;
        let map_data = &self.data[map_tag + 5..map_end];

        // Map entries are 8-byte u64 each
        if map_data.len() % 8 != 0 {
            return Err(DtaError::InvalidFormat("Map data not aligned".into()));
        }

        let num_entries = map_data.len() / 8;
        let mut offsets = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            offsets.push(self.byteorder.read_u64(&map_data[i * 8..(i + 1) * 8]));
        }
        self.map_offsets = offsets;
        Ok(())
    }

    fn parse_data(&mut self) -> Result<DataSet, DtaError> {
        let mut ds = DataSet::new();
        ds.datalabel = self.label.clone();
        ds.time_stamp = self.timestamp.clone();

        // Parse variable types
        let var_types = self.read_variable_types()?;

        // Parse variable names
        let varnames = self.read_varnames()?;
        if varnames.len() != self.k as usize {
            return Err(DtaError::InvalidFormat(format!(
                "Expected {} varnames, got {}", self.k, varnames.len()
            )));
        }

        // Parse sort list
        let _sortlist = self.read_sortlist();

        // Parse formats
        let formats = self.read_formats();

        // Parse value label names
        let val_label_names = self.read_value_label_names();

        // Parse variable labels
        let var_labels = self.read_variable_labels();

        // Build variables
        let mut record_size = 0usize;
        for i in 0..self.k as usize {
            let vtype = var_types.get(i).copied().unwrap_or(VarType::Double);
            record_size += vtype.storage_width();
            let mut var = Variable::new(varnames[i].clone(), vtype);
            if let Some(fmt) = formats.get(i) {
                var.format = fmt.clone();
            }
            if let Some(vlab) = val_label_names.get(i) {
                var.value_label_name = vlab.clone();
            }
            if let Some(vlbl) = var_labels.get(i) {
                var.label = vlbl.clone();
            }
            ds.add_variable(var).map_err(|e| DtaError::InvalidFormat(e))?;
        }

        // Parse characteristics
        let _chars = self.read_characteristics();

        // Parse data matrix
        self.read_data_matrix(&mut ds, record_size)?;

        // Parse strls
        let strls = self.read_strls();
        ds.strls = strls;

        // Parse value labels
        let vlabels = self.read_value_labels();
        ds.value_labels = vlabels;

        Ok(ds)
    }

    fn read_variable_types(&self) -> Result<Vec<VarType>, DtaError> {
        let start = Self::find_tag(&self.data, b"<variable_types>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing <variable_types>".into()))?;
        let end = Self::find_tag(&self.data, b"</variable_types>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing </variable_types>".into()))?;
        let data = &self.data[start + 16..end];
        let mut types = Vec::with_capacity(self.k as usize);
        for i in 0..self.k as usize {
            let offset = i * 2;
            if offset + 2 > data.len() {
                return Err(DtaError::InvalidFormat("variable_types truncated".into()));
            }
            let code = self.byteorder.read_u16(&data[offset..offset + 2]);
            match VarType::from_type_code(code) {
                Some(vt) => types.push(vt),
                None => return Err(DtaError::InvalidFormat(format!(
                    "Unknown variable type code: {}", code
                ))),
            }
        }
        Ok(types)
    }

    fn read_varnames(&self) -> Result<Vec<String>, DtaError> {
        let start = Self::find_tag(&self.data, b"<varnames>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing <varnames>".into()))?;
        let end = Self::find_tag(&self.data, b"</varnames>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing </varnames>".into()))?;
        let data = &self.data[start + 10..end];
        let width = data.len() / self.k as usize;
        let mut names = Vec::with_capacity(self.k as usize);
        for i in 0..self.k as usize {
            let chunk = &data[i * width..(i + 1) * width];
            let s = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
            names.push(String::from_utf8_lossy(s).into_owned());
        }
        Ok(names)
    }

    fn read_sortlist(&self) -> Vec<i16> {
        let Some(start) = Self::find_tag(&self.data, b"<sortlist>") else { return vec![] };
        let Some(end) = Self::find_tag(&self.data, b"</sortlist>") else { return vec![] };
        let data = &self.data[start + 10..end];
        let mut list = Vec::new();
        for chunk in data.chunks(2) {
            if chunk.len() == 2 {
                let val = self.byteorder.read_i16(chunk);
                list.push(val);
                if val == 0 { break; }
            }
        }
        list
    }

    fn read_formats(&self) -> Vec<String> {
        let Some(start) = Self::find_tag(&self.data, b"<formats>") else { return vec![] };
        let Some(end) = Self::find_tag(&self.data, b"</formats>") else { return vec![] };
        let data = &self.data[start + 9..end];
        let width = data.len() / self.k as usize;
        (0..self.k as usize)
            .map(|i| {
                let chunk = &data[i * width..(i + 1) * width];
                let s = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
                String::from_utf8_lossy(s).into_owned()
            })
            .collect()
    }

    fn read_value_label_names(&self) -> Vec<String> {
        let Some(start) = Self::find_tag(&self.data, b"<value_label_names>") else { return vec![] };
        let Some(end) = Self::find_tag(&self.data, b"</value_label_names>") else { return vec![] };
        let data = &self.data[start + 19..end];
        let width = data.len() / self.k as usize;
        (0..self.k as usize)
            .map(|i| {
                let chunk = &data[i * width..(i + 1) * width];
                let s = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
                String::from_utf8_lossy(s).into_owned()
            })
            .collect()
    }

    fn read_variable_labels(&self) -> Vec<String> {
        let Some(start) = Self::find_tag(&self.data, b"<variable_labels>") else { return vec![] };
        let Some(end) = Self::find_tag(&self.data, b"</variable_labels>") else { return vec![] };
        let data = &self.data[start + 17..end];
        let width = data.len() / self.k as usize;
        (0..self.k as usize)
            .map(|i| {
                let chunk = &data[i * width..(i + 1) * width];
                let s = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
                String::from_utf8_lossy(s).into_owned()
            })
            .collect()
    }

    fn read_characteristics(&self) -> HashMap<String, HashMap<String, String>> {
        let Some(start) = Self::find_tag(&self.data, b"<characteristics>") else {
            return HashMap::new();
        };
        let Some(end) = Self::find_tag(&self.data, b"</characteristics>") else {
            return HashMap::new();
        };
        let data = &self.data[start + 17..end];
        let mut chars: HashMap<String, HashMap<String, String>> = HashMap::new();

        // Parse <ch> name_len name char_len char </ch> entries
        let mut pos = 0usize;
        while pos + 6 < data.len() {
            if &data[pos..pos + 4] != b"<ch>" {
                pos += 1;
                continue;
            }
            pos += 4;
            if pos + 1 >= data.len() { break; }
            let name_len = self.byteorder.read_u16(&data[pos..pos + 2]) as usize;
            pos += 2;
            if pos + name_len + 4 > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).into_owned();
            pos += name_len;

            // Read element name
            let el_len = self.byteorder.read_u16(&data[pos..pos + 2]) as usize;
            pos += 2;
            if pos + el_len + 4 > data.len() { break; }
            let el = String::from_utf8_lossy(&data[pos..pos + el_len]).into_owned();
            pos += el_len;

            // Read char value
            let val_len = self.byteorder.read_u32(&data[pos..pos + 4]) as usize;
            pos += 4;
            if pos + val_len > data.len() { break; }
            let val = String::from_utf8_lossy(&data[pos..pos + val_len]).into_owned();
            pos += val_len;

            // Skip </ch>
            if pos + 5 <= data.len() && &data[pos..pos + 5] == b"</ch>" {
                pos += 5;
            }

            chars.entry(name).or_insert_with(HashMap::new).insert(el, val);
        }
        chars
    }

    fn read_data_matrix(&self, ds: &mut DataSet, record_size: usize) -> Result<(), DtaError> {
        let start = Self::find_tag(&self.data, b"<data>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing <data>".into()))?;
        let end = Self::find_tag(&self.data, b"</data>")
            .ok_or_else(|| DtaError::InvalidFormat("Missing </data>".into()))?;
        let data = &self.data[start + 6..end];

        let expected_size = self.n as usize * record_size;
        if data.len() != expected_size {
            // Some formats may have padding; allow reading the minimum
            if data.len() < expected_size {
                return Err(DtaError::InvalidFormat(format!(
                    "Data size mismatch: expected {}, got {}", expected_size, data.len()
                )));
            }
        }

        // Pre-allocate
        for var in ds.vars.iter_mut() {
            var.data.reserve(self.n as usize);
        }

        for obs_idx in 0..self.n as usize {
            let offset = obs_idx * record_size;
            if offset + record_size > data.len() {
                break;
            }
            let mut cursor = offset;
            for var in ds.vars.iter_mut() {
                let width = var.var_type.storage_width();
                let chunk = &data[cursor..cursor + width];
                let value = match var.var_type {
                    VarType::Byte => {
                        let raw = chunk[0] as i8;
                        if missing::is_missing_byte(raw) {
                            Value::Double(f64::from_bits(0x7FF8000000000000))
                        } else {
                            Value::Byte(raw)
                        }
                    }
                    VarType::Int => {
                        let raw = self.byteorder.read_i16(chunk);
                        if missing::is_missing_int(raw) {
                            Value::Double(f64::from_bits(0x7FF8000000000000))
                        } else {
                            Value::Int(raw)
                        }
                    }
                    VarType::Long => {
                        let raw = self.byteorder.read_i32(chunk);
                        if missing::is_missing_long(raw) {
                            Value::Double(f64::from_bits(0x7FF8000000000000))
                        } else {
                            Value::Long(raw)
                        }
                    }
                    VarType::Float => {
                        let raw = self.byteorder.read_f32(chunk);
                        if missing::is_missing_float(raw) {
                            Value::Double(f64::from_bits(0x7FF8000000000000))
                        } else {
                            Value::Float(raw)
                        }
                    }
                    VarType::Double => {
                        let raw = self.byteorder.read_f64(chunk);
                        if missing::is_missing_double(raw) {
                            Value::Double(f64::from_bits(0x7FF8000000000000))
                        } else {
                            Value::Double(raw)
                        }
                    }
                    VarType::Str(_n) => {
                        let s = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
                        Value::Str(String::from_utf8_lossy(s).into_owned())
                    }
                    VarType::StrL => {
                        let v = self.byteorder.read_u32(&chunk[0..4]);
                        let o = self.byteorder.read_u32(&chunk[4..8]);
                        Value::StrL(v, o)
                    }
                };
                var.data.push(value);
                cursor += width;
            }
        }

        ds.n_obs = self.n as usize;
        Ok(())
    }

    fn read_strls(&self) -> HashMap<(u32, u32), String> {
        let Some(start) = Self::find_tag(&self.data, b"<strls>") else {
            return HashMap::new();
        };
        let Some(end) = Self::find_tag(&self.data, b"</strls>") else {
            return HashMap::new();
        };
        let data = &self.data[start + 7..end];
        let mut result = HashMap::new();
        let mut pos = 0usize;
        while pos + 12 < data.len() {
            // Each strL entry: v(4) o(4) len(4) text(len)
            if pos + 12 > data.len() { break; }
            let v = self.byteorder.read_u32(&data[pos..pos + 4]);
            let o = self.byteorder.read_u32(&data[pos + 4..pos + 8]);
            let len = self.byteorder.read_u32(&data[pos + 8..pos + 12]) as usize;
            pos += 12;
            if pos + len > data.len() { break; }
            let text = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
            pos += len;
            result.insert((v, o), text);
        }
        result
    }

    fn read_value_labels(&self) -> HashMap<String, ValueLabel> {
        let Some(start) = Self::find_tag(&self.data, b"<value_labels>") else {
            return HashMap::new();
        };
        let Some(end) = Self::find_tag(&self.data, b"</value_labels>") else {
            return HashMap::new();
        };
        let data = &self.data[start + 14..end];
        let mut result = HashMap::new();
        let mut pos = 0usize;
        while pos + 8 < data.len() {
            if &data[pos..pos + 5] != b"<lbl>" { pos += 1; continue; }
            pos += 5;
            if pos + 8 > data.len() { break; }
            let len = self.byteorder.read_u32(&data[pos..pos + 4]) as usize;
            let txtlen = self.byteorder.read_u32(&data[pos + 4..pos + 8]) as usize;
            pos += 8;
            if pos + len + txtlen > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
            pos += len;
            let txtdata = &data[pos..pos + txtlen];
            pos += txtlen;
            if pos + 6 <= data.len() && &data[pos..pos + 6] == b"</lbl>" {
                pos += 6;
            }

            let mut vlab = ValueLabel::new(&name);
            // Parse label table: off(n) + val(n) + txt(txtlen)
            // The txtdata contains: n(4) txtlen(4) off[] val[] text[]
            if txtdata.len() < 8 { continue; }
            let n = self.byteorder.read_u32(&txtdata[0..4]) as usize;
            let stored_txtlen = self.byteorder.read_u32(&txtdata[4..8]) as usize;
            let table_offset = 8;
            let off_size = n * 4;
            let val_size = n * 8;
            if txtdata.len() < table_offset + off_size + val_size + stored_txtlen { continue; }

            let mut offsets = Vec::with_capacity(n);
            for i in 0..n {
                let off = self.byteorder.read_u32(&txtdata[table_offset + i * 4..table_offset + (i + 1) * 4]) as usize;
                offsets.push(off);
            }
            let val_offset = table_offset + off_size;
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let val = self.byteorder.read_f64(&txtdata[val_offset + i * 8..val_offset + (i + 1) * 8]);
                values.push(val);
            }
            let text_offset = val_offset + val_size;
            for i in 0..n {
                let start = offsets.get(i).copied().unwrap_or(0);
                let end = offsets.get(i + 1).copied().unwrap_or(stored_txtlen);
                let label = String::from_utf8_lossy(&txtdata[text_offset + start..text_offset + end])
                    .trim_end_matches('\0')
                    .to_string();
                if i < values.len() {
                    vlab.add(values[i], label);
                }
            }
            result.insert(name, vlab);
        }
        result
    }

    fn find_tag(data: &[u8], tag: &[u8]) -> Option<usize> {
        data.windows(tag.len()).position(|w| w == tag)
    }
}

/// DTA writer (basic support for format 118).
pub struct DtaWriter;

impl DtaWriter {
    pub fn write<W: Write>(writer: &mut W, ds: &DataSet) -> Result<(), DtaError> {
        // For simplicity, write CSV-like intermediate or full DTA
        // Full DTA writing is complex; we'll implement a simplified version.
        let k = ds.n_vars() as u16;
        let n = ds.n_obs as u64;

        // Build sections
        let _sections: Vec<(String, Vec<u8>)> = Vec::new();

        // Header
        let mut header = Vec::new();
        header.extend_from_slice(b"<release>118</release>");
        header.extend_from_slice(b"<byteorder>LSF</byteorder>");
        header.extend_from_slice(b"<K>");
        header.extend_from_slice(&k.to_le_bytes());
        header.extend_from_slice(b"</K>");
        header.extend_from_slice(b"<N>");
        header.extend_from_slice(&n.to_le_bytes());
        header.extend_from_slice(b"</N>");
        let label_bytes = ds.datalabel.as_bytes();
        header.extend_from_slice(b"<label>");
        header.extend_from_slice(&(label_bytes.len() as u16).to_le_bytes());
        header.extend_from_slice(label_bytes);
        header.extend_from_slice(b"</label>");
        let ts_bytes = ds.time_stamp.as_bytes();
        header.extend_from_slice(b"<timestamp>");
        header.push(ts_bytes.len() as u8);
        header.extend_from_slice(ts_bytes);
        header.extend_from_slice(b"</timestamp>");

        // Variable types
        let mut var_types = Vec::new();
        for var in &ds.vars {
            var_types.extend_from_slice(&var.var_type.type_code().to_le_bytes());
        }

        // Varnames: fixed-width 129 bytes, null-terminated
        let mut varnames = Vec::new();
        for var in &ds.vars {
            let mut buf = vec![0u8; 129];
            let b = var.name.as_bytes();
            let len = b.len().min(128);
            buf[..len].copy_from_slice(&b[..len]);
            varnames.extend_from_slice(&buf);
        }

        // Sortlist: (k+1) × 2 bytes
        let mut sortlist = vec![0u8; (k as usize + 1) * 2];
        for (i, name) in ds.sort_by.iter().enumerate() {
            if let Some(&idx) = ds.var_index.get(name) {
                let pos = i * 2;
                sortlist[pos..pos + 2].copy_from_slice(&((idx + 1) as u16).to_le_bytes());
            }
        }

        // Formats: fixed-width 57 bytes, null-terminated
        let mut formats = Vec::new();
        for var in &ds.vars {
            let mut buf = vec![0u8; 57];
            let b = var.format.as_bytes();
            let len = b.len().min(56);
            buf[..len].copy_from_slice(&b[..len]);
            formats.extend_from_slice(&buf);
        }

        // Value label names: fixed-width 129 bytes
        let mut val_label_names = Vec::new();
        for var in &ds.vars {
            let mut buf = vec![0u8; 129];
            let b = var.value_label_name.as_bytes();
            let len = b.len().min(128);
            buf[..len].copy_from_slice(&b[..len]);
            val_label_names.extend_from_slice(&buf);
        }

        // Variable labels: fixed-width 81 bytes
        let mut var_labels = Vec::new();
        for var in &ds.vars {
            let mut buf = vec![0u8; 81];
            let b = var.label.as_bytes();
            let len = b.len().min(80);
            buf[..len].copy_from_slice(&b[..len]);
            var_labels.extend_from_slice(&buf);
        }

        // Characteristics
        let mut chars = Vec::new();
        for (name, map) in &ds.characteristics {
            for (el, val) in map {
                chars.extend_from_slice(b"<ch>");
                chars.extend_from_slice(&(name.len() as u16).to_le_bytes());
                chars.extend_from_slice(name.as_bytes());
                chars.extend_from_slice(&(el.len() as u16).to_le_bytes());
                chars.extend_from_slice(el.as_bytes());
                chars.extend_from_slice(&(val.len() as u32).to_le_bytes());
                chars.extend_from_slice(val.as_bytes());
                chars.extend_from_slice(b"</ch>");
            }
        }

        // Data matrix
        let mut data_mat = Vec::new();
        let _record_size: usize = ds.vars.iter().map(|v| v.var_type.storage_width()).sum();
        for obs in 0..ds.n_obs {
            for var in &ds.vars {
                match (&var.data[obs], var.var_type) {
                    (Value::Byte(v), _) => data_mat.push(*v as u8),
                    (Value::Int(v), _) => data_mat.extend_from_slice(&v.to_le_bytes()),
                    (Value::Long(v), _) => data_mat.extend_from_slice(&v.to_le_bytes()),
                    (Value::Float(v), _) => data_mat.extend_from_slice(&v.to_le_bytes()),
                    (Value::Double(v), _) => data_mat.extend_from_slice(&v.to_bits().to_le_bytes()),
                    (Value::Str(s), VarType::Str(n)) => {
                        let mut buf = vec![0u8; n];
                        let b = s.as_bytes();
                        let len = b.len().min(n);
                        buf[..len].copy_from_slice(&b[..len]);
                        data_mat.extend_from_slice(&buf);
                    }
                    (Value::StrL(v, o), VarType::StrL) => {
                        data_mat.extend_from_slice(&v.to_le_bytes());
                        data_mat.extend_from_slice(&o.to_le_bytes());
                    }
                    _ => {
                        // write zeros for missing/unknown
                        let w = var.var_type.storage_width();
                        data_mat.extend_from_slice(&vec![0u8; w]);
                    }
                }
            }
        }

        // strls
        let mut strls = Vec::new();
        for ((v, o), text) in &ds.strls {
            strls.extend_from_slice(&v.to_le_bytes());
            strls.extend_from_slice(&o.to_le_bytes());
            strls.extend_from_slice(&(text.len() as u32).to_le_bytes());
            strls.extend_from_slice(text.as_bytes());
        }

        // Value labels
        let mut vlabels = Vec::new();
        for (name, vlab) in &ds.value_labels {
            vlabels.extend_from_slice(b"<lbl>");
            let name_bytes = name.as_bytes();
            vlabels.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            // Build label table
            let n = vlab.len() as u32;
            let mut txt = Vec::new();
            let mut offsets = Vec::new();
            let mut values = Vec::new();
            for (val, label) in &vlab.mapping {
                offsets.push(txt.len() as u32);
                values.push(val.0);
                let mut lb = label.as_bytes().to_vec();
                lb.push(0); // null terminate
                txt.extend_from_slice(&lb);
            }
            let txtlen = txt.len() as u32;
            vlabels.extend_from_slice(&(txtlen).to_le_bytes());
            vlabels.extend_from_slice(name_bytes);
            vlabels.extend_from_slice(&n.to_le_bytes());
            vlabels.extend_from_slice(&txtlen.to_le_bytes());
            for off in offsets {
                vlabels.extend_from_slice(&off.to_le_bytes());
            }
            for val in values {
                vlabels.extend_from_slice(&val.to_bits().to_le_bytes());
            }
            vlabels.extend_from_slice(&txt);
            vlabels.extend_from_slice(b"</lbl>");
        }

        // Now assemble the file and compute offsets for the map
        let mut file = Vec::new();
        file.extend_from_slice(b"<stata_dta>");
        file.extend_from_slice(b"<header>");
        file.extend_from_slice(&header);
        file.extend_from_slice(b"</header>");

        let map_pos = file.len();
        file.extend_from_slice(b"<map>");
        // We'll fill in the offsets after computing them
        let map_data_start = file.len();
        let mut map_entries = vec![0u64; 14];
        map_entries[0] = 0; // start of file (already at 0)
        map_entries[1] = map_pos as u64;

        // Placeholder for map entries
        for e in &map_entries {
            file.extend_from_slice(&e.to_le_bytes());
        }
        file.extend_from_slice(b"</map>");

        map_entries[2] = file.len() as u64;
        file.extend_from_slice(b"<variable_types>");
        file.extend_from_slice(&var_types);
        file.extend_from_slice(b"</variable_types>");

        map_entries[3] = file.len() as u64;
        file.extend_from_slice(b"<varnames>");
        file.extend_from_slice(&varnames);
        file.extend_from_slice(b"</varnames>");

        map_entries[4] = file.len() as u64;
        file.extend_from_slice(b"<sortlist>");
        file.extend_from_slice(&sortlist);
        file.extend_from_slice(b"</sortlist>");

        map_entries[5] = file.len() as u64;
        file.extend_from_slice(b"<formats>");
        file.extend_from_slice(&formats);
        file.extend_from_slice(b"</formats>");

        map_entries[6] = file.len() as u64;
        file.extend_from_slice(b"<value_label_names>");
        file.extend_from_slice(&val_label_names);
        file.extend_from_slice(b"</value_label_names>");

        map_entries[7] = file.len() as u64;
        file.extend_from_slice(b"<variable_labels>");
        file.extend_from_slice(&var_labels);
        file.extend_from_slice(b"</variable_labels>");

        map_entries[8] = file.len() as u64;
        file.extend_from_slice(b"<characteristics>");
        file.extend_from_slice(&chars);
        file.extend_from_slice(b"</characteristics>");

        map_entries[9] = file.len() as u64;
        file.extend_from_slice(b"<data>");
        file.extend_from_slice(&data_mat);
        file.extend_from_slice(b"</data>");

        map_entries[10] = file.len() as u64;
        file.extend_from_slice(b"<strls>");
        file.extend_from_slice(&strls);
        file.extend_from_slice(b"</strls>");

        map_entries[11] = file.len() as u64;
        file.extend_from_slice(b"<value_labels>");
        file.extend_from_slice(&vlabels);
        file.extend_from_slice(b"</value_labels>");

        map_entries[12] = file.len() as u64;
        file.extend_from_slice(b"</stata_dta>");
        map_entries[13] = file.len() as u64;

        // Fix up map entries
        for (i, e) in map_entries.iter().enumerate() {
            let pos = map_data_start + i * 8;
            file[pos..pos + 8].copy_from_slice(&e.to_le_bytes());
        }

        writer.write_all(&file)?;
        Ok(())
    }
}
