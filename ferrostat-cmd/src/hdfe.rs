//! High-dimensional fixed effects: group indexes, singleton detection,
//! partialling out several sets of fixed effects (symmetric Kaczmarz sweeps
//! accelerated by conjugate gradient), redundant-level counting and recovery
//! of the fixed-effect estimates.
//!
//! Everything works on contiguous `f64` columns over the estimation sample;
//! group indexes and per-group normalisers are computed once, and the
//! iteration reuses preallocated work vectors.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// ---------------------------------------------------------------------------
// Fast hashing for group keys
// ---------------------------------------------------------------------------

/// Fast hasher for integer keys (group codes, f64 bit patterns).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        // final avalanche: keys such as f64 bit patterns of integers have
        // all-zero low bits, which the table uses to pick buckets
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Dense group codes (0..G) for numeric keys, in order of first appearance.
pub fn codes_f64(keys: &[f64]) -> (Vec<u32>, usize) {
    let n = keys.len();
    // integer keys in a compact range: direct addressing
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut ints = true;
    for &k in keys {
        if k.fract() != 0.0 || !k.is_finite() {
            ints = false;
            break;
        }
        lo = lo.min(k);
        hi = hi.max(k);
    }
    if ints && n > 0 && hi - lo < (4 * n + 1_000_000) as f64 {
        let span = (hi - lo) as usize + 1;
        let mut slot = vec![u32::MAX; span];
        let mut next = 0u32;
        let codes = keys
            .iter()
            .map(|&k| {
                let s = &mut slot[(k - lo) as usize];
                if *s == u32::MAX {
                    *s = next;
                    next += 1;
                }
                *s
            })
            .collect();
        return (codes, next as usize);
    }
    let mut map: FxMap<u64, u32> = FxMap::default();
    let codes = keys
        .iter()
        .map(|&k| {
            let k = if k == 0.0 { 0.0f64 } else { k };
            let len = map.len() as u32;
            *map.entry(k.to_bits()).or_insert(len)
        })
        .collect();
    (codes, map.len())
}

/// Dense group codes for string keys.
pub fn codes_str<S: AsRef<str>>(keys: &[S]) -> (Vec<u32>, usize) {
    let mut map: HashMap<&str, u32> = HashMap::new();
    let codes = keys
        .iter()
        .map(|k| {
            let len = map.len() as u32;
            *map.entry(k.as_ref()).or_insert(len)
        })
        .collect();
    (codes, map.len())
}

/// Codes of the interaction of two groupings.
pub fn combine(a: &[u32], ga: usize, b: &[u32], gb: usize) -> (Vec<u32>, usize) {
    let n = a.len();
    if (ga as u128) * (gb as u128) <= (4 * n + 1_000_000) as u128 {
        let mut slot = vec![u32::MAX; ga * gb];
        let mut next = 0u32;
        let codes = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let s = &mut slot[x as usize * gb + y as usize];
                if *s == u32::MAX {
                    *s = next;
                    next += 1;
                }
                *s
            })
            .collect();
        return (codes, next as usize);
    }
    let mut map: FxMap<u64, u32> = FxMap::default();
    let codes = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let len = map.len() as u32;
            *map.entry(((x as u64) << 32) | y as u64).or_insert(len)
        })
        .collect();
    (codes, map.len())
}

/// Renumber codes after dropping observations (keeps first-appearance order).
pub fn recode(codes: &[u32], g: usize) -> (Vec<u32>, usize) {
    let mut slot = vec![u32::MAX; g];
    let mut next = 0u32;
    let out = codes
        .iter()
        .map(|&c| {
            let s = &mut slot[c as usize];
            if *s == u32::MAX {
                *s = next;
                next += 1;
            }
            *s
        })
        .collect();
    (out, next as usize)
}

// ---------------------------------------------------------------------------
// Fixed-effect factors
// ---------------------------------------------------------------------------

/// One absorbed term: group indicators (`intercept`), group-specific slopes
/// on the `slopes` variables, or both (`id##c.x`).
#[derive(Clone, Debug)]
pub struct Factor {
    /// Display name of the grouping (`id`, `id#year`).
    pub label: String,
    /// Names of the slope variables (without `c.`).
    pub slope_names: Vec<String>,
    pub intercept: bool,
    /// Variable name(s) for the saved estimates (`name=` or `savefe`).
    pub save_as: Option<String>,
    pub ids: Vec<u32>,
    pub ng: usize,
    /// Slope variables over the sample.
    pub slopes: Vec<Vec<f64>>,
    /// Intercept-only: 1 / (sum of weights) per group.
    inv_w: Vec<f64>,
    /// General: per-group inverse of Z'WZ (p x p, row-major).
    ginv: Vec<f64>,
}

impl Factor {
    pub fn new(label: String, ids: Vec<u32>, ng: usize, intercept: bool, slopes: Vec<Vec<f64>>, slope_names: Vec<String>) -> Self {
        Factor { label, slope_names, intercept, save_as: None, ids, ng, slopes, inv_w: Vec::new(), ginv: Vec::new() }
    }

    /// Coefficients per group.
    pub fn p(&self) -> usize {
        self.intercept as usize + self.slopes.len()
    }

    fn simple(&self) -> bool {
        self.slopes.is_empty()
    }

    /// Keep only the observations flagged in `keep` and renumber groups.
    pub fn subset(&mut self, keep: &[bool]) {
        let ids: Vec<u32> = self.ids.iter().zip(keep).filter(|(_, k)| **k).map(|(i, _)| *i).collect();
        let (ids, ng) = recode(&ids, self.ng);
        self.ids = ids;
        self.ng = ng;
        for s in &mut self.slopes {
            *s = s.iter().zip(keep).filter(|(_, k)| **k).map(|(v, _)| *v).collect();
        }
    }

    /// Precompute the per-group normalisers for weights `w` (None = unit).
    pub fn prepare(&mut self, w: Option<&[f64]>) {
        let ng = self.ng;
        if self.simple() {
            let mut sw = vec![0.0; ng];
            match w {
                Some(w) => {
                    for (&g, &wi) in self.ids.iter().zip(w) {
                        sw[g as usize] += wi;
                    }
                }
                None => {
                    for &g in &self.ids {
                        sw[g as usize] += 1.0;
                    }
                }
            }
            self.inv_w = sw.into_iter().map(|s| if s > 0.0 { 1.0 / s } else { 0.0 }).collect();
            return;
        }
        let p = self.p();
        let mut gm = vec![0.0; ng * p * p];
        let mut z = vec![0.0; p];
        for i in 0..self.ids.len() {
            self.z_row(i, &mut z);
            let wi = w.map_or(1.0, |w| w[i]);
            let base = self.ids[i] as usize * p * p;
            for a in 0..p {
                for b in 0..p {
                    gm[base + a * p + b] += wi * z[a] * z[b];
                }
            }
        }
        for g in 0..ng {
            let blk = &mut gm[g * p * p..(g + 1) * p * p];
            sym_pinv(blk, p);
        }
        self.ginv = gm;
    }

    #[inline]
    fn z_row(&self, i: usize, z: &mut [f64]) {
        let mut a = 0;
        if self.intercept {
            z[0] = 1.0;
            a = 1;
        }
        for s in &self.slopes {
            z[a] = s[i];
            a += 1;
        }
    }

    /// Group coefficients of the weighted projection of `v` on this factor.
    pub fn coefs(&self, v: &[f64], w: Option<&[f64]>, out: &mut Vec<f64>) {
        let p = self.p();
        out.clear();
        out.resize(self.ng * p, 0.0);
        if self.simple() {
            match w {
                Some(w) => {
                    for ((&g, &x), &wi) in self.ids.iter().zip(v).zip(w) {
                        out[g as usize] += wi * x;
                    }
                }
                None => {
                    for (&g, &x) in self.ids.iter().zip(v) {
                        out[g as usize] += x;
                    }
                }
            }
            for (o, &iw) in out.iter_mut().zip(&self.inv_w) {
                *o *= iw;
            }
            return;
        }
        let mut z = vec![0.0; p];
        for i in 0..v.len() {
            self.z_row(i, &mut z);
            let f = w.map_or(1.0, |w| w[i]) * v[i];
            let base = self.ids[i] as usize * p;
            for a in 0..p {
                out[base + a] += z[a] * f;
            }
        }
        let mut tmp = vec![0.0; p];
        for g in 0..self.ng {
            let gi = &self.ginv[g * p * p..(g + 1) * p * p];
            let s = &mut out[g * p..(g + 1) * p];
            for a in 0..p {
                tmp[a] = (0..p).map(|b| gi[a * p + b] * s[b]).sum();
            }
            s.copy_from_slice(&tmp);
        }
    }

    /// Fitted value of observation `i` given group coefficients.
    #[inline]
    pub fn fitted(&self, coefs: &[f64], i: usize) -> f64 {
        let p = self.p();
        let g = self.ids[i] as usize;
        if self.simple() {
            return coefs[g];
        }
        let c = &coefs[g * p..(g + 1) * p];
        let mut a = 0;
        let mut s = 0.0;
        if self.intercept {
            s += c[0];
            a = 1;
        }
        for sl in &self.slopes {
            s += c[a] * sl[i];
            a += 1;
        }
        s
    }

    /// v <- v - P v (weighted projection on this factor's span).
    #[inline]
    fn annihilate(&self, v: &mut [f64], w: Option<&[f64]>, buf: &mut Vec<f64>) {
        self.coefs(v, w, buf);
        if self.simple() {
            for (x, &g) in v.iter_mut().zip(&self.ids) {
                *x -= buf[g as usize];
            }
        } else {
            for i in 0..v.len() {
                v[i] -= self.fitted(buf, i);
            }
        }
    }
}

/// In-place pseudo-inverse of a small symmetric PSD matrix (sweep with
/// tolerance; dropped pivots get zero rows and columns).
fn sym_pinv(a: &mut [f64], p: usize) {
    let diag0: Vec<f64> = (0..p).map(|i| a[i * p + i].abs()).collect();
    let mut swept = vec![false; p];
    for k in 0..p {
        let d = a[k * p + k];
        if d.abs() <= 1e-10 * diag0[k].max(1e-300) || d.abs() < 1e-300 {
            continue;
        }
        for i in 0..p {
            if i == k {
                continue;
            }
            for j in 0..p {
                if j != k {
                    a[i * p + j] -= a[i * p + k] * a[k * p + j] / d;
                }
            }
        }
        for i in 0..p {
            if i != k {
                a[i * p + k] /= d;
                a[k * p + i] /= d;
            }
        }
        a[k * p + k] = -1.0 / d;
        swept[k] = true;
    }
    for i in 0..p {
        for j in 0..p {
            a[i * p + j] = if swept[i] && swept[j] { -a[i * p + j] } else { 0.0 };
        }
    }
}

// ---------------------------------------------------------------------------
// Singletons
// ---------------------------------------------------------------------------

/// Iteratively flag observations that are alone in a group of any factor.
/// Returns the keep mask and the number dropped.
pub fn singletons(factors: &[Factor], n: usize) -> (Vec<bool>, usize) {
    let mut keep = vec![true; n];
    let mut counts: Vec<Vec<u32>> = factors.iter().map(|f| vec![0u32; f.ng]).collect();
    for (f, c) in factors.iter().zip(counts.iter_mut()) {
        for &g in &f.ids {
            c[g as usize] += 1;
        }
    }
    let mut dropped = 0;
    loop {
        let mut changed = false;
        for k in 0..factors.len() {
            for i in 0..n {
                if keep[i] && counts[k][factors[k].ids[i] as usize] == 1 {
                    keep[i] = false;
                    dropped += 1;
                    changed = true;
                    for (j, f) in factors.iter().enumerate() {
                        counts[j][f.ids[i] as usize] -= 1;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    (keep, dropped)
}

// ---------------------------------------------------------------------------
// Partialling out
// ---------------------------------------------------------------------------

/// Work vectors for one column at a time.
pub struct Workspace {
    r: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    buf: Vec<f64>,
    /// one buffer per factor (group means in the fused sweep)
    means: Vec<Vec<f64>>,
    /// direct two-way solver scratch
    tw: Vec<Vec<f64>>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Workspace { r: vec![0.0; n], p: vec![0.0; n], ap: vec![0.0; n], buf: Vec::new(), means: Vec::new(), tw: Vec::new() }
    }
}

/// Symmetric Kaczmarz sweep: (I-P1)...(I-PK)...(I-P1) v.
fn sweep(factors: &[Factor], v: &mut [f64], w: Option<&[f64]>, buf: &mut Vec<f64>, means: &mut Vec<Vec<f64>>) {
    let k = factors.len();
    if factors.iter().all(|f| f.simple()) {
        return sweep_simple(factors, v, w, means);
    }
    for f in factors {
        f.annihilate(v, w, buf);
    }
    for f in factors[..k - 1].iter().rev() {
        f.annihilate(v, w, buf);
    }
}

/// The sweep for intercept-only factors, fusing the subtraction of one
/// factor's group means with the accumulation of the next factor's sums
/// (2K passes over the data instead of 4K-2).
fn sweep_simple(factors: &[Factor], v: &mut [f64], w: Option<&[f64]>, means: &mut Vec<Vec<f64>>) {
    let k = factors.len();
    if means.len() < k {
        means.resize_with(k, Vec::new);
    }
    let order: Vec<usize> = (0..k).chain((0..k - 1).rev()).collect();
    let first = &factors[order[0]];
    first.coefs(v, w, &mut means[order[0]]);
    for step in 1..order.len() {
        let (pi, ci) = (order[step - 1], order[step]);
        let (prev, cur) = (&factors[pi], &factors[ci]);
        let mut acc = std::mem::take(&mut means[ci]);
        acc.clear();
        acc.resize(cur.ng, 0.0);
        let pm = &means[pi];
        match w {
            Some(w) => {
                for i in 0..v.len() {
                    let x = v[i] - pm[prev.ids[i] as usize];
                    v[i] = x;
                    acc[cur.ids[i] as usize] += w[i] * x;
                }
            }
            None => {
                for i in 0..v.len() {
                    let x = v[i] - pm[prev.ids[i] as usize];
                    v[i] = x;
                    acc[cur.ids[i] as usize] += x;
                }
            }
        }
        for (a, &iw) in acc.iter_mut().zip(&cur.inv_w) {
            *a *= iw;
        }
        means[ci] = acc;
    }
    let li = *order.last().unwrap();
    let (lf, lm) = (&factors[li], &means[li]);
    for (x, &g) in v.iter_mut().zip(&lf.ids) {
        *x -= lm[g as usize];
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        Some(w) => a.iter().zip(b).zip(w).map(|((x, y), wi)| x * y * wi).sum(),
        None => {
            // four accumulators for instruction-level parallelism
            let mut s = [0.0f64; 4];
            let ch = a.len() / 4 * 4;
            for i in (0..ch).step_by(4) {
                s[0] += a[i] * b[i];
                s[1] += a[i + 1] * b[i + 1];
                s[2] += a[i + 2] * b[i + 2];
                s[3] += a[i + 3] * b[i + 3];
            }
            let mut t = s[0] + s[1] + s[2] + s[3];
            for i in ch..a.len() {
                t += a[i] * b[i];
            }
            t
        }
    }
}

/// Outcome of partialling out one column.
#[derive(Clone, Copy, Debug)]
pub struct Convergence {
    pub iterations: usize,
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Direct solver for two sets of fixed effects
// ---------------------------------------------------------------------------

/// Two intercept-only factors, one of them with few groups: the large
/// factor is eliminated and the small dense Schur complement
/// M = diag(n_s) - C' diag(1/n_b) C (C = crosstab of weights) is inverted
/// once. Each column then needs four passes over the data.
pub struct TwoWay {
    big: usize,
    small: usize,
    /// generalized inverse of M (gs x gs, row-major)
    minv: Vec<f64>,
}

const TWOWAY_MAX_SMALL: usize = 400;

impl TwoWay {
    pub fn prepare(factors: &[Factor], w: Option<&[f64]>) -> Option<TwoWay> {
        if factors.len() != 2 || !factors.iter().all(|f| f.simple()) {
            return None;
        }
        let (big, small) = if factors[0].ng >= factors[1].ng { (0, 1) } else { (1, 0) };
        let (fb, fs) = (&factors[big], &factors[small]);
        let gs = fs.ng;
        if gs > TWOWAY_MAX_SMALL || gs == 0 {
            return None;
        }
        let mut m = vec![0.0; gs * gs];
        let mut acc = vec![0.0; gs];
        let mut touched: Vec<u32> = Vec::with_capacity(gs);
        let mut cost = 0usize;
        // add one large-factor group's row of the crosstab to M
        let mut add_row = |g: usize, acc: &mut [f64], touched: &mut Vec<u32>, cost: &mut usize| -> bool {
            *cost += touched.len() * touched.len();
            if *cost > 400_000_000 {
                return false;
            }
            let ib = fb.inv_w[g];
            for &u in touched.iter() {
                let au = acc[u as usize];
                m[u as usize * gs + u as usize] += au;
                let f = au * ib;
                for &v in touched.iter() {
                    m[u as usize * gs + v as usize] -= f * acc[v as usize];
                }
            }
            for &u in touched.iter() {
                acc[u as usize] = 0.0;
            }
            touched.clear();
            true
        };
        if fb.ng * gs <= 16_000_000 {
            // dense crosstab built in one sequential pass
            let mut cells = vec![0.0; fb.ng * gs];
            match w {
                Some(w) => {
                    for ((&a, &b), &wi) in fb.ids.iter().zip(&fs.ids).zip(w) {
                        cells[a as usize * gs + b as usize] += wi;
                    }
                }
                None => {
                    for (&a, &b) in fb.ids.iter().zip(&fs.ids) {
                        cells[a as usize * gs + b as usize] += 1.0;
                    }
                }
            }
            for g in 0..fb.ng {
                for (h, &c) in cells[g * gs..(g + 1) * gs].iter().enumerate() {
                    if c != 0.0 {
                        acc[h] = c;
                        touched.push(h as u32);
                    }
                }
                if !add_row(g, &mut acc, &mut touched, &mut cost) {
                    return None;
                }
            }
        } else {
            // observations grouped by the large factor (counting sort)
            let n = fb.ids.len();
            let mut start = vec![0u32; fb.ng + 1];
            for &g in &fb.ids {
                start[g as usize + 1] += 1;
            }
            for g in 0..fb.ng {
                start[g + 1] += start[g];
            }
            let mut fill = start.clone();
            let mut perm = vec![0u32; n];
            for (i, &g) in fb.ids.iter().enumerate() {
                perm[fill[g as usize] as usize] = i as u32;
                fill[g as usize] += 1;
            }
            drop(fill);
            for g in 0..fb.ng {
                for &i in &perm[start[g] as usize..start[g + 1] as usize] {
                    let i = i as usize;
                    let h = fs.ids[i];
                    if acc[h as usize] == 0.0 {
                        touched.push(h);
                    }
                    acc[h as usize] += w.map_or(1.0, |w| w[i]);
                }
                if !add_row(g, &mut acc, &mut touched, &mut cost) {
                    return None;
                }
            }
        }
        drop(add_row);
        sym_pinv(&mut m, gs);
        Some(TwoWay { big, small, minv: m })
    }

    /// Largest group mean of `x` in either factor (zero once the fixed
    /// effects are fully removed).
    fn check(&self, factors: &[Factor], x: &[f64], w: Option<&[f64]>, scratch: &mut Vec<Vec<f64>>) -> f64 {
        let (fb, fs) = (&factors[self.big], &factors[self.small]);
        if scratch.len() < 4 {
            scratch.resize_with(4, Vec::new);
        }
        let [sb, ss, ..] = &mut scratch[..4] else { unreachable!() };
        sb.clear();
        sb.resize(fb.ng, 0.0);
        ss.clear();
        ss.resize(fs.ng, 0.0);
        for i in 0..x.len() {
            let v = w.map_or(x[i], |w| w[i] * x[i]);
            sb[fb.ids[i] as usize] += v;
            ss[fs.ids[i] as usize] += v;
        }
        let mb = sb.iter().zip(&fb.inv_w).fold(0.0f64, |m, (s, iw)| m.max((s * iw).abs()));
        ss.iter().zip(&fs.inv_w).fold(mb, |m, (s, iw)| m.max((s * iw).abs()))
    }

    /// One direct elimination of both sets of fixed effects from `x`.
    fn apply(&self, factors: &[Factor], x: &mut [f64], w: Option<&[f64]>, scratch: &mut Vec<Vec<f64>>) {
        let (fb, fs) = (&factors[self.big], &factors[self.small]);
        let (gb, gs) = (fb.ng, fs.ng);
        if scratch.len() < 4 {
            scratch.resize_with(4, Vec::new);
        }
        let [sb, ss, rhs, gam] = &mut scratch[..4] else { unreachable!() };
        sb.clear();
        sb.resize(gb, 0.0);
        ss.clear();
        ss.resize(gs, 0.0);
        let (ib, is) = (&fb.ids, &fs.ids);
        match w {
            Some(w) => {
                for i in 0..x.len() {
                    let v = w[i] * x[i];
                    sb[ib[i] as usize] += v;
                    ss[is[i] as usize] += v;
                }
            }
            None => {
                for i in 0..x.len() {
                    sb[ib[i] as usize] += x[i];
                    ss[is[i] as usize] += x[i];
                }
            }
        }
        // a0 = S_b / n_b
        for (s, &iw) in sb.iter_mut().zip(&fb.inv_w) {
            *s *= iw;
        }
        // rhs = S_s - C' a0
        rhs.clear();
        rhs.extend_from_slice(ss);
        match w {
            Some(w) => {
                for i in 0..x.len() {
                    rhs[is[i] as usize] -= w[i] * sb[ib[i] as usize];
                }
            }
            None => {
                for i in 0..x.len() {
                    rhs[is[i] as usize] -= sb[ib[i] as usize];
                }
            }
        }
        gam.clear();
        gam.resize(gs, 0.0);
        for a in 0..gs {
            let row = &self.minv[a * gs..(a + 1) * gs];
            gam[a] = row.iter().zip(rhs.iter()).map(|(m, r)| m * r).sum();
        }
        // alpha = a0 - (C gamma) / n_b, accumulated in ss-sized pass
        let mut cg = std::mem::take(ss);
        cg.clear();
        cg.resize(gb, 0.0);
        match w {
            Some(w) => {
                for i in 0..x.len() {
                    cg[ib[i] as usize] += w[i] * gam[is[i] as usize];
                }
            }
            None => {
                for i in 0..x.len() {
                    cg[ib[i] as usize] += gam[is[i] as usize];
                }
            }
        }
        for g in 0..gb {
            sb[g] -= cg[g] * fb.inv_w[g];
        }
        *ss = cg;
        for i in 0..x.len() {
            x[i] -= sb[ib[i] as usize] + gam[is[i] as usize];
        }
    }
}

/// Replace `x` by its residual from the fixed effects (in place).
pub fn partial_out(factors: &[Factor], x: &mut [f64], w: Option<&[f64]>, tol: f64, maxiter: usize, ws: &mut Workspace) -> Convergence {
    partial_out_with(factors, None, x, w, tol, maxiter, ws)
}

fn partial_out_with(factors: &[Factor], tw: Option<&TwoWay>, x: &mut [f64], w: Option<&[f64]>, tol: f64, maxiter: usize, ws: &mut Workspace) -> Convergence {
    if factors.is_empty() {
        return Convergence { iterations: 0, converged: true };
    }
    if factors.len() == 1 {
        factors[0].annihilate(x, w, &mut ws.buf);
        return Convergence { iterations: 1, converged: true };
    }
    if let Some(tw) = tw {
        // direct solve with iterative refinement
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        for round in 1..=3 {
            tw.apply(factors, x, w, &mut ws.tw);
            if tw.check(factors, x, w, &mut ws.tw) <= tol * scale {
                return Convergence { iterations: round, converged: true };
            }
        }
        // not accurate enough: finish with the iterative method
    }
    let n = x.len();
    let Workspace { r, p, ap, buf, means, .. } = ws;
    let (r, p, ap) = (&mut r[..n], &mut p[..n], &mut ap[..n]);
    // conjugate gradient on (I - T) x = 0 started at x0 = y: the iterates
    // stay in y + span(FE) and converge to the annihilated vector
    r.copy_from_slice(x);
    sweep(factors, r, w, buf, means);
    for i in 0..n {
        r[i] -= x[i];
    }
    p.copy_from_slice(r);
    let mut rr = dot(r, r, w);
    let rr0 = rr;
    let mut potential = dot(x, x, w);
    if rr <= 1e-30 * potential.max(1e-300) || potential == 0.0 {
        return Convergence { iterations: 1, converged: true };
    }
    let mut recent = [0.0f64; 2];
    for iter in 1..=maxiter {
        ap.copy_from_slice(p);
        sweep(factors, ap, w, buf, means);
        let mut pap = 0.0;
        for i in 0..n {
            let a = p[i] - ap[i];
            ap[i] = a;
            pap += w.map_or(1.0, |w| w[i]) * p[i] * a;
        }
        if pap <= 0.0 {
            return Convergence { iterations: iter, converged: true };
        }
        let alpha = rr / pap;
        let mut rr_new = 0.0;
        for i in 0..n {
            x[i] += alpha * p[i];
            let ri = r[i] - alpha * ap[i];
            r[i] = ri;
            rr_new += w.map_or(1.0, |w| w[i]) * ri * ri;
        }
        // Hestenes-Stiefel estimate of the remaining error
        recent[iter % 2] = alpha * rr;
        potential -= alpha * rr;
        if potential <= 0.0 || ((recent[0] + recent[1]) / potential).sqrt() <= tol {
            return Convergence { iterations: iter, converged: true };
        }
        // residual at rounding level: further steps only amplify noise
        if rr_new <= 1e-26 * rr0 {
            return Convergence { iterations: iter, converged: true };
        }
        let beta = rr_new / rr;
        rr = rr_new;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }
    Convergence { iterations: maxiter, converged: false }
}

/// Partial out the fixed effects from several columns (in parallel where
/// threads are available). Returns the convergence of each column.
pub fn partial_out_all(factors: &[Factor], cols: &mut [&mut Vec<f64>], w: Option<&[f64]>, tol: f64, maxiter: usize) -> Vec<Convergence> {
    let n = cols.first().map_or(0, |c| c.len());
    let tw = TwoWay::prepare(factors, w);
    let threads = worker_count(cols.len(), n, factors.len());
    if threads <= 1 {
        let mut ws = Workspace::new(n);
        return cols.iter_mut().map(|c| partial_out_with(factors, tw.as_ref(), c, w, tol, maxiter, &mut ws)).collect();
    }
    run_parallel(factors, tw.as_ref(), cols, w, tol, maxiter, threads)
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_count(ncols: usize, n: usize, nf: usize) -> usize {
    if n < 50_000 || ncols < 2 || nf < 2 {
        return 1;
    }
    std::thread::available_parallelism().map_or(1, |p| p.get()).min(ncols).min(8)
}

#[cfg(target_arch = "wasm32")]
fn worker_count(_ncols: usize, _n: usize, _nf: usize) -> usize {
    1
}

#[cfg(not(target_arch = "wasm32"))]
fn run_parallel(factors: &[Factor], tw: Option<&TwoWay>, cols: &mut [&mut Vec<f64>], w: Option<&[f64]>, tol: f64, maxiter: usize, threads: usize) -> Vec<Convergence> {
    let n = cols[0].len();
    let ncols = cols.len();
    let mut res = vec![Convergence { iterations: 0, converged: true }; ncols];
    // deal columns round-robin to the workers
    let mut buckets: Vec<Vec<(usize, &mut Vec<f64>)>> = (0..threads).map(|_| Vec::new()).collect();
    for (j, c) in cols.iter_mut().enumerate() {
        buckets[j % threads].push((j, &mut **c));
    }
    let out: Vec<Vec<(usize, Convergence)>> = std::thread::scope(|s| {
        let handles: Vec<_> = buckets
            .into_iter()
            .map(|b| {
                s.spawn(move || {
                    let mut ws = Workspace::new(n);
                    b.into_iter().map(|(j, c)| (j, partial_out_with(factors, tw, c, w, tol, maxiter, &mut ws))).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    for (j, c) in out.into_iter().flatten() {
        res[j] = c;
    }
    res
}

#[cfg(target_arch = "wasm32")]
fn run_parallel(factors: &[Factor], tw: Option<&TwoWay>, cols: &mut [&mut Vec<f64>], w: Option<&[f64]>, tol: f64, maxiter: usize, _threads: usize) -> Vec<Convergence> {
    let n = cols[0].len();
    let mut ws = Workspace::new(n);
    cols.iter_mut().map(|c| partial_out_with(factors, tw, c, w, tol, maxiter, &mut ws)).collect()
}

// ---------------------------------------------------------------------------
// Degrees of freedom
// ---------------------------------------------------------------------------

/// Connected components of the bipartite graph linking the groups of two
/// groupings through shared observations.
pub fn components(a: &[u32], ga: usize, b: &[u32], gb: usize) -> usize {
    let mut parent: Vec<u32> = (0..(ga + gb) as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[parent[x as usize] as usize];
            parent[x as usize] = p;
            x = p;
        }
        x
    }
    let mut comps = ga + gb;
    for (&x, &y) in a.iter().zip(b) {
        let rx = find(&mut parent, x);
        let ry = find(&mut parent, ga as u32 + y);
        if rx != ry {
            parent[rx as usize] = ry;
            comps -= 1;
        }
    }
    comps
}

/// Is every group of `fe` contained in a single cluster?
pub fn nested(fe: &[u32], ng: usize, cl: &[u32]) -> bool {
    let mut first = vec![u32::MAX; ng];
    for (&g, &c) in fe.iter().zip(cl) {
        let s = &mut first[g as usize];
        if *s == u32::MAX {
            *s = c;
        } else if *s != c {
            return false;
        }
    }
    true
}

/// Groups in which a slope variable is collinear with the rest of the
/// group's terms (zero, or constant when the group also has an intercept).
pub fn degenerate_slope_groups(f: &Factor, s: usize, w: Option<&[f64]>) -> usize {
    let ng = f.ng;
    let mut sw = vec![0.0; ng];
    let mut sx = vec![0.0; ng];
    let mut sxx = vec![0.0; ng];
    let x = &f.slopes[s];
    for i in 0..x.len() {
        let g = f.ids[i] as usize;
        let wi = w.map_or(1.0, |w| w[i]);
        sw[g] += wi;
        sx[g] += wi * x[i];
        sxx[g] += wi * x[i] * x[i];
    }
    (0..ng)
        .filter(|&g| {
            let v = if f.intercept { sxx[g] - sx[g] * sx[g] / sw[g] } else { sxx[g] };
            v <= 1e-12 * sxx[g].max(1e-300)
        })
        .count()
}

// ---------------------------------------------------------------------------
// Fixed-effect estimates
// ---------------------------------------------------------------------------

/// Split `d` (the sum of all fixed effects, observation by observation) into
/// one component per factor by block Gauss-Seidel (backfitting). Returns the
/// per-observation contribution of every factor (intercept and slopes
/// separated: `out[k][0]` is the intercept, `out[k][1..]` the slopes).
pub fn recover(factors: &[Factor], d: &[f64], w: Option<&[f64]>, tol: f64, maxiter: usize) -> Vec<Vec<Vec<f64>>> {
    let n = d.len();
    let k = factors.len();
    let mut coefs: Vec<Vec<f64>> = factors.iter().map(|f| vec![0.0; f.ng * f.p()]).collect();
    let mut fit: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    let mut resid = d.to_vec();
    let mut buf = Vec::new();
    let scale = dot(d, d, w).max(1e-300);
    for _ in 0..maxiter.max(1) {
        let mut change = 0.0;
        for (j, f) in factors.iter().enumerate() {
            // add back this factor's fit and re-project
            for i in 0..n {
                resid[i] += fit[j][i];
            }
            f.coefs(&resid, w, &mut buf);
            for i in 0..n {
                let nv = f.fitted(&buf, i);
                let dlt = nv - fit[j][i];
                change += dlt * dlt * w.map_or(1.0, |w| w[i]);
                fit[j][i] = nv;
                resid[i] -= nv;
            }
            coefs[j].copy_from_slice(&buf);
        }
        if k == 1 || (change / scale).sqrt() < tol {
            break;
        }
    }
    factors
        .iter()
        .zip(&coefs)
        .map(|(f, c)| {
            let p = f.p();
            (0..p)
                .map(|a| {
                    (0..n)
                        .map(|i| {
                            let g = f.ids[i] as usize;
                            let coef = c[g * p + a];
                            let slope_ix = a as isize - f.intercept as isize;
                            if slope_ix < 0 { coef } else { coef * f.slopes[slope_ix as usize][i] }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_way_demeaning_matches_closed_form_on_balanced_panel() {
        // balanced panel: two-way demeaning is x - xbar_i - xbar_t + xbar
        let (ni, nt) = (7usize, 5usize);
        let n = ni * nt;
        let a: Vec<u32> = (0..n).map(|i| (i / nt) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| (i % nt) as u32).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64).sin() + i as f64 * 0.1).collect();
        let mut fa = Factor::new("a".into(), a.clone(), ni, true, vec![], vec![]);
        let mut fb = Factor::new("b".into(), b.clone(), nt, true, vec![], vec![]);
        fa.prepare(None);
        fb.prepare(None);
        let mut v = x.clone();
        let mut ws = Workspace::new(n);
        let c = partial_out(&[fa, fb], &mut v, None, 1e-12, 1000, &mut ws);
        assert!(c.converged);
        let mean = x.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            let mi = (0..nt).map(|t| x[a[i] as usize * nt + t]).sum::<f64>() / nt as f64;
            let mt = (0..ni).map(|j| x[j * nt + b[i] as usize]).sum::<f64>() / ni as f64;
            assert!((v[i] - (x[i] - mi - mt + mean)).abs() < 1e-9);
        }
    }

    #[test]
    fn components_and_singletons() {
        // two disconnected blocks
        let a = [0u32, 0, 1, 1, 2];
        let b = [0u32, 1, 2, 2, 3];
        assert_eq!(components(&a, 3, &b, 4), 3);
        let fa = Factor::new("a".into(), a.to_vec(), 3, true, vec![], vec![]);
        let fb = Factor::new("b".into(), b.to_vec(), 4, true, vec![], vec![]);
        // obs 4 is alone in a=2; then obs 0 and 1 are alone in b=0, b=1
        let (keep, dropped) = singletons(&[fa, fb], 5);
        assert_eq!(keep, vec![false, false, true, true, false]);
        assert_eq!(dropped, 3);
    }
}

#[cfg(test)]
mod unbalanced_tests {
    use super::*;

    #[test]
    fn unbalanced_two_way_matches_alternating_projections() {
        let n = 400usize;
        let a: Vec<f64> = (1..=n).map(|i| (1 + (i * 7) % 40) as f64).collect();
        let b: Vec<f64> = (1..=n).map(|i| (1 + (i * 3) % 9) as f64).collect();
        let (ia, ga) = codes_f64(&a);
        let (ib, gb) = codes_f64(&b);
        let x: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64).sin() + a[i] / 7.0 + b[i] / 3.0).collect();
        let mut fa = Factor::new("a".into(), ia, ga, true, vec![], vec![]);
        let mut fb = Factor::new("b".into(), ib, gb, true, vec![], vec![]);
        fa.prepare(None);
        fb.prepare(None);
        let fs = [fa, fb];
        let mut v = x.clone();
        let mut ws = Workspace::new(n);
        let c = partial_out(&fs, &mut v, None, 1e-12, 1000, &mut ws);
        let mut slow = x.clone();
        let mut buf = Vec::new();
        for _ in 0..5000 {
            for f in &fs {
                f.annihilate(&mut slow, None, &mut buf);
            }
        }
        let err = v.iter().zip(&slow).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(err < 1e-8, "err {err} after {:?}; {:?} vs {:?}", c, &v[..4], &slow[..4]);
    }
}

#[cfg(test)]
mod twoway_tests {
    use super::*;

    #[test]
    fn direct_two_way_solver_matches_alternating_projections_with_weights() {
        let n = 3000usize;
        let a: Vec<f64> = (0..n).map(|i| ((i * 7919) % 211) as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i * 104729 + i / 13) % 17) as f64).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 5) as f64).collect();
        let (ia, ga) = codes_f64(&a);
        let (ib, gb) = codes_f64(&b);
        let x: Vec<f64> = (0..n).map(|i| ((i * 31 % 101) as f64).sin() * 3.0 + a[i] / 7.0 + b[i] / 3.0).collect();
        let mut fa = Factor::new("a".into(), ia, ga, true, vec![], vec![]);
        let mut fb = Factor::new("b".into(), ib, gb, true, vec![], vec![]);
        fa.prepare(Some(&w));
        fb.prepare(Some(&w));
        let fs = [fa, fb];
        let tw = TwoWay::prepare(&fs, Some(&w)).expect("direct solver applies");
        let mut v = x.clone();
        let mut ws = Workspace::new(n);
        let c = partial_out_with(&fs, Some(&tw), &mut v, Some(&w), 1e-12, 1000, &mut ws);
        assert!(c.converged);
        let mut slow = x.clone();
        let mut buf = Vec::new();
        for _ in 0..20000 {
            for f in &fs {
                f.annihilate(&mut slow, Some(&w), &mut buf);
            }
        }
        let err = v.iter().zip(&slow).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(err < 1e-8, "err {err}");
        // the CG path on the same problem
        let mut v2 = x.clone();
        partial_out(&fs, &mut v2, Some(&w), 1e-13, 1000, &mut ws);
        let err2 = v2.iter().zip(&slow).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(err2 < 1e-8, "cg err {err2}");
    }
}
