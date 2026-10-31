//! Regression trees grown on gradient/weight sums, shared by the gradient
//! boosting and random-forest learners, plus exact TreeSHAP attributions.
//!
//! A node is split on the predictor and threshold maximizing
//! `sum_k G_k^2/H` over the two children (G: sum of targets/gradients,
//! H: sum of weights/hessians).  With unit weights and the response as
//! target this is the least-squares (variance-reduction) criterion; with
//! one-hot targets it is the Gini criterion.  Leaves hold `scale * G/H`.
//! Missing predictor values follow the child that received more rows.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Node {
    /// predictor index; `usize::MAX` for a leaf
    pub feat: usize,
    pub thr: f64,
    pub na_left: bool,
    pub left: usize,
    pub right: usize,
    pub value: Vec<f64>,
    /// number of training rows reaching the node
    pub cover: f64,
    /// improvement of the split criterion
    pub gain: f64,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.feat == usize::MAX
    }
}

#[derive(Clone, Debug, Default)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

pub struct GrowParams {
    pub max_depth: usize,
    pub min_leaf: f64,
    /// predictors sampled at each split (None: all)
    pub mtry: Option<usize>,
    pub scale: f64,
}

/// Grow a tree. `x[j][i]` is predictor j at row i (NaN = missing),
/// `g[k][i]` the k-th target and `h[i]` the weight of row i.
pub fn grow(x: &[Vec<f64>], rows: &[usize], g: &[Vec<f64>], h: &[f64], prm: &GrowParams, rng: &mut Rng) -> Tree {
    let kk = g.len();
    let p = x.len();
    let mut nodes: Vec<Node> = Vec::new();
    let leaf_of = |rs: &[usize]| -> (Vec<f64>, f64) {
        let hs: f64 = rs.iter().map(|&i| h[i]).sum();
        let v = (0..kk)
            .map(|k| {
                let gs: f64 = rs.iter().map(|&i| g[k][i]).sum();
                if hs.abs() > 1e-12 { prm.scale * gs / hs } else { 0.0 }
            })
            .collect();
        (v, hs)
    };
    let (v0, _) = leaf_of(rows);
    nodes.push(Node { feat: usize::MAX, thr: 0.0, na_left: false, left: 0, right: 0, value: v0, cover: rows.len() as f64, gain: 0.0 });
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, rows.to_vec(), 0)];
    let mut gs = vec![0.0; kk];
    let mut gl = vec![0.0; kk];
    while let Some((id, rs, depth)) = stack.pop() {
        if depth >= prm.max_depth || (rs.len() as f64) < 2.0 * prm.min_leaf {
            continue;
        }
        let feats: Vec<usize> = match prm.mtry {
            Some(m) if m < p => {
                let mut all: Vec<usize> = (0..p).collect();
                for i in 0..m {
                    let j = i + (rng.uniform() * (p - i) as f64) as usize;
                    all.swap(i, j.min(p - 1));
                }
                all.truncate(m);
                all
            }
            _ => (0..p).collect(),
        };
        let mut best: Option<(f64, usize, f64)> = None;
        for &j in &feats {
            let mut v: Vec<(f64, usize)> = rs.iter().filter(|&&i| !x[j][i].is_nan()).map(|&i| (x[j][i], i)).collect();
            if v.len() < 2 {
                continue;
            }
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            for k in 0..kk {
                gs[k] = v.iter().map(|&(_, i)| g[k][i]).sum();
                gl[k] = 0.0;
            }
            let hs: f64 = v.iter().map(|&(_, i)| h[i]).sum();
            let parent: f64 = if hs.abs() > 1e-12 { gs.iter().map(|a| a * a).sum::<f64>() / hs } else { 0.0 };
            let mut hl = 0.0;
            let n = v.len();
            for t in 0..n - 1 {
                let i = v[t].1;
                for k in 0..kk {
                    gl[k] += g[k][i];
                }
                hl += h[i];
                if v[t].0 == v[t + 1].0 {
                    continue;
                }
                let nl = (t + 1) as f64;
                if nl < prm.min_leaf || ((n - t - 1) as f64) < prm.min_leaf {
                    continue;
                }
                let hr = hs - hl;
                if hl <= 1e-12 || hr <= 1e-12 {
                    continue;
                }
                let mut s = 0.0;
                for k in 0..kk {
                    let r = gs[k] - gl[k];
                    s += gl[k] * gl[k] / hl + r * r / hr;
                }
                let gain = s - parent;
                if gain > 1e-10 * (1.0 + parent.abs()) && best.map_or(true, |b| gain > b.0) {
                    best = Some((gain, j, 0.5 * (v[t].0 + v[t + 1].0)));
                }
            }
        }
        let Some((gain, j, thr)) = best else { continue };
        let (mut l, mut r): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut na = Vec::new();
        for &i in &rs {
            let xv = x[j][i];
            if xv.is_nan() {
                na.push(i);
            } else if xv < thr {
                l.push(i);
            } else {
                r.push(i);
            }
        }
        let na_left = l.len() >= r.len();
        if na_left { l.extend(na) } else { r.extend(na) }
        let (vl, _) = leaf_of(&l);
        let (vr, _) = leaf_of(&r);
        let li = nodes.len();
        nodes.push(Node { feat: usize::MAX, thr: 0.0, na_left: false, left: 0, right: 0, value: vl, cover: l.len() as f64, gain: 0.0 });
        nodes.push(Node { feat: usize::MAX, thr: 0.0, na_left: false, left: 0, right: 0, value: vr, cover: r.len() as f64, gain: 0.0 });
        let nd = &mut nodes[id];
        nd.feat = j;
        nd.thr = thr;
        nd.na_left = na_left;
        nd.left = li;
        nd.right = li + 1;
        nd.gain = gain;
        stack.push((li + 1, r, depth + 1));
        stack.push((li, l, depth + 1));
    }
    Tree { nodes }
}

impl Tree {
    fn next(&self, n: &Node, xv: f64) -> usize {
        let left = if xv.is_nan() { n.na_left } else { xv < n.thr };
        if left { n.left } else { n.right }
    }

    /// Leaf reached by a row (`xr(j)` returns predictor j).
    pub fn leaf(&self, xr: &dyn Fn(usize) -> f64) -> &Node {
        let mut id = 0;
        loop {
            let n = &self.nodes[id];
            if n.is_leaf() {
                return n;
            }
            id = self.next(n, xr(n.feat));
        }
    }

    pub fn depth(&self) -> usize {
        fn d(t: &Tree, id: usize) -> usize {
            let n = &t.nodes[id];
            if n.is_leaf() { 0 } else { 1 + d(t, n.left).max(d(t, n.right)) }
        }
        d(self, 0)
    }

    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Cover-weighted mean of the leaf values of output k.
    pub fn expected(&self, k: usize) -> f64 {
        let root = self.nodes[0].cover.max(1e-300);
        self.nodes.iter().filter(|n| n.is_leaf()).map(|n| n.cover * n.value[k]).sum::<f64>() / root
    }

    /// Add the TreeSHAP attributions of output k for one row to `phi`.
    pub fn shap(&self, xr: &dyn Fn(usize) -> f64, k: usize, phi: &mut [f64]) {
        self.shap_rec(xr, k, phi, 0, Vec::new(), 1.0, 1.0, -1);
    }

    #[allow(clippy::too_many_arguments)]
    fn shap_rec(&self, xr: &dyn Fn(usize) -> f64, k: usize, phi: &mut [f64], id: usize, mut m: Vec<PathEl>, pz: f64, po: f64, pi: isize) {
        extend(&mut m, pz, po, pi);
        let n = &self.nodes[id];
        if n.is_leaf() {
            for i in 1..m.len() {
                let w = unwound_sum(&m, i);
                let el = &m[i];
                phi[el.d as usize] += w * (el.o - el.z) * n.value[k];
            }
            return;
        }
        let hot = self.next(n, xr(n.feat));
        let cold = if hot == n.left { n.right } else { n.left };
        let cov = n.cover.max(1e-300);
        let (mut iz, mut io) = (1.0, 1.0);
        if let Some(pos) = m.iter().position(|e| e.d == n.feat as isize) {
            iz = m[pos].z;
            io = m[pos].o;
            unwind(&mut m, pos);
        }
        self.shap_rec(xr, k, phi, hot, m.clone(), iz * self.nodes[hot].cover / cov, io, n.feat as isize);
        self.shap_rec(xr, k, phi, cold, m, iz * self.nodes[cold].cover / cov, 0.0, n.feat as isize);
    }
}

#[derive(Clone, Debug)]
struct PathEl {
    d: isize,
    z: f64,
    o: f64,
    w: f64,
}

fn extend(m: &mut Vec<PathEl>, pz: f64, po: f64, pi: isize) {
    let l = m.len();
    m.push(PathEl { d: pi, z: pz, o: po, w: if l == 0 { 1.0 } else { 0.0 } });
    for i in (0..l).rev() {
        m[i + 1].w += po * m[i].w * (i + 1) as f64 / (l + 1) as f64;
        m[i].w = pz * m[i].w * (l - i) as f64 / (l + 1) as f64;
    }
}

fn unwind(m: &mut Vec<PathEl>, pos: usize) {
    let ud = m.len() - 1;
    let (o, z) = (m[pos].o, m[pos].z);
    let mut next = m[ud].w;
    for i in (0..ud).rev() {
        if o != 0.0 {
            let tmp = m[i].w;
            m[i].w = next * (ud + 1) as f64 / ((i + 1) as f64 * o);
            next = tmp - m[i].w * z * (ud - i) as f64 / (ud + 1) as f64;
        } else {
            m[i].w = m[i].w * (ud + 1) as f64 / (z * (ud - i) as f64);
        }
    }
    for i in pos..ud {
        m[i].d = m[i + 1].d;
        m[i].z = m[i + 1].z;
        m[i].o = m[i + 1].o;
    }
    m.pop();
}

fn unwound_sum(m: &[PathEl], pos: usize) -> f64 {
    let ud = m.len() - 1;
    let (o, z) = (m[pos].o, m[pos].z);
    let mut next = m[ud].w;
    let mut total = 0.0;
    for i in (0..ud).rev() {
        if o != 0.0 {
            let tmp = next * (ud + 1) as f64 / ((i + 1) as f64 * o);
            total += tmp;
            next = m[i].w - tmp * z * (ud - i) as f64 / (ud + 1) as f64;
        } else if z != 0.0 {
            total += m[i].w / z / ((ud - i) as f64 / (ud + 1) as f64);
        }
    }
    total
}

/// Exact Shapley values of one tree output by enumerating coalitions
/// (exponential in the number of predictors; used to check TreeSHAP).
pub fn shap_bruteforce(t: &Tree, xr: &dyn Fn(usize) -> f64, k: usize, p: usize) -> Vec<f64> {
    // expected value of the tree when only predictors in `s` are known
    fn cond(t: &Tree, id: usize, xr: &dyn Fn(usize) -> f64, k: usize, s: u32) -> f64 {
        let n = &t.nodes[id];
        if n.is_leaf() {
            return n.value[k];
        }
        if s & (1 << n.feat) != 0 {
            return cond(t, t.next(n, xr(n.feat)), xr, k, s);
        }
        let c = n.cover.max(1e-300);
        (t.nodes[n.left].cover * cond(t, n.left, xr, k, s) + t.nodes[n.right].cover * cond(t, n.right, xr, k, s)) / c
    }
    let fact = |n: usize| (1..=n).map(|v| v as f64).product::<f64>();
    let mut phi = vec![0.0; p];
    for j in 0..p {
        for s in 0u32..(1 << p) {
            if s & (1 << j) != 0 {
                continue;
            }
            let sz = s.count_ones() as usize;
            let w = fact(sz) * fact(p - sz - 1) / fact(p);
            phi[j] += w * (cond(t, 0, xr, k, s | (1 << j)) - cond(t, 0, xr, k, s));
        }
    }
    phi
}
