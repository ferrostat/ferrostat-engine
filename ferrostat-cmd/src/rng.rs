//! Random-number generation: 64-bit Mersenne Twister (MT19937-64) with
//! derived generators for the common distributions.

const NN: usize = 312;
const MM: usize = 156;
const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
const UM: u64 = 0xFFFF_FFFF_8000_0000;
const LM: u64 = 0x7FFF_FFFF;

#[derive(Debug, Clone)]
pub struct Rng {
    mt: Vec<u64>,
    mti: usize,
    pub seed: u64,
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng { mt: vec![0; NN], mti: NN + 1, seed, spare_normal: None };
        r.reseed(seed);
        r
    }

    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.mt[0] = seed;
        for i in 1..NN {
            self.mt[i] = 6364136223846793005u64
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        self.mti = NN;
        self.spare_normal = None;
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.mti >= NN {
            for i in 0..NN {
                let x = (self.mt[i] & UM) | (self.mt[(i + 1) % NN] & LM);
                let mut xa = x >> 1;
                if x & 1 != 0 {
                    xa ^= MATRIX_A;
                }
                self.mt[i] = self.mt[(i + MM) % NN] ^ xa;
            }
            self.mti = 0;
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;
        x
    }

    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Uniform on (0, 1).
    fn open_uniform(&mut self) -> f64 {
        loop {
            let u = self.uniform();
            if u > 0.0 {
                return u;
            }
        }
    }

    pub fn uniform_ab(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }

    pub fn uniform_int(&mut self, a: f64, b: f64) -> f64 {
        (a + ((b - a + 1.0) * self.uniform()).floor()).min(b)
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // polar Box-Muller
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * m);
                return u * m;
            }
        }
    }

    /// Gamma(shape a, scale 1), Marsaglia-Tsang.
    pub fn gamma(&mut self, a: f64) -> f64 {
        if a < 1.0 {
            let u = self.open_uniform();
            return self.gamma(a + 1.0) * u.powf(1.0 / a);
        }
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.normal();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.open_uniform();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        x / (x + y)
    }

    pub fn chi2(&mut self, df: f64) -> f64 {
        2.0 * self.gamma(df / 2.0)
    }

    pub fn t(&mut self, df: f64) -> f64 {
        self.normal() / (self.chi2(df) / df).sqrt()
    }

    pub fn exponential(&mut self, b: f64) -> f64 {
        -b * self.open_uniform().ln()
    }

    pub fn poisson(&mut self, m: f64) -> f64 {
        if m < 30.0 {
            let l = (-m).exp();
            let mut k = 0.0;
            let mut p = 1.0;
            loop {
                p *= self.uniform();
                if p <= l {
                    return k;
                }
                k += 1.0;
            }
        }
        // gamma-Poisson splitting for large means
        let n = (0.875 * m).floor();
        let x = self.gamma(n);
        if x > m {
            self.binomial(n - 1.0, m / x)
        } else {
            n + self.poisson(m - x)
        }
    }

    pub fn binomial(&mut self, n: f64, p: f64) -> f64 {
        if n < 40.0 {
            return (0..n as u64).filter(|_| self.uniform() < p).count() as f64;
        }
        // beta-binomial splitting
        let a = (n / 2.0).floor() + 1.0;
        let b = n - a + 1.0;
        let x = self.beta(a, b);
        if x >= p {
            self.binomial(a - 1.0, p / x)
        } else {
            a + self.binomial(b - 1.0, (p - x) / (1.0 - x))
        }
    }

    /// State string for `c(rngstate)` (seed only; full state is internal).
    pub fn state(&self) -> String {
        format!("X{:016x}{:04x}", self.seed, self.mti)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt64_reference_output() {
        // First output of the reference MT19937-64 implementation seeded
        // with init_genrand64(5489).
        let mut r = Rng::new(5489);
        assert_eq!(r.next_u64(), 14514284786278117030);
    }

    #[test]
    fn moments() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let m = xs.iter().sum::<f64>() / n as f64;
        let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02);
        let g: f64 = (0..n).map(|_| r.gamma(3.0)).sum::<f64>() / n as f64;
        assert!((g - 3.0).abs() < 0.03);
    }
}
