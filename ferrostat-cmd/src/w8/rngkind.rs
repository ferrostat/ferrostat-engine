//! Generator selection for `set rng` and `set rngstream`.
//!
//! * `mt64` (also `default`): the 64-bit Mersenne Twister of `crate::rng`,
//!   seeded with the standard `init_genrand64` scheme; its streams are those
//!   of earlier releases.
//! * `kiss32`: Marsaglia's KISS generator (multiply-with-carry, 3-shift
//!   register and congruential components combined); 64-bit draws are made
//!   from two 32-bit outputs.
//! * `mt64s`: stream-split MT64. Stream `s` under seed `k` is the MT19937-64
//!   sequence initialised by `init_by_array64([k, s])`, so every (seed,
//!   stream) pair gives a reproducible sequence and different streams start
//!   from unrelated states (substream scheme by key, not jump-ahead).
//!
//! `c(rngstate)` holds the complete state (kind, stream, pending normal
//! deviate and generator words) and `set rngstate` restores it.

use crate::eval::rc_err;
use crate::rng::Rng;

const NN: usize = 312;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Mt64,
    Kiss32,
    Mt64s,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Mt64 => "mt64",
            Kind::Kiss32 => "kiss32",
            Kind::Mt64s => "mt64s",
        }
    }
    fn letter(self) -> char {
        match self {
            Kind::Mt64 => 'M',
            Kind::Kiss32 => 'K',
            Kind::Mt64s => 'S',
        }
    }
}

fn init_genrand64(mt: &mut [u64], seed: u64) {
    mt[0] = seed;
    for i in 1..NN {
        mt[i] = 6364136223846793005u64.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_add(i as u64);
    }
}

/// MT19937-64 initialisation by a key array.
fn init_by_array64(mt: &mut [u64], key: &[u64]) {
    init_genrand64(mt, 19650218);
    let (mut i, mut j) = (1usize, 0usize);
    let mut k = NN.max(key.len());
    while k > 0 {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_mul(3935559000370003845)).wrapping_add(key[j]).wrapping_add(j as u64);
        i += 1;
        j += 1;
        if i >= NN {
            mt[0] = mt[NN - 1];
            i = 1;
        }
        if j >= key.len() {
            j = 0;
        }
        k -= 1;
    }
    k = NN - 1;
    while k > 0 {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_mul(2862933555777941757)).wrapping_sub(i as u64);
        i += 1;
        if i >= NN {
            mt[0] = mt[NN - 1];
            i = 1;
        }
        k -= 1;
    }
    mt[0] = 1u64 << 63;
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed a non-default generator (the default MT64 is seeded by `Rng`).
pub fn reseed(r: &mut Rng, seed: u64) {
    r.seed = seed;
    let (kind, stream) = (r.kind, r.stream);
    match kind {
        Kind::Kiss32 => {
            let mut x = seed;
            let mut w = [0u32; 4];
            for v in w.iter_mut() {
                *v = (splitmix(&mut x) >> 32) as u32;
            }
            // the multiply-with-carry halves and the shift register must not be zero
            w[0] = w[0].max(1);
            w[1] = w[1].max(1);
            w[2] = w[2].max(1);
            r.kiss = w;
            let (_, mti, spare) = r.mt_parts();
            *mti = NN;
            *spare = None;
        }
        Kind::Mt64 | Kind::Mt64s => {
            let (mt, mti, spare) = r.mt_parts();
            if kind == Kind::Mt64 {
                init_genrand64(mt, seed);
            } else {
                init_by_array64(mt, &[seed, stream]);
            }
            *mti = NN;
            *spare = None;
        }
    }
}

fn kiss32(k: &mut [u32; 4]) -> u32 {
    k[0] = 36969u32.wrapping_mul(k[0] & 65535).wrapping_add(k[0] >> 16);
    k[1] = 18000u32.wrapping_mul(k[1] & 65535).wrapping_add(k[1] >> 16);
    let mwc = (k[0] << 16).wrapping_add(k[1]);
    k[2] ^= k[2] << 17;
    k[2] ^= k[2] >> 13;
    k[2] ^= k[2] << 5;
    k[3] = 69069u32.wrapping_mul(k[3]).wrapping_add(1234567);
    (mwc ^ k[3]).wrapping_add(k[2])
}

/// A 64-bit draw of the KISS32 generator.
pub fn kiss_u64(r: &mut Rng) -> u64 {
    let hi = kiss32(&mut r.kiss) as u64;
    let lo = kiss32(&mut r.kiss) as u64;
    (hi << 32) | lo
}

/// Full state string.
pub fn state(r: &Rng) -> String {
    let mut c = r.clone();
    let (kind, stream, seed, kiss) = (c.kind, c.stream, c.seed, c.kiss);
    let (mt, mti, spare) = c.mt_parts();
    let mut s = String::with_capacity(5100);
    s.push(kind.letter());
    s.push_str(&format!("{stream:04x}{seed:016x}"));
    match spare {
        Some(z) => s.push_str(&format!("1{:016x}", z.to_bits())),
        None => s.push_str(&format!("0{:016x}", 0)),
    }
    if kind == Kind::Kiss32 {
        for w in kiss {
            s.push_str(&format!("{w:08x}"));
        }
    } else {
        s.push_str(&format!("{:04x}", *mti));
        for w in mt.iter() {
            s.push_str(&format!("{w:016x}"));
        }
    }
    s
}

/// Restore a state made by `state`.
pub fn set_state(r: &mut Rng, s: &str) -> Result<(), String> {
    let bad = || rc_err(198, "invalid rngstate");
    let s = s.trim();
    let kind = match s.chars().next() {
        Some('M') => Kind::Mt64,
        Some('K') => Kind::Kiss32,
        Some('S') => Kind::Mt64s,
        _ => return Err(bad()),
    };
    let hex = |a: usize, b: usize| -> Result<u64, String> { s.get(a..b).and_then(|t| u64::from_str_radix(t, 16).ok()).ok_or_else(bad) };
    let stream = hex(1, 5)?;
    let seed = hex(5, 21)?;
    let has_spare = s.get(21..22) == Some("1");
    let spare_bits = hex(22, 38)?;
    let mut pos = 38;
    if kind == Kind::Kiss32 {
        let mut w = [0u32; 4];
        for v in w.iter_mut() {
            *v = hex(pos, pos + 8)? as u32;
            pos += 8;
        }
        if s.len() != pos {
            return Err(bad());
        }
        r.kiss = w;
    } else {
        let mti = hex(pos, pos + 4)? as usize;
        pos += 4;
        if s.len() != pos + 16 * NN || mti > NN + 1 {
            return Err(bad());
        }
        let mut words = Vec::with_capacity(NN);
        for k in 0..NN {
            words.push(hex(pos + 16 * k, pos + 16 * k + 16)?);
        }
        let (mt, m, _) = r.mt_parts();
        *mt = words;
        *m = mti;
    }
    r.kind = kind;
    r.stream = stream;
    r.seed = seed;
    let (_, _, spare) = r.mt_parts();
    *spare = has_spare.then(|| f64::from_bits(spare_bits));
    Ok(())
}

/// `set rng kind`: switching generators seeds the new one with the current
/// seed; selecting the generator in use changes nothing.
pub fn set_kind(r: &mut Rng, name: &str) -> Result<(), String> {
    let kind = match name.trim() {
        "default" | "mt64" => Kind::Mt64,
        "kiss32" => Kind::Kiss32,
        "mt64s" => Kind::Mt64s,
        o => return Err(rc_err(198, format!("set rng {o} not allowed; default, mt64, mt64s or kiss32"))),
    };
    if kind != r.kind {
        r.kind = kind;
        let seed = r.seed;
        if kind == Kind::Mt64 {
            r.reseed(seed);
        } else {
            reseed(r, seed);
        }
    }
    Ok(())
}

/// `set rngstream #` (1 to 32767): restarts the mt64s generator at the
/// beginning of that stream under the current seed.
pub fn set_stream(r: &mut Rng, v: &str) -> Result<(), String> {
    let n: u64 = v.trim().parse().ok().filter(|n| (1..=32767).contains(n)).ok_or_else(|| rc_err(198, format!("rngstream must be an integer between 1 and 32767; {} not allowed", v.trim())))?;
    r.stream = n;
    if r.kind == Kind::Mt64s {
        let seed = r.seed;
        reseed(r, seed);
    }
    Ok(())
}
