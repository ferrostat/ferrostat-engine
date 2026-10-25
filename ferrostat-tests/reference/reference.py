"""Independent reference values for the Ferrostat test suite.

Pure Python (standard library only), written from textbook formulas. It
deliberately shares no code with the Rust implementation. Re-run with
`python3 reference.py` and paste the output into tests/common/reference.rs
if the fixture changes.
"""
import math

ID = list(range(1, 13))
GROUP = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
X = [float(v) for v in range(1, 13)]
Y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 19.9, 22.2, 23.8]
Z = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
D = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
CNT = [0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]


def mean(v): return sum(v) / len(v)
def var(v):
    m = mean(v); return sum((a - m) ** 2 for a in v) / (len(v) - 1)
def sd(v): return math.sqrt(var(v))


def solve(a, b):
    n = len(b); a = [r[:] + [b[i]] for i, r in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c])); a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def inv(a):
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def xtx(cols, w=None):
    n = len(cols[0]); w = w or [1.0] * n
    return [[sum(w[t] * ci[t] * cj[t] for t in range(n)) for cj in cols] for ci in cols]


def ols(y, xs, cons=True):
    n = len(y); cols = [*xs] + ([[1.0] * n] if cons else [])
    A = xtx(cols); b = solve(A, [sum(c[t] * y[t] for t in range(n)) for c in cols])
    fit = [sum(b[j] * cols[j][t] for j in range(len(cols))) for t in range(n)]
    e = [y[t] - fit[t] for t in range(n)]
    k = len(cols); rss = sum(r * r for r in e)
    tss = sum((v - mean(y)) ** 2 for v in y) if cons else sum(v * v for v in y)
    df_m = k - (1 if cons else 0); df_r = n - k
    s2 = rss / df_r; Ai = inv(A)
    se = [math.sqrt(s2 * Ai[j][j]) for j in range(k)]
    # HC1 robust
    meat = [[sum(e[t] ** 2 * cols[i][t] * cols[j][t] for t in range(n)) for j in range(k)] for i in range(k)]
    V = [[sum(Ai[i][a] * meat[a][b_] * Ai[b_][j] for a in range(k) for b_ in range(k)) for j in range(k)] for i in range(k)]
    rse = [math.sqrt(V[j][j] * n / (n - k)) for j in range(k)]
    return dict(b=b, se=se, rse=rse, rss=rss, mss=tss - rss, tss=tss, df_m=df_m, df_r=df_r,
                r2=1 - rss / tss, adj_r2=1 - (rss / tss) * (n - (1 if cons else 0)) / df_r,
                rmse=math.sqrt(s2), F=((tss - rss) / df_m) / s2, fit=fit, resid=e)


def phi(x): return 0.5 * math.erfc(-x / math.sqrt(2))
def dphi(x): return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def mle(y, xs, kind):
    n = len(y); cols = [*xs, [1.0] * n]; k = len(cols); b = [0.0] * k
    def ll_of(b):
        s = 0.0
        for t in range(n):
            xb = sum(b[j] * cols[j][t] for j in range(k))
            if kind == "logit":
                p = 1 / (1 + math.exp(-xb)); s += y[t] * math.log(p) + (1 - y[t]) * math.log(1 - p)
            elif kind == "probit":
                p = phi(xb); s += y[t] * math.log(p) + (1 - y[t]) * math.log(1 - p)
            else:
                mu = math.exp(xb); s += y[t] * xb - mu - math.lgamma(y[t] + 1)
        return s
    for _ in range(200):
        g = [0.0] * k; H = [[0.0] * k for _ in range(k)]
        for t in range(n):
            xb = sum(b[j] * cols[j][t] for j in range(k))
            if kind == "logit":
                p = 1 / (1 + math.exp(-xb)); s = y[t] - p; w = p * (1 - p)
            elif kind == "probit":
                p = phi(xb); q = dphi(xb)
                lam = q / p if y[t] else -q / (1 - p)
                s = lam; w = lam * (lam + xb)
            else:
                mu = math.exp(xb); s = y[t] - mu; w = mu
            for i in range(k):
                g[i] += s * cols[i][t]
                for j in range(k):
                    H[i][j] += w * cols[i][t] * cols[j][t]
        step = solve(H, g); b = [b[i] + step[i] for i in range(k)]
        if max(abs(s) for s in step) < 1e-13: break
    Vi = inv(H); se = [math.sqrt(Vi[j][j]) for j in range(k)]
    ll = ll_of(b)
    # constant-only model
    ybar = mean(y)
    if kind == "logit": ll0 = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    elif kind == "probit": ll0 = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    else: ll0 = sum(v * math.log(ybar) - ybar - math.lgamma(v + 1) for v in y)
    return dict(b=b, se=se, ll=ll, ll0=ll0, chi2=2 * (ll - ll0), r2_p=1 - ll / ll0)


def corr(a, b):
    ma, mb = mean(a), mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]: j += 1
        for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pctile(v, p):
    """Percentile as defined for `summarize, detail` (empirical CDF, averaging at ties)."""
    s = sorted(v); n = len(s); np_ = n * p / 100
    i = math.floor(np_)
    if np_ == i: return (s[i - 1] + s[i]) / 2 if i > 0 else s[0]
    return s[i]


def emit(name, v):
    if isinstance(v, list): print(f"pub const {name}: [f64; {len(v)}] = [{', '.join(repr(x) for x in v)}];")
    else: print(f"pub const {name}: f64 = {v!r};")


print("// @generated by reference/reference.py -- do not edit by hand")
for nm, v in [("X", X), ("Y", Y), ("Z", Z)]:
    emit(f"{nm}_MEAN", mean(v)); emit(f"{nm}_SD", sd(v)); emit(f"{nm}_VAR", var(v))
s = sorted(Y)
emit("Y_P50", pctile(Y, 50)); emit("Y_P25", pctile(Y, 25)); emit("Y_P75", pctile(Y, 75))
emit("Y_P10", pctile(Y, 10)); emit("Y_P90", pctile(Y, 90))
m = mean(Y); n = len(Y)
m2 = sum((a - m) ** 2 for a in Y) / n; m3 = sum((a - m) ** 3 for a in Y) / n; m4 = sum((a - m) ** 4 for a in Y) / n
emit("Y_SKEW", m3 / m2 ** 1.5); emit("Y_KURT", m4 / m2 ** 2)
emit("CORR_XY", corr(X, Y)); emit("CORR_XZ", corr(X, Z)); emit("CORR_YZ", corr(Y, Z))
emit("SPEARMAN_XZ", corr(rank(X), rank(Z)))
cov = sum((a - mean(X)) * (b - mean(Y)) for a, b in zip(X, Y)) / (n - 1); emit("COV_XY", cov)

r = ols(Y, [X, Z])
emit("REG_B", r["b"]); emit("REG_SE", r["se"]); emit("REG_RSE", r["rse"])
for k in ["rss", "mss", "tss", "r2", "adj_r2", "rmse", "F"]: emit(f"REG_{k.upper()}", r[k])
emit("REG_FIT", r["fit"]); emit("REG_RESID", r["resid"])
r1 = ols(Y, [X]); emit("REG1_B", r1["b"]); emit("REG1_SE", r1["se"]); emit("REG1_R2", r1["r2"])
rn = ols(Y, [X], cons=False); emit("REGNC_B", rn["b"]); emit("REGNC_SE", rn["se"])
# regression on first 11 obs (12th y treated as missing -> listwise deletion)
r11 = ols(Y[:11], [X[:11], Z[:11]]); emit("REG11_B", r11["b"])
# weighted (aweight = z): WLS with weights normalised to sum to N
wsum = sum(Z); w = [v * n / wsum for v in Z]
cols = [X, [1.0] * n]; A = xtx(cols, w)
bw = solve(A, [sum(w[t] * c[t] * Y[t] for t in range(n)) for c in cols]); emit("REGAW_B", bw)

for kind in ["logit", "probit"]:
    f = mle(D, [X], kind); K = kind.upper()
    emit(f"{K}_B", f["b"]); emit(f"{K}_SE", f["se"]); emit(f"{K}_LL", f["ll"]); emit(f"{K}_LL0", f["ll0"])
    emit(f"{K}_CHI2", f["chi2"]); emit(f"{K}_R2P", f["r2_p"])
f = mle(CNT, [X], "poisson")
emit("POIS_B", f["b"]); emit("POIS_SE", f["se"]); emit("POIS_LL", f["ll"]); emit("POIS_CHI2", f["chi2"])

# t tests
se = sd(Y) / math.sqrt(n); emit("TT1_T", (mean(Y) - 10) / se); emit("TT1_SE", se)
dif = [a - b for a, b in zip(Y, X)]; emit("TTP_T", mean(dif) / (sd(dif) / math.sqrt(n))); emit("TTP_MEAN", mean(dif))
g0 = [y for y, d in zip(Y, D) if d == 0]; g1 = [y for y, d in zip(Y, D) if d == 1]
n0, n1 = len(g0), len(g1); sp = ((n0 - 1) * var(g0) + (n1 - 1) * var(g1)) / (n0 + n1 - 2)
emit("TTG_T", (mean(g0) - mean(g1)) / math.sqrt(sp * (1 / n0 + 1 / n1)))
v0, v1 = var(g0) / n0, var(g1) / n1
emit("TTW_T", (mean(g0) - mean(g1)) / math.sqrt(v0 + v1))
emit("TTW_DF", (v0 + v1) ** 2 / (v0 ** 2 / (n0 - 1) + v1 ** 2 / (n1 - 1)))
# two-sample unpaired t test of two variables (x vs y)
spv = ((n - 1) * var(X) + (n - 1) * var(Y)) / (2 * n - 2)
emit("TTU_T", (mean(X) - mean(Y)) / math.sqrt(spv * 2 / n))

# chi2 group x d
tab = {}
for g, d in zip(GROUP, D): tab[(g, d)] = tab.get((g, d), 0) + 1
chi2 = 0.0
for g in (1, 2, 3):
    for d in (0.0, 1.0):
        e = sum(1 for a in GROUP if a == g) * sum(1 for b in D if b == d) / n
        chi2 += (tab.get((g, d), 0) - e) ** 2 / e
emit("CHI2_GD", chi2)

# one-way anova y by group
gm = {g: mean([y for y, gg in zip(Y, GROUP) if gg == g]) for g in (1, 2, 3)}
ssb = sum(4 * (gm[g] - mean(Y)) ** 2 for g in gm); ssw = sum((y - gm[g]) ** 2 for y, g in zip(Y, GROUP))
emit("ANOVA_SSB", ssb); emit("ANOVA_SSW", ssw); emit("ANOVA_F", (ssb / 2) / (ssw / 9))
emit("GROUP_MEAN_Y", [gm[1], gm[2], gm[3]])
emit("GROUP_SUM_Y", [sum(y for y, g in zip(Y, GROUP) if g == k) for k in (1, 2, 3)])
emit("GROUP_SD_Y", [sd([y for y, g in zip(Y, GROUP) if g == k]) for k in (1, 2, 3)])

# fixed effects y x, i(group)
gx = {g: mean([x for x, gg in zip(X, GROUP) if gg == g]) for g in (1, 2, 3)}
xd = [x - gx[g] for x, g in zip(X, GROUP)]; yd = [y - gm[g] for y, g in zip(Y, GROUP)]
bfe = sum(a * b for a, b in zip(xd, yd)) / sum(a * a for a in xd)
ufe = [b - bfe * a for a, b in zip(xd, yd)]
s2 = sum(u * u for u in ufe) / (n - 3 - 1)
emit("FE_B", bfe); emit("FE_SE", math.sqrt(s2 / sum(a * a for a in xd)))
emit("FE_CONS", mean(Y) - bfe * mean(X))

# Gaussian kernel density of x at x=6.5 with Silverman bandwidth (Stata-style: 0.9*min(sd, iqr/1.349)*n^-1/5)
sx = sorted(X); iqr = pctile(X, 75) - pctile(X, 25)
h = 0.9 * min(sd(X), iqr / 1.349) * n ** (-0.2)
emit("KD_H", h)
emit("KD_AT_6_5", sum(math.exp(-((6.5 - x) / h) ** 2 / 2) / math.sqrt(2 * math.pi) for x in X) / (n * h))

# ---- linear IV: y on x (endogenous) with instruments z and cnt (overidentified) ----
def mmul(a, b): return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def tr(a): return [list(r) for r in zip(*a)]
def rows(cols): return [list(r) for r in zip(*cols)]
def proj(cols_a, cols_z):
    """columns of a projected on the columns of z"""
    Zr = rows(cols_z); Zi = inv(xtx(cols_z))
    out = []
    for a in cols_a:
        c = mmul(Zi, [[sum(z[t] * a[t] for t in range(n))] for z in cols_z])
        out.append([sum(Zr[t][k] * c[k][0] for k in range(len(cols_z))) for t in range(n)])
    return out

ONE = [1.0] * n
IVX = [X, ONE]; IVZ = [Z, CNT, ONE]
def kclass(kappa):
    xh = proj(IVX, IVZ)
    xt = [[x[t] - kappa * (x[t] - h[t]) for t in range(n)] for x, h in zip(IVX, xh)]
    A = [[sum(a[t] * b[t] for t in range(n)) for b in IVX] for a in xt]
    b = solve(A, [sum(a[t] * Y[t] for t in range(n)) for a in xt])
    u = [Y[t] - sum(b[j] * IVX[j][t] for j in range(2)) for t in range(n)]
    Ai = inv(A); s2 = sum(e * e for e in u) / n
    return b, [math.sqrt(s2 * Ai[j][j]) for j in range(2)], u
b2, se2, u2 = kclass(1.0)
emit("IV2_B", b2); emit("IV2_SE", se2)
# Sargan: N * uncentred R2 of the 2SLS residuals on the instruments
fu = proj([u2], IVZ)[0]
emit("IV2_SARGAN", n * sum(f * f for f in fu) / sum(e * e for e in u2))
# LIML kappa: smallest root of det(W1 - k W0) = 0, W1 = Y'M_1 Y, W0 = Y'M_Z Y, Y = [y, x]
Yc = [Y, X]
m1 = [[v - mean(c) for v in c] for c in Yc]
mz = [[c[t] - p[t] for t in range(n)] for c, p in zip(Yc, proj(Yc, IVZ))]
W1 = xtx(m1); W0 = xtx(mz)
qa = W0[0][0] * W0[1][1] - W0[0][1] ** 2
qb = -(W1[0][0] * W0[1][1] + W1[1][1] * W0[0][0] - 2 * W1[0][1] * W0[0][1])
qc = W1[0][0] * W1[1][1] - W1[0][1] ** 2
kap = (-qb - math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
bl, sel, ul = kclass(kap)
emit("LIML_KAPPA", kap); emit("LIML_B", bl); emit("LIML_SE", sel)
# two-step GMM, heteroskedasticity-robust weight matrix from the 2SLS residuals
def S_of(u): return [[sum(u[t] ** 2 * a[t] * b[t] for t in range(n)) for b in IVZ] for a in IVZ]
S1i = inv(S_of(u2))
ZX = [[sum(z[t] * x[t] for t in range(n)) for x in IVX] for z in IVZ]
Zy = [[sum(z[t] * Y[t] for t in range(n))] for z in IVZ]
B = inv(mmul(mmul(tr(ZX), S1i), ZX))
bg = [r[0] for r in mmul(B, mmul(mmul(tr(ZX), S1i), Zy))]
ug = [Y[t] - sum(bg[j] * IVX[j][t] for j in range(2)) for t in range(n)]
S2 = S_of(ug)
mid = mmul(mmul(mmul(mmul(tr(ZX), S1i), S2), S1i), ZX)
Vg = mmul(mmul(B, mid), B)
g = [[sum(z[t] * ug[t] for t in range(n))] for z in IVZ]
emit("GMM_B", bg); emit("GMM_SE", [math.sqrt(Vg[j][j]) for j in range(2)])
emit("GMM_J", mmul(mmul(tr(g), inv(S2)), g)[0][0])

# ---- panel IV: 6 groups x 4 periods, x endogenous, instruments z1 z2 ----
PN = 24
pid = [math.ceil(i / 4) for i in range(1, PN + 1)]
pz1 = [math.sin(i * 0.7) for i in range(1, PN + 1)]
pz2 = [math.cos(i * 1.3) for i in range(1, PN + 1)]
pe = [math.sin(i * 2.1) * 0.5 for i in range(1, PN + 1)]
px = [pz1[t] + 0.5 * pz2[t] + 0.6 * pe[t] + 0.3 * (pid[t] / 5) for t in range(PN)]
py = [1 + 2 * px[t] + pe[t] + pid[t] / 5 for t in range(PN)]
G = sorted(set(pid))
def gmean(v): return {g: mean([v[t] for t in range(PN) if pid[t] == g]) for g in G}

def tsls_cols(y, xs, zs):
    """2SLS: returns b, bread (X^'X^)^-1, residuals"""
    m = len(y)
    Zi = inv(xtx(zs))
    xh = []
    for x in xs:
        c = [sum(Zi[a][b_] * sum(zs[b_][t] * x[t] for t in range(m)) for b_ in range(len(zs))) for a in range(len(zs))]
        xh.append([sum(c[a] * zs[a][t] for a in range(len(zs))) for t in range(m)])
    A = xtx(xh)
    b = solve(A, [sum(h[t] * y[t] for t in range(m)) for h in xh])
    u = [y[t] - sum(b[j] * xs[j][t] for j in range(len(xs))) for t in range(m)]
    return b, inv(A), u

def se_of(Ai, u, dof): s2 = sum(e * e for e in u) / dof; return [math.sqrt(s2 * Ai[j][j]) for j in range(len(Ai))]

# between
mx, my, m1, m2 = gmean(px), gmean(py), gmean(pz1), gmean(pz2)
bb, Ab, ub = tsls_cols([my[g] for g in G], [[mx[g] for g in G], [1.0] * 6], [[m1[g] for g in G], [m2[g] for g in G], [1.0] * 6])
emit("XTIV_BE_B", bb); emit("XTIV_BE_SE", se_of(Ab, ub, 6 - 2))
# first differences (consecutive periods within a group)
idx = [t for t in range(PN) if t > 0 and pid[t] == pid[t - 1]]
d = lambda v: [v[t] - v[t - 1] for t in idx]
bf, Af, uf = tsls_cols(d(py), [d(px), [1.0] * len(idx)], [d(pz1), d(pz2), [1.0] * len(idx)])
emit("XTIV_FD_B", bf); emit("XTIV_FD_SE", se_of(Af, uf, len(idx) - 2))
# random effects: variance components from within and between 2SLS
w = lambda v, m: [v[t] - m[pid[t]] for t in range(PN)]
_, _, uw = tsls_cols(w(py, my), [w(px, mx)], [w(pz1, m1), w(pz2, m2)])
s2e = sum(e * e for e in uw) / (PN - 6 - 1)
s2u = max(0.0, sum(e * e for e in ub) / (6 - 2) - s2e / 4)
th = 1 - math.sqrt(s2e / (4 * s2u + s2e))
q = lambda v, m: [v[t] - th * m[pid[t]] for t in range(PN)]
one = [1 - th] * PN
bg, Ag, ug = tsls_cols(q(py, my), [q(px, mx), one], [q(pz1, m1), q(pz2, m2), one])
emit("XTIV_RE_SIGMA_E", math.sqrt(s2e)); emit("XTIV_RE_SIGMA_U", math.sqrt(s2u))
emit("XTIV_G2SLS_B", bg); emit("XTIV_G2SLS_SE", se_of(Ag, ug, PN - 2))
ec_z = [w(pz1, m1), w(pz2, m2), [m1[pid[t]] for t in range(PN)], [m2[pid[t]] for t in range(PN)], [1.0] * PN]
be2, Ae, ue = tsls_cols(q(py, my), [q(px, mx), one], ec_z)
emit("XTIV_EC2SLS_B", be2); emit("XTIV_EC2SLS_SE", se_of(Ae, ue, PN - 2))
