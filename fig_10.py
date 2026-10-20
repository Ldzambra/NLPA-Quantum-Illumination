
import math
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from matplotlib.lines import Line2D


N_MAX = 32
N_ENV = 42
N_RETURN = 46

NTH = 1.0
KAPPA = 0.01
ETA = 0.1
SQUEEZED_NS = 0.05

R_ORIGINAL = np.arcsinh(np.sqrt(SQUEEZED_NS))
LAMBDA = np.tanh(R_ORIGINAL)
STATE_TRUNC_TOL = 1e-14

T_fixed = {"NLPA": 0.001, "PA": 0.99999999, "PS": 0.001, "PC": 0.6022}

KINDS = ["NLPA", "PA", "PS", "PC"]
ORDER = ["NLPA", "PA", "PS", "PC", "TMSS", "CI"]

STYLES = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PS": ("#008A2E", "--", "^"),
    "PC": ("#0050FF", ":", "o"),
    "TMSS": ("#ff7f0e", ":", None),
    "CI": ("#AED814", "--", None)
}


def prepared_state(kind, T=0.5):
    psi = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    lam = LAMBDA

    if kind == "TMSS":
        for n in range(N_MAX):
            psi[n, n] = lam**n

    elif kind == "NLPA":
        for n in range(N_MAX - 1):
            coefficient = np.sqrt(n + 1.0) * (lam * T)**n / np.sqrt(2.0)
            psi[n + 1, n] += coefficient
            psi[n, n + 1] += coefficient

    elif kind == "PA":
        for n in range(N_MAX - 1):
            psi[n, n + 1] = np.sqrt(n + 1.0) * (lam * np.sqrt(T))**n

    elif kind == "PS":
        for n in range(1, N_MAX):
            psi[n, n - 1] = np.sqrt(n) * (lam * np.sqrt(T))**n

    elif kind == "PC":
        for n in range(N_MAX):
            psi[n, n] = (n * (1.0 - T) - T) * (lam * np.sqrt(T))**n

    else:
        raise ValueError(f"Unknown state: {kind}")

    norm = np.linalg.norm(psi)

    if norm <= 0:
        raise ValueError(f"Zero-norm state: {kind}, T={T}")

    return psi / norm


def signal_energy(psi):
    probabilities_B = np.sum(np.abs(psi)**2, axis=0)
    return float(np.dot(np.arange(psi.shape[1]), probabilities_B))


def thermal_probs(mean, dim):
    if mean < 0:
        raise ValueError("Negative thermal occupation.")

    if mean == 0:
        p = np.zeros(dim)
        p[0] = 1.0
        return p

    ratio = mean / (1.0 + mean)
    p = (1.0 - ratio) * ratio**np.arange(dim)
    return p / p.sum()


@lru_cache(maxsize=200000)
def beamsplitter_amp(j, m, p, kappa_eff):
    q = j + m - p

    if p < 0 or q < 0:
        return 0.0

    prefactor = np.exp(0.5 * (
        gammaln(p + 1) + gammaln(q + 1)
        - gammaln(j + 1) - gammaln(m + 1)
    ))

    total = 0.0
    lower = max(0, p - m)
    upper = min(j, p)

    for i in range(lower, upper + 1):
        b = p - i

        term = (
            math.comb(j, i) * math.comb(m, b)
            * kappa_eff**((i + m - b) / 2)
            * (1.0 - kappa_eff)**((j - i + b) / 2)
        )

        if b % 2:
            term = -term

        total += term

    return float(prefactor * total)


def compress_state(psi):
    pA = np.sum(np.abs(psi)**2, axis=1)
    pB = np.sum(np.abs(psi)**2, axis=0)

    idxA = np.flatnonzero(pA > STATE_TRUNC_TOL)
    idxB = np.flatnonzero(pB > STATE_TRUNC_TOL)

    dA = int(idxA[-1] + 1) if len(idxA) else 1
    dB = int(idxB[-1] + 1) if len(idxB) else 1

    compact = psi[:dA, :dB].copy()
    discarded = 1.0 - float(np.sum(np.abs(compact)**2))

    if discarded > 1e-10:
        raise RuntimeError(f"State compression discarded {discarded:.3e}")

    return compact / np.linalg.norm(compact)


def build_hypotheses_pre_target(psi_pre, eta=ETA):
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be between 0 and 1")

    psi = compress_state(psi_pre)

    dA, dB = psi.shape
    dR = N_RETURN
    dim = dA * dR

    k_eff = eta * KAPPA
    n_env_eff = NTH / (1.0 - k_eff)

    rhoA = psi @ psi.conj().T
    p0 = thermal_probs(NTH, dR)
    rho0 = np.kron(rhoA, np.diag(p0))

    p_env = thermal_probs(n_env_eff, N_ENV)
    rho1 = np.zeros((dim, dim), dtype=np.float64)

    active_signal_levels = np.flatnonzero(np.any(np.abs(psi) > 1e-16, axis=0))

    for m in range(N_ENV):
        V = np.zeros((dim, m + dB), dtype=np.float64)

        for j in active_signal_levels:
            j = int(j)
            idler_amplitudes = psi[:, j]
            p_max = min(dR, j + m + 1)

            for p in range(p_max):
                q = j + m - p
                amplitude = beamsplitter_amp(j, m, p, k_eff)

                if abs(amplitude) > 1e-18:
                    V[p::dR, q] += amplitude * idler_amplitudes

        rho1 += p_env[m] * (V @ V.T)

    trace1 = float(np.trace(rho1))

    if abs(trace1 - 1.0) > 1e-7:
        raise RuntimeError(
            f"Insufficient return/environment cutoff: Tr(rho1) = {trace1:.10f}"
        )

    rho1 /= trace1

    return rho0, rho1, rhoA, p0, dA, dR


def quantum_chernoff(rho1, rhoA, p0, dA, dR):
    eigA, vecA = eigh(rhoA, check_finite=False)
    eig1, vec1 = eigh(rho1, check_finite=False)

    eigA = np.maximum(eigA, 0.0)
    eig1 = np.maximum(eig1, 0.0)

    eig0 = (eigA[:, None] * p0[None, :]).reshape(-1)

    vec1_in_rho0_basis = np.einsum(
        "ak,ard->krd", vecA.conj(),
        vec1.reshape(dA, dR, -1), optimize=True
    )

    overlaps = np.abs(vec1_in_rho0_basis.reshape(dA * dR, -1))**2

    def q_of_s(s):
        pow0 = (eig0 > 1e-14).astype(float) if s == 0.0 else eig0**s
        pow1 = (eig1 > 1e-14).astype(float) if s == 1.0 else eig1**(1.0 - s)
        return float(pow0 @ overlaps @ pow1)

    result = minimize_scalar(
        q_of_s, bounds=(1e-6, 1.0 - 1e-6),
        method="bounded", options={"xatol": 1e-7}
    )

    Q = min(q_of_s(0.0), q_of_s(1.0), result.fun)

    if not (-1e-7 <= Q <= 1.0 + 1e-7):
        raise RuntimeError(f"Unphysical Chernoff quantity: {Q}")

    return float(np.clip(Q, 0.0, 1.0))


CACHE = {}


def calculate_zeta(kind, T=0.5):
    T = float(T)
    key = (kind, round(T, 12))

    if key in CACHE:
        return CACHE[key]

    psi = prepared_state(kind, T)
    Ns = signal_energy(psi)

    if Ns <= 1e-13:
        raise ValueError(f"Signal energy too small: {kind}, T={T}")

    rho0, rho1, rhoA, p0, dA, dR = build_hypotheses_pre_target(psi, eta=ETA)
    Q = quantum_chernoff(rho1, rhoA, p0, dA, dR)

    if not (0.0 < Q <= 1.0):
        raise RuntimeError(f"Invalid Q: {kind}, T={T}, Q={Q}")

    zeta = -np.log(Q) / Ns
    CACHE[key] = (zeta, Ns, Q)

    return CACHE[key]


zeta_TMSS, Ns_TMSS, Q_TMSS = calculate_zeta("TMSS", 0.5)

zeta_CI = ETA * KAPPA * (np.sqrt(NTH + 1.0) - np.sqrt(NTH))**2

zeta_fixed = {"TMSS": zeta_TMSS, "CI": zeta_CI}

print("\nREFERENCE EXPONENTS")
print(f"TMSS Ns = {Ns_TMSS:.9f}")
print(f"TMSS Q = {Q_TMSS:.12f}")
print(f"TMSS zeta = {zeta_TMSS:.12e}")
print(f"CI zeta = {zeta_CI:.12e}")

for kind in KINDS:
    T = T_fixed[kind]
    zeta, Ns, Q = calculate_zeta(kind, T)
    zeta_fixed[kind] = zeta
    print(f"{kind:5s} | T={T:.8f} | Ns={Ns:.9f} | Q={Q:.12f} | zeta={zeta:.10e}")


N_T = 15

T_grid = np.unique(np.concatenate([
    [1e-3],
    np.linspace(0.01, 0.99, N_T),
    np.linspace(0.43, 0.63, 15),
    [1.0 - 1e-3]
]))

zeta_vs_T = {}
Ns_vs_T = {}

for kind in KINDS:
    zeta_list = []
    Ns_list = []

    print(f"\nCalculating {kind}...")

    for i, T in enumerate(T_grid):
        zeta, Ns, Q = calculate_zeta(kind, T)
        zeta_list.append(zeta)
        Ns_list.append(Ns)

        print(
            f"{kind:5s} | {i+1:2d}/{len(T_grid)} | "
            f"T={T:.5f} | Ns={Ns:.7f} | zeta={zeta:.9e}"
        )

    zeta_vs_T[kind] = np.array(zeta_list)
    Ns_vs_T[kind] = np.array(Ns_list)


log10E = np.linspace(2.0, 6.0, 250)
E_vals = 10.0**log10E

log10Pe = {
    kind: np.log10(0.5) - E_vals * zeta_fixed[kind] / np.log(10.0)
    for kind in ORDER
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white"
})


def format_axes(ax):
    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = -1
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.7, width=0.6)


fig, axs = plt.subplots(1, 2, figsize=(3.1, 1.5), sharex=False)

for ax in axs:
    format_axes(ax)


ax = axs[0]

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    if kind in KINDS:
        ax.plot(T_grid, zeta_vs_T[kind], color=color, linestyle=linestyle,
                linewidth=1.1, zorder=3)

        i_best = np.argmax(zeta_vs_T[kind])

        ax.plot(T_grid[i_best], zeta_vs_T[kind][i_best], marker="x",
                color=color, markersize=3.5, markeredgewidth=0.9,
                linestyle="none", zorder=5)

    else:
        ax.axhline(zeta_fixed[kind], color=color, linestyle=linestyle,
                   linewidth=1.1, zorder=3)

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$\zeta$")
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(bottom=0)
ax.ticklabel_format(axis="y", style="plain", useOffset=False)

ax.text(0.05, 0.12, "(a)", transform=ax.transAxes,
        fontsize=7, ha="left", va="bottom")


ax = axs[1]

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    ax.plot(
        log10E, log10Pe[kind],
        color=color,
        linestyle=linestyle,
        linewidth=1.1,
        marker=marker,
        markersize=3.0 if marker else 0,
        markevery=18 if marker else None,
        markerfacecolor="white" if marker else None,
        markeredgecolor=color,
        markeredgewidth=0.85,
        zorder=3
    )

ax.set_xlabel(r"$\log_{10}(K N_S)$")
ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
ax.set_xlim(2, 6)
ax.set_ylim(-10, 0)
ax.set_xticks([2, 3, 4, 5, 6])
ax.set_yticks([-10, -5, 0])

ax.text(0.05, 0.12, "(b)", transform=ax.transAxes,
        fontsize=7, ha="left", va="bottom")


handles = []

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    handles.append(
        Line2D(
            [0], [0],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            marker=marker,
            markersize=3.2 if marker else 0,
            markerfacecolor="white" if marker else None,
            markeredgecolor=color,
            markeredgewidth=0.85,
            label=kind
        )
    )

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=6,
    frameon=False,
    fontsize=6.2,
    handlelength=1.7,
    handletextpad=0.35,
    columnspacing=0.9,
    labelspacing=0.35,
    borderaxespad=0.0
)

fig.subplots_adjust(
    left=0.17,
    right=0.94,
    bottom=0.22,
    top=0.80,
    wspace=0.48
)

fig.savefig("Fig_10.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
