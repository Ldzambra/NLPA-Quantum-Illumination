
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from scipy.linalg import expm, eigh
from scipy.optimize import minimize_scalar


DIM = int(os.environ.get("QI_DIM", "32"))
DIM_ENV = DIM

NTH = 1.0
KAPPA = 0.01

R_SQUEEZING = np.arcsinh(np.sqrt(0.05))
LAM = np.tanh(R_SQUEEZING)

T_MIN = 1e-6
T_MAX = 1.0 - 1e-6
T_DENSE = np.linspace(T_MIN, T_MAX, 150)

T_ENT = {"NLPA": T_MAX, "PA": T_MAX, "PS": T_MAX, "PC": 0.125}
T_SUCC = {"NLPA": T_MIN, "PA": T_MIN, "PS": T_MIN, "PC": T_MAX}

KINDS = ["NLPA", "PA", "PS", "PC"]
ORDER = ["NLPA", "PA", "PS", "PC", "TMSS"]

STYLES = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PS": ("#008A2E", "--", "^"),
    "PC": ("#0050FF", ":", "o"),
    "TMSS": ("#ff7f0e", ":", None)
}


def state_coefficients(kind, T, dim=DIM):
    c = np.zeros((dim, dim), dtype=np.complex128)
    n = np.arange(dim)
    tmss = np.sqrt(1.0 - LAM**2) * LAM**n

    if kind == "TMSS":
        c[n, n] = tmss

    elif kind == "NLPA":
        x = LAM**2 * T**2

        for k in range(dim - 2):
            amplitude = 0.5 * (1.0 - x)**1.5 * (LAM * T)**k * np.sqrt((k + 1) * (k + 2))
            c[k + 2, k] = amplitude
            c[k, k + 2] = -amplitude

    elif kind == "PA":
        c[n[1:], n[1:]] = tmss[:-1] * (1.0 - T) * n[1:] * T**n[:-1]

    elif kind == "PS":
        c[n[:-1], n[:-1]] = tmss[1:] * (1.0 - T) * n[1:] * T**n[:-1]

    elif kind == "PC":
        c[n, n] = tmss * T**(n - 1) * ((1.0 - T) * n - T)**2

    else:
        raise ValueError(f"Unknown protocol: {kind}")

    norm = np.linalg.norm(c)

    if norm <= 0:
        raise ValueError(f"Zero state norm for {kind} at T={T}")

    return c / norm


def entropy_bits(c):
    probabilities = np.linalg.svd(c, compute_uv=False)**2
    probabilities = probabilities[probabilities > 1e-15]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def success_probability(kind, T):
    l2 = LAM**2
    x = l2 * T**2
    pref = 1.0 - l2

    if kind == "NLPA":
        return pref * (1.0 - T)**2 / (1.0 - x)**3

    elif kind == "PA":
        return pref * (1.0 - T)**2 * (1.0 + x) / (1.0 - x)**3

    elif kind == "PS":
        return pref * (1.0 - T)**2 * l2 * (1.0 + x) / (1.0 - x)**3

    elif kind == "PC":
        S0 = 1.0 / (1.0 - x)
        S1 = x / (1.0 - x)**2
        S2 = x * (1.0 + x) / (1.0 - x)**3
        S3 = x * (1.0 + 4.0*x + x**2) / (1.0 - x)**4
        S4 = x * (1.0 + 11.0*x + 11.0*x**2 + x**3) / (1.0 - x)**5

        terms = (T**4 * S0 - 4.0*T**3 * (1.0 - T) * S1
                 + 6.0*T**2 * (1.0 - T)**2 * S2
                 - 4.0*T * (1.0 - T)**3 * S3
                 + (1.0 - T)**4 * S4)

        return pref * terms / T**2

    elif kind == "TMSS":
        return 1.0

    else:
        raise ValueError(f"Unknown protocol: {kind}")


def annihilation(dim):
    a = np.zeros((dim, dim), dtype=np.complex128)

    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)

    return a


def thermal_weights(mean, dim):
    n = np.arange(dim)
    weights = (mean / (1.0 + mean))**n / (1.0 + mean)
    return weights / weights.sum()


def beam_splitter_unitary():
    a = annihilation(DIM)
    b = annihilation(DIM_ENV)

    xi = np.arctan(np.sqrt((1.0 - KAPPA) / KAPPA))
    G = xi * (np.kron(a, b.conj().T) - np.kron(a.conj().T, b))

    print("Building target beam splitter...")

    return expm(G).reshape(DIM, DIM_ENV, DIM, DIM_ENV)


def hypothesis_states(coeff, U4, p_env0, p_env1):
    row_p = np.sum(np.abs(coeff)**2, axis=1)
    relevant_rows = np.flatnonzero(row_p > 1e-14)

    c = coeff[relevant_rows, :]
    c /= np.linalg.norm(c)

    idim = len(relevant_rows)

    rho_idler = c @ c.conj().T
    rho0 = np.kron(rho_idler, np.diag(p_env0))

    states = np.einsum("ab,rcbj->arcj", c, U4, optimize=True)
    states *= np.sqrt(p_env1)[None, None, None, :]

    X = states.reshape(idim * DIM, DIM_ENV * DIM_ENV)
    rho1 = X @ X.conj().T

    return rho0, rho1, rho_idler


def chernoff_quantity(rho0, rho1, rho_idler, p_env0):
    eigA, vecA = eigh(rho_idler, check_finite=False)
    eig0 = np.outer(np.maximum(eigA, 0.0), p_env0).reshape(-1)
    vec0 = np.kron(vecA, np.eye(DIM_ENV))

    eig1, vec1 = eigh(rho1, check_finite=False)
    eig1 = np.maximum(eig1, 0.0)

    good0 = eig0 > 1e-14
    good1 = eig1 > 1e-14

    lam0 = eig0[good0]
    lam1 = eig1[good1]

    overlap = np.abs(vec0[:, good0].conj().T @ vec1[:, good1])**2

    def f(s):
        return float((lam0**s) @ overlap @ (lam1**(1.0 - s)))

    result = minimize_scalar(f, bounds=(1e-8, 1.0 - 1e-8),method="bounded", options={"xatol": 1e-7})

    Q = min(f(1e-8), result.fun, f(1.0 - 1e-8))

    if Q <= 0 or Q > 1.0 + 1e-8:
        raise ValueError(f"Unphysical Chernoff value Q={Q}")

    return min(Q, 1.0), result.x


def calculate_q(kind, T, U4, p_env0, p_env1):
    coeff = state_coefficients(kind, T)
    rho0, rho1, rho_idler = hypothesis_states(coeff, U4, p_env0, p_env1)

    Q, s = chernoff_quantity(rho0, rho1, rho_idler, p_env0)

    print(f"{kind:5s} | T={T:.6f} | Q={Q:.10f} | s={s:.5f}", flush=True)

    return Q


def log10_error_bound(Q, K):
    return np.log10(0.5) + K * np.log10(Q)


entanglement = {}
success = {}

for kind in KINDS:
    entanglement[kind] = np.array([
        entropy_bits(state_coefficients(kind, T)) for T in T_DENSE
    ])

    success[kind] = np.array([
        100.0 * success_probability(kind, T) for T in T_DENSE
    ])

entanglement_TMSS = entropy_bits(state_coefficients("TMSS", 0.5))

U4 = beam_splitter_unitary()
p_env0 = thermal_weights(NTH, DIM_ENV)
p_env1 = thermal_weights(NTH / (1.0 - KAPPA), DIM_ENV)

print("\nTMSS reference")
Q_TMSS = calculate_q("TMSS", 0.5, U4, p_env0, p_env1)

Q_ENT = {"TMSS": Q_TMSS}
Q_SUCC = {"TMSS": Q_TMSS}

print("\nMaximum-entanglement configuration")

for kind in KINDS:
    Q_ENT[kind] = calculate_q(kind, T_ENT[kind], U4, p_env0, p_env1)

print("\nMaximum-success configuration")

for kind in KINDS:
    Q_SUCC[kind] = calculate_q(kind, T_SUCC[kind], U4, p_env0, p_env1)


log10K = np.linspace(3.0, 6.0, 250)
K_vals = 10.0**log10K

log10Pe_ENT = {kind: log10_error_bound(Q_ENT[kind], K_vals) for kind in ORDER}
log10Pe_SUCC = {kind: log10_error_bound(Q_SUCC[kind], K_vals) for kind in ORDER}


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


def panel_label(ax, letter):
    ax.text(0.05, 0.12, f"({letter})", transform=ax.transAxes,
            fontsize=7, ha="left", va="bottom")


def format_axes(ax):
    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = -1
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.7, width=0.6)


def plot_curves(ax, x, data):
    for kind in ORDER:
        color, linestyle, marker = STYLES[kind]

        ax.plot(x, data[kind],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.0 if marker else 0,markevery=8 if marker else None,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,zorder=3)


fig, axs = plt.subplots(2, 2, figsize=(3.1, 3.1), sharex=False, sharey=False)
axs = axs.ravel()

for ax in axs:
    format_axes(ax)


ax = axs[0]

for kind in KINDS:
    color, linestyle, marker = STYLES[kind]

    ax.plot(T_DENSE, entanglement[kind],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.0 if marker else 0,markevery=8 if marker else None,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,zorder=3)

ax.axhline(entanglement_TMSS, color=STYLES["TMSS"][0],linestyle=STYLES["TMSS"][1], linewidth=1.1)

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$E_V$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.6)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.61, 0.4))

panel_label(ax, "a")


ax = axs[1]

for kind in KINDS:
    color, linestyle, marker = STYLES[kind]

    ax.plot(T_DENSE, success[kind],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.0 if marker else 0,markevery=8 if marker else None,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,zorder=3)

ax.axhline(100.0, color=STYLES["TMSS"][0],linestyle=STYLES["TMSS"][1], linewidth=1.1)

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$P_{\mathrm{succ}}\ (\%)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 105)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 101, 20))

panel_label(ax, "b")


ax = axs[2]

plot_curves(ax, log10K, log10Pe_ENT)

ax.set_xlabel(r"$\log_{10}K$")
ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
ax.set_xlim(3, 6)
ax.set_ylim(-10, 0)
ax.set_xticks([3, 4, 5, 6])
ax.set_yticks([-10, -5, 0])

panel_label(ax, "c")


ax = axs[3]

plot_curves(ax, log10K, log10Pe_SUCC)

ax.set_xlabel(r"$\log_{10}K$")
ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
ax.set_xlim(3, 6)
ax.set_ylim(-10, 0)
ax.set_xticks([3, 4, 5, 6])
ax.set_yticks([-10, -5, 0])

panel_label(ax, "d")


handles = []

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    handles.append(Line2D([0], [0],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.2 if marker else 0,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,label=kind))

fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5, 0.98),ncol=5,frameon=False,fontsize=6.2,handlelength=1.7,handletextpad=0.35,columnspacing=0.9,labelspacing=0.35,borderaxespad=0.0)

fig.subplots_adjust(top=0.93,bottom=0.08,left=0.12,right=0.95,hspace=0.19,wspace=0.31)

fig.savefig("Fig_7.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
