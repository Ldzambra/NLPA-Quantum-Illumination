import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.linalg import expm, eigh
from scipy.optimize import minimize_scalar
from matplotlib.lines import Line2D


n_max = 24
Nth = 1.0
R = 0.01
r = np.arcsinh(np.sqrt(0.05))
lam = np.tanh(r)

T_dense = np.linspace(1e-6, 1.0 - 1e-6, 150)
T_mark = np.linspace(1e-6, 1.0 - 1e-6, 20)
T_ent = {"NLPA": 1.0 - 1e-6, "PA": 1.0 - 1e-6, "PS": 1.0 - 1e-6, "PC": 0.041}
T_succ = {"NLPA": 1e-6, "PA": 1e-6, "PS": 1e-6, "PC": 1.0 - 1e-6}

kinds = ["NLPA", "PA", "PS", "PC"]
order = ["NLPA", "PA", "PS", "PC", "TMSS"]

styles = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PS": ("#008A2E", "--", "^"),
    "PC": ("#0050FF", ":", "o"),
    "TMSS": ("#ff7f0e", ":", None)
}


def state_coefficients(kind, T=0.5):
    c = np.zeros((n_max, n_max), dtype=float)

    if kind == "TMSS":
        for n in range(n_max):
            c[n, n] = lam**n

    elif kind == "NLPA":
        for n in range(n_max - 1):
            amplitude = np.sqrt(n + 1.0) * (lam * T)**n
            c[n + 1, n] += amplitude
            c[n, n + 1] += amplitude

    elif kind == "PA":
        for n in range(n_max - 1):
            c[n, n + 1] = np.sqrt(n + 1.0) * (lam * np.sqrt(T))**n

    elif kind == "PS":
        for n in range(1, n_max):
            c[n, n - 1] = np.sqrt(n) * (lam * np.sqrt(T))**n

    elif kind == "PC":
        for n in range(n_max):
            c[n, n] = (n * (1.0 - T) - T) * (lam * np.sqrt(T))**n

    else:
        raise ValueError(f"Unknown state: {kind}")

    norm = np.linalg.norm(c)

    if norm <= 0:
        raise ValueError(f"Zero-norm state: {kind}, T={T}")

    return c / norm


def entropy_bits(c):
    values = np.linalg.svd(c, compute_uv=False)**2
    values = values[values > 1e-15]
    return float(-np.sum(values * np.log2(values)))


def success_probability(kind, T):
    lam2 = lam**2
    sech2 = 1.0 / np.cosh(r)**2
    x = lam2 * T

    if kind == "NLPA":
        return sech2 * (1.0 - T) / (1.0 - lam2 * T**2)**2

    if kind == "PA":
        return sech2 * (1.0 - T) / (1.0 - x)**2

    if kind == "PS":
        return sech2 * (1.0 - T) * lam2 / (1.0 - x)**2

    if kind == "PC":
        term1 = T**2 / (1.0 - x)
        term2 = -2.0 * T * (1.0 - T) * x / (1.0 - x)**2
        term3 = (1.0 - T)**2 * x * (1.0 + x) / (1.0 - x)**3
        return sech2 * (term1 + term2 + term3) / T

    if kind == "TMSS":
        return 1.0

    raise ValueError(kind)


def thermal_weights(mean, dim):
    n = np.arange(dim)
    weights = (mean / (1.0 + mean))**n / (1.0 + mean)
    return weights / weights.sum()


def beam_splitter_tensor(dim, reflectivity):
    xi = np.arctan(np.sqrt((1.0 - reflectivity) / reflectivity))
    U = np.zeros((dim, dim, dim, dim), dtype=float)

    for total in range(2 * dim - 1):
        levels = np.arange(max(0, total - dim + 1), min(dim - 1, total) + 1)
        positions = {int(b): i for i, b in enumerate(levels)}
        G = np.zeros((len(levels), len(levels)), dtype=float)

        for col, b in enumerate(levels):
            b = int(b)
            e = total - b

            if b - 1 in positions:
                G[positions[b - 1], col] += np.sqrt(b * (e + 1))

            if b + 1 in positions:
                G[positions[b + 1], col] -= np.sqrt((b + 1) * e)

        block = expm(xi * G)

        for i, b_out in enumerate(levels):
            e_out = total - int(b_out)

            for j, b_in in enumerate(levels):
                U[int(b_out), e_out, int(b_in), total - int(b_in)] = block[i, j]

    return U


print("Building beam-splitter transformation...")
U_BC = beam_splitter_tensor(n_max, R)
p0 = thermal_weights(Nth, n_max)
p1 = thermal_weights(Nth / (1.0 - R), n_max)


def calculate_q(kind, T):
    c = state_coefficients(kind, T)
    active = np.flatnonzero(np.sum(c**2, axis=1) > 1e-14)
    c = c[active, :].copy()
    c /= np.linalg.norm(c)
    dA = len(active)

    rhoA = c @ c.T
    states = np.einsum("ab,rcbj->arcj", c, U_BC, optimize=True)
    states *= np.sqrt(p1)[None, None, None, :]
    X = states.reshape(dA * n_max, n_max * n_max)
    rho1 = X @ X.T
    rho1 = 0.5 * (rho1 + rho1.T)
    rho1 /= np.trace(rho1)

    eigA, vecA = eigh(rhoA, check_finite=False)
    eig1, vec1 = eigh(rho1, check_finite=False)
    eig0 = (np.maximum(eigA, 0.0)[:, None] * p0[None, :]).reshape(-1)
    eig1 = np.maximum(eig1, 0.0)
    overlap_vectors = np.einsum("ak,ard->krd", vecA, vec1.reshape(dA, n_max, -1), optimize=True)

    mask0 = eig0 > 1e-13
    mask1 = eig1 > 1e-13
    a = eig0[mask0]
    b = eig1[mask1]
    overlap = overlap_vectors.reshape(dA * n_max, -1)[mask0][:, mask1]**2

    def q_of_s(s):
        return float((a**s) @ overlap @ (b**(1.0 - s)))

    result = minimize_scalar(q_of_s, bounds=(1e-7, 1.0 - 1e-7), method="bounded", options={"xatol": 1e-9})
    Q = min(result.fun, q_of_s(1e-9), q_of_s(1.0 - 1e-9))

    if not np.isfinite(Q) or Q <= 0.0 or Q > 1.0 + 1e-8:
        raise ValueError(f"Invalid QCB for {kind} at T={T}: {Q}")

    return float(min(Q, 1.0))


entropy = {kind: np.array([entropy_bits(state_coefficients(kind, T)) for T in T_dense]) for kind in kinds}
entropy_mark = {kind: np.array([entropy_bits(state_coefficients(kind, T)) for T in T_mark]) for kind in kinds}
success = {kind: 100.0 * np.array([success_probability(kind, T) for T in T_dense]) for kind in kinds}
success_mark = {kind: 100.0 * np.array([success_probability(kind, T) for T in T_mark]) for kind in kinds}
entropy_TMSS = entropy_bits(state_coefficients("TMSS"))

Q_TMSS = calculate_q("TMSS", 0.5)
Q_ent = {"TMSS": Q_TMSS}
Q_succ = {"TMSS": Q_TMSS}

for kind in kinds:
    Q_ent[kind] = calculate_q(kind, T_ent[kind])
    Q_succ[kind] = calculate_q(kind, T_succ[kind])
    print(f"{kind:5s} | Q_ent={Q_ent[kind]:.10f} | Q_succ={Q_succ[kind]:.10f}")

log10K = np.linspace(3.0, 6.0, 250)
K_vals = 10.0**log10K
log10Pe_ent = {kind: np.log10(0.5) + K_vals * np.log10(Q_ent[kind]) for kind in order}
log10Pe_succ = {kind: np.log10(0.5) + K_vals * np.log10(Q_succ[kind]) for kind in order}


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


def panel_label(ax, letter):
    ax.text(0.05, 0.12, f"({letter})", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")


def plot_dense_sparse(ax, y_dense, y_mark, kind):
    color, linestyle, marker = styles[kind]
    ax.plot(T_dense, y_dense, color=color, linestyle=linestyle, linewidth=1.1, zorder=2)
    ax.plot(T_mark, y_mark, color=color, linestyle="none", marker=marker, markersize=3.0,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.85, zorder=3)


def plot_qcb(ax, values):
    for kind in order:
        color, linestyle, marker = styles[kind]
        ax.plot(log10K, values[kind], color=color, linestyle=linestyle, linewidth=1.1,
                marker=marker, markersize=3.0 if marker else 0, markevery=18 if marker else None,
                markerfacecolor="white" if marker else None, markeredgecolor=color,
                markeredgewidth=0.85, zorder=3)


fig, axs = plt.subplots(2, 2, figsize=(3.1, 3.1), sharex=False, sharey=False)
axs = axs.ravel()

for ax in axs:
    format_axes(ax)

for kind in kinds:
    plot_dense_sparse(axs[0], entropy[kind], entropy_mark[kind], kind)
    plot_dense_sparse(axs[1], success[kind], success_mark[kind], kind)

axs[0].axhline(entropy_TMSS, color=styles["TMSS"][0], linestyle=styles["TMSS"][1], linewidth=1.1)
axs[0].set_xlabel(r"$T$")
axs[0].set_ylabel(r"$E_V$")
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1.2)
axs[0].set_xticks(np.arange(0, 1.01, 0.2))
axs[0].set_yticks(np.arange(0, 1.21, 0.2))
panel_label(axs[0], "a")

axs[1].axhline(100.0, color=styles["TMSS"][0], linestyle=styles["TMSS"][1], linewidth=1.1)
axs[1].set_xlabel(r"$T$")
axs[1].set_ylabel(r"$P_{\mathrm{succ}}\ (\%)$")
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 105)
axs[1].set_xticks(np.arange(0, 1.01, 0.2))
axs[1].set_yticks(np.arange(0, 101, 20))
panel_label(axs[1], "b")

for ax, values, letter in [(axs[2], log10Pe_ent, "c"), (axs[3], log10Pe_succ, "d")]:
    plot_qcb(ax, values)
    ax.set_xlabel(r"$\log_{10}K$")
    ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
    ax.set_xlim(3, 6)
    ax.set_ylim(-10, 0)
    ax.set_xticks([3, 4, 5, 6])
    ax.set_yticks([-10, -5, 0])
    panel_label(ax, letter)

handles = []

for kind in order:
    color, linestyle, marker = styles[kind]
    handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.1,marker=marker, markersize=3.2 if marker else 0,markerfacecolor="white" if marker else None,markeredgecolor=color, markeredgewidth=0.85, label=kind))

fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5,frameon=False, fontsize=6.2, handlelength=1.7, handletextpad=0.35,columnspacing=0.9, labelspacing=0.35, borderaxespad=0.0)

fig.subplots_adjust(top=0.93, bottom=0.08, left=0.12, right=0.95, hspace=0.19, wspace=0.31)
fig.savefig("Fig_3.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
