
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from functools import lru_cache
from scipy.linalg import expm, eigh
from scipy.optimize import minimize_scalar
from matplotlib.lines import Line2D


n_max = 32
nE_max = n_max
Nth = 1.0
R = 0.01

r = np.arcsinh(np.sqrt(0.05))
lam = np.tanh(r)

IDLER_POP_TOL = 1e-15
EIGEN_TOL = 1e-13

T_fixed = {"NLPA": 0.001, "PA": 0.99999999, "PS": 0.001, "PC": 0.6022}
T_success = {"NLPA": 1e-6, "PA": 1e-6, "PS": 1e-3, "PC": 1.0 - 1e-6}

kinds = ["NLPA", "PA", "PS", "PC"]
order = ["NLPA", "PA", "PS", "PC", "TMSS", "CI"]

styles = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PS": ("#008A2E", "--", "^"),
    "PC": ("#0050FF", ":", "o"),
    "TMSS": ("#ff7f0e", ":", None),
    "CI": ("#AED814", "--", None)
}


def state_coefficients(kind, T=0.5):
    c = np.zeros((n_max, n_max), dtype=float)

    if kind == "TMSS":
        for n in range(n_max):
            c[n, n] = lam**n

    elif kind == "NLPA":
        for n in range(n_max - 1):
            amplitude = np.sqrt(n + 1) * (lam * T)**n
            c[n + 1, n] += amplitude
            c[n, n + 1] += amplitude

    elif kind == "PA":
        for n in range(n_max - 1):
            c[n, n + 1] = np.sqrt(n + 1) * (lam * np.sqrt(T))**n

    elif kind == "PS":
        for n in range(1, n_max):
            c[n, n - 1] = np.sqrt(n) * (lam * np.sqrt(T))**n

    elif kind == "PC":
        for n in range(n_max):
            c[n, n] = (n * (1.0 - T) - T) * (lam * np.sqrt(T))**n

    else:
        raise ValueError(f"Unknown protocol: {kind}")

    norm = np.linalg.norm(c)

    if norm <= 0:
        raise ValueError(f"Zero-norm state: {kind}, T={T}")

    return c / norm


def signal_energy(c):
    probabilities = np.sum(np.abs(c)**2, axis=0)
    return float(np.dot(np.arange(n_max), probabilities))


def thermal_weights(N, dim):
    n = np.arange(dim)
    p = (N / (1.0 + N))**n / (1.0 + N)
    return p / np.sum(p)


thermal0 = thermal_weights(Nth, nE_max)
thermal1 = thermal_weights(Nth / (1.0 - R), nE_max)


def beam_splitter_tensor(dim, reflectivity):
    xi = np.arctan(np.sqrt((1.0 - reflectivity) / reflectivity))
    U = np.zeros((dim, dim, dim, dim), dtype=float)

    for total in range(2 * dim - 1):
        b_values = np.arange(max(0, total - dim + 1), min(dim - 1, total) + 1)
        positions = {int(b): i for i, b in enumerate(b_values)}
        G = np.zeros((len(b_values), len(b_values)), dtype=float)

        for col, b in enumerate(b_values):
            b = int(b)
            e = total - b

            if b - 1 in positions:
                G[positions[b - 1], col] += np.sqrt(b * (e + 1))

            if b + 1 in positions:
                G[positions[b + 1], col] -= np.sqrt((b + 1) * e)

        U_block = expm(xi * G)

        for i, b_out in enumerate(b_values):
            e_out = total - int(b_out)

            for j, b_in in enumerate(b_values):
                e_in = total - int(b_in)
                U[int(b_out), e_out, int(b_in), e_in] = U_block[i, j]

    return U


print("\nBuilding beam-splitter transformation...")

U_BC = beam_splitter_tensor(n_max, R)
sqrt_thermal1 = np.sqrt(thermal1)

F = [
    (U_BC[:, :, n, :] * sqrt_thermal1[None, None, :]).reshape(n_max, -1)
    for n in range(n_max)
]


@lru_cache(maxsize=None)
def channel_on_fock_outer(n, m):
    return F[n] @ F[m].T


def build_hypotheses(c):
    idler_weights = np.sum(np.abs(c)**2, axis=1)
    active = np.flatnonzero(idler_weights > IDLER_POP_TOL)

    if len(active) == 0:
        raise ValueError("No active idler basis states.")

    c = c[active, :].copy()
    c /= np.linalg.norm(c)

    dA = len(active)
    dB = n_max
    rhoA = c @ c.T
    rho1 = np.zeros((dA * dB, dA * dB), dtype=float)
    supports = [np.flatnonzero(c[i] != 0.0) for i in range(dA)]

    for i in range(dA):
        for j in range(i, dA):
            block = np.zeros((dB, dB), dtype=float)

            for n in supports[i]:
                for m in supports[j]:
                    block += c[i, n] * c[j, m] * channel_on_fock_outer(int(n), int(m))

            i0 = i * dB
            j0 = j * dB
            rho1[i0:i0 + dB, j0:j0 + dB] = block

            if i != j:
                rho1[j0:j0 + dB, i0:i0 + dB] = block.T

    rho1 = 0.5 * (rho1 + rho1.T)
    rho1 /= np.trace(rho1)

    return rhoA, rho1, dA


def quantum_chernoff(rhoA, rho1, dA):
    eigA, vecA = eigh(rhoA, check_finite=False)
    eig1, vec1 = eigh(rho1, check_finite=False)

    eigA = np.maximum(eigA, 0.0)
    eig1 = np.maximum(eig1, 0.0)
    eig0 = (eigA[:, None] * thermal0[None, :]).reshape(-1)

    vec1_in_rho0_basis = np.einsum(
        "ak,ard->krd", vecA.conj(),
        vec1.reshape(dA, n_max, -1), optimize=True
    )

    mask0 = eig0 > EIGEN_TOL
    mask1 = eig1 > EIGEN_TOL
    a = eig0[mask0]
    b = eig1[mask1]
    overlap = np.abs(vec1_in_rho0_basis.reshape(dA * n_max, -1)[mask0][:, mask1])**2

    def Qs(s):
        return float((a**s) @ overlap @ (b**(1.0 - s)))

    result = minimize_scalar(
        Qs, bounds=(1e-7, 1.0 - 1e-7),
        method="bounded", options={"xatol": 1e-10}
    )

    Q = min(result.fun, Qs(1e-9), Qs(1.0 - 1e-9))

    if not np.isfinite(Q) or Q <= 0 or Q > 1.0 + 1e-8:
        raise ValueError(f"Invalid Chernoff quantity: {Q}")

    return float(np.clip(Q, 1e-300, 1.0))


cache = {}


def calculate_q(kind, T=0.5):
    T = float(T)
    key = (kind, round(T, 12))

    if key in cache:
        return cache[key]

    c = state_coefficients(kind, T)
    Ns = signal_energy(c)

    if Ns <= 0:
        raise ValueError(f"Invalid signal energy: {kind}, T={T}")

    rhoA, rho1, dA = build_hypotheses(c)
    Q = quantum_chernoff(rhoA, rho1, dA)
    cache[key] = (Q, Ns)

    return cache[key]


Q_TMSS, Ns_TMSS = calculate_q("TMSS", 0.5)
zeta_CI = R * (np.sqrt(Nth + 1.0) - np.sqrt(Nth))**2

print("\nREFERENCE EXPONENTS")
print(f"TMSS Ns = {Ns_TMSS:.9f}")
print(f"TMSS Q = {Q_TMSS:.12f}")
print(f"TMSS zeta = {-np.log1p(Q_TMSS - 1.0) / Ns_TMSS:.10e}")
print(f"CI zeta = {zeta_CI:.10e}")

zeta_a = {"TMSS": -np.log1p(Q_TMSS - 1.0) / Ns_TMSS, "CI": zeta_CI}
zeta_b = zeta_a.copy()

for label, config, results in [("MAXIMUM ENTANGLEMENT", T_fixed, zeta_a), ("MAXIMUM SUCCESS", T_success, zeta_b)]:
    print(f"\n{label}")

    for kind in kinds:
        Q, Ns = calculate_q(kind, config[kind])
        results[kind] = -np.log1p(Q - 1.0) / Ns
        print(f"{kind:5s} | T={config[kind]:.8f} | Ns={Ns:.9f} | Q={Q:.12f} | zeta={results[kind]:.10e}")


log10E = np.linspace(2.0, 6.0, 250)
E_vals = 10.0**log10E


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


fig, axs = plt.subplots(1, 2, figsize=(3.1, 1.5), sharex=False, sharey=False)

for ax in axs:
    format_axes(ax)

for ax, (letter, zeta_data) in zip(axs, [("a", zeta_a), ("b", zeta_b)]):
    for kind in order:
        color, linestyle, marker = styles[kind]
        y = np.log10(0.5) - E_vals * zeta_data[kind] / np.log(10.0)

        ax.plot(log10E, y, color=color, linestyle=linestyle,linewidth=1.1, marker=marker,markersize=3.0 if marker else 0,markevery=18 if marker else None,markerfacecolor="white" if marker else None,markeredgecolor=color, markeredgewidth=0.85, zorder=3)

    ax.set_xlabel(r"$\log_{10}(KN_S)$")
    ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
    ax.set_xlim(2, 6)
    ax.set_ylim(-10, 0)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_yticks([-10, -5, 0])
    ax.text(0.05, 0.12, f"({letter})", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")


handles = []

for kind in order:
    color, linestyle, marker = styles[kind]handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.1,marker=marker, markersize=3.2 if marker else 0,markerfacecolor="white" if marker else None,markeredgecolor=color, markeredgewidth=0.85, label=kind))

fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),ncol=6, frameon=False, fontsize=6.2, handlelength=1.7,handletextpad=0.35, columnspacing=0.9, labelspacing=0.35,borderaxespad=0.0)

fig.subplots_adjust(left=0.15, right=0.94, bottom=0.22, top=0.80, wspace=0.42)
fig.savefig("Fig_5.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
