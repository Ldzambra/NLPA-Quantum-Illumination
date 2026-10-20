
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from matplotlib.lines import Line2D
from functools import lru_cache


n_max = 32
nE_max = n_max

Nth = 1.0
R = 0.01

r = np.arcsinh(np.sqrt(0.05))
lam = np.tanh(r)

IDLER_POP_TOL = 1e-15
EIGEN_TOL = 1e-15

T_grid = np.unique(np.concatenate([
    [0.001],
    np.linspace(0.01, 0.99, 15),
    np.linspace(0.43, 0.63, 15),
    [0.999]
]))

REFINE_OPTIMUM = True
PE_MODE = "optimized"

T_manual = {"NLPA": 0.001, "PA": 0.999, "PS": 0.001, "PC": 0.6022}

log10E_vals = np.linspace(2.0, 6.0, 250)
E_vals = 10.0**log10E_vals

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
        for k in range(n_max - 2):
            amp = (lam * T)**k * np.sqrt((k + 1) * (k + 2))
            c[k + 2, k] += amp
            c[k, k + 2] -= amp

    elif kind == "PA":
        for n in range(n_max - 1):
            c[n + 1, n + 1] = (n + 1) * (lam * T)**n

    elif kind == "PS":
        for n in range(n_max - 1):
            c[n, n] = (n + 1) * (lam * T)**n

    elif kind == "PC":
        for n in range(n_max):
            factor = T**(n + 1) * (((1.0 - T) / T) * n - 1.0)**2
            c[n, n] = lam**n * factor

    else:
        raise ValueError(f"Unknown protocol: {kind}")

    norm = np.linalg.norm(c)

    if norm <= 0:
        raise ValueError(f"Zero-norm state: {kind}, T={T}")

    return c / norm


def signal_energy(c):
    probabilities = np.sum(np.abs(c)**2, axis=0)
    return float(np.dot(np.arange(n_max), probabilities))


def Ns_NLPA_two_analytic(T):
    x = lam**2 * T**2
    return 1.0 + 3.0 * x / (1.0 - x)


def thermal_weights(N, dim):
    n = np.arange(dim)
    p = (1.0 / (N + 1.0)) * (N / (N + 1.0))**n
    return p / np.sum(p)


thermal0 = thermal_weights(Nth, nE_max)
Nth_prime = Nth / (1.0 - R)
thermal1 = thermal_weights(Nth_prime, nE_max)


def beam_splitter_tensor(dim, reflectivity):
    xi = np.arctan(np.sqrt((1.0 - reflectivity) / reflectivity))
    U = np.zeros((dim, dim, dim, dim), dtype=float)

    for total in range(2 * dim - 1):
        b_values = np.arange(max(0, total - (dim - 1)), min(dim - 1, total) + 1)
        size = len(b_values)

        if size == 0:
            continue

        positions = {int(b): i for i, b in enumerate(b_values)}
        G = np.zeros((size, size), dtype=float)

        for col, b in enumerate(b_values):
            b = int(b)
            e = total - b

            if b - 1 in positions:
                row = positions[b - 1]
                G[row, col] += np.sqrt(b * (e + 1))

            if b + 1 in positions:
                row = positions[b + 1]
                G[row, col] -= np.sqrt((b + 1) * e)

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

F = []

for n in range(n_max):
    F_n = U_BC[:, :, n, :] * sqrt_thermal1[None, None, :]
    F.append(F_n.reshape(n_max, -1))


@lru_cache(maxsize=None)
def channel_on_fock_outer(n, m):
    return F[n] @ F[m].T


def build_rho0_rho1(c):
    idler_weights = np.sum(np.abs(c)**2, axis=1)
    active = np.flatnonzero(idler_weights > IDLER_POP_TOL)

    if len(active) == 0:
        raise ValueError("No active idler basis states.")

    c = c[active, :].copy()
    c /= np.linalg.norm(c)

    dA = len(active)
    dB = n_max

    rhoA = c @ c.T
    rho0 = np.kron(rhoA, np.diag(thermal0))
    rho1 = np.zeros((dA * dB, dA * dB), dtype=float)

    supports = [np.flatnonzero(c[i] != 0.0) for i in range(dA)]

    for i in range(dA):
        si = supports[i]

        for j in range(i, dA):
            sj = supports[j]
            block = np.zeros((dB, dB), dtype=float)

            for n in si:
                for m in sj:
                    block += c[i, n] * c[j, m] * channel_on_fock_outer(int(n), int(m))

            i0 = i * dB
            j0 = j * dB

            rho1[i0:i0 + dB, j0:j0 + dB] = block

            if i != j:
                rho1[j0:j0 + dB, i0:i0 + dB] = block.T

    rho0 = 0.5 * (rho0 + rho0.T)
    rho1 = 0.5 * (rho1 + rho1.T)

    rho0 /= np.trace(rho0)
    rho1 /= np.trace(rho1)

    return rho0, rho1


def qcb_Q_from_states(rho0, rho1):
    e0, V0 = np.linalg.eigh(rho0)
    e1, V1 = np.linalg.eigh(rho1)

    e0 = np.maximum(e0, 0.0)
    e1 = np.maximum(e1, 0.0)

    mask0 = e0 > EIGEN_TOL
    mask1 = e1 > EIGEN_TOL

    a = e0[mask0]
    b = e1[mask1]

    overlap = (V0[:, mask0].T @ V1[:, mask1])**2

    def Qs(s):
        return float((a**s) @ overlap @ (b**(1.0 - s)))

    result = minimize_scalar(
        Qs, bounds=(1e-7, 1.0 - 1e-7),
        method="bounded", options={"xatol": 1e-9}
    )

    Q = min(result.fun, Qs(1e-9), Qs(1.0 - 1e-9))

    if not np.isfinite(Q):
        raise ValueError("Invalid numerical QCB.")

    if Q > 1.0 + 1e-8:
        raise ValueError(f"QCB exceeds unity: {Q}")

    return float(np.clip(Q, 1e-300, 1.0))


cache = {}


def calculate_zeta(kind, T):
    T = float(T)

    if not (0.0 < T < 1.0):
        raise ValueError("Require 0 < T < 1")

    key = (kind, round(T, 12))

    if key in cache:
        return cache[key]

    c = state_coefficients(kind, T)
    Ns = signal_energy(c)

    if Ns <= 0.0:
        raise ValueError(f"Invalid signal energy for {kind}")

    if Ns < 1e-5:
        warnings.warn(
            f"{kind} at T={T:.6g} has very small Ns={Ns:.3e}. "
            "Verify QCB convergence before interpreting zeta."
        )

    rho0, rho1 = build_rho0_rho1(c)
    Q = qcb_Q_from_states(rho0, rho1)
    zeta = -np.log1p(Q - 1.0) / Ns

    cache[key] = (zeta, Ns, Q)

    return cache[key]


c_tmss = state_coefficients("TMSS")
Ns_TMSS = signal_energy(c_tmss)

rho0_tmss, rho1_tmss = build_rho0_rho1(c_tmss)
Q_TMSS = qcb_Q_from_states(rho0_tmss, rho1_tmss)

zeta_TMSS = -np.log1p(Q_TMSS - 1.0) / Ns_TMSS
zeta_CI = R * (np.sqrt(Nth + 1.0) - np.sqrt(Nth))**2

print("\nFIXED SOURCE PARAMETERS")
print(f"n_max = {n_max}")
print(f"sinh^2(r) = {np.sinh(r)**2:.7f}")
print(f"lambda = {lam:.9f}")
print(f"Nth = {Nth}")
print(f"kappa = {R}")
print(f"TMSS Ns = {Ns_TMSS:.9f}")
print(f"TMSS zeta = {zeta_TMSS:.10e}")
print(f"CI zeta = {zeta_CI:.10e}")


zeta_vs_T = {}
Ns_vs_T = {}

for kind in kinds:
    zeta_list = []
    Ns_list = []

    print(f"\nCalculating {kind}...")

    for i, T in enumerate(T_grid):
        zeta, Ns, Q = calculate_zeta(kind, T)
        zeta_list.append(zeta)
        Ns_list.append(Ns)

        print(
            f"{kind:5s} | {i+1:2d}/{len(T_grid)} | "
            f"T={T:.6f} | Ns={Ns:.7e} | zeta={zeta:.9e}"
        )

    zeta_vs_T[kind] = np.array(zeta_list)
    Ns_vs_T[kind] = np.array(Ns_list)


optimal_results = {}

print("\nOPTIMAL TWO-PHOTON TRANSMISSIVITIES")

for kind in kinds:
    values = zeta_vs_T[kind]
    i_best = int(np.argmax(values))

    T_best = float(T_grid[i_best])
    zeta_best = float(values[i_best])

    if REFINE_OPTIMUM and 0 < i_best < len(T_grid) - 1:
        left = T_grid[i_best - 1]
        right = T_grid[i_best + 1]

        result = minimize_scalar(
            lambda T: -calculate_zeta(kind, T)[0],
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 25}
        )

        if result.success and -result.fun > zeta_best:
            T_best = float(result.x)
            zeta_best = float(-result.fun)

    optimal_results[kind] = (T_best, zeta_best)

    print(
        f"{kind:5s} | T_opt={T_best:.8f} | "
        f"zeta_max={zeta_best:.10e} | "
        f"zeta/TMSS={zeta_best/zeta_TMSS:.7f} | "
        f"zeta/CI={zeta_best/zeta_CI:.7f}"
    )


if PE_MODE == "manual":
    T_error = T_manual.copy()

elif PE_MODE == "optimized":
    T_error = {kind: optimal_results[kind][0] for kind in kinds}

else:
    raise ValueError("PE_MODE must be 'manual' or 'optimized'")


Q_error = {"TMSS": Q_TMSS}
Ns_error = {"TMSS": Ns_TMSS}
zeta_error = {"TMSS": zeta_TMSS, "CI": zeta_CI}

print(f"\nTRANSMISSIVITIES FOR ERROR FIGURE: {PE_MODE}")

for kind in kinds:
    T = T_error[kind]
    zeta, Ns, Q = calculate_zeta(kind, T)

    Q_error[kind] = Q
    Ns_error[kind] = Ns
    zeta_error[kind] = zeta

    print(
        f"{kind:5s} | T={T:.8f} | Ns={Ns:.9e} | "
        f"Q={Q:.10f} | zeta={zeta:.10e}"
    )


log10Pe = {
    kind: np.log10(0.5) - zeta_error[kind] * E_vals / np.log(10.0)
    for kind in order
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


fig, axs = plt.subplots(1, 2, figsize=(3.1, 1.5), sharex=False, sharey=False)

for ax in axs:
    format_axes(ax)


ax = axs[0]

for kind in order:
    color, linestyle, marker = styles[kind]

    if kind in kinds:
        ax.plot(
            T_grid, zeta_vs_T[kind],
            color=color, linestyle=linestyle,
            linewidth=1.1, zorder=3
        )

        T_opt, zeta_opt = optimal_results[kind]

        ax.plot(
            T_opt, zeta_opt,
            marker="x", color=color,
            markersize=3.5, markeredgewidth=0.9,
            linestyle="none", zorder=5
        )

    else:
        reference = zeta_TMSS if kind == "TMSS" else zeta_CI

        ax.axhline(
            reference, color=color,
            linestyle=linestyle,
            linewidth=1.1, zorder=3
        )

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$\zeta$")
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(bottom=0)
ax.ticklabel_format(axis="y", style="plain", useOffset=False)

ax.text(0.05, 0.12, "(a)", transform=ax.transAxes,fontsize=7, ha="left", va="bottom")


ax = axs[1]

for kind in order:
    color, linestyle, marker = styles[kind]

    ax.plot(log10E_vals, log10Pe[kind],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.0 if marker else 0,markevery=18 if marker else None,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,zorder=3)

ax.set_xlabel(r"$\log_{10}(K N_S)$")
ax.set_ylabel(r"$\log_{10}P_{\mathrm{error}}$")
ax.set_xlim(2, 6)
ax.set_ylim(-10, 0)
ax.set_xticks([2, 3, 4, 5, 6])
ax.set_yticks([-10, -5, 0])

ax.text(0.05, 0.12, "(b)", transform=ax.transAxes,fontsize=7, ha="left", va="bottom")


handles = []

for kind in order:
    color, linestyle, marker = styles[kind]

    handles.append(Line2D([0], [0],color=color,linestyle=linestyle,linewidth=1.1,marker=marker,markersize=3.2 if marker else 0,markerfacecolor="white" if marker else None,markeredgecolor=color,markeredgewidth=0.85,label=kind))

fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5, 0.95),ncol=6,frameon=False,fontsize=6.2,handlelength=1.7,handletextpad=0.35,columnspacing=0.9,labelspacing=0.35,borderaxespad=0.0)

fig.subplots_adjust(left=0.17,right=0.94,bottom=0.22,top=0.80,wspace=0.48)

fig.savefig("Fig_8.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
