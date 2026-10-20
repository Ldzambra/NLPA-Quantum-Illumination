
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from scipy.linalg import expm, eigh
from scipy.optimize import minimize_scalar


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.labelsize": 7,
    "font.size": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white"
})

STYLES = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PS": ("#008A2E", "--", "^"),
    "PC": ("#0050FF", ":", "o"),
    "TMSS": ("#ff7f0e", ":", None)
}

NTH = 1.0
KAPPA = 0.01

r = np.arcsinh(np.sqrt(0.05))
LAM = np.tanh(r)

D_LIST = np.array([12, 16, 20, 24, 28, 32, 36, 40], dtype=int)
D_REFERENCE = 32

T_SINGLE = {"NLPA": 1.0 - 1e-6, "PA": 1.0 - 1e-6, "PS": 1.0 - 1e-6, "PC": 0.041}
T_DOUBLE = {"NLPA": 1.0 - 1e-6, "PA": 1.0 - 1e-6, "PS": 1.0 - 1e-6, "PC": 0.125}

ORDER = ["NLPA", "PA", "PS", "PC", "TMSS"]
SUPPORT_TOL = 1e-14


def normalized_coefficients(C):
    norm = np.linalg.norm(C)

    if norm <= 0.0:
        raise ValueError("Zero state norm.")

    return C / norm


def state_coefficients(kind, T, D, auxiliary=1):
    C = np.zeros((D, D), dtype=np.complex128)
    n = np.arange(D)
    tmss = np.sqrt(1.0 - LAM**2) * LAM**n

    if kind == "TMSS":
        C[n, n] = tmss
        return normalized_coefficients(C)

    if auxiliary == 1:

        if kind == "NLPA":
            x = LAM**2 * T**2

            for k in range(D - 1):
                amplitude = (1.0 - x) * (LAM * T)**k * np.sqrt(k + 1.0) / np.sqrt(2.0)
                C[k + 1, k] = amplitude
                C[k, k + 1] = amplitude

        elif kind == "PA":
            for k in range(D - 1):
                amplitude = tmss[k] * np.sqrt(1.0 - T) * np.sqrt(k + 1.0) * T**(k / 2.0)
                C[k, k + 1] = amplitude

        elif kind == "PS":
            for k in range(1, D):
                amplitude = tmss[k] * np.sqrt(1.0 - T) * np.sqrt(k) * T**((k - 1.0) / 2.0)
                C[k, k - 1] = amplitude

        elif kind == "PC":
            amplitude = tmss * T**((n - 1.0) / 2.0) * ((1.0 - T) * n - T)
            C[n, n] = amplitude

        else:
            raise ValueError(f"Unknown protocol: {kind}")

    elif auxiliary == 2:

        if kind == "NLPA":
            x = LAM**2 * T**2

            for k in range(D - 2):
                amplitude = 0.5 * (1.0 - x)**1.5 * (LAM * T)**k * np.sqrt((k + 1.0) * (k + 2.0))
                C[k + 2, k] = amplitude
                C[k, k + 2] = -amplitude

        elif kind == "PA":
            C[n[1:], n[1:]] = tmss[:-1] * (1.0 - T) * n[1:] * T**n[:-1]

        elif kind == "PS":
            C[n[:-1], n[:-1]] = tmss[1:] * (1.0 - T) * n[1:] * T**n[:-1]

        elif kind == "PC":
            C[n, n] = tmss * T**(n - 1) * ((1.0 - T) * n - T)**2

        else:
            raise ValueError(f"Unknown protocol: {kind}")

    else:
        raise ValueError("auxiliary must be 1 or 2.")

    return normalized_coefficients(C)


def mean_signal_photons(C):
    D = C.shape[1]
    photon_numbers = np.arange(D)
    probabilities = np.abs(C)**2
    Ns = np.sum(probabilities * photon_numbers[None, :])

    return float(np.real(Ns))


def annihilation(D):
    a = np.zeros((D, D), dtype=np.complex128)

    for n in range(1, D):
        a[n - 1, n] = np.sqrt(n)

    return a


def thermal_weights(mean, D):
    n = np.arange(D)
    weights = (mean / (1.0 + mean))**n / (1.0 + mean)

    return weights / np.sum(weights)


def beam_splitter_unitary(D):
    a = annihilation(D)
    b = annihilation(D)

    xi = np.arctan(np.sqrt((1.0 - KAPPA) / KAPPA))
    generator = xi * (np.kron(a, b.conj().T) - np.kron(a.conj().T, b))
    U = expm(generator)

    return U.reshape(D, D, D, D)


def hypothesis_states(C, U4, p_env0, p_env1):
    D = C.shape[1]
    row_probabilities = np.sum(np.abs(C)**2, axis=1)
    relevant_rows = np.flatnonzero(row_probabilities > SUPPORT_TOL)
    discarded_weight = np.sum(row_probabilities[row_probabilities <= SUPPORT_TOL])

    C = C[relevant_rows, :]
    C = C / np.linalg.norm(C)

    idim = len(relevant_rows)

    rho_idler = C @ C.conj().T
    rho0 = np.kron(rho_idler, np.diag(p_env0))

    states = np.einsum("ab,rcbj->arcj", C, U4, optimize=True)
    states *= np.sqrt(p_env1)[None, None, None, :]

    X = states.reshape(idim * D, D * D)
    rho1 = X @ X.conj().T
    rho1 = 0.5 * (rho1 + rho1.conj().T)

    trace0 = np.real(np.trace(rho0))
    trace1 = np.real(np.trace(rho1))

    if abs(trace0 - 1.0) > 1e-8:
        raise ValueError(f"Invalid H0 trace: {trace0}")

    if abs(trace1 - 1.0) > 1e-8:
        raise ValueError(f"Invalid H1 trace: {trace1}")

    return rho0, rho1, rho_idler, float(discarded_weight)


def chernoff_quantity(rho0, rho1, rho_idler, p_env0):
    D = len(p_env0)

    eigI, vecI = eigh(rho_idler, check_finite=False)
    eigI = np.maximum(eigI, 0.0)
    eig0 = np.outer(eigI, p_env0).reshape(-1)
    vec0 = np.kron(vecI, np.eye(D))

    eig1, vec1 = eigh(rho1, check_finite=False)

    if np.min(eig1) < -1e-9:
        raise ValueError("Significantly negative density-matrix eigenvalue.")

    eig1 = np.maximum(eig1, 0.0)

    good0 = eig0 > SUPPORT_TOL
    good1 = eig1 > SUPPORT_TOL

    lam0 = eig0[good0]
    lam1 = eig1[good1]

    overlap = np.abs(vec0[:, good0].conj().T @ vec1[:, good1])**2

    def Q_of_s(s):
        left = lam0**(1.0 - s)
        right = lam1**s
        return float(left @ overlap @ right)

    eps = 1e-8

    result = minimize_scalar(Q_of_s, bounds=(eps, 1.0 - eps), method="bounded", options={"xatol": 1e-8})

    candidates = [(eps, Q_of_s(eps)), (float(result.x), float(result.fun)), (1.0 - eps, Q_of_s(1.0 - eps))]

    s_opt, Q = min(candidates, key=lambda item: item[1])

    if Q <= 0.0 or Q > 1.0 + 1e-8:
        raise ValueError(f"Unphysical Chernoff quantity Q={Q}")

    return min(Q, 1.0), s_opt


def calculate_zeta(kind, T, D, auxiliary, U4, p_env0, p_env1):
    C = state_coefficients(kind, T, D, auxiliary=auxiliary)
    Ns = mean_signal_photons(C)

    if Ns <= 0.0:
        raise ValueError(f"Zero signal energy: {kind}, D={D}")

    rho0, rho1, rhoI, discarded = hypothesis_states(C, U4, p_env0, p_env1)
    Q, s_opt = chernoff_quantity(rho0, rho1, rhoI, p_env0)
    zeta = -np.log(Q) / Ns

    print(f"  {kind:5s} | T={T:.6f} | Ns={Ns:.8f} | Q={Q:.12f} | s={s_opt:.6f} | zeta={zeta:.10e} | discarded={discarded:.2e}", flush=True)

    return zeta


RESULTS_SINGLE = {kind: [] for kind in ORDER}
RESULTS_DOUBLE = {kind: [] for kind in ORDER}


for D in D_LIST:
    D = int(D)

    print("\n" + "=" * 60)
    print(f"FOCK-SPACE DIMENSION D = {D}")
    print("=" * 60, flush=True)

    D_ENV = D

    print("Building target beam-splitter unitary...", flush=True)

    U4 = beam_splitter_unitary(D)

    p_env0 = thermal_weights(NTH, D_ENV)
    p_env1 = thermal_weights(NTH / (1.0 - KAPPA), D_ENV)

    print("\nTMSS reference", flush=True)

    zeta_tmss = calculate_zeta("TMSS", T=0.5, D=D, auxiliary=1, U4=U4, p_env0=p_env0, p_env1=p_env1)

    RESULTS_SINGLE["TMSS"].append(zeta_tmss)
    RESULTS_DOUBLE["TMSS"].append(zeta_tmss)

    print("\nSINGLE-AUXILIARY-PHOTON CASE", flush=True)

    for kind in ORDER[:-1]:
        zeta = calculate_zeta(kind, T_SINGLE[kind], D, auxiliary=1, U4=U4, p_env0=p_env0, p_env1=p_env1)
        RESULTS_SINGLE[kind].append(zeta)

    print("\nTWO-AUXILIARY-PHOTON CASE", flush=True)

    for kind in ORDER[:-1]:
        zeta = calculate_zeta(kind, T_DOUBLE[kind], D, auxiliary=2, U4=U4, p_env0=p_env0, p_env1=p_env1)
        RESULTS_DOUBLE[kind].append(zeta)

    del U4


for kind in ORDER:
    RESULTS_SINGLE[kind] = np.array(RESULTS_SINGLE[kind], dtype=float)
    RESULTS_DOUBLE[kind] = np.array(RESULTS_DOUBLE[kind], dtype=float)


print("\n" + "=" * 65)
print("NUMERICAL CONVERGENCE: D=32 vs D=40")
print("=" * 65)


def print_convergence(results, title):
    index32 = np.where(D_LIST == 32)[0][0]
    index40 = np.where(D_LIST == 40)[0][0]

    print(f"\n{title}")
    print(f"{'State':6s} {'zeta(32)':>15s} {'zeta(40)':>15s} {'Difference (%)':>17s}")

    for kind in ORDER:
        z32 = results[kind][index32]
        z40 = results[kind][index40]
        relative_difference = 100.0 * abs(z40 - z32) / max(abs(z40), 1e-15)

        print(f"{kind:6s} {z32:15.9e} {z40:15.9e} {relative_difference:17.7f}")


print_convergence(RESULTS_SINGLE, "Single-auxiliary-photon case")
print_convergence(RESULTS_DOUBLE, "Two-auxiliary-photon case")


fig, axs = plt.subplots(1, 2, figsize=(3.1, 1.5), sharex=False, sharey=False)

for ax in axs:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=2.5, width=0.7)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.4, width=0.5)
    ax.axvline(D_REFERENCE, color="0.45", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)


def plot_convergence_panel(ax, results, letter):
    for kind in ORDER:
        color, linestyle, marker = STYLES[kind]
        y = 1e3 * results[kind]

        ax.plot(D_LIST, y, color=color, linestyle=linestyle, linewidth=1.05, marker=marker, markersize=2.6 if marker else 0, markerfacecolor="white" if marker else None, markeredgecolor=color, markeredgewidth=0.75, label=kind, zorder=3)

    ax.set_xlim(11, 41)
    ax.set_xticks([12, 20, 28, 32, 40])
    ax.set_xlabel(r"$D$")
    ax.text(0.05, 0.07, f"({letter})", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")


plot_convergence_panel(axs[0], RESULTS_SINGLE, "a")
plot_convergence_panel(axs[1], RESULTS_DOUBLE, "b")

axs[0].set_ylabel(r"$10^3\zeta$")
axs[1].set_ylabel(r"$10^3\zeta$")


handles = []

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.1, marker=marker, markersize=3.0 if marker else 0, markerfacecolor="white" if marker else None, markeredgecolor=color, markeredgewidth=0.75, label=kind))


fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False, fontsize=6.0, handlelength=1.6, handletextpad=0.30, columnspacing=0.9, borderaxespad=0.0)

fig.subplots_adjust(left=0.15, right=0.97, bottom=0.27, top=0.8, wspace=0.30)

filename_fig = "Fig_13.pdf"

fig.savefig(filename_fig, format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)
