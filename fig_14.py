
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


T_NLPA = 0.999999

NB_REFERENCE = 1.0
KAPPA_REFERENCE = 0.01

r0 = np.arcsinh(np.sqrt(0.05))
lam0 = np.tanh(r0)

x0 = (lam0 * T_NLPA)**2
NS_REFERENCE = 0.5 + 2.0 * x0 / (1.0 - x0)

print(f"Original NLPA signal brightness: Ns = {NS_REFERENCE:.10f}")


NS_VALUES = np.array([0.50, 0.55, NS_REFERENCE, 0.70, 0.80, 0.90, 1.00])
NB_VALUES = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
KAPPA_VALUES = np.array([0.001, 0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.030])

SOURCE_TAIL_TOL = 1e-10
OUTPUT_TAIL_TOL = 1e-10

SPECTRAL_TOL = 1e-14
AMPLITUDE_TOL = 1e-16

TRACE_TOL = 2e-8


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white"
})

STYLES = {"NLPA": ("#6A00A8", "--", "v"), "TMSS": ("#ff7f0e", ":", None), "CI": ("#AED814", "--", None)}

ORDER = ["NLPA", "TMSS", "CI"]

folder = Path("Numerical_Data")
folder.mkdir(parents=True, exist_ok=True)


def normalize_coefficients(C):
    norm = np.linalg.norm(C)

    if norm <= 0:
        raise ValueError("Zero state norm.")

    return C / norm


def nlpa_coefficients(Ns):
    if Ns < 0.5:
        raise ValueError("Single-photon NLPA requires Ns >= 0.5.")

    x = (2.0 * Ns - 1.0) / (2.0 * Ns + 3.0)
    lam = np.sqrt(x) / T_NLPA

    if lam >= 1.0:
        raise ValueError("Requested Ns is incompatible with fixed T.")

    r = np.arctanh(lam)
    D = 3

    while True:
        k = np.arange(D - 1)
        amplitudes = (1.0 - x) * np.sqrt(k + 1.0) * x**(k / 2.0)
        retained = np.sum(amplitudes**2)

        if 1.0 - retained < SOURCE_TAIL_TOL:
            break

        D += 1

        if D > 100:
            raise RuntimeError("NLPA source dimension did not converge.")

    C = np.zeros((D, D), dtype=np.complex128)

    C[k + 1, k] = amplitudes / np.sqrt(2.0)
    C[k, k + 1] = amplitudes / np.sqrt(2.0)

    return normalize_coefficients(C), r


def tmss_coefficients(Ns):
    q = Ns / (Ns + 1.0)
    D = max(3, int(np.ceil(np.log(SOURCE_TAIL_TOL) / np.log(q))))
    n = np.arange(D)

    amplitudes = np.sqrt(1.0 - q) * q**(n / 2.0)
    C = np.diag(amplitudes.astype(np.complex128))

    return normalize_coefficients(C)


def signal_brightness(C):
    D = C.shape[1]
    photon_numbers = np.arange(D)

    return float(np.sum(np.abs(C)**2 * photon_numbers[None, :]))


def return_dimension(NB):
    q = NB / (NB + 1.0)
    D = int(np.ceil(np.log(OUTPUT_TAIL_TOL) / np.log(q)))

    return max(12, D + 8)


def thermal_weights(NB, D):
    q = NB / (NB + 1.0)
    n = np.arange(D)
    probabilities = (1.0 - q) * q**n

    return probabilities / probabilities.sum()


def make_blocks(NI, NO, kind):
    a = np.arange(NI)[:, None]
    b = np.arange(NO)[None, :]

    if kind == "TMSS":
        block_labels = b - a

    elif kind == "NLPA":
        block_labels = (b - a) % 2

    else:
        raise ValueError(kind)

    blocks = {}
    maps = {}

    for label in np.unique(block_labels):
        indices = np.flatnonzero((block_labels == label).ravel())
        block = np.zeros((len(indices), len(indices)), dtype=np.complex128)
        index_map = np.full(NI * NO, -1, dtype=int)
        index_map[indices] = np.arange(len(indices))

        blocks[int(label)] = block
        maps[int(label)] = index_map

    return blocks, maps


def target_output_blocks(C, NB, kappa, kind):
    NI = C.shape[0]
    NO = return_dimension(NB)

    p0 = thermal_weights(NB, NO)
    blocks, index_maps = make_blocks(NI, NO, kind)

    G = NB + 1.0
    tau = kappa / G

    log_tau = np.log(tau)
    log_loss = np.log1p(-tau)
    log_G = np.log(G)
    log_q = np.log(NB / G)

    for ell in range(NI):
        b = np.arange(ell, NI)
        p = b - ell

        log_loss_amplitude = 0.5 * (gammaln(b + 1) - gammaln(ell + 1) - gammaln(p + 1) + ell * log_loss + p * log_tau)

        for j in range(NO):
            valid = (p + j) < NO

            if not np.any(valid):
                break

            bb = b[valid]
            pp = p[valid]

            log_amp_amplitude = 0.5 * (gammaln(pp + j + 1) - gammaln(j + 1) - gammaln(pp + 1) + j * log_q - (pp + 1) * log_G)
            scale = np.exp(log_loss_amplitude[valid] + log_amp_amplitude)
            entries = C[:, bb] * scale[None, :]

            aa, kk = np.nonzero(np.abs(entries) > AMPLITUDE_TOL)

            if len(aa) == 0:
                continue

            returned = pp[kk] + j
            indices = aa * NO + returned
            values = entries[aa, kk]

            if kind == "TMSS":
                labels = returned - aa
            else:
                labels = (returned - aa) % 2

            if not np.all(labels == labels[0]):
                raise RuntimeError("Unexpected symmetry-sector mixing.")

            label = int(labels[0])
            local = index_maps[label][indices]

            blocks[label][np.ix_(local, local)] += values[:, None] * values.conj()[None, :]

    trace1 = sum(np.trace(block).real for block in blocks.values())

    if abs(trace1 - 1.0) > TRACE_TOL:
        raise RuntimeError(f"Thermal channel is insufficiently truncated: Tr(rho1)={trace1:.12f}. Decrease OUTPUT_TAIL_TOL.")

    return blocks, index_maps, p0, NI, NO, trace1


def quantum_chernoff(C, NB, kappa, kind):
    blocks, index_maps, p0, NI, NO, trace1 = target_output_blocks(C, NB, kappa, kind)

    spectral_terms = []

    if kind == "NLPA":
        rho_idler = C @ C.conj().T
        eigI, vecI = eigh(rho_idler, check_finite=False)
        eigI = np.maximum(eigI, 0.0)
        eig0 = (eigI[:, None] * p0[None, :]).ravel()
        good0 = eig0 > SPECTRAL_TOL
        lambda0 = eig0[good0]

    else:
        probabilities_idler = np.sum(np.abs(C)**2, axis=1)

    for label, rho1_block in blocks.items():
        eig1, vec1 = eigh(rho1_block, check_finite=False, overwrite_a=True)

        if np.min(eig1) < -1e-9:
            raise RuntimeError("rho1 has significantly negative eigenvalues.")

        good1 = eig1 > SPECTRAL_TOL

        if not np.any(good1):
            continue

        lambda1 = eig1[good1]
        V1 = vec1[:, good1]

        indices = np.flatnonzero(index_maps[label] >= 0)

        if kind == "TMSS":
            idler_n = indices // NO
            return_n = indices % NO
            eig0_block = probabilities_idler[idler_n] * p0[return_n]
            keep0 = eig0_block > SPECTRAL_TOL
            lam0 = eig0_block[keep0]
            overlap = np.abs(V1[keep0, :])**2

        else:
            n_eig = V1.shape[1]
            full_vectors = np.zeros((NI * NO, n_eig), dtype=np.complex128)
            full_vectors[indices, :] = V1

            rotated = vecI.conj().T @ full_vectors.reshape(NI, NO * n_eig)
            rotated = rotated.reshape(NI * NO, n_eig)

            overlap = np.abs(rotated[good0, :])**2
            lam0 = lambda0

        spectral_terms.append((lam0, overlap, lambda1))

    def Q_of_s(s):
        result = 0.0

        for lam0, overlap, lam1 in spectral_terms:
            result += lam0**(1.0 - s) @ overlap @ lam1**s

        return float(np.real(result))

    eps = 1e-7

    optimum = minimize_scalar(Q_of_s, bounds=(eps, 1.0 - eps), method="bounded", options={"xatol": 1e-8})

    candidates = [(eps, Q_of_s(eps)), (float(optimum.x), float(optimum.fun)), (1.0 - eps, Q_of_s(1.0 - eps))]

    s_opt, Q = min(candidates, key=lambda pair: pair[1])

    if Q <= 0.0 or Q > 1.0 + 1e-8:
        raise RuntimeError(f"Unphysical Chernoff value Q={Q}")

    return min(Q, 1.0), s_opt, NI, NO, trace1


def zeta_from_state(C, NB, kappa, kind):
    Ns = signal_brightness(C)
    Q, s, DI, DO, trace1 = quantum_chernoff(C, NB, kappa, kind)
    zeta = -np.log(Q) / Ns

    print(f"  {kind:4s} | Ns={Ns:.8f} | NB={NB:.3f} | kappa={kappa:.4f} | Din={DI:2d} | Dout={DO:3d} | Q={Q:.12f} | s={s:.6f} | zeta={zeta:.10e} | Tr1={trace1:.10f}", flush=True)

    return zeta


def zeta_classical(NB, kappa):
    return kappa * (np.sqrt(NB + 1.0) - np.sqrt(NB))**2


CACHE = {}


def calculate_point(Ns, NB, kappa):
    key = (float(Ns), float(NB), float(kappa))

    if key in CACHE:
        return CACHE[key]

    print("\n" + "-" * 65)
    print(f"Ns={Ns:.8f}, NB={NB:.3f}, kappa={kappa:.5f}")
    print("-" * 65, flush=True)

    C_NLPA, r_NLPA = nlpa_coefficients(Ns)
    Ns_actual = signal_brightness(C_NLPA)

    C_TMSS = tmss_coefficients(Ns_actual)
    Ns_TMSS = signal_brightness(C_TMSS)

    if abs(Ns_actual - Ns_TMSS) > 1e-7:
        raise RuntimeError("TMSS and NLPA signal energies are not matched.")

    z_NLPA = zeta_from_state(C_NLPA, NB, kappa, "NLPA")
    z_TMSS = zeta_from_state(C_TMSS, NB, kappa, "TMSS")
    z_CI = zeta_classical(NB, kappa)

    print(f"  CI   | Ns={Ns_actual:.8f} | zeta={z_CI:.10e}")

    answer = np.array([z_NLPA, z_TMSS, z_CI])

    CACHE[key] = answer

    return answer


print("\n" + "=" * 65)
print("PANEL (a): zeta versus Ns")
print("=" * 65)

RESULTS_NS = np.array([calculate_point(Ns, NB_REFERENCE, KAPPA_REFERENCE) for Ns in NS_VALUES])


print("\n" + "=" * 65)
print("PANEL (b): zeta versus NB")
print("=" * 65)

RESULTS_NB = np.array([calculate_point(NS_REFERENCE, NB, KAPPA_REFERENCE) for NB in NB_VALUES])


print("\n" + "=" * 65)
print("PANEL (c): zeta versus kappa")
print("=" * 65)

RESULTS_KAPPA = np.array([calculate_point(NS_REFERENCE, NB_REFERENCE, kappa) for kappa in KAPPA_VALUES])


fig, axs = plt.subplots(3, 1, figsize=(3.1, 4), sharey=False, sharex=False)


def configure_axis(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.7, width=0.6)


def plot_panel(ax, x, results, letter):
    for i, kind in enumerate(ORDER):
        color, linestyle, marker = STYLES[kind]

        ax.plot(x, 1e3 * results[:, i], color=color, linestyle=linestyle, linewidth=1.25, marker=marker, markersize=3.3 if marker else 0, markerfacecolor="white" if marker else None, markeredgecolor=color, markeredgewidth=0.85, label=kind, zorder=3)

    ax.text(0.05, 0.94, f"({letter})", transform=ax.transAxes, fontsize=8, ha="left", va="top")


for ax in axs:
    configure_axis(ax)


plot_panel(axs[0], NS_VALUES, RESULTS_NS, "a")

axs[0].set_xlabel(r"$N_S$")

axs[0].set_ylabel(r"$10^3\zeta$")
axs[1].set_ylabel(r"$10^3\zeta$")
axs[2].set_ylabel(r"$10^3\zeta$")

axs[0].set_xlim(0.48, 1.02)
axs[0].set_xticks([0.5, 0.6, 0.8, 1.0])


plot_panel(axs[1], NB_VALUES, RESULTS_NB, "b")

axs[1].set_xlabel(r"$\bar{n}_{\mathrm{th}}$")
axs[1].set_xscale("log")
axs[1].set_xlim(0.09, 11.0)
axs[1].xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0]))
axs[1].xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))


plot_panel(axs[2], KAPPA_VALUES, RESULTS_KAPPA, "c")

axs[2].set_xlabel(r"$\kappa$")
axs[2].set_xlim(0.0, 0.032)
axs[2].set_xticks([0.0, 0.01, 0.02, 0.03])


handles = []

for kind in ORDER:
    color, linestyle, marker = STYLES[kind]

    handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.3, marker=marker, markersize=3.5 if marker else 0, markerfacecolor="white" if marker else None, markeredgecolor=color, markeredgewidth=0.85, label=kind))


fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=8, handlelength=2.0, handletextpad=0.4, columnspacing=1.8, borderaxespad=0.0)

fig.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.94, hspace=0.5)


filename = "Fig_14.pdf"

fig.savefig(filename, format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)


print("\n" + "=" * 65)
print("FINISHED")
print("=" * 65)

print("Saved:", filename)

print("Saved numerical datasets:")
print("  Numerical_Data/FigB2a_vs_Ns.txt")
print("  Numerical_Data/FigB2b_vs_NB.txt")
print("  Numerical_Data/FigB2c_vs_kappa.txt")


ia = np.argmin(abs(NS_VALUES - NS_REFERENCE))
ib = np.argmin(abs(NB_VALUES - NB_REFERENCE))
ic = np.argmin(abs(KAPPA_VALUES - KAPPA_REFERENCE))

print("\n========== CHECK 1: PANEL CONSISTENCY ==========")

for j, kind in enumerate(ORDER):
    a = RESULTS_NS[ia, j]
    b = RESULTS_NB[ib, j]
    c = RESULTS_KAPPA[ic, j]

    difference = 100 * (max(a, b, c) - min(a, b, c)) / max(abs(a), abs(b), abs(c))

    print(f"\n{kind}")
    print(f"Panel (a): {a:.12e}")
    print(f"Panel (b): {b:.12e}")
    print(f"Panel (c): {c:.12e}")
    print(f"Difference: {difference:.8f}%")
    print("PASS" if difference < 0.01 else "CHECK")


print("\n========== CHECK 2: HIGH-NOISE CONVERGENCE ==========")

ib_high = np.argmin(abs(NB_VALUES - 10.0))
original_values = RESULTS_NB[ib_high, :].copy()
original_tolerance = OUTPUT_TAIL_TOL

try:
    OUTPUT_TAIL_TOL = 1e-11
    CACHE.clear()

    print(f"New output dimension: {return_dimension(10.0)}")

    new_values = calculate_point(NS_REFERENCE, 10.0, KAPPA_REFERENCE)

    for j, kind in enumerate(ORDER):
        old = original_values[j]
        new = new_values[j]

        difference = 100 * abs(new - old) / max(abs(new), 1e-15)

        print(f"\n{kind}")
        print(f"Original zeta: {old:.12e}")
        print(f"New zeta:      {new:.12e}")
        print(f"Difference:    {difference:.8f}%")
        print("PASS" if difference < 0.1 else "CHECK")

finally:
    OUTPUT_TAIL_TOL = original_tolerance
    CACHE.clear()

print("\n========== VALIDATION FINISHED ==========")
