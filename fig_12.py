
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from qutip import basis, tensor, destroy, qeye, coherent, expect


n_max = 32
Nth = 1.0
R = 0.01
T_PROP_best = 0.999999

r = np.arcsinh(np.sqrt(0.05))
lam = np.tanh(r)

K_hom = int(1e6)
kappa_list = np.linspace(1e-5, 0.1, 200)

G_OPA = 1.0 + 7.4e-5
NU_PC = 1.0
MU_PC = np.sqrt(1.0 + NU_PC**2)
MEAN_TOL = 1e-11


def tmss_state(dim, lam):
    psi = 0
    for n in range(dim):
        psi += lam**n * tensor(basis(dim, n), basis(dim, n))
    return psi.unit()


def nlpa_state(dim, r, T):
    lam = np.tanh(r)
    x = lam**2 * T**2
    psi = 0

    for k in range(dim - 1):
        alpha_k = (1.0 - x) * x**(k/2) * np.sqrt(k + 1)
        phi_k = (tensor(basis(dim, k + 1), basis(dim, k)) +
                 tensor(basis(dim, k), basis(dim, k + 1))) / np.sqrt(2.0)
        psi += alpha_k * phi_k

    return psi.unit()


def mean_photons(psi, operator):
    return float(np.real(expect(operator, psi)))


I = qeye(n_max)
aA = tensor(destroy(n_max), I)
aB = tensor(I, destroy(n_max))
I_AB = tensor(I, I)

nA = aA.dag() * aA
nB = aB.dag() * aB

O_exchange = aA.dag() * aB + aB.dag() * aA
O_pair = aA * aB + aA.dag() * aB.dag()

psi_NLPA = nlpa_state(n_max, r, T_PROP_best)

Ns_target = mean_photons(psi_NLPA, nB)
r_TMSS_equal = np.arcsinh(np.sqrt(Ns_target))
lam_TMSS_equal = np.tanh(r_TMSS_equal)

psi_TMSS = tmss_state(n_max, lam_TMSS_equal)

Ns_NLPA = mean_photons(psi_NLPA, nB)
Ni_NLPA = mean_photons(psi_NLPA, nA)

Ns_TMSS = mean_photons(psi_TMSS, nB)
Ni_TMSS = mean_photons(psi_TMSS, nA)

alpha_I = np.sqrt(Ni_NLPA)
alpha_S = np.sqrt(Ns_NLPA)

psi_CI = tensor(coherent(n_max, alpha_I), coherent(n_max, alpha_S))

Ns_CI = mean_photons(psi_CI, nB)
Ni_CI = mean_photons(psi_CI, nA)

print("\n=== EQUAL SIGNAL ENERGY CHECK ===")
print(f"NLPA Ns = {Ns_NLPA:.10f}")
print(f"TMSS Ns = {Ns_TMSS:.10f}")
print(f"CI   Ns = {Ns_CI:.10f}")
print(f"TMSS adjusted squeezing r = {r_TMSS_equal:.10f}")

assert np.isclose(Ns_NLPA, Ns_TMSS, atol=1e-8)
assert np.isclose(Ns_NLPA, Ns_CI, atol=1e-8)

STATES = {"NLPA": psi_NLPA, "TMSS": psi_TMSS, "CI": psi_CI}


def receiver_coefficients(receiver):
    if receiver == "COUNT":
        return {"c0": 0.0, "cI": 0.0, "cR": 0.0, "cE": 1.0, "cP": 0.0, "cV": 0.0}

    elif receiver == "dHD":
        return {"c0": 1.0, "cI": 1.0, "cR": 1.0, "cE": 0.0, "cP": -1.0, "cV": 0.0}

    elif receiver == "OPA":
        return {
            "c0": G_OPA - 1.0,
            "cI": G_OPA,
            "cR": G_OPA - 1.0,
            "cE": 0.0,
            "cP": np.sqrt(G_OPA * (G_OPA - 1.0)),
            "cV": 0.0
        }

    elif receiver == "PC":
        return {"c0": 0.0, "cI": 0.0, "cR": 0.0, "cE": 0.0, "cP": NU_PC, "cV": MU_PC}

    else:
        raise ValueError(f"Unknown receiver: {receiver}")


def receiver_moments(psi, kappa, receiver, hypothesis):
    c = receiver_coefficients(receiver)

    c0 = c["c0"]
    cI = c["cI"]
    cR = c["cR"]
    cE = c["cE"]
    cP = c["cP"]
    cV = c["cV"]

    if hypothesis == 0:
        t, s, nenv = 0.0, 1.0, Nth

    elif hypothesis == 1:
        if not (0.0 <= kappa < 1.0):
            raise ValueError("kappa must satisfy 0 <= kappa < 1")

        t = np.sqrt(kappa)
        s = np.sqrt(1.0 - kappa)
        nenv = Nth / (1.0 - kappa)

    else:
        raise ValueError("hypothesis must be 0 or 1")

    D = (c0 * I_AB + cI * nA + cR * t**2 * nB +
         t * cE * O_exchange + t * cP * O_pair)

    u = cR * s**2
    F = s * (cR * t * aB.dag() + cE * aA.dag() + cP * aA)

    mean_D = float(np.real(expect(D, psi)))
    mean_O = mean_D + u * nenv

    Dpsi = D * psi
    second_D = Dpsi.norm()**2
    var_D = second_D - mean_D**2

    thermal_number_variance = u**2 * nenv * (nenv + 1.0)

    Fpsi = F * psi
    Fdagpsi = F.dag() * psi

    thermal_linear_variance = (
        (nenv + 1.0) * Fdagpsi.norm()**2 +
        nenv * Fpsi.norm()**2
    )

    vacuum_variance = cV**2 * mean_photons(psi, nA)

    var_O = (var_D + thermal_number_variance +
             thermal_linear_variance + vacuum_variance)

    if var_O < -1e-9:
        raise RuntimeError(
            f"Negative variance: {var_O}, receiver={receiver}, kappa={kappa}"
        )

    var_O = max(0.0, float(np.real(var_O)))

    return mean_O, var_O


def snr_from_moments(mu0, var0, mu1, var1, K):
    delta = mu1 - mu0

    if abs(delta) < MEAN_TOL:
        return np.nan

    denominator = 2.0 * (np.sqrt(var0) + np.sqrt(var1))**2

    if denominator <= 0.0:
        return np.nan

    snr = K * delta**2 / denominator

    if snr <= 0.0:
        return np.nan

    return float(10.0 * np.log10(snr))


RECEIVERS = ["COUNT", "dHD", "OPA", "PC"]
ORDER = ["NLPA", "CI", "TMSS"]

RESULTS = {}
H0_CACHE = {}

for receiver in RECEIVERS:
    RESULTS[receiver] = {}

    for kind in ORDER:
        psi = STATES[kind]

        mu0, var0 = receiver_moments(psi, 0.0, receiver, 0)
        H0_CACHE[(receiver, kind)] = (mu0, var0)

        results = []

        print(f"\nCalculating {receiver}: {kind}")

        for kappa in kappa_list:
            mu1, var1 = receiver_moments(psi, kappa, receiver, 1)
            snr_db = snr_from_moments(mu0, var0, mu1, var1, K_hom)
            results.append(snr_db)

        RESULTS[receiver][kind] = np.array(results, dtype=float)

        y = RESULTS[receiver][kind]
        valid = np.isfinite(y)

        if not np.any(valid):
            print(f"  {kind}: zero mean-discrimination SNR for {receiver}")
        else:
            print(f"  {kind}: SNR range = [{np.min(y[valid]):.3f}, "
                  f"{np.max(y[valid]):.3f}] dB")


print("\n========================================")
print("SNR AT kappa = 0.01")
print("========================================")

for receiver in RECEIVERS:
    print(f"\n--- {receiver} ---")

    for kind in ORDER:
        psi = STATES[kind]
        mu0, var0 = H0_CACHE[(receiver, kind)]
        mu1, var1 = receiver_moments(psi, R, receiver, 1)
        snr = snr_from_moments(mu0, var0, mu1, var1, K_hom)

        print(f"{kind:5s} | mu0={mu0:.8e} | mu1={mu1:.8e} | "
              f"V0={var0:.8e} | V1={var1:.8e} | SNR={snr:.6f} dB")


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

styles = {
    "NLPA": ("#6A00A8", "--", "v"),
    "CI": ("#AED814", "-", "s"),
    "TMSS": ("#ff7f0e", ":", None)
}


def panel_label(ax, letter):
    ax.text(0.05, 0.12, f"({letter})", transform=ax.transAxes,
            fontsize=7, ha="left", va="bottom")


def plot_receiver_panel(ax, receiver, letter):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)

    ax.tick_params(which="major", direction="in", top=True,
                   right=True, length=3.0, width=0.8)

    ax.tick_params(which="minor", direction="in", top=True,
                   right=True, length=1.7, width=0.6)

    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = -1

    for kind in ORDER:
        y = RESULTS[receiver][kind]
        color, linestyle, marker = styles[kind]

        ax.plot(
            kappa_list, y,
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            marker=marker,
            markersize=3.0 if marker else 0,
            markevery=16 if marker else None,
            markerfacecolor="white" if marker else None,
            markeredgecolor=color,
            markeredgewidth=0.85,
            zorder=3
        )

    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\mathrm{SNR}\ (\mathrm{dB})$")
    ax.set_xlim(0.0, 0.1)
    ax.set_xticks([0.0, 0.05, 0.10])

    if receiver == "COUNT":
        ax.set_ylim(20, 40)
    elif receiver == "dHD":
        ax.set_ylim(10, 45)
    elif receiver == "OPA":
        ax.set_ylim(-13, 7)
    elif receiver == "PC":
        ax.set_ylim(14, 45)

    panel_label(ax, letter)


fig, axs = plt.subplots(2, 2, figsize=(3.1, 3.1), sharex=False, sharey=False)
axs = axs.ravel()

panel_specs = [
    ("COUNT", "a"),
    ("dHD", "b"),
    ("OPA", "c"),
    ("PC", "d")
]

for ax, (receiver, letter) in zip(axs, panel_specs):
    plot_receiver_panel(ax, receiver, letter)


handles = []

for kind in ORDER:
    color, linestyle, marker = styles[kind]

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
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
    fontsize=6.2,
    handlelength=1.7,
    handletextpad=0.35,
    columnspacing=1.0,
    labelspacing=0.35,
    borderaxespad=0.0
)

fig.subplots_adjust(
    top=0.93,
    bottom=0.08,
    left=0.17,
    right=0.96,
    hspace=0.30,
    wspace=0.45
)

fig.savefig("Fig_12.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)

print("\nSaved: Fig_12.pdf")
