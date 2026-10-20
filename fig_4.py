
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from qutip import basis, tensor, destroy, qeye
from matplotlib.lines import Line2D


n_max = 32
nE_max = n_max

Nth = 1.0
R = 0.01

r = np.arcsinh(np.sqrt(0.05))
lam = np.tanh(r)

T_qcb = np.linspace(1e-6, 1.0 - 1e-6, 20)
protocols = ["NLPA", "PA", "PC", "PS"]

styles = {
    "NLPA": ("#6A00A8", "--", "v"),
    "PA": ("#D00000", "-.", "s"),
    "PC": ("#0050FF", ":", "o"),
    "PS": ("#008A2E", "--", "^")
}


def tmss_state(dim, r):
    lam = np.tanh(r)
    psi = 0

    for n in range(dim):
        psi += lam**n * tensor(basis(dim, n), basis(dim, n))

    return psi.unit()


def photon_added_tmss_single_side(dim, r, T):
    psi = tmss_state(dim, r)
    a = destroy(dim)
    n_op = a.dag() * a
    O = -np.sqrt(1.0 - T) * a.dag() * (0.5 * np.log(T) * n_op).expm()
    return (tensor(qeye(dim), O) * psi).unit()


def photon_subtracted_tmss_single_side(dim, r, T):
    psi = tmss_state(dim, r)
    a = destroy(dim)
    n_op = a.dag() * a
    O = np.sqrt((1.0 - T) / T) * a * (0.5 * np.log(T) * n_op).expm()
    return (tensor(qeye(dim), O) * psi).unit()


def photon_catalysis_tmss_single_side(dim, r, T):
    psi = tmss_state(dim, r)
    a = destroy(dim)
    n_op = a.dag() * a
    O = np.sqrt(T) * (((1.0 - T) / T) * n_op - qeye(dim)) * (0.5 * np.log(T) * n_op).expm()
    return (tensor(qeye(dim), O) * psi).unit()


def psi_f_state(dim, r, T):
    lam = np.tanh(r)
    x = lam**2 * T**2
    psi = 0

    for k in range(dim - 1):
        alpha = (1.0 - x) * x**(k / 2.0) * np.sqrt(k + 1.0)
        phi = (tensor(basis(dim, k + 1), basis(dim, k)) +
               tensor(basis(dim, k), basis(dim, k + 1))) / np.sqrt(2.0)
        psi += alpha * phi

    return psi.unit()


def get_state(kind, T):
    if kind == "NLPA":
        return psi_f_state(n_max, r, T)
    elif kind == "PA":
        return photon_added_tmss_single_side(n_max, r, T)
    elif kind == "PC":
        return photon_catalysis_tmss_single_side(n_max, r, T)
    elif kind == "PS":
        return photon_subtracted_tmss_single_side(n_max, r, T)
    elif kind == "TMSS":
        return tmss_state(n_max, r)
    else:
        raise ValueError(kind)


def success_probability(kind, T):
    lam2 = lam**2
    sech2 = 1.0 / np.cosh(r)**2

    if kind == "NLPA":
        return sech2 * (1.0 - T) / (1.0 - lam2 * T**2)**2

    elif kind == "PA":
        return sech2 * (1.0 - T) / (1.0 - lam2 * T)**2

    elif kind == "PS":
        return sech2 * (1.0 - T) * lam2 / (1.0 - lam2 * T)**2

    elif kind == "PC":
        x = lam2 * T
        term1 = T**2 / (1.0 - x)
        term2 = -2.0 * T * (1.0 - T) * x / (1.0 - x)**2
        term3 = (1.0 - T)**2 * x * (1.0 + x) / (1.0 - x)**3
        return sech2 * (term1 + term2 + term3) / T

    elif kind == "TMSS":
        return 1.0

    else:
        raise ValueError(kind)


def thermal_state(N, dim):
    weights = (N / (N + 1.0))**np.arange(dim) / (N + 1.0)
    weights /= weights.sum()
    return np.diag(weights)


rhoC0 = thermal_state(Nth, nE_max)
rhoC1 = thermal_state(Nth / (1.0 - R), nE_max)

a = destroy(n_max)
b = destroy(nE_max)

xi = np.arctan(np.sqrt((1.0 - R) / R))
generator = xi * (tensor(a, b.dag()) - tensor(a.dag(), b))
U_array = generator.expm().full().reshape(n_max, nE_max, n_max, nE_max)


def build_rho0_rho1(psi):
    c = psi.full().reshape(n_max, n_max)
    rhoA = c @ c.conj().T
    rho0 = np.kron(rhoA, rhoC0)

    dim_joint = n_max * n_max
    rho1 = np.zeros((dim_joint, dim_joint), dtype=complex)

    for j, weight in enumerate(np.diag(rhoC1)):
        if weight < 1e-15:
            continue

        Uj = U_array[:, :, :, j]
        output = np.einsum("ab,rcb->arc", c, Uj, optimize=True)
        M = output.reshape(dim_joint, nE_max)
        rho1 += weight * (M @ M.conj().T)

    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho1 = 0.5 * (rho1 + rho1.conj().T)

    rho0 /= np.trace(rho0).real
    rho1 /= np.trace(rho1).real

    return rho0, rho1


def qcb_Q_from_states(rho0, rho1):
    e0, V0 = np.linalg.eigh(rho0)
    e1, V1 = np.linalg.eigh(rho1)

    e0 = np.maximum(e0.real, 0.0)
    e1 = np.maximum(e1.real, 0.0)

    mask0 = e0 > 1e-13
    mask1 = e1 > 1e-13

    a = e0[mask0]
    b = e1[mask1]
    overlap = np.abs(V0[:, mask0].conj().T @ V1[:, mask1])**2

    def Qs(s):
        return float((a**s) @ overlap @ (b**(1.0 - s)))

    result = minimize_scalar(
        Qs, bounds=(1e-7, 1.0 - 1e-7),
        method="bounded", options={"xatol": 1e-9}
    )

    Q = float(result.fun)

    if not np.isfinite(Q) or Q <= 0 or Q > 1.0 + 1e-8:
        raise ValueError(f"Invalid Chernoff quantity: {Q}")

    return min(Q, 1.0)


def calculate_Q(kind, T):
    psi = get_state(kind, T)
    rho0, rho1 = build_rho0_rho1(psi)
    return qcb_Q_from_states(rho0, rho1)


Q_TMSS = calculate_Q("TMSS", 0.5)
epsilon_TMSS = -np.log1p(Q_TMSS - 1.0)

print("\nTMSS REFERENCE")
print(f"Q_TMSS = {Q_TMSS:.12f}")
print(f"epsilon_TMSS = {epsilon_TMSS:.12e}")

P_data = {}
G_data = {}

for kind in protocols:
    P_data[kind] = np.zeros_like(T_qcb)
    G_data[kind] = np.zeros_like(T_qcb)

    print(f"\nCalculating {kind}...")

    for i, T in enumerate(T_qcb):
        P = success_probability(kind, T)
        Q = calculate_Q(kind, T)
        epsilon = -P * np.log1p(Q - 1.0)
        G = epsilon / epsilon_TMSS

        P_data[kind][i] = P
        G_data[kind][i] = G

        print(f"{kind:5s} | {i+1:2d}/{len(T_qcb)} | T={T:.5f} | P={P:.6f} | G={G:.8f}")


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
    "savefig.facecolor": "white",
    "axes.unicode_minus": False
})


fig, axs = plt.subplots(2, 2, figsize=(3.1, 3.1), sharex=False, sharey=False)
axs = axs.ravel()

panel_data = [("NLPA", "a"), ("PA", "b"), ("PC", "c"), ("PS", "d")]

for ax, (kind, letter) in zip(axs, panel_data):
    color, linestyle, marker = styles[kind]

    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = -1
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.7, width=0.6)
    ax.grid(True, which="major", color="0.75", linewidth=0.45, alpha=0.30)

    ax.plot(T_qcb, G_data[kind], color=color, linestyle=linestyle,linewidth=1.15, marker=marker, markersize=2.5,markevery=3, markerfacecolor="white",markeredgecolor=color, markeredgewidth=0.8, zorder=3)

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$G$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0.0, 0.5, 1.0])

    ax.text(0.91, 0.10, f"({letter})", transform=ax.transAxes,fontsize=7, ha="right", va="bottom")


handles = []

for kind in protocols:
    color, linestyle, marker = styles[kind]

    handles.append(Line2D([0], [0], color=color, linestyle=linestyle,linewidth=1.15, marker=marker, markersize=3.0,markerfacecolor="white", markeredgecolor=color,markeredgewidth=0.8, label=kind))

fig.legend(handles=handles,labels=[r"$G_{\mathrm{NLPA}}$",r"$G_{\mathrm{PA}}$",r"$G_{\mathrm{PC}}$",r"$G_{\mathrm{PS}}$"],loc="upper center",bbox_to_anchor=(0.5, 0.98),ncol=4,frameon=False,fontsize=6.2,handlelength=1.4,handletextpad=0.30,columnspacing=0.65,labelspacing=0.35,borderaxespad=0.0)

fig.subplots_adjust(top=0.93, bottom=0.08, left=0.12, right=0.95, hspace=0.21, wspace=0.42)

fig.savefig("Fig_4.pdf", format="pdf", dpi=300, facecolor="white")

plt.show()
plt.close(fig)

print("\nSaved: Fig_4.pdf")
print("\nMAXIMUM SAMPLED GAINS")

for kind in protocols:
    i_best = int(np.argmax(G_data[kind]))
    print(f"{kind:5s} | T={T_qcb[i_best]:.6f} | P={P_data[kind][i_best]:.6f} | G_max={G_data[kind][i_best]:.8f}")
