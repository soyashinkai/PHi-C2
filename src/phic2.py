import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats
import click
# -----------------------------------------------------------------------------


def Calc_C_normalized(C):
    N = C.shape[0]
    C_normalized = np.zeros((N, N))
    # Diagonal elements
    for i in range(N):
        C_normalized[i, i] = 1
    # Non-diagonal elements
    for i in range(N-1):
        for j in range(i+1, N):
            if C[i, i] > 0 and C[j, j] > 0:
                tmp = C[i, j] / np.sqrt(C[i, i] * C[j, j])
                C_normalized[i, j] = C_normalized[j, i] = tmp
    return C_normalized
# -----------------------------------------------------------------------------


def Calc_C_normalized_high_resolution(C):
    N = C.shape[0]
    C_normalized = np.zeros((N, N))
    # Diagonal elements
    for i in range(N):
        C_normalized[i, i] = 1
    # Non-diagonal elements
    for i in range(N-1):
        for j in range(i+1, N):
            if (C[i, i] + C[i, i+1] + C[i+1, i+1]) > 0 and (C[j-1, j-1] + C[j-1, j] + C[j, j]) > 0:
                tmp = C[i, j] / np.sqrt((C[i, i] + C[i, i+1] + C[i+1, i+1]) * (C[j-1, j-1] + C[j-1, j] + C[j, j]))
                C_normalized[i, j] = C_normalized[j, i] = tmp
    return C_normalized
# -----------------------------------------------------------------------------


def Calc_P(C, RES):
    N = C.shape[0]
    P = np.zeros((N, 2))
    for n in range(0, N):
        P[n, 0] = RES * n
        for m in range(0, N - n):
            P[n, 1] += C[m, m + n]
        P[n, 1] /= (N - n)
    return P
# -----------------------------------------------------------------------------


def Read_Normalized_C(FILE_READ):
    C = np.loadtxt(FILE_READ)
    N = C.shape[0]
    return C, N
# -----------------------------------------------------------------------------


def Set_Init_K(N, INIT_K_BACKBONE):
    K = np.zeros((N, N))
    for i in range(N - 1):
        j = i + 1
        K[i, j] = K[j, i] = INIT_K_BACKBONE
    return K
# -----------------------------------------------------------------------------


@numba.jit(nopython=True)
def Convert_K_into_C(K, N):
    # K to Laplacian matrix
    d = np.sum(K, axis=0)
    D = np.diag(d)
    L = D - K
    # Eigenvalues and eigenvectors
    lam, Q = np.linalg.eigh(L)
    inv_lam = 1 / lam   # inverse of the eigenvalues
    inv_lam[0] = 0
    inv_Lam = np.diag(inv_lam)
    # L to M
    M = np.dot(Q, np.dot(inv_Lam, Q.T))
    # M to Σ^2
    M_diag = np.diag(np.diag(M))
    A = np.dot(M_diag, np.ones((N, N)))
    Sigma2 = (A + A.T - 2 * M) / 3
    # Σ^2 to C
    C = (1 + Sigma2)**(-1.5)
    return C
# -----------------------------------------------------------------------------


@numba.jit(nopython=True)
def Calc_Diff_Cost(A, B, N):
    Diff = A - B
    Cost = np.sqrt(np.trace(np.dot(Diff.T, Diff))) / N
    return Diff, Cost
# -----------------------------------------------------------------------------


def Calc_Correlation(A, B, N):
    list_X = []
    list_Y = []
    # -------------------------------------------------------------------------
    for i in range(N):
        for j in range(i + 1, N):
            list_X.append(A[i, j])
            list_Y.append(B[i, j])
    # -------------------------------------------------------------------------
    X = np.array(list_X)
    Y = np.array(list_Y)
    # -------------------------------------------------------------------------
    r, p = scipy.stats.pearsonr(X, Y)
    return r, X, Y
# -----------------------------------------------------------------------------


def Calc_Distance_Corrected_Correlation(A, B, N):
    P_A = np.zeros(N)
    P_B = np.zeros(N)
    for n in range(0, N):
        for m in range(0, N - n):
            P_A[n] += A[m, m + n]
            P_B[n] += B[m, m + n]
        P_A[n] /= (N - n)
        P_B[n] /= (N - n)
    # -------------------------------------------------------------------------
    tmp_A = np.zeros((N, N))
    tmp_B = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            tmp_A[i, j] = tmp_A[j, i] = A[i, j] - P_A[j - i]
            tmp_B[i, j] = tmp_B[j, i] = B[i, j] - P_B[j - i]
    # -------------------------------------------------------------------------
    list_X = []
    list_Y = []
    # -------------------------------------------------------------------------
    for i in range(N):
        for j in range(i + 1, N):
            list_X.append(tmp_A[i, j])
            list_Y.append(tmp_B[i, j])
    # -------------------------------------------------------------------------
    X = np.array(list_X)
    Y = np.array(list_Y)
    # -------------------------------------------------------------------------
    r, p = scipy.stats.pearsonr(X, Y)
    return r, X, Y
# -----------------------------------------------------------------------------


def Read_K(FILE_READ_K):
    K = np.loadtxt(FILE_READ_K)
    N = K.shape[0]
    return K, N
# -----------------------------------------------------------------------------


def Transform_K_into_L(K):
    d = np.sum(K, axis=0)
    D = np.diag(d)
    L = D - K
    return L
# -----------------------------------------------------------------------------


def Equilibrium_Conformation_of_Normal_Coordinates(lam, N):
    Xx = np.zeros(N)
    Xy = np.zeros(N)
    Xz = np.zeros(N)
    for p in range(1, N):
        sd = np.sqrt(1 / 3 / lam[p])
        Xx[p] = sd * np.random.randn()
        Xy[p] = sd * np.random.randn()
        Xz[p] = sd * np.random.randn()
    return Xx, Xy, Xz
# -----------------------------------------------------------------------------


def Convert_X_to_R(Xx, Xy, Xz, Q):
    Rx = np.dot(Q, Xx)
    Ry = np.dot(Q, Xy)
    Rz = np.dot(Q, Xz)
    return Rx, Ry, Rz
# -----------------------------------------------------------------------------


def Integrate_Polymer_Network(x, y, z, L, N, NOISE, F_Coefficient):
    noise_x = NOISE * np.random.randn(N)
    noise_y = NOISE * np.random.randn(N)
    noise_z = NOISE * np.random.randn(N)

    force_x = - F_Coefficient * np.dot(L, x)
    force_y = - F_Coefficient * np.dot(L, y)
    force_z = - F_Coefficient * np.dot(L, z)

    x_dt = x + force_x + noise_x
    y_dt = y + force_y + noise_y
    z_dt = z + force_z + noise_z

    force_x = - F_Coefficient * np.dot(L, x_dt)
    force_y = - F_Coefficient * np.dot(L, y_dt)
    force_z = - F_Coefficient * np.dot(L, z_dt)

    x_2dt = x_dt + force_x + noise_x
    y_2dt = y_dt + force_y + noise_y
    z_2dt = z_dt + force_z + noise_z

    X = 0.5 * (x + x_2dt)
    Y = 0.5 * (y + y_2dt)
    Z = 0.5 * (z + z_2dt)
    return X, Y, Z
# -----------------------------------------------------------------------------


def Write_Psfdata(DIR, NAME, N):
    FILE_PSF = DIR + "/polymer_N{0:d}.psf".format(N)
    fp = open(FILE_PSF, "w")
    print("PSF\n\n       1 !NTITLE\n REMARKS %s\n" % NAME, file=fp)
    print(" %7d !NATOM" % N, file=fp)
    for n in range(N):
        print(" %7d A    %04d GLY  CA   CT1    0.070000       12.0110           0"
              % (n + 1, n + 1), file=fp)
    print("\n %7d !NBOND: bonds" % (N - 1), file=fp)
    j = 0
    for i in range(N):
        if i % N != 0:
            print(" %7d %7d" % (i, i + 1), end="", file=fp)
            j += 1
            if j % 4 == 0:
                print("\n", end="", file=fp)
    print("\n", end="", file=fp)
    fp.close()
# -----------------------------------------------------------------------------


@click.group()
def cli():
    pass
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--input", "FILE_DUMPED", required=True,
              help="Input contact matrix file dumped by Straw for a hic file")
@click.option("--res", "RES", type=int, required=True,
              help="Resolution of the bin size")
@click.option("--plt-max-c", "PLT_MAX_C", type=float, required=True,
              help="Maximum value of contact map")
@click.option("--for-high-resolution", "HIGH_RESOLUTION", type=int, default=0,
              help="Normalization of contact map for high-resolution case (ex. 1-kb, 500-bp, 200-bp)  [default=0]")
def preprocessing(FILE_DUMPED, RES, PLT_MAX_C, HIGH_RESOLUTION):
    DIR, EXT = os.path.splitext(FILE_DUMPED)
    os.makedirs(DIR, exist_ok=True)
    FILE_OUT_C_NORMALIZED = DIR + "/C_normalized.txt"
    FILE_OUT_P_NORMALIZED = DIR + "/P_normalized.txt"
    FILE_FIG_C_NORMALIZED = DIR + "/C_normalized.svg"
    FILE_FIG_P_NORMALIZED = DIR + "/P_normalized.svg"
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    C = np.loadtxt(FILE_DUMPED)
    if HIGH_RESOLUTION:
        C_normalized = Calc_C_normalized_high_resolution(C)
    else:
        C_normalized = Calc_C_normalized(C)
    P_normalized = Calc_P(C_normalized, RES)
    # -------------------------------------------------------------------------
    np.savetxt(FILE_OUT_C_NORMALIZED, C_normalized, fmt="%e")
    np.savetxt(FILE_OUT_P_NORMALIZED, P_normalized, fmt="%d\t%e")
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(C_normalized, cmap="magma_r", clim=(0, PLT_MAX_C))
    plt.colorbar(ticks=[0, PLT_MAX_C], shrink=0.5, orientation="vertical",
                 label="Normalized contact probability")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FILE_FIG_C_NORMALIZED)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Genomic distance [bp]", fontweight="bold")
    plt.ylabel("Normalized contact probability", fontweight="bold")
    plt.plot(P_normalized[1:, 0], P_normalized[1:, 1], linewidth=4)
    plt.tight_layout()
    plt.savefig(FILE_FIG_P_NORMALIZED)
    plt.close()
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--init-k-backbone", "INIT_K_BACKBONE", type=float, default=0.5,
              help="Initial parameter of K_i,i+1  [default=0.5]")
@click.option("--learning-rate", "ETA", type=float, default=1e-4,
              help="Learning rate  [default=1e-4]")
@click.option("--stop-condition-parameter", "ALPHA", type=float, default=1e-4,
              help="Parameter for the stop condition  [default=1e-4]")
def optimization(NAME, INIT_K_BACKBONE, ETA, ALPHA):
    STOP_DELTA = ETA * ALPHA
    # -------------------------------------------------------------------------
    FILE_READ = NAME + "/C_normalized.txt"
    DIR_OPT = NAME + "/data_optimization"
    os.makedirs(DIR_OPT, exist_ok=True)
    FILE_LOG = DIR_OPT + "/optimization.log"
    # -------------------------------------------------------------------------
    C_normalized, N = Read_Normalized_C(FILE_READ)
    K = Set_Init_K(N, INIT_K_BACKBONE)
    tmp_C = Convert_K_into_C(K, N)
    Diff, Cost = Calc_Diff_Cost(tmp_C, C_normalized, N)
    print("Initial Cost = %f" % Cost)
    step = 0
    fp = open(FILE_LOG, "w")
    # -------------------------------------------------------------------------
    while True:
        step += 1
        tmp_Cost = Cost

        K -= ETA * Diff
        C = Convert_K_into_C(K, N)
        Diff, Cost = Calc_Diff_Cost(C, C_normalized, N)

        print("%d\t%e" % (step, Cost), file=fp)
        delta = tmp_Cost - Cost
        if 0 < delta < STOP_DELTA:
            FILE_OUT = DIR_OPT + "/K_optimized.txt"
            np.savetxt(FILE_OUT, K, fmt="%e")
            break
    # -------------------------------------------------------------------------
    fp.close()
    # -------------------------------------------------------------------------
    # Check whether the optimizaed K is physically acceptable or unrealistic
    L = Transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    flag = False
    for n in range(N - 1):
        if K[n, n + 1] < 0:
            flag = True

    if lam[1] < 0:
        flag = True

    if flag:
        print(
            "[Caution] Optimization failed! The optimized K is physically unrealistic.")
        print("Please carry out the optimization with different initial parameters.")
    else:
        print("Optimization succeeded! The optimized K is physically acceptable.")
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--res", "RES", type=int, required=True,
              help="Resolution of the bin size")
@click.option("--plt-max-c", "PLT_MAX_C", type=float, required=True,
              help="Maximum value of contact map")
@click.option("--plt-max-k-backbone", "PLT_MAX_K_BACKBONE", type=float, required=True,
              help="Maximum value of K_i,i+1 profile")
@click.option("--plt-max-k", "PLT_MAX_K", type=float, required=True,
              help="Maximum and minimum values of optimized K map")
@click.option("--plt-k-dis-bins", "PLT_K_DIS_BINS", type=int, required=True,
              help="The number of bins of distribution of optimized K values")
@click.option("--plt-max-k-dis", "PLT_MAX_K_DIS", type=float, required=True,
              help="Maximum value of the K distributioin")
def plot_optimization(NAME, RES, PLT_MAX_C, PLT_MAX_K_BACKBONE, PLT_MAX_K, PLT_K_DIS_BINS, PLT_MAX_K_DIS):
    # READ & OUTPUT FILES
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_C = NAME + "/C_normalized.txt"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    FILE_READ_Cost = DIR_OPT + "/optimization.log"
    FILE_OUT_C_OPT = DIR_OPT + "/C_optimized.txt"
    FILE_OUT_K_BACKBONE = DIR_OPT + "/K_backbone.txt"
    FILE_FIG_CORRELATION = DIR_OPT + "/Correlation.png"
    FILE_FIG_DC_CORRELATION = DIR_OPT + "/Correlation_distance_corrected.png"
    FILE_FIG_K_BACKBONE = DIR_OPT + "/K_backbone.svg"
    FILE_FIG_C = DIR_OPT + "/C.svg"
    FILE_FIG_K = DIR_OPT + "/K.svg"
    FILE_FIG_P = DIR_OPT + "/P.svg"
    FILE_FIG_K_DIS = DIR_OPT + "/K_distribution.svg"
    FILE_FIG_Cost = DIR_OPT + "/Cost.svg"
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    C_normalized, N = Read_Normalized_C(FILE_READ_C)
    K = np.loadtxt(FILE_READ_K)
    # -------------------------------------------------------------------------
    # CALC cost and correlation
    C_optimized = Convert_K_into_C(K, N)
    np.savetxt(FILE_OUT_C_OPT, C_optimized, fmt="%e")
    Diff, Cost = Calc_Diff_Cost(C_optimized, C_normalized, N)
    r, Optimized, Normalized = Calc_Correlation(C_optimized, C_normalized, N)
    dcr, Optimized_dcr, Normalized_dcr = Calc_Distance_Corrected_Correlation(
        C_optimized, C_normalized, N)
    # -------------------------------------------------------------------------
    P_normalized = Calc_P(C_normalized, RES)
    P_optimized = Calc_P(C_optimized, RES)
    # -------------------------------------------------------------------------
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i > j:
                C[i, j] = C_optimized[i, j]
            else:
                C[i, j] = C_normalized[i, j]
    # -------------------------------------------------------------------------
    list_K_wo_backbone = []
    for i in range(0, N - 2):
        for j in range(i + 2, N):
            list_K_wo_backbone.append(K[i, j])
    K_wo_backbone = np.array(list_K_wo_backbone)
    # -------------------------------------------------------------------------
    K_backbone = np.zeros((N - 1, 2))
    for i in range(N - 1):
        K_backbone[i, 0] = i
        K_backbone[i, 1] = K[i, i + 1]
    np.savetxt(FILE_OUT_K_BACKBONE, K_backbone, fmt="%d\t%f")
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.axes().set_aspect("equal")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.title(r"$r$={0:.3f}".format(r))
    plt.xlabel("PHi-C", fontweight="bold")
    plt.ylabel("Hi-C", fontweight="bold")
    x = np.linspace(0, 1)
    plt.plot(x, x, linestyle="dashed", color="gray", linewidth=3)
    plt.scatter(Optimized, Normalized, color="blue", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    plt.savefig(FILE_FIG_CORRELATION)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.axes().set_aspect("equal")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.title(r"$r'$={0:.3f}".format(dcr))
    plt.xlabel("PHi-C subtracted the average\nat each genomic distance", fontweight="bold")
    plt.ylabel("Hi-C subtracted the average\nat each genomic distance", fontweight="bold")
    x = np.linspace(-0.5, 0.5)
    plt.plot(x, x, linestyle="dashed", color="gray", linewidth=3)
    plt.scatter(Optimized_dcr, Normalized_dcr, color="blue", alpha=0.5)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xticks([-0.4, -0.2, 0, 0.2, 0.4])
    plt.yticks([-0.4, -0.2, 0, 0.2, 0.4])
    plt.tight_layout()
    plt.savefig(FILE_FIG_DC_CORRELATION)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(20, 6))
    plt.xlabel(r"index $i$ ({0:d}-bp bins)".format(RES), fontweight="bold")
    plt.ylabel(r"Nomrlaized $K_{i, i+1}$", fontweight="bold")
    plt.ylim(0, PLT_MAX_K_BACKBONE)
    plt.bar(K_backbone[:, 0], K_backbone[:, 1],
            width=1.0,
            color="#FF0000",
            linewidth=0)
    plt.tight_layout()
    plt.savefig(FILE_FIG_K_BACKBONE)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.text(N - 1, 0, "Hi-C", fontweight="bold", ha="right", va="top")
    plt.text(0, N - 1, "PHi-C\n" + r"($r$={0:.3f}, $r'$={1:.3f})".format(r, dcr),
             fontweight="bold", ha="left", va="bottom")
    plt.imshow(C, cmap="magma_r", clim=(0, PLT_MAX_C))
    plt.colorbar(ticks=[0, PLT_MAX_C], shrink=0.5, orientation="vertical",
                 label="Normalized contact probability")
    plt.axis("off")
    plt.savefig(FILE_FIG_C)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(K, cmap="bwr", clim=(-PLT_MAX_K, PLT_MAX_K))
    plt.colorbar(ticks=[-PLT_MAX_K, 0, PLT_MAX_K], shrink=0.5, orientation="vertical",
                 label=r"Nomrlaized $K_{ij}$")
    plt.axis("off")
    plt.savefig(FILE_FIG_K)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Genomic distance [bp]", fontweight="bold")
    plt.ylabel("Normalized contact probability", fontweight="bold")
    plt.plot(P_normalized[1:, 0], P_normalized[1:, 1],
             label="Hi-C", linewidth=6)
    plt.plot(P_optimized[1:, 0], P_optimized[1:, 1],
             label="PHi-C", linewidth=3)
    plt.legend(handlelength=1, loc="upper right")
    plt.tight_layout()
    plt.savefig(FILE_FIG_P)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.xlabel(r"Nomrlaized $K_{ij}$", fontweight="bold")
    plt.ylabel("Probability density", fontweight="bold")
    plt.hist(K_wo_backbone, bins=PLT_K_DIS_BINS,
             range=(-PLT_MAX_K, PLT_MAX_K), density=True)
    plt.xlim(-PLT_MAX_K, PLT_MAX_K)
    plt.ylim(0, PLT_MAX_K_DIS)
    plt.xticks([-PLT_MAX_K, -PLT_MAX_K / 2, 0, PLT_MAX_K / 2, PLT_MAX_K])
    plt.tight_layout()
    plt.savefig(FILE_FIG_K_DIS)
    plt.close()
    # -------------------------------------------------------------------------
    data = np.loadtxt(FILE_READ_Cost)
    plt.figure(figsize=(10, 5))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.xlabel("Iteration", fontweight="bold")
    plt.ylabel("Cost", fontweight="bold")
    plt.xlim(0, data[-1, 0])
    plt.ylim(0, data[0, 1])
    plt.plot(data[:, 0], data[:, 1], linewidth=4)
    plt.tight_layout()
    plt.savefig(FILE_FIG_Cost)
    plt.close()
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--eps", "EPS", type=float, default=1e-3,
              help="Stepsize in the Langevin dynamics  [default=1e-3]")
@click.option("--interval", "INTERVAL", type=int, required=True,
              help="The number of steps between output frames")
@click.option("--frame", "FRAME", type=int, required=True,
              help="The number of output frames")
@click.option("--sample", "SAMPLE", type=int, default=1,
              help="The number of output dynamics  [default=1]")
@click.option("--seed", "SEED", type=int, default=12345678,
              help="Seed of the random numbers  [default=12345678]")
def dynamics(NAME, EPS, INTERVAL, FRAME, SAMPLE, SEED):
    NOISE = np.sqrt(2 * EPS)
    F_Coefficient = 3 * EPS
    np.random.seed(SEED)
    # -------------------------------------------------------------------------
    FILE_READ_K = NAME + "/data_optimization/K_optimized.txt"
    DIR_4D = NAME + "/data_dynamics"
    os.makedirs(DIR_4D, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    Write_Psfdata(DIR_4D, NAME, N)
    L = Transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    for sample in range(SAMPLE):
        Xx, Xy, Xz = Equilibrium_Conformation_of_Normal_Coordinates(lam, N)
        Rx, Ry, Rz = Convert_X_to_R(Xx, Xy, Xz, Q)
        # ---------------------------------------------------------------------
        FILE_OUT = DIR_4D + "/sample{0:d}.xyz".format(sample)
        fp = open(FILE_OUT, "w")
        for frame in range(FRAME + 1):
            # -----------------------------------------------------------------
            print("%d" % N, file=fp)
            print("frame = %d" % frame, file=fp)
            for n in range(N):
                print("CA\t%f\t%f\t%f" % (Rx[n], Ry[n], Rz[n]), file=fp)
            # -----------------------------------------------------------------
            for step in range(INTERVAL):
                Rx, Ry, Rz = Integrate_Polymer_Network(
                    Rx, Ry, Rz, L, N, NOISE, F_Coefficient)
        fp.close()
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--sample", "SAMPLE", type=int, required=True,
              help="The number of output conformations")
@click.option("--seed", "SEED", type=int, default=12345678,
              help="Seed of the random numbers  [default=12345678]")
def sampling(NAME, SAMPLE, SEED):
    np.random.seed(SEED)
    # -------------------------------------------------------------------------
    FILE_READ_K = NAME + "/data_optimization/K_optimized.txt"
    DIR_3D = NAME + "/data_sampling"
    os.makedirs(DIR_3D, exist_ok=True)
    FILE_OUT = DIR_3D + "/conformations.xyz"
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    Write_Psfdata(DIR_3D, NAME, N)
    L = Transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    # -------------------------------------------------------------------------
    fp = open(FILE_OUT, "w")
    for sample in range(SAMPLE):
        Xx, Xy, Xz = Equilibrium_Conformation_of_Normal_Coordinates(lam, N)
        Rx, Ry, Rz = Convert_X_to_R(Xx, Xy, Xz, Q)
        # ---------------------------------------------------------------------
        print("%d" % N, file=fp)
        print("sample = %d" % sample, file=fp)
        for n in range(N):
            print("CA\t%f\t%f\t%f" % (Rx[n], Ry[n], Rz[n]), file=fp)
        # ---------------------------------------------------------------------
    fp.close()
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=1,
              help="Upper value of the exponent of the angular frequency  [default=1]")
@click.option("--lower", "LOWER", type=int, default=-5,
              help="Lower value of the exponent of the angular frequency  [default=-5]")
def rheology(NAME, UPPER, LOWER):
    M = 100 * (UPPER - LOWER)
    # -------------------------------------------------------------------------
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    DIR = NAME + "/data_rheology"
    os.makedirs(DIR, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    L = Transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    lam[0] = 0
    omega_1 = 6 * np.pi * lam[1]
    with open("{0:s}/data_normalized_omega1.txt".format(DIR), "w") as fp:
        print("ω1 = %e" % omega_1, file=fp)
        print("log10(ω1) = %f" % (np.log10(omega_1)), file=fp)
    # -------------------------------------------------------------------------
    for n in range(N):
        data = np.zeros((M + 1, 8))
        for m in range(M + 1):
            # -----------------------------------------------------------------
            omega = 10**(0.01 * (m + LOWER * 100))
            data[m, 0] = omega
            # -----------------------------------------------------------------
            for p in range(1, N):
                tmp = Q[n, p] * Q[n, p] / (omega * omega + 9 * lam[p] * lam[p])
                data[m, 1] += 3 * lam[p] * tmp  # J'
                data[m, 2] += omega * tmp  # J''
            # -----------------------------------------------------------------
            J2 = data[m, 1] * data[m, 1] + data[m, 2] * data[m, 2]
            data[m, 3] = np.sqrt(J2)  # |J*|
            data[m, 4] = data[m, 1] / J2  # G'
            data[m, 5] = data[m, 2] / J2  # G''
            data[m, 6] = np.sqrt(data[m, 4] * data[m, 4] +
                                 data[m, 5] * data[m, 5])  # |G*|
            data[m, 7] = data[m, 2] / data[m, 1]  # tanδ
        # ---------------------------------------------------------------------
        FILE_OUT = DIR + "/n{0:d}.txt".format(n)
        np.savetxt(FILE_OUT, data, fmt="%e")
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=1,
              help="Upper value of the exponent of the angular frequency  [default=1]")
@click.option("--lower", "LOWER", type=int, default=-5,
              help="Lower value of the exponent of the angular frequency  [default=-5]")
@click.option("--plt-upper", "PLT_UPPER", type=int, required=True,
              help="Upper value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-lower", "PLT_LOWER", type=int, required=True,
              help="Lower value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-max-log", "PLT_MAX_LOG", type=float, required=True,
              help="Maximum value of log10 |J*|")
@click.option("--plt-min-log", "PLT_MIN_LOG", type=float, required=True,
              help="Minimum value of log10 |J*|")
@click.option("--aspect", "ASPECT", type=float, default=0.8,
              help="Aspect ratio of the spectrum  [default=0.8]")
def plot_compliance(NAME, UPPER, LOWER, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, PLT_MIN_LOG, ASPECT):
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    DIR = NAME + "/data_rheology"
    FILE_OUT_SPECTRUM_STORAGE = DIR + "/data_J_storage_spectrum.txt"
    FILE_OUT_SPECTRUM_LOSS = DIR + "/data_J_loss_spectrum.txt"
    FILE_OUT_SPECTRUM_ABS = DIR + "/data_J_abs_spectrum.txt"
    DIR_FIG = DIR + "/figs"
    os.makedirs(DIR_FIG, exist_ok=True)
    FILE_FIG_CURVES = DIR_FIG + "/J_curves.png"
    FILE_FIG_SPECTRUM_STORAGE = DIR_FIG + "/J_storage_spectrum.svg"
    FILE_FIG_SPECTRUM_LOSS = DIR_FIG + "/J_loss_spectrum.svg"
    FILE_FIG_SPECTRUM_ABS = DIR_FIG + "/J_abs_spectrum.svg"
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 9))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**(LOWER), 10**(UPPER))
    plt.xlabel(r"$\mathrm{\mathbf{\bar{\omega}}}$")
    plt.ylabel(r"$\mathrm{\mathbf{Normalized\ compliance}}$")
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        plt.plot(data[:, 0], data[:, 1], linewidth=1, color="blue", alpha=0.1)
        plt.plot(data[:, 0], data[:, 2], linewidth=1, color="red", alpha=0.1)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.savefig(FILE_FIG_CURVES)
    plt.close()
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 24
    # -------------------------------------------------------------------------
    M = 100 * (UPPER - LOWER)
    START = 100 * (PLT_LOWER - LOWER)
    END = 100 * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = []
    for n in range(PLT_LOWER, PLT_UPPER + 1):
        YTICKS_LABELS.append(n)
    J_storage = np.zeros((M + 1, N))
    J_loss = np.zeros((M + 1, N))
    J_abs = np.zeros((M + 1, N))
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        J_storage[:, n] = data[:, 1]
        J_loss[:, n] = data[:, 2]
        J_abs[:, n] = data[:, 3]
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(J_storage[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \bar{J}'(\bar{\omega})}}$",
                 ticks=[0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_STORAGE)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(J_loss[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \bar{J}''(\bar{\omega})}}$",
                 ticks=[0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_LOSS)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(J_abs[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} |\bar{J}*(\bar{\omega})|}}$",
                 ticks=[0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_ABS)
    plt.close()
    # -------------------------------------------------------------------------
    np.savetxt(FILE_OUT_SPECTRUM_STORAGE, J_storage[START:END, :], fmt="%e")
    np.savetxt(FILE_OUT_SPECTRUM_LOSS, J_loss[START:END, :], fmt="%e")
    np.savetxt(FILE_OUT_SPECTRUM_ABS, J_abs[START:END, :], fmt="%e")
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=1,
              help="Upper value of the exponent of the angular frequency  [default=1]")
@click.option("--lower", "LOWER", type=int, default=-5,
              help="Lower value of the exponent of the angular frequency  [default=-5]")
@click.option("--plt-upper", "PLT_UPPER", type=int, required=True,
              help="Upper value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-lower", "PLT_LOWER", type=int, required=True,
              help="Lower value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-max-log", "PLT_MAX_LOG", type=float, required=True,
              help="Maximum value of log10 |G*|")
@click.option("--plt-min-log", "PLT_MIN_LOG", type=float, required=True,
              help="Minimum value of log10 |G*|")
@click.option("--aspect", "ASPECT", type=float, default=0.8,
              help="Aspect ratio of the spectrum  [default=0.8]")
def plot_modulus(NAME, UPPER, LOWER, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, PLT_MIN_LOG, ASPECT):
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    DIR = NAME + "/data_rheology"
    FILE_OUT_SPECTRUM_STORAGE = DIR + "/data_G_storage_spectrum.txt"
    FILE_OUT_SPECTRUM_LOSS = DIR + "/data_G_loss_spectrum.txt"
    FILE_OUT_SPECTRUM_ABS = DIR + "/data_G_abs_spectrum.txt"
    DIR_FIG = DIR + "/figs"
    os.makedirs(DIR_FIG, exist_ok=True)
    FILE_FIG_CURVES = DIR_FIG + "/G_curves.png"
    FILE_FIG_SPECTRUM_STORAGE = DIR_FIG + "/G_storage_spectrum.svg"
    FILE_FIG_SPECTRUM_LOSS = DIR_FIG + "/G_loss_spectrum.svg"
    FILE_FIG_SPECTRUM_ABS = DIR_FIG + "/G_abs_spectrum.svg"
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 9))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**(LOWER), 10**(UPPER))
    plt.xlabel(r"$\mathrm{\mathbf{\bar{\omega}}}$")
    plt.ylabel(r"$\mathrm{\mathbf{Normalized\ modulus}}$")
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        plt.plot(data[:, 0], data[:, 4], linewidth=1, color="blue", alpha=0.1)
        plt.plot(data[:, 0], data[:, 5], linewidth=1, color="red", alpha=0.1)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.savefig(FILE_FIG_CURVES)
    plt.close()
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 24
    # -------------------------------------------------------------------------
    M = 100 * (UPPER - LOWER)
    START = 100 * (PLT_LOWER - LOWER)
    END = 100 * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = []
    for n in range(PLT_LOWER, PLT_UPPER + 1):
        YTICKS_LABELS.append(n)
    G_storage = np.zeros((M + 1, N))
    G_loss = np.zeros((M + 1, N))
    G_abs = np.zeros((M + 1, N))
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        G_storage[:, n] = data[:, 4]
        G_loss[:, n] = data[:, 5]
        G_abs[:, n] = data[:, 6]
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(G_storage[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \bar{G}'(\bar{\omega})}}$",
                 ticks=[-1, 0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_STORAGE)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(G_loss[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \bar{G}''(\bar{\omega})}}$",
                 ticks=[-1, 0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_LOSS)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(G_abs[START:END, :]),
               cmap="jet",
               clim=(PLT_MIN_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} |\bar{G}*(\bar{\omega})|}}$",
                 ticks=[-1, 0, 1])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_ABS)
    plt.close()
    # -------------------------------------------------------------------------
    np.savetxt(FILE_OUT_SPECTRUM_STORAGE, G_storage[START:END, :], fmt="%e")
    np.savetxt(FILE_OUT_SPECTRUM_LOSS, G_loss[START:END, :], fmt="%e")
    np.savetxt(FILE_OUT_SPECTRUM_ABS, G_abs[START:END, :], fmt="%e")
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=1,
              help="Upper value of the exponent of the angular frequency  [default=1]")
@click.option("--lower", "LOWER", type=int, default=-5,
              help="Lower value of the exponent of the angular frequency  [default=-5]")
@click.option("--plt-upper", "PLT_UPPER", type=int, required=True,
              help="Upper value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-lower", "PLT_LOWER", type=int, required=True,
              help="Lower value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-max-log", "PLT_MAX_LOG", type=float, required=True,
              help="Maximum value of log10 tanδ")
@click.option("--aspect", "ASPECT", type=float, default=0.8,
              help="Aspect ratio of the spectrum  [default=0.8]")
def plot_tangent(NAME, UPPER, LOWER, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, ASPECT):
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    DIR = NAME + "/data_rheology"
    FILE_OUT_SPECTRUM = DIR + "/data_tan_spectrum.txt"
    DIR_FIG = DIR + "/figs"
    os.makedirs(DIR_FIG, exist_ok=True)
    FILE_FIG_SPECTRUM = DIR_FIG + "/tan_spectrum.svg"
    # -------------------------------------------------------------------------
    K, N = Read_K(FILE_READ_K)
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 24
    # -------------------------------------------------------------------------
    M = 100 * (UPPER - LOWER)
    START = 100 * (PLT_LOWER - LOWER)
    END = 100 * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = []
    for n in range(PLT_LOWER, PLT_UPPER + 1):
        YTICKS_LABELS.append(n)
    tan = np.zeros((M + 1, N))
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        tan[:, n] = data[:, 7]
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(tan[START:END, :]),
               cmap="coolwarm",
               clim=(-PLT_MAX_LOG, PLT_MAX_LOG),
               aspect=ASPECT,
               origin="lower")
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \mathbf{tan δ(\bar{\omega})}}}$",
                 ticks=[-PLT_MAX_LOG, 0, PLT_MAX_LOG])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM)
    plt.close()
    # -------------------------------------------------------------------------
    np.savetxt(FILE_OUT_SPECTRUM, tan[START:END, :], fmt="%e")
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    cli()
