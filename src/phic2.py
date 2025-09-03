import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import click
import hicstraw # for version >= 2.1.0
import pandas as pd # for version >= 2.1.0
import cooler # for version >= 2.1.1
import h5py # for version >= 2.1.1
# -----------------------------------------------------------------------------

def calc_C_normalized(C):
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

def calc_C_normalized_high_resolution(C):
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

def calc_Ps(C, RES):
    N = C.shape[0]
    P = np.full((N, 2), np.nan)
    for n in range(N):
        P[n, 0] = RES * n

        C_elements = []
        for i in range(N-n):
            C_elements.append(C[i, i+n])

        C_elements = [x for x in C_elements if not np.isnan(x)]
        
        if C_elements:
            P[n, 1] = np.mean(C_elements)            
    return P
# -----------------------------------------------------------------------------

def read_normalized_C(FILE_READ):
    C = np.loadtxt(FILE_READ)
    N = C.shape[0]
    return C, N
# -----------------------------------------------------------------------------

def set_init_K(N, INIT_K_BACKBONE):
    K = np.zeros((N, N))
    for i in range(N - 1):
        j = i + 1
        K[i, j] = K[j, i] = INIT_K_BACKBONE
    return K
# -----------------------------------------------------------------------------

def convert_K_into_C(K, N):
    # K to L
    d = np.sum(K, axis=0)
    L = np.diag(d) - K

    # L to M
    lam, Q = np.linalg.eigh(L)
    inv_lam = np.zeros_like(lam)
    nonzero = lam > 1e-12
    inv_lam[nonzero] = 1.0 / lam[nonzero]
    inv_lam[0] = 0.0
    M = (Q * inv_lam) @ Q.T

    # M to Σ^2
    md = np.diag(M)
    Sigma2 = (md[:, None] + md[None, :] - 2.0 * M) / 3.0

    # Σ^2 to C
    C = (1.0 + Sigma2)**(-1.5)

    # All eigenvalues have to be positive
    error_flag = lam[0] < -1e-10

    return C, error_flag
# -----------------------------------------------------------------------------

def calc_diff_cost(A, B, N):
    diff = A - B
    cost = np.linalg.norm(diff, "fro") / N
    return diff, cost
# -----------------------------------------------------------------------------

def calc_correlation(A, B, N):
    upper_indices = np.triu_indices(N, k=1)
    X = A[upper_indices]
    Y = B[upper_indices]
    valid_indices = ~np.isnan(X)
    X_filtered = X[valid_indices]
    Y_filtered = Y[valid_indices]
    # -------------------------------------------------------------------------
    r, p = scipy.stats.pearsonr(X_filtered, Y_filtered)
    return r, X_filtered, Y_filtered
# -----------------------------------------------------------------------------

def calc_distance_corrected_correlation(A, B, N, P_A, P_B):
    tmp_A = np.zeros((N, N))
    tmp_B = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if A[i, j] != np.nan:
                tmp_A[i, j] = A[i, j] - P_A[j - i]
                tmp_B[i, j] = B[i, j] - P_B[j - i]
    # -------------------------------------------------------------------------
    upper_indices = np.triu_indices(N, k=1)
    X = tmp_A[upper_indices]
    Y = tmp_B[upper_indices]
    valid_indices = ~np.isnan(X)
    X_filtered = X[valid_indices]
    Y_filtered = Y[valid_indices]
    # -------------------------------------------------------------------------
    r, p = scipy.stats.pearsonr(X_filtered, Y_filtered)
    return r, X_filtered, Y_filtered
# -----------------------------------------------------------------------------

def read_K(FILE_READ_K):
    K = np.loadtxt(FILE_READ_K)
    N = K.shape[0]
    return K, N
# -----------------------------------------------------------------------------

def transform_K_into_L(K):
    d = np.sum(K, axis=0)
    D = np.diag(d)
    L = D - K
    return L
# -----------------------------------------------------------------------------

def equilibrium_conformation_of_normal_coordinates(lam, N):
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

def convert_X_to_R(Xx, Xy, Xz, Q):
    Rx = np.dot(Q, Xx)
    Ry = np.dot(Q, Xy)
    Rz = np.dot(Q, Xz)
    return Rx, Ry, Rz
# -----------------------------------------------------------------------------

def integrate_polymer_network(x, y, z, L, N, NOISE, F_Coefficient):
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

def write_psfdata(psf_path, NAME, N):
    fp = open(psf_path, "w")
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

# Supports both .hic and .mcool files.
def make_input_contact_matrix(FILE_INPUT, RES, CHR, START, END, NORM):
    NAME, EXT = os.path.splitext(os.path.basename(FILE_INPUT))

    if EXT == ".hic": # for version >= 2.1.0
        if not (isinstance(START, int) and isinstance(END, int)): # for the whole single chromosome
            # Set START and END for the target chromosome ID
            hic = hicstraw.HiCFile(FILE_INPUT)
            START = int(0)
            for chrom in hic.getChromosomes():
                if chrom.name == CHR:
                    END = chrom.length
                    break
            # Set an input raw contact matrix with nan-values
            N_input = int(END / RES) + 1
            C_input = np.full((N_input, N_input), np.nan)
            result = hicstraw.straw("observed", NORM, FILE_INPUT, CHR, CHR, "BP", RES)
            for k in range(len(result)):
                l = int((result[k].binX - START) / RES)
                m = int((result[k].binY - START) / RES)
                C_input[l, m] = C_input[m, l] = result[k].counts
            # Set the name of the working directory
            DIR = f"{NAME}_{NORM}_chr{CHR}_res{RES}bp"
        else:
            # Set an input raw contact matrix with nan-values
            N_input = int((END - START) / RES)
            C_input = np.full((N_input, N_input), np.nan)
            #ROI = "{0:s}:{1:d}:{2:d}".format(CHR, START, END - RES)
            ROI = f"{CHR}:{START}:{END - RES}"
            result = hicstraw.straw("observed", NORM, FILE_INPUT, ROI, ROI, "BP", RES)
            for k in range(len(result)):
                l = int((result[k].binX - START) / RES)
                m = int((result[k].binY - START) / RES)
                C_input[l, m] = C_input[m, l] = result[k].counts
            # Set the name of the working directory
            DIR = f"{NAME}_{NORM}_chr{CHR}_{START}-{END}_res{RES}bp"
    elif EXT == ".mcool": # for mcool files # for version >= 2.1.1
        if not (isinstance(START, int) and isinstance(END, int)):  # for the whole single chromosome
            hic = cooler.Cooler(f"{FILE_INPUT}::resolutions/{RES}")
            START = int(0)
            for chrom in hic.chromnames:
                if chrom == CHR:
                    END = hic.chromsizes[chrom]
                    break
            coo = hic.matrix(balance=NORM, sparse=True).fetch(f"{CHR}") # sparse=True. Returns scipy.sparse.coo_matrix.
            C_input = np.full(coo.shape, np.nan)
            C_input[coo.row, coo.col] = coo.data
            # Set the name of the working directory
            DIR = f"{NAME}_{NORM}_{CHR}_res{RES}bp"
        else:
            hic = cooler.Cooler(f"{FILE_INPUT}::resolutions/{RES}")
            coo = hic.matrix(balance=NORM, sparse=True).fetch(f"{CHR}:{START}-{END}") # sparse=True. Returns scipy.sparse.coo_matrix.
            C_input = np.full(coo.shape, np.nan)
            C_input[coo.row, coo.col] = coo.data
            # Set the name of the working directory
            DIR = f"{NAME}_{NORM}_{CHR}_{START}-{END}_res{RES}bp"
        N_input = C_input.shape[0]
    else: # for version <= 2.0.13
        print("Version 2.1.0 and above no longer support input in formats other than .hic or .mcool.")

    os.makedirs(DIR, exist_ok=True)
    return C_input, N_input, DIR, START, END
# -----------------------------------------------------------------------------

def remove_invalid_segments(C_input, N_input, TOLERANCE):
    nan_ratio = np.isnan(C_input).sum(axis=1) / N_input
    nan_indices = np.where(nan_ratio > TOLERANCE)[0]

    C_input_nan_filtered = np.delete(np.delete(C_input, nan_indices, axis=0), nan_indices, axis=1)
    C_for_phic = np.nan_to_num(C_input_nan_filtered, nan=0)
    N_for_phic = C_for_phic.shape[0]

    return C_for_phic, N_for_phic, nan_indices
# -----------------------------------------------------------------------------

def write_meta_data(DIR, FILE_INPUT, NORM, CHR, START, END, RES, N_input, N_for_phic, nan_indices, TOLERANCE):
    DIR_META = DIR + "/_meta_data"
    os.makedirs(DIR_META, exist_ok=True)

    with open(DIR_META + "/_fetched_data_info.txt", "w") as file:
        file.write(f"filename,{FILE_INPUT}\n")
        file.write(f"normalization,{NORM}\n")
        file.write(f"chromosome ID,{CHR}\n")
        file.write(f"start genomic position,{START}\n")
        file.write(f"end genomic position,{END}\n")
        file.write(f"resolution,{RES}\n")
        file.write(f"input contact matrix size,{N_input}x{N_input}\n")
        file.write(f"phic contact matrix size,{N_for_phic}x{N_for_phic}\n")
        file.write(f"tolerance,{TOLERANCE}\n")

    with open(DIR_META + "/_removed_segments.txt", "w") as file:
        file.write(f"index,chrom,chromStart,chromEnd\n")
        for i in range(len(nan_indices)):
            grs = START + nan_indices[i] * RES
            gre = START + (nan_indices[i] + 1) * RES
            file.write(f"{nan_indices[i]},{CHR},{grs},{gre}\n")

    all_indices = list(range(N_input))
    valid_indices = list(set(all_indices) - set(nan_indices))
    with open(DIR_META + "/_remaining_segments.txt", "w") as file:
        file.write(f"index,chrom,chromStart,chromEnd,new index\n")
        for i in range(len(valid_indices)):
            grs = START + valid_indices[i] * RES
            gre = START + (valid_indices[i] + 1) * RES
            file.write(f"{valid_indices[i]},{CHR},{grs},{gre},{i}\n")
# -----------------------------------------------------------------------------

def write_optimizatioin_meta_data(DIR, INIT_K_BACKBONE, ETA, ALPHA):
    DIR_META = DIR + "/_meta_data"
    os.makedirs(DIR_META, exist_ok=True)

    with open(DIR_META + "/_optimization_hyper_parameters.txt", "w") as file:
        file.write(f"initial K along backbone,{INIT_K_BACKBONE}\n")
        file.write(f"learning rate,{ETA:.2e}\n")
        file.write(f"stop condition parameter,{ALPHA:.2e}\n")
# -----------------------------------------------------------------------------

def calc_plot_correlations(C_optimized, C_normalized, N, P_optimized, P_normalized, DIR_OPT):
    r, Optimized, Normalized = calc_correlation(C_optimized, C_normalized, N)
    dcr, Optimized_dcr, Normalized_dcr = calc_distance_corrected_correlation(C_optimized, C_normalized, N, P_optimized[:, 1], P_normalized[:, 1])
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    FILE_FIG_CORRELATION = DIR_OPT + "/Correlation.png"
    FILE_FIG_DC_CORRELATION = DIR_OPT + "/Correlation_distance_corrected.png"
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

    return r, dcr
# -----------------------------------------------------------------------------

@click.group()
def cli():
    pass
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--input", "FILE_INPUT", required=True,
              help="Input Hi-C file (.hic or .mcool format)")
def fetch_fileinfo(FILE_INPUT):
    NAME, EXT = os.path.splitext(os.path.basename(FILE_INPUT))
    if EXT == ".hic":  # for .hic files
        hic = hicstraw.HiCFile(FILE_INPUT)
        print("chromosome, length")
        [print(chrom.name, chrom.length) for chrom in hic.getChromosomes()]
        print("------------------------")
        print("(usually) available normalization methods:", ["NONE", "VC", "VC_SQRT", "KR", "SCALE"])
        print("-------------------------")
        print("available resolutions:", hic.getResolutions())
        print("-------------------------")
    elif EXT == ".mcool":  # for .mcool files
        with h5py.File(FILE_INPUT, "r") as f:
            resolutions = list(f["resolutions"].keys())
        hic = cooler.Cooler(f'{FILE_INPUT}::resolutions/{resolutions[0]}')  # Use a default resolution to access metadata
        print("chromosome, length")
        [print(chromname, hic.chromsizes[chromname]) for chromname in hic.chromnames]
        print("------------------------")
        print("(usually) available normalization methods:", [None, True, "KR", "VC", "VC_SQRT"])
        print("-------------------------")
        print("available resolutions:", resolutions)
        print("-------------------------")      
    else:
        print("Unsupported file format.")
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--input", "FILE_INPUT", required=True,
              help="Input Hi-C file (.hic or .mcool format)")
@click.option("--res", "RES", type=int, required=True,
              help="Resolution of the bin size")
@click.option("--plt-max-c", "PLT_MAX_C", type=float, required=True,
              help="Maximum value of contact map")
@click.option("--for-high-resolution", "HIGH_RESOLUTION", type=int, default=0,
              help="Normalization of contact map for high-resolution case (ex. 1-kb, 500-bp, 200-bp)  [default=0]")
@click.option("--chr", "CHR", type=str, required=True,
              help="Target chromosome")
@click.option("--grs", "START", type=int, default=None,
              help="Start position of the target genomic region")
@click.option("--gre", "END", type=int, default=None,
              help="End position of the target genomic region")
@click.option("--norm", "NORM", type=str, default=None,
              help="Type of normalization to apply")
@click.option("--tolerance", "TOLERANCE", type=float, required=True,
              help="Threshold used to remove segments containing NaN values")
def preprocessing(FILE_INPUT, RES, PLT_MAX_C, HIGH_RESOLUTION, CHR, START, END, NORM, TOLERANCE):
    C_input, N_input, DIR, START, END = make_input_contact_matrix(FILE_INPUT, RES, CHR, START, END, NORM)
    C_for_phic, N_for_phic, nan_indices = remove_invalid_segments(C_input, N_input, TOLERANCE)
    write_meta_data(DIR, FILE_INPUT, NORM, CHR, START, END, RES, N_input, N_for_phic, nan_indices, TOLERANCE)
    # -------------------------------------------------------------------------
    FILE_OUT_C_NORMALIZED = DIR + "/C_normalized.txt"
    FILE_OUT_P_NORMALIZED = DIR + "/P_normalized.txt"
    FILE_FIG_C_NORMALIZED = DIR + "/C_normalized.svg"
    FILE_FIG_P_NORMALIZED = DIR + "/P_normalized.svg"
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    cmap_for_C = plt.get_cmap("magma_r")
    cmap_for_C.set_bad(color=(0.8, 0.8, 0.8))
    # -------------------------------------------------------------------------
    if HIGH_RESOLUTION:
        C_normalized = calc_C_normalized_high_resolution(C_for_phic)
    else:
        C_normalized = calc_C_normalized(C_for_phic)
    np.savetxt(FILE_OUT_C_NORMALIZED, C_normalized, fmt="%e")
    # -------------------------------------------------------------------------
    for idx in nan_indices:
        N = C_normalized.shape[0]
        C_normalized = np.insert(np.insert(C_normalized, idx, np.full(N, np.nan), axis=0), idx, np.full(N+1, np.nan), axis=1)
    P_normalized = calc_Ps(C_normalized, RES)
    np.savetxt(FILE_OUT_P_NORMALIZED, P_normalized, fmt="%d\t%e")
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(C_normalized,
               cmap=cmap_for_C,
               interpolation="none", vmin=0, vmax=PLT_MAX_C)
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
    write_optimizatioin_meta_data(NAME, INIT_K_BACKBONE, ETA, ALPHA)
    STOP_DELTA = ETA * ALPHA
    # -------------------------------------------------------------------------
    FILE_READ = NAME + "/C_normalized.txt"
    DIR_OPT = NAME + "/data_optimization"
    os.makedirs(DIR_OPT, exist_ok=True)
    FILE_LOG = DIR_OPT + "/optimization.log"
    # -------------------------------------------------------------------------
    C_normalized, N = read_normalized_C(FILE_READ)
    K = set_init_K(N, INIT_K_BACKBONE)
    tmp_C, error_flag = convert_K_into_C(K, N)
    diff, cost = calc_diff_cost(tmp_C, C_normalized, N)
    print("Initial Cost = %f" % cost)
    step = 0
    fp = open(FILE_LOG, "w")
    print("%d\t%e" % (step, cost), file=fp)
    # -------------------------------------------------------------------------
    while True:
        step += 1
        tmp_K = K.copy()
        tmp_cost = cost

        K -= ETA * diff
        C, error_flag = convert_K_into_C(K, N)

        if error_flag:
            print(f"Stopping optimization at step {step} due to negative eigenvalue")
            FILE_OUT = DIR_OPT + "/K_optimized.txt"
            np.savetxt(FILE_OUT, tmp_K, fmt="%e")
            break

        diff, cost = calc_diff_cost(C, C_normalized, N)
        print("%d\t%e" % (step, cost), file=fp)
        delta = tmp_cost - cost
        if 0 < delta < STOP_DELTA:
            FILE_OUT = DIR_OPT + "/K_optimized.txt"
            np.savetxt(FILE_OUT, K, fmt="%e")
            break
    # -------------------------------------------------------------------------
    fp.close()
    # -------------------------------------------------------------------------
    # Check whether the optimizaed K is physically acceptable or unrealistic
    if not error_flag:
        L = transform_K_into_L(K)
        lam, Q = np.linalg.eigh(L)

        # The following condition, which was required for version <= 2.0.13, has been removed.
        # The positive semidefiniteness of L is a critical condition for the system's stability.
        #flag = False
        #for n in range(N - 1):
        #    if K[n, n + 1] < 0:
        #        flag = True
        #if lam[1] < 0:
        #    flag = True
        
        if lam[0] < -1e-10:
            print("[Caution] Optimization failed! The optimized K is physically unrealistic.")
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
@click.option("--plt-max-k", "PLT_MAX_K", type=float, required=True,
              help="Maximum and minimum values of optimized K map")
def plot_optimization(NAME, RES, PLT_MAX_C, PLT_MAX_K):
    # READ & OUTPUT FILES
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_C = NAME + "/C_normalized.txt"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    FILE_READ_Cost = DIR_OPT + "/optimization.log"
    FILE_OUT_C_OPT = DIR_OPT + "/C_optimized.txt"
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    FILE_FIG_C = DIR_OPT + "/C.svg"
    FILE_FIG_K = DIR_OPT + "/K.svg"
    FILE_FIG_P = DIR_OPT + "/P.svg"
    FILE_FIG_Cost = DIR_OPT + "/Cost.svg"
    # The following files, which was output for version <= 2.0.13, has been removed.
    #FILE_OUT_K_BACKBONE = DIR_OPT + "/K_backbone.txt"
    #FILE_FIG_K_BACKBONE = DIR_OPT + "/K_backbone.svg"
    #FILE_FIG_K_DIS = DIR_OPT + "/K_distribution.svg"
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    cmap_for_C = plt.get_cmap("magma_r")
    cmap_for_C.set_bad(color=(0.8, 0.8, 0.8))
    cmap_for_K = plt.get_cmap("bwr")
    cmap_for_K.set_bad(color=(0.8, 0.8, 0.8))
    # -------------------------------------------------------------------------
    C_normalized, N = read_normalized_C(FILE_READ_C)
    K = np.loadtxt(FILE_READ_K)
    C_optimized, error_flag = convert_K_into_C(K, N)
    np.savetxt(FILE_OUT_C_OPT, C_optimized, fmt="%e")
    # -------------------------------------------------------------------------
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            N = C_normalized.shape[0]
            C_normalized = np.insert(np.insert(C_normalized, idx, np.full(N, np.nan), axis=0), idx, np.full(N+1, np.nan), axis=1)
            C_optimized = np.insert(np.insert(C_optimized, idx, np.full(N, np.nan), axis=0), idx, np.full(N+1, np.nan), axis=1)
            K = np.insert(np.insert(K, idx, np.full(N, np.nan), axis=0), idx, np.full(N+1, np.nan), axis=1)

    N = C_normalized.shape[0]
    C = np.zeros((N, N))
    C[np.triu_indices(N, k=0)] = C_normalized[np.triu_indices(N, k=0)]
    C[np.tril_indices(N, k=-1)] = C_optimized[np.tril_indices(N, k=-1)]

    P_normalized = calc_Ps(C_normalized, RES)
    P_optimized = calc_Ps(C_optimized, RES)
    # -------------------------------------------------------------------------
    r, dcr = calc_plot_correlations(C_optimized, C_normalized, N, P_optimized, P_normalized, DIR_OPT)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.text(N - 1, 0, "Hi-C", fontweight="bold", ha="right", va="top")
    plt.text(0, N - 1, "PHi-C\n" + r"($r$={0:.3f}, $r'$={1:.3f})".format(r, dcr),
             fontweight="bold", ha="left", va="bottom")
    plt.imshow(C, 
               cmap=cmap_for_C,
               interpolation="none", vmin=0, vmax=PLT_MAX_C)
    plt.colorbar(ticks=[0, PLT_MAX_C], shrink=0.5, orientation="vertical",
                 label="Normalized contact probability")
    plt.axis("off")
    plt.savefig(FILE_FIG_C)
    plt.close()
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(K,
               cmap=cmap_for_K,
               interpolation="none", vmin=-PLT_MAX_K, vmax=PLT_MAX_K)
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
    K, N = read_K(FILE_READ_K)
    psf_path = os.path.join(DIR_4D, f"polymer_N{N}.psf")
    write_psfdata(psf_path, NAME, N)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    for sample in range(SAMPLE):
        Xx, Xy, Xz = equilibrium_conformation_of_normal_coordinates(lam, N)
        Rx, Ry, Rz = convert_X_to_R(Xx, Xy, Xz, Q)
        # ---------------------------------------------------------------------
        xyz_path = os.path.join(DIR_4D, f"sample{sample}.xyz")
        fp = open(xyz_path, "w")
        for frame in range(FRAME + 1):
            # -----------------------------------------------------------------
            print("%d" % N, file=fp)
            print("frame = %d" % frame, file=fp)
            for n in range(N):
                print("CA\t%f\t%f\t%f" % (Rx[n], Ry[n], Rz[n]), file=fp)
            # -----------------------------------------------------------------
            for step in range(INTERVAL):
                Rx, Ry, Rz = integrate_polymer_network(Rx, Ry, Rz, L, N,
                                                       NOISE, F_Coefficient)
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
    K, N = read_K(FILE_READ_K)
    write_psfdata(DIR_3D, NAME, N)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    # -------------------------------------------------------------------------
    fp = open(FILE_OUT, "w")
    for sample in range(SAMPLE):
        Xx, Xy, Xz = equilibrium_conformation_of_normal_coordinates(lam, N)
        Rx, Ry, Rz = convert_X_to_R(Xx, Xy, Xz, Q)
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
    K, N = read_K(FILE_READ_K)
    L = transform_K_into_L(K)
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
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
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
    K, N = read_K(FILE_READ_K)
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
    cmap_for_J = plt.get_cmap("jet")
    cmap_for_J.set_bad(color=(0.8, 0.8, 0.8))
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
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            J_storage = np.insert(J_storage, idx, np.full(M+1, np.nan), axis=1)
            J_loss = np.insert(J_loss, idx, np.full(M+1, np.nan), axis=1)
            J_abs = np.insert(J_abs, idx, np.full(M+1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(J_storage[START:END, :]),
               cmap=cmap_for_J,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
               cmap=cmap_for_J,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
               cmap=cmap_for_J,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
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
    K, N = read_K(FILE_READ_K)
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
    cmap_for_G = plt.get_cmap("jet")
    cmap_for_G.set_bad(color=(0.8, 0.8, 0.8))
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
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            G_storage = np.insert(G_storage, idx, np.full(M+1, np.nan), axis=1)
            G_loss = np.insert(G_loss, idx, np.full(M+1, np.nan), axis=1)
            G_abs = np.insert(G_abs, idx, np.full(M+1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(G_storage[START:END, :]),
               cmap=cmap_for_G,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
               cmap=cmap_for_G,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
               cmap=cmap_for_G,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
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
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    DIR = NAME + "/data_rheology"
    FILE_OUT_SPECTRUM = DIR + "/data_tan_spectrum.txt"
    DIR_FIG = DIR + "/figs"
    os.makedirs(DIR_FIG, exist_ok=True)
    FILE_FIG_SPECTRUM = DIR_FIG + "/tan_spectrum.svg"
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 24
    cmap_for_tan = plt.get_cmap("coolwarm")
    cmap_for_tan.set_bad(color=(0.5, 0.5, 0.5))
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
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            tan = np.insert(tan, idx, np.full(M+1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{\omega}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(tan[START:END, :]),
               cmap=cmap_for_tan,
               interpolation="none", vmin=-PLT_MAX_LOG, vmax=PLT_MAX_LOG,
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

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=5,
              help="Upper value of the exponent of the normalized time  [default=5]")
@click.option("--lower", "LOWER", type=int, default=-1,
              help="Lower value of the exponent of the normalized time  [default=-1]")
def msd(NAME, UPPER, LOWER):
    M = 100 * (UPPER - LOWER)
    # -------------------------------------------------------------------------
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    DIR = NAME + "/data_MSD"
    os.makedirs(DIR, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    # -------------------------------------------------------------------------
    for n in range(N):
        data = np.zeros((M + 1, 2))
        for m in range(M + 1):
            # -----------------------------------------------------------------
            t = 10**(0.01 * (m + LOWER * 100))
            data[m, 0] = t
            # -----------------------------------------------------------------
            for p in range(1, N):
                data[m, 1] += 2 * Q[n, p] * Q[n, p] / lam[p] * (1 - np.exp(- 3 * lam[p] * t))
        # ---------------------------------------------------------------------
        FILE_OUT = DIR + "/n{0:d}.txt".format(n)
        np.savetxt(FILE_OUT, data, fmt="%e")
# -----------------------------------------------------------------------------


@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--upper", "UPPER", type=int, default=5,
              help="Upper value of the exponent of the normalized time  [default=5]")
@click.option("--lower", "LOWER", type=int, default=-1,
              help="Lower value of the exponent of the normalized time  [default=-1]")
@click.option("--plt-upper", "PLT_UPPER", type=int, required=True,
              help="Upper value of the exponent of the normalized time in the spectrum")
@click.option("--plt-lower", "PLT_LOWER", type=int, required=True,
              help="Lower value of the exponent of the normalized time in the spectrum")
@click.option("--plt-max-log", "PLT_MAX_LOG", type=float, required=True,
              help="Maximum value of log10 MSD")
@click.option("--plt-min-log", "PLT_MIN_LOG", type=float, required=True,
              help="Minimum value of log10 MSD")
@click.option("--aspect", "ASPECT", type=float, default=0.8,
              help="Aspect ratio of the spectrum  [default=0.8]")
def plot_msd(NAME, UPPER, LOWER, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, PLT_MIN_LOG, ASPECT):
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.txt"
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    DIR = NAME + "/data_MSD"
    FILE_OUT_SPECTRUM_MSD = DIR + "/data_MSD_spectrum.txt"
    DIR_FIG = DIR + "/figs"
    os.makedirs(DIR_FIG, exist_ok=True)
    FILE_FIG_CURVES = DIR_FIG + "/MSD_curves.png"
    FILE_FIG_SPECTRUM_MSD = DIR_FIG + "/MSD_spectrum.svg"
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 9))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**(LOWER), 10**(UPPER))
    plt.xlabel(r"$\mathrm{\mathbf{\bar{t}}}$")
    plt.ylabel(r"$\mathrm{\mathbf{\overline{MSD}(\bar{t})}}$")
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        plt.plot(data[:, 0], data[:, 1], linewidth=1, color="green", alpha=0.1)
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
    cmap_for_MSD = plt.get_cmap("jet")
    cmap_for_MSD.set_bad(color=(0.8, 0.8, 0.8))
    # -------------------------------------------------------------------------
    M = 100 * (UPPER - LOWER)
    START = 100 * (PLT_LOWER - LOWER)
    END = 100 * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = []
    for n in range(PLT_LOWER, PLT_UPPER + 1):
        YTICKS_LABELS.append(n)
    MSD = np.zeros((M + 1, N))
    for n in range(N):
        FILE_READ = DIR + "/n{0:d}.txt".format(n)
        data = np.loadtxt(FILE_READ)
        MSD[:, n] = data[:, 1]
    # -------------------------------------------------------------------------
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            MSD = np.insert(MSD, idx, np.full(M+1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.ylabel(r"$\mathrm{\mathbf{log_{10} \bar{t}}}$")
    plt.yticks(np.arange(0, END - START, 100), YTICKS_LABELS)

    plt.imshow(np.log10(MSD[START:END, :]),
               cmap=cmap_for_MSD,
               interpolation="none", vmin=PLT_MIN_LOG, vmax=PLT_MAX_LOG,
               aspect=ASPECT)
    plt.colorbar(shrink=0.8,
                 label=r"$\mathrm{\mathbf{log_{10} \overline{MSD}(\bar{t})}}$",
                 ticks=[0, 1, 2])

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_MSD)
    plt.close()
    # -------------------------------------------------------------------------
    np.savetxt(FILE_OUT_SPECTRUM_MSD, MSD[START:END, :], fmt="%e")
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    cli()

