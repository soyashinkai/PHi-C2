import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import click
# for version >= 2.1.0
import hicstraw
import pandas as pd
# for version >= 2.1.1
import cooler
import h5py
# for version >= 2.2.0
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
from tqdm import tqdm
from scipy.linalg import solve_triangular
# 2026-05-22: standard-library and new external imports for --json feature
# for version >= 2.2.1
import sys
import json
import time
import hashlib
import socket
import platform
import tempfile
import re
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
import psutil  # 2026-05-22: hardware info for runtime_profiles in phic.json
import hictkpy # 2026-05-22: .hic/.mcool metadata for hic_file_info in phic.json
# -----------------------------------------------------------------------------
POINTS_PER_DECADE = 100
# -----------------------------------------------------------------------------

def calc_C_normalized(C):
    C_normalized = np.full_like(C, np.nan, dtype=float)
    
    # Non-diagonal elements
    d = np.sqrt(np.diag(C))
    denom = np.outer(d, d)

    mask = (denom > 0)  & ~np.isnan(C)
    C_normalized[mask] = C[mask] / denom[mask]

    # Diagonal elements
    np.fill_diagonal(C_normalized, 1.0)
    return C_normalized
# -----------------------------------------------------------------------------

def calc_C_normalized_high_resolution(C):
    N = C.shape[0]
    C_normalized = np.full_like(C, np.nan, dtype=float)

    # Non-diagonal elements
    for i in range(N-1):
        for j in range(i+1, N):
            freq = C[i, j]
            wi = C[i, i] + C[i, i+1] + C[i+1, i+1]
            wj = C[j-1, j-1] + C[j-1, j] + C[j, j]
            if (not np.isnan(freq)) and (wi > 0) and (wj > 0):
                tmp = freq / np.sqrt(wi * wj)
                C_normalized[i, j] = C_normalized[j, i] = tmp

    # Diagonal elements
    np.fill_diagonal(C_normalized, 1.0)
    return C_normalized
# -----------------------------------------------------------------------------

def calc_Ps(C, RES):
    N = C.shape[0]
    P = np.full((N, 2), np.nan)
    P[:, 0] = RES * np.arange(N)
    for n in range(N):
        d = np.diag(C, k=n)
        P[n, 1] = np.nanmean(d) if np.isfinite(d).any() else np.nan
    return P
# -----------------------------------------------------------------------------

def read_normalized_C(FILE_READ):
    C = np.load(FILE_READ)["C_normalized"]
    N = C.shape[0]
    return C, N
# -----------------------------------------------------------------------------

def set_init_K(N, init_k_backbone):
    K = init_k_backbone * (np.eye(N, k=1) + np.eye(N, k=-1))
    return K
# -----------------------------------------------------------------------------

def convert_K_into_C(K, J_over_N, I_N):
    # K to L
    d = np.sum(K, axis=0)
    L = np.diag(d) - K

    # L to M (Cholesky-based pseudoinverse; for version >= 2.2.0)
    L_tilde = L + J_over_N
    try:
        chol = np.linalg.cholesky(L_tilde)
    except np.linalg.LinAlgError:
        return np.zeros_like(K), True
    Y = solve_triangular(chol, I_N, lower=True)
    X = solve_triangular(chol.T, Y, lower=False)
    M = X - J_over_N

    # L to M (eigh-based pseudoinverse; used until version 2.1.x)
    # lam, Q = np.linalg.eigh(L)
    # inv_lam = np.zeros_like(lam)
    # nonzero = lam > 1e-12
    # inv_lam[nonzero] = 1.0 / lam[nonzero]
    # inv_lam[0] = 0.0
    # M = (Q * inv_lam) @ Q.T
    # error_flag = lam[0] < -1e-10

    # M to Σ^2
    md = np.diag(M)
    Sigma2 = (md[:, None] + md[None, :] - 2.0 * M) / 3.0

    # Σ^2 to C
    C = (1.0 + Sigma2)**(-1.5)

    return C, False
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
    valid_indices = ~np.isnan(X) & ~np.isnan(Y)
    X_filtered = X[valid_indices]
    Y_filtered = Y[valid_indices]
    # -------------------------------------------------------------------------
    r, _ = pearsonr(X_filtered, Y_filtered)
    return r, X_filtered, Y_filtered
# -----------------------------------------------------------------------------

def calc_distance_corrected_correlation(A, B, N, P_A, P_B):
    i_idx, j_idx = np.triu_indices(N, k=1)
    dist = j_idx - i_idx
    tmp_A = A[i_idx, j_idx] - P_A[dist]
    tmp_B = B[i_idx, j_idx] - P_B[dist]
    valid = ~np.isnan(tmp_A) & ~np.isnan(tmp_B)
    r, _ = pearsonr(tmp_A[valid], tmp_B[valid])
    return r, tmp_A[valid], tmp_B[valid]
# -----------------------------------------------------------------------------

def read_K(FILE_READ_K):
    K = np.load(FILE_READ_K)["K_optimized"]
    N = K.shape[0]
    return K, N
# -----------------------------------------------------------------------------

def transform_K_into_L(K):
    d = np.sum(K, axis=0)
    L = np.diag(d) - K
    return L
# -----------------------------------------------------------------------------

def equilibrium_conformation_of_normal_coordinates(lam, N):
    Xx = np.zeros(N)
    Xy = np.zeros(N)
    Xz = np.zeros(N)
    for p in range(1, N):
        sd = np.sqrt(1.0 / 3.0 / lam[p])
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
    with open(psf_path, "w") as fp:
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
# -----------------------------------------------------------------------------

# Supports both .hic and .mcool files.
# def make_input_contact_matrix(FILE_INPUT, RES, CHR, START, END, NORM):  # 2026-05-22: original
def make_input_contact_matrix(FILE_INPUT, RES, CHR, START, END, NORM, output_dir=None):  # 2026-05-22: added output_dir
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
        raise click.UsageError("Version 2.1.0 and above no longer support input in formats other than .hic or .mcool.")

    # 2026-05-22: override auto-computed DIR with canonical output path if provided
    if output_dir is not None:
        DIR = output_dir
    os.makedirs(DIR, exist_ok=True)
    return C_input, N_input, DIR, START, END
# -----------------------------------------------------------------------------

def remove_invalid_segments(C_input, N_input, TOLERANCE):
    nan_ratio = np.isnan(C_input).sum(axis=1) / N_input
    nan_indices = np.where(nan_ratio > TOLERANCE)[0]

    C_for_phic = np.delete(np.delete(C_input, nan_indices, axis=0), nan_indices, axis=1)
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

    nan_set = set(nan_indices)
    valid_indices = [i for i in range(N_input) if i not in nan_set]
    with open(DIR_META + "/_remaining_segments.txt", "w") as file:
        file.write(f"index,chrom,chromStart,chromEnd,new index\n")
        for i in range(len(valid_indices)):
            grs = START + valid_indices[i] * RES
            gre = START + (valid_indices[i] + 1) * RES
            file.write(f"{valid_indices[i]},{CHR},{grs},{gre},{i}\n")
# -----------------------------------------------------------------------------

def write_optimization_meta_data(DIR, init_k_backbone, alpha, beta, gradient_degree, version):
    DIR_META = DIR + "/_meta_data"
    os.makedirs(DIR_META, exist_ok=True)

    with open(DIR_META + "/_optimization_hyper_parameters.txt", "w") as file:
        file.write(f"PHi-C version,{version}\n")
        file.write(f"initial K along backbone,{init_k_backbone}\n")
        file.write(f"gradient degree,{gradient_degree}\n")
        file.write(f"stop condition parameter,{alpha:.2e}\n")
        file.write(f"backtracking factor,{beta:.2e}\n")
# -----------------------------------------------------------------------------

def write_correlations_meta_data(DIR, r, dcr):
    DIR_META = DIR + "/_meta_data"
    os.makedirs(DIR_META, exist_ok=True)

    with open(DIR_META + "/_correlations.txt", "w") as file:
        file.write(f"Pearson correlation coefficient,{r}\n")
        file.write(f"distance-corrected Pearson correlation coefficient,{dcr}\n")
# -----------------------------------------------------------------------------

# =============================================================================
# 2026-05-22: helpers for --json output feature (phic.json manifest)
# =============================================================================

# Key packages to record in runtime_profiles.environment.packages
_JSON_KEY_PACKAGES = [
    "click", "cooler", "h5py", "hic-straw", "hictkpy",
    "matplotlib", "MDAnalysis", "numpy", "pandas",
    "psutil", "scipy", "tqdm",
]


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path, data):
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".phic.", suffix=".json.tmp", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _load_or_init_phic_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return _empty_phic_json()


def _empty_phic_json():
    # 2026-05-25: updated schema to 2026-05-25; removed source_metadata block
    # return {  # 2026-05-22: original
    #     "$schema": "./schemas/phic-json-schema_2026-05-22.json",
    #     "_comment": "PHi-C2 analysis log. See the schema for field descriptions.",
    #     "phic_version": "2.2.1",
    #     "schema_version": "2026-05-22",
    #     "created_at": _now_iso(),
    #     "updated_at": _now_iso(),
    #     "hic_file_info": None,
    #     "source_metadata": None,
    #     "runtime_profiles": {},
    #     "run": None,
    # }
    return {
        "$schema": "./schemas/phic-json-schema_2026-05-25.json",
        "_comment": "PHi-C2 analysis log. See the schema for field descriptions.",
        "phic_version": "2.2.1",
        "schema_version": "2026-05-25",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "hic_file_info": None,
        "runtime_profiles": {},
        "run": None,
    }


def _get_key_packages():
    out = {}
    for name in _JSON_KEY_PACKAGES:
        try:
            out[name] = _pkg_version(name)
        except Exception:
            out[name] = None
    return dict(sorted(out.items()))


def _detect_environment():
    if os.environ.get("CONDA_PREFIX"):
        env_type = "conda"
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
        env_prefix = os.environ.get("CONDA_PREFIX", "")
    elif sys.prefix != sys.base_prefix:
        env_type = "venv"
        env_name = os.path.basename(sys.prefix)
        env_prefix = sys.prefix
    elif hasattr(sys, "real_prefix"):
        env_type = "virtualenv"
        env_name = os.path.basename(sys.prefix)
        env_prefix = sys.prefix
    else:
        env_type = "system"
        env_name = ""
        env_prefix = sys.prefix
    return {
        "type": env_type,
        "name": env_name,
        "prefix": env_prefix,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": _get_key_packages(),
    }


def _detect_gpu():
    import subprocess as _sub, json as _j
    try:
        r = _sub.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
            parts = lines[0].split(", ")
            return {"vendor": "NVIDIA", "name": parts[0], "count": len(lines),
                    "driver": parts[1] if len(parts) > 1 else None}
    except Exception:
        pass
    try:
        r = _sub.run(["rocm-smi", "--showproductname", "--json"],
                     capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            data = _j.loads(r.stdout)
            cards = [v for k, v in data.items() if k.startswith("card")]
            if cards:
                return {"vendor": "AMD", "name": cards[0].get("Card series", ""),
                        "count": len(cards), "driver": "ROCm"}
    except Exception:
        pass
    return None


def _detect_math_libraries():
    blas_backend = None
    mkl_version = None
    try:
        try:
            config = np.show_config(mode="dicts")
            blas_backend = (
                config.get("Build Dependencies", {})
                      .get("blas", {})
                      .get("name")
            )
        except TypeError:
            libs = getattr(np.__config__, "blas_opt_info", {}).get("libraries", [])
            if any("mkl" in l for l in libs):
                blas_backend = "mkl"
            elif any("openblas" in l for l in libs):
                blas_backend = "openblas"
    except Exception:
        pass
    try:
        mkl_version = _pkg_version("mkl")
    except Exception:
        pass
    return {
        "blas_backend": blas_backend,
        "mkl": mkl_version,
        "cusolver": None,  # not applicable for CPU-only version
    }


def _gather_runtime():
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "cpu": {
            "processor": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": _detect_gpu(),
        "math_libraries": _detect_math_libraries(),
        "phic_version": "2.2.1",
        "environment": _detect_environment(),
    }


def _runtime_profile_id(runtime):
    canonical = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"rt_{digest[:8]}"


def _upsert_runtime_profile(phic_json, runtime):
    profile_id = _runtime_profile_id(runtime)
    profiles = phic_json.setdefault("runtime_profiles", {})
    if profile_id not in profiles:
        profiles[profile_id] = runtime
    return profile_id


def _get_or_init_run(phic_json, run_id, output_dir, run_uuid=None):
    if phic_json.get("run") is None:
        phic_json["run"] = {
            "run_id": run_id,
            "run_uuid": run_uuid,
            "output_dir": output_dir,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "preprocessing": None,
            "optimization": None,
            "plot_optimization": None,
            "msd": None,
            "losstangent": None,
        }
    else:
        if run_uuid is not None and phic_json["run"].get("run_uuid") is None:
            phic_json["run"]["run_uuid"] = run_uuid
    return phic_json["run"]


def _read_hic_fileinfo(file_input):
    _, ext = os.path.splitext(file_input)
    ext = ext.lower()
    if ext not in (".hic", ".mcool"):
        raise click.UsageError("Unsupported file format. Use .hic or .mcool.")
    mrf = hictkpy.MultiResFile(file_input)
    resolutions = [int(r) for r in mrf.resolutions()]
    f = hictkpy.File(file_input, resolutions[-1])
    attrs = f.attributes()
    assembly = attrs.get("assembly", None)
    chroms_raw = f.chromosomes()
    chromosomes = [{"name": k, "length": int(v)} for k, v in chroms_raw.items()
                   if k.upper() != "ALL"]
    normalizations = list(f.avail_normalizations())
    if ext == ".hic" and "NONE" not in normalizations:
        normalizations = ["NONE"] + normalizations
    fmt = "hic" if ext == ".hic" else "mcool"
    return {
        "input_file": file_input,
        "format": fmt,
        "genome_assembly": assembly,
        "chromosomes": chromosomes,
        "resolutions": resolutions,
        "normalizations": normalizations,
    }
# =============================================================================

# def calc_plot_correlations(C_optimized, C_normalized, N, P_optimized, P_normalized, DIR_OPT):  # 2026-05-22: original
def calc_plot_correlations(C_optimized, C_normalized, N, P_optimized, P_normalized, DIR_OPT, draw_figures=True):  # 2026-05-22: added draw_figures parameter
    r, Optimized, Normalized = calc_correlation(C_optimized, C_normalized, N)
    dcr, Optimized_dcr, Normalized_dcr = calc_distance_corrected_correlation(C_optimized, C_normalized, N, P_optimized[:, 1], P_normalized[:, 1])
    if not draw_figures:
        return r, dcr
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    FILE_FIG_CORRELATION = DIR_OPT + "/Correlation.png"
    FILE_FIG_DC_CORRELATION = DIR_OPT + "/Correlation_distance_corrected.png"
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect("equal")
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
    plt.gca().set_aspect("equal")
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
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write hic_file_info block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
# def fetch_fileinfo(FILE_INPUT):  # 2026-05-22: original
def fetch_fileinfo(FILE_INPUT, WRITE_JSON, JSON_PATH):  # 2026-05-22: added WRITE_JSON, JSON_PATH
    _, EXT = os.path.splitext(FILE_INPUT)
    if EXT == ".hic":  # for .hic files
        hic = hicstraw.HiCFile(FILE_INPUT)
        print("chromosome, length")
        for chrom in hic.getChromosomes():
            print(chrom.name, chrom.length)
        print("------------------------")
        print("(usually) available normalization methods:", ["NONE", "VC", "VC_SQRT", "KR", "SCALE"])
        print("-------------------------")
        print("available resolutions:", hic.getResolutions())
        print("-------------------------")
    elif EXT == ".mcool":  # for .mcool files
        with h5py.File(FILE_INPUT, "r") as f:
            resolutions = list(f["resolutions"].keys())
        hic = cooler.Cooler(f'{FILE_INPUT}::resolutions/{resolutions[0]}')
        print("chromosome, length")
        for chromname in hic.chromnames:
            print(chromname, hic.chromsizes[chromname])
        print("------------------------")
        print("(usually) available normalization methods:", [None, "weight"])
        print("-------------------------")
        print("available resolutions:", resolutions)
        print("-------------------------")
    else:
        raise click.UsageError("Unsupported file format. Use .hic or .mcool.")
    # 2026-05-22: write hic_file_info block to phic.json when --json is given
    if WRITE_JSON:
        phic_data = _load_or_init_phic_json(JSON_PATH)
        runtime = _gather_runtime()
        profile_id = _upsert_runtime_profile(phic_data, runtime)
        fileinfo = _read_hic_fileinfo(FILE_INPUT)
        fileinfo["fetched_at"] = _now_iso()
        fileinfo["runtime_profile_id"] = profile_id
        phic_data["hic_file_info"] = fileinfo
        phic_data["updated_at"] = _now_iso()
        _atomic_write_json(JSON_PATH, phic_data)
        click.echo(f"Written hic_file_info to {JSON_PATH}")
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--input", "FILE_INPUT", required=True,
              help="Input Hi-C file (.hic or .mcool format)")
@click.option("--res", "RES", type=int, required=True,
              help="Resolution of the bin size")
@click.option("--plt-max-c", "PLT_MAX_C", type=float, required=True,
              help="Maximum value of contact map")
@click.option("--for-high-resolution", "HIGH_RESOLUTION", is_flag=True, default=False,
              help="Normalization of contact map for high-resolution case (ex. 1-kb, 500-bp, 200-bp)")
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
# 2026-05-22: added --name option to write output directly to canonical path
@click.option("--name", "NAME", default=None,
              help="Output directory name (canonical path); if omitted, auto-named by script")
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write preprocessing block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
# 2026-05-22: added --run-uuid for Data Conductor job UUID7 linkage
@click.option("--run-uuid", "RUN_UUID", default=None,
              help="Data Conductor job UUID7; written to run.run_uuid in phic.json (null if omitted)")
# def preprocessing(FILE_INPUT, RES, PLT_MAX_C, HIGH_RESOLUTION, CHR, START, END, NORM, TOLERANCE):  # 2026-05-22: original
def preprocessing(FILE_INPUT, RES, PLT_MAX_C, HIGH_RESOLUTION, CHR, START, END, NORM, TOLERANCE, NAME, WRITE_JSON, JSON_PATH, RUN_UUID):  # 2026-05-22: added NAME, WRITE_JSON, JSON_PATH, RUN_UUID
    # 2026-05-22: capture start time and gather runtime info before any computation
    started_at = _now_iso()
    t0 = time.monotonic()
    if WRITE_JSON:
        _runtime_json = _gather_runtime()
        _profile_id_json = None  # set after DIR is known (run_id needs DIR)
    C_input, N_input, DIR, START, END = make_input_contact_matrix(FILE_INPUT, RES, CHR, START, END, NORM, output_dir=NAME)
    # 2026-05-22: write status "running" once DIR (= run_id) is known
    if WRITE_JSON:
        try:
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, DIR, DIR + "/", run_uuid=RUN_UUID)
            _run_json["preprocessing"] = {
                "status": "running",
                "started_at": started_at,
                "runtime_profile_id": _profile_id_json,
                "parameters": {
                    "input": FILE_INPUT, "chr": CHR, "res": RES,
                    "grs": START, "gre": END, "norm": NORM,
                    "tolerance": TOLERANCE, "plt_max_c": PLT_MAX_C,
                    "for_high_resolution": HIGH_RESOLUTION,
                },
            }
            _run_json["updated_at"] = started_at
            _phic_data_json["updated_at"] = started_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
        except Exception as _e:
            click.echo(f"Warning: could not write running status to {JSON_PATH}: {_e}", err=True)
    C_for_phic, N_for_phic, nan_indices = remove_invalid_segments(C_input, N_input, TOLERANCE)
    write_meta_data(DIR, FILE_INPUT, NORM, CHR, START, END, RES, N_input, N_for_phic, nan_indices, TOLERANCE)
    # -------------------------------------------------------------------------
    FILE_OUT_C_NORMALIZED = DIR + "/C_normalized.npz"
    FILE_OUT_P_NORMALIZED = DIR + "/P_normalized.npz"
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
    np.savez_compressed(FILE_OUT_C_NORMALIZED, C_normalized=C_normalized)
    # 2026-05-22: capture contact_stats from normalized matrix (0-1 range) before NaN rows are reinserted
    if WRITE_JSON:
        _cn_upper = C_normalized[np.triu_indices(C_normalized.shape[0], k=1)]
        _cn_valid = _cn_upper[~np.isnan(_cn_upper)]
        _contact_stats_json = {
            "min": float(_cn_valid.min()) if len(_cn_valid) > 0 else None,
            "max": float(_cn_valid.max()) if len(_cn_valid) > 0 else None,
            "mean": float(_cn_valid.mean()) if len(_cn_valid) > 0 else None,
            "nonzero_fraction": float((_cn_valid > 0).sum() / len(_cn_valid)) if len(_cn_valid) > 0 else None,
        }
    # -------------------------------------------------------------------------
    for idx in nan_indices:
        N = C_normalized.shape[0]
        C_normalized = np.insert(np.insert(C_normalized, idx, np.full(N, np.nan), axis=0), idx, np.full(N+1, np.nan), axis=1)
    P_normalized = calc_Ps(C_normalized, RES)
    np.savez_compressed(FILE_OUT_P_NORMALIZED, P_normalized=P_normalized)
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
    # -------------------------------------------------------------------------
    # 2026-05-22: write completed preprocessing block to phic.json
    if WRITE_JSON:
        try:
            elapsed_sec = round(time.monotonic() - t0, 3)
            completed_at = _now_iso()
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, DIR, DIR + "/", run_uuid=RUN_UUID)
            _run_json["preprocessing"] = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "runtime_profile_id": _profile_id_json,
                "parameters": {
                    "input": FILE_INPUT, "chr": CHR, "res": RES,
                    "grs": START, "gre": END, "norm": NORM,
                    "tolerance": TOLERANCE, "plt_max_c": PLT_MAX_C,
                    "for_high_resolution": HIGH_RESOLUTION,
                },
                "matrix_shape": [N_input, N_input],
                "n_bins_total": N_input,
                "n_bins_removed_nan": int(len(nan_indices)),
                "n_bins_kept": N_for_phic,
                "start_position": START,
                "end_position": END,
                "input_matrix_size": N_input,
                "phic_matrix_size": N_for_phic,
                "removed_bin_indices": [int(i) for i in nan_indices],
                "contact_stats": _contact_stats_json,
                "files": {
                    "C_normalized": DIR + "/C_normalized.npz",
                    "P_normalized": DIR + "/P_normalized.npz",
                    "C_normalized_figure": DIR + "/C_normalized.svg",
                    "P_normalized_figure": DIR + "/P_normalized.svg",
                    "removed_segments": DIR + "/_meta_data/_removed_segments.txt",
                    "valid_segments":   DIR + "/_meta_data/_remaining_segments.txt",
                },
            }
            _run_json["updated_at"] = completed_at
            _phic_data_json["updated_at"] = completed_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
            click.echo(f"Written preprocessing block to {JSON_PATH}")
        except Exception as _e:
            click.echo(f"Warning: could not write completed block to {JSON_PATH}: {_e}", err=True)
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--init-k-backbone", "init_k_backbone", type=float, default=0.5,
              help="Initial parameter of K_i,i+1  [default=0.5]")
@click.option("--stop-condition-parameter", "alpha", type=float, default=1e-7,
              help="Parameter for the stop condition  [default=1e-7]")
@click.option("--backtracking-factor", "beta", type=float, default=0.7,
              help="Backtracking factor  [default=0.7]")
@click.option("--gradient-degree", "gradient_degree", type=int, default=2,
              help="Gradient used for optimizing of K  [default=2]")
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write optimization block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
# 2026-05-22: added --run-uuid for Data Conductor job UUID7 linkage
@click.option("--run-uuid", "RUN_UUID", default=None,
              help="Data Conductor job UUID7; written to run.run_uuid in phic.json (null if omitted)")
# def optimization(NAME, init_k_backbone, alpha, beta, gradient_degree):  # 2026-05-22: original
def optimization(NAME, init_k_backbone, alpha, beta, gradient_degree, WRITE_JSON, JSON_PATH, RUN_UUID):  # 2026-05-22: added WRITE_JSON, JSON_PATH, RUN_UUID
    def gradient_degree_1(diff, C):
        return diff

    def gradient_degree_2(diff, C):
        return diff * C**(5.0/3.0)

    available_gradients = {
        1: gradient_degree_1,
        2: gradient_degree_2,
    }
    # -------------------------------------------------------------------------
    # 2026-05-22: capture start time before any computation
    started_at = _now_iso()
    t0 = time.monotonic()
    if WRITE_JSON:
        _runtime_json = _gather_runtime()
    # -------------------------------------------------------------------------
    write_optimization_meta_data(NAME, init_k_backbone, alpha, beta, gradient_degree, version="2.2.0")
    # -------------------------------------------------------------------------
    FILE_READ = NAME + "/C_normalized.npz"
    DIR_OPT = NAME + "/data_optimization"
    os.makedirs(DIR_OPT, exist_ok=True)
    FILE_LOG = DIR_OPT + "/optimization.log"
    # -------------------------------------------------------------------------
    eta = 8.0 # Set initial eta, which is better to be large value
    stop_delta = eta * alpha
    gradient = available_gradients[gradient_degree]
    max_backtracks = 1000

    C_normalized, N = read_normalized_C(FILE_READ)
    C_normalized = np.nan_to_num(C_normalized, nan=0.0)
    J_over_N = np.full((N, N), 1.0 / N)
    I_N = np.eye(N)
    K = set_init_K(N, init_k_backbone)
    C, error_flag = convert_K_into_C(K, J_over_N, I_N)
    diff, cost = calc_diff_cost(C, C_normalized, N)
    # -------------------------------------------------------------------------
    step = 0
    fp = open(FILE_LOG, "w")
    print("step\tcost\teta", file=fp)
    print(f"{step:d}\t{cost:e}\t{eta:e}", file=fp)
    # -------------------------------------------------------------------------
    while True:
        step += 1

        # Temporarily store the current state
        K_prev = K.copy()
        C_prev = C
        diff_prev = diff
        cost_prev = cost

        # Backtracking within this step to adjust eta
        eta_trial = eta
        bt = 0
        while True:
            # Trial update (not accepted yet)
            K_trial = K_prev - eta_trial * gradient(diff_prev, C_prev)

            # Check positive semi-definiteness (PSD) of the induced Laplacian matrix
            C_trial, err_trial = convert_K_into_C(K_trial, J_over_N, I_N)
            if err_trial:
                bt += 1
                if bt > max_backtracks:
                    raise click.ClickException("Optimization failed due to an infinite loop in backtracking.")
                eta_trial *= beta # reduce eta and retry
                continue

            # Evaluate cost (check monotonic decrease)
            diff_trial, cost_trial = calc_diff_cost(C_trial, C_normalized, N)
            if cost_trial < cost_prev:
                # Accept the update and move one step forward
                K = K_trial
                C = C_trial
                diff = diff_trial
                cost = cost_trial
                eta = eta_trial # Use the found eta as the initial value for the next step
                stop_delta = eta * alpha # Update the stopping criterion accordingly
                break
            else:
                bt += 1
                if bt > max_backtracks:
                    raise click.ClickException("Optimization failed due to an infinite loop in backtracking.")
                # Non-monotonic → reduce eta and retry
                eta_trial *= beta
                continue

        # Logging
        print(f"{step:d}\t{cost:e}\t{eta:e}", file=fp)

        # Convergence check (stop if cost reduction is sufficiently small)
        delta = cost_prev - cost
        if 0.0 < delta < stop_delta:
            np.savez_compressed(DIR_OPT + "/K_optimized.npz", K_optimized=K)
            break
    # -------------------------------------------------------------------------
    fp.close()
    # -------------------------------------------------------------------------
    # 2026-05-22: write completed optimization block to phic.json
    if WRITE_JSON:
        try:
            elapsed_sec = round(time.monotonic() - t0, 3)
            completed_at = _now_iso()
            # gradient norm at final step (Frobenius norm of the gradient tensor)
            final_grad = gradient(diff, C)
            final_gradient_norm = float(np.linalg.norm(final_grad, "fro"))
            # K stats (non-NaN elements)
            k_stats = {
                "min": float(np.nanmin(K)),
                "max": float(np.nanmax(K)),
                "mean": float(np.nanmean(K)),
            }
            # correlations between C_optimized and C_normalized
            C_normalized_corr, N_corr = read_normalized_C(FILE_READ)  # fresh load with NaN preserved
            C_optimized_corr, _ = convert_K_into_C(K, J_over_N, I_N)
            P_normalized_corr = np.load(NAME + "/P_normalized.npz")["P_normalized"]
            # infer resolution from saved P_normalized (P[:,0] = RES * arange(N))
            RES_corr = int(P_normalized_corr[1, 0])
            P_optimized_corr = calc_Ps(C_optimized_corr, RES_corr)
            r_corr, _, _ = calc_correlation(C_optimized_corr, C_normalized_corr, N_corr)
            dcr_corr, _, _ = calc_distance_corrected_correlation(
                C_optimized_corr, C_normalized_corr, N_corr,
                P_optimized_corr[:, 1], P_normalized_corr[:, 1])
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, NAME, NAME + "/", run_uuid=RUN_UUID)
            _run_json["optimization"] = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "runtime_profile_id": _profile_id_json,
                "hyperparameters": {
                    "initial_k_backbone": init_k_backbone,
                    "gradient_degree": gradient_degree,
                    "stop_condition": alpha,
                    "backtracking_factor": beta,
                },
                "iterations": step,
                "final_cost": float(cost),
                "final_gradient_norm": final_gradient_norm,
                "converged": True,
                "K_stats": k_stats,
                "correlation": {
                    "pearson": round(float(r_corr), 6),
                },
                "correlation_distance_corrected": {
                    "pearson": round(float(dcr_corr), 6),
                },
                "files": {
                    "K_optimized": DIR_OPT + "/K_optimized.npz",
                    "log": FILE_LOG,
                },
            }
            _run_json["updated_at"] = completed_at
            _phic_data_json["updated_at"] = completed_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
            click.echo(f"Written optimization block to {JSON_PATH}")
        except Exception as _e:
            click.echo(f"Warning: could not write optimization block to {JSON_PATH}: {_e}", err=True)
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
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write plot_optimization block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
@click.option("--no-figures", "NO_FIGURES", is_flag=True, default=False,
              help="Skip all matplotlib figure generation (faster; use when figures are not needed)")
# 2026-05-22: added --run-uuid for Data Conductor job UUID7 linkage
@click.option("--run-uuid", "RUN_UUID", default=None,
              help="Data Conductor job UUID7; written to run.run_uuid in phic.json (null if omitted)")
# def plot_optimization(NAME, RES, PLT_MAX_C, PLT_MAX_K):  # 2026-05-22: original
def plot_optimization(NAME, RES, PLT_MAX_C, PLT_MAX_K, WRITE_JSON, JSON_PATH, NO_FIGURES, RUN_UUID):  # 2026-05-22: added WRITE_JSON, JSON_PATH, NO_FIGURES, RUN_UUID
    # 2026-05-22: capture start time before any computation
    started_at = _now_iso()
    t0 = time.monotonic()
    if WRITE_JSON:
        _runtime_json = _gather_runtime()
    # READ & OUTPUT FILES
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_C = NAME + "/C_normalized.npz"
    FILE_READ_K = DIR_OPT + "/K_optimized.npz"
    FILE_READ_LOG = DIR_OPT + "/optimization.log"
    FILE_OUT_C_OPT = DIR_OPT + "/C_optimized.npz"
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    FILE_FIG_C = DIR_OPT + "/C.svg"
    FILE_FIG_K = DIR_OPT + "/K.svg"
    FILE_FIG_P = DIR_OPT + "/P.svg"
    FILE_FIG_Cost = DIR_OPT + "/Cost.svg"
    FILE_FIG_ETA = DIR_OPT + "/Eta.svg"
    # -------------------------------------------------------------------------
    if not NO_FIGURES:
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 36
        cmap_for_C = plt.get_cmap("magma_r")
        cmap_for_C.set_bad(color=(0.8, 0.8, 0.8))
        cmap_for_K = plt.get_cmap("bwr")
        cmap_for_K.set_bad(color=(0.8, 0.8, 0.8))
    # -------------------------------------------------------------------------
    C_normalized, N = read_normalized_C(FILE_READ_C)
    K = np.load(FILE_READ_K)["K_optimized"]
    J_over_N = np.full((N, N), 1.0 / N)
    I_N = np.eye(N)
    C_optimized, _ = convert_K_into_C(K, J_over_N, I_N)
    mask_nan = np.isnan(C_normalized)
    C_optimized[mask_nan] = np.nan
    np.savez_compressed(FILE_OUT_C_OPT, C_optimized=C_optimized)
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
    r, dcr = calc_plot_correlations(C_optimized, C_normalized, N, P_optimized, P_normalized, DIR_OPT, draw_figures=not NO_FIGURES)
    write_correlations_meta_data(NAME, r, dcr)
    if not NO_FIGURES:
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
                     label=r"Normalized $K_{ij}$")
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
        log_data = np.loadtxt(FILE_READ_LOG, delimiter="\t", skiprows=1)
        data = log_data[:, 0:2]  # step and cost columns
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
        # -------------------------------------------------------------------------
        data = log_data[:, [0, 2]]  # step and eta columns
        plt.figure(figsize=(10, 5))
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().yaxis.set_ticks_position("left")
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.xlabel("Iteration", fontweight="bold")
        plt.ylabel("Learning rate", fontweight="bold")
        plt.xlim(0, data[-1, 0])
        # plt.ylim(0, data[0, 1])
        plt.plot(data[:, 0], data[:, 1], linewidth=4)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(FILE_FIG_ETA)
        plt.close()
    # -------------------------------------------------------------------------
    # 2026-05-22: write completed plot_optimization block to phic.json
    if WRITE_JSON:
        try:
            elapsed_sec = round(time.monotonic() - t0, 3)
            completed_at = _now_iso()
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, NAME, NAME + "/", run_uuid=RUN_UUID)
            _run_json["plot_optimization"] = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "runtime_profile_id": _profile_id_json,
                "parameters": {
                    "res": RES,
                    "plt_max_c": PLT_MAX_C,
                    "plt_max_k": PLT_MAX_K,
                },
                "correlations": {
                    "pearson_r": round(float(r), 16),
                    "pearson_r_distance_corrected": round(float(dcr), 16),
                },
                "files": {
                    "C_optimized": FILE_OUT_C_OPT,
                    **({"figures": {
                        "C":    FILE_FIG_C,
                        "P":    FILE_FIG_P,
                        "K":    FILE_FIG_K,
                        "Cost": FILE_FIG_Cost,
                        "Eta":  FILE_FIG_ETA,
                        "Correlation": DIR_OPT + "/Correlation.png",
                        "Correlation_distance_corrected": DIR_OPT + "/Correlation_distance_corrected.png",
                    }} if not NO_FIGURES else {}),
                },
            }
            _run_json["updated_at"] = completed_at
            _phic_data_json["updated_at"] = completed_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
            click.echo(f"Written plot_optimization block to {JSON_PATH}")
        except Exception as _e:
            click.echo(f"Warning: could not write plot_optimization block to {JSON_PATH}: {_e}", err=True)
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
    NOISE = np.sqrt(2.0 * EPS)
    F_Coefficient = 3.0 * EPS
    np.random.seed(SEED)
    # -------------------------------------------------------------------------
    FILE_READ_K = NAME + "/data_optimization/K_optimized.npz"
    DIR_4D = NAME + "/data_dynamics"
    os.makedirs(DIR_4D, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    psf_path = os.path.join(DIR_4D, f"polymer_N{N}.psf")
    write_psfdata(psf_path, NAME, N)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    u = mda.Universe(psf_path)
    u.load_new(np.zeros((1, N, 3), dtype=np.float32))
    for sample in tqdm(range(SAMPLE), desc="dynamics"):
        Xx, Xy, Xz = equilibrium_conformation_of_normal_coordinates(lam, N)
        Rx, Ry, Rz = convert_X_to_R(Xx, Xy, Xz, Q)
        # ---------------------------------------------------------------------
        xyz_path = os.path.join(DIR_4D, f"sample{sample}.xyz")
        dcd_path = os.path.join(DIR_4D, f"sample{sample}.dcd")
        with open(xyz_path, "w") as fp, DCDWriter(dcd_path, u.atoms.n_atoms) as dcd:
            for frame in range(FRAME + 1):
                # -------------------------------------------------------------
                print("%d" % N, file=fp)
                print("frame = %d" % frame, file=fp)
                coords = np.column_stack([Rx, Ry, Rz])
                np.savetxt(fp, coords, fmt="CA\t%f\t%f\t%f")
                u.atoms.positions = coords
                dcd.write(u.atoms)
                # -------------------------------------------------------------
                for step in range(INTERVAL):
                    Rx, Ry, Rz = integrate_polymer_network(Rx, Ry, Rz, L, N,
                                                           NOISE, F_Coefficient)
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
    FILE_READ_K = NAME + "/data_optimization/K_optimized.npz"
    DIR_3D = NAME + "/data_sampling"
    os.makedirs(DIR_3D, exist_ok=True)
    FILE_OUT = DIR_3D + "/conformations.xyz"
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    psf_path = os.path.join(DIR_3D, f"polymer_N{N}.psf")
    write_psfdata(psf_path, NAME, N)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    # -------------------------------------------------------------------------
    dcd_path = os.path.join(DIR_3D, "conformations.dcd")
    u = mda.Universe(psf_path)
    u.load_new(np.zeros((1, N, 3), dtype=np.float32))
    with open(FILE_OUT, "w") as fp, DCDWriter(dcd_path, u.atoms.n_atoms) as dcd:
        for sample in tqdm(range(SAMPLE), desc="sampling"):
            Xx, Xy, Xz = equilibrium_conformation_of_normal_coordinates(lam, N)
            Rx, Ry, Rz = convert_X_to_R(Xx, Xy, Xz, Q)
            # -----------------------------------------------------------------
            print("%d" % N, file=fp)
            print("sample = %d" % sample, file=fp)
            coords = np.column_stack([Rx, Ry, Rz])
            np.savetxt(fp, coords, fmt="CA\t%f\t%f\t%f")
            u.atoms.positions = coords
            dcd.write(u.atoms)
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write msd block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
# 2026-05-22: added --run-uuid for Data Conductor job UUID7 linkage
@click.option("--run-uuid", "RUN_UUID", default=None,
              help="Data Conductor job UUID7; written to run.run_uuid in phic.json (null if omitted)")
def msd(NAME, WRITE_JSON, JSON_PATH, RUN_UUID):
    started_at = _now_iso()
    t0 = time.monotonic()
    if WRITE_JSON:
        _runtime_json = _gather_runtime()
    # -------------------------------------------------------------------------
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.npz"
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    DIR = NAME + "/data_MSD"
    FILE_OUT_MSD = DIR + "/MSD_matrix.npz"
    os.makedirs(DIR, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    lam1 = lam[1:]
    Q1   = Q[:, 1:]
    # Each non-trivial mode has relaxation time tau_p = 1 / (3 * lam_p);
    # set [LOWER, UPPER] to the integer log10 endpoints covering all modes.
    # UPPER uses 3 * tau_max so the window reaches the equilibrium plateau.
    tau1 = 1.0 / (3.0 * lam1)
    # tau[0]=NaN for the center-of-mass mode; tau[p] corresponds to mode p (p=1..N-1)
    tau = np.concatenate(([np.nan], tau1))
    tau_min = tau1.min()
    tau_max = tau1.max()
    LOWER = int(np.floor(np.log10(tau_min)))
    UPPER = int(np.ceil(np.log10(3.0 * tau_max)))
    click.echo(f"Relaxation times: tau_min={tau_min:.3e}, tau_max={tau_max:.3e}")
    click.echo(f"Auto time range from eigenvalues: LOWER={LOWER}, UPPER={UPPER}")
    M = POINTS_PER_DECADE * (UPPER - LOWER)
    exponents = LOWER + np.arange(M + 1) / POINTS_PER_DECADE
    t = 10.0 ** exponents
    # W[n,p] = 6 * Q[n,p]^2 * tau[p]  -> (N, N-1)  (mode-p amplitude weight on bead n)
    W = 6.0 * (Q1 ** 2) * tau1
    # F[p,m] = 1 - exp(-t[m] / tau[p])  -> (N-1, M+1)  (mode-p relaxation)
    F = 1.0 - np.exp(-t[None, :] / tau1[:, None])
    # MSD[m,n] = sum_p W[n,p] * F[p,m]  -> (M+1, N)
    MSD = (W @ F).T
    # -------------------------------------------------------------------------
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            MSD = np.insert(MSD, idx, np.full(M + 1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    np.savez_compressed(FILE_OUT_MSD, MSD=MSD, t=t, tau=tau)
    # -------------------------------------------------------------------------
    if WRITE_JSON:
        try:
            elapsed_sec = round(time.monotonic() - t0, 3)
            completed_at = _now_iso()
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, NAME, NAME + "/", run_uuid=RUN_UUID)
            _run_json["msd"] = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "runtime_profile_id": _profile_id_json,
                "parameters": {
                    "upper": UPPER,
                    "lower": LOWER,
                },
                "files": {
                    "MSD_matrix": FILE_OUT_MSD,
                },
            }
            _run_json["updated_at"] = completed_at
            _phic_data_json["updated_at"] = completed_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
            click.echo(f"Written msd block to {JSON_PATH}")
        except Exception as _e:
            click.echo(f"Warning: could not write msd block to {JSON_PATH}: {_e}", err=True)
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
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
def plot_msd(NAME, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, PLT_MIN_LOG, ASPECT):
    DIR = NAME + "/data_MSD"
    FILE_READ_MSD = DIR + "/MSD_matrix.npz"
    FILE_FIG_CURVES = DIR + "/fig_MSD_curves.png"
    FILE_FIG_SPECTRUM_MSD = DIR + "/fig_MSD_spectrum.svg"
    # -------------------------------------------------------------------------
    data = np.load(FILE_READ_MSD)
    MSD  = data["MSD"]
    t    = data["t"]
    LOWER = int(np.round(np.log10(t[0])))
    UPPER = int(np.round(np.log10(t[-1])))
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 36
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 9))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(t[0], t[-1])
    plt.xlabel(r"$\mathrm{\mathbf{\bar{t}}}$")
    plt.ylabel(r"$\mathrm{\mathbf{\overline{MSD}(\bar{t})}}$")
    for n in range(MSD.shape[1]):
        col = MSD[:, n]
        if not np.all(np.isnan(col)):
            plt.plot(t, col, linewidth=1, color="green", alpha=0.1)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.savefig(FILE_FIG_CURVES)
    plt.close()
    # -------------------------------------------------------------------------
    plt.rcParams["font.size"] = 24
    cmap_for_MSD = plt.get_cmap("plasma")
    cmap_for_MSD.set_bad(color=(0.8, 0.8, 0.8))
    # -------------------------------------------------------------------------
    START = POINTS_PER_DECADE * (PLT_LOWER - LOWER)
    END = POINTS_PER_DECADE * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = list(range(PLT_LOWER, PLT_UPPER + 1))
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
                 ticks=list(range(int(np.ceil(PLT_MIN_LOG)), int(np.floor(PLT_MAX_LOG)) + 1)))

    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.tick_params(labelbottom=0, bottom=0, labelleft=1)

    plt.savefig(FILE_FIG_SPECTRUM_MSD)
    plt.close()
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
# 2026-05-22: added --json and --json-path for phic.json output
@click.option("--json", "WRITE_JSON", is_flag=True, default=False,
              help="Write losstangent block to phic.json in the workspace")
@click.option("--json-path", "JSON_PATH", default="phic.json",
              help="Path to phic.json  [default: ./phic.json]")
# 2026-05-22: added --run-uuid for Data Conductor job UUID7 linkage
@click.option("--run-uuid", "RUN_UUID", default=None,
              help="Data Conductor job UUID7; written to run.run_uuid in phic.json (null if omitted)")
def losstangent(NAME, WRITE_JSON, JSON_PATH, RUN_UUID):
    started_at = _now_iso()
    t0 = time.monotonic()
    if WRITE_JSON:
        _runtime_json = _gather_runtime()
    # -------------------------------------------------------------------------
    DIR_OPT = NAME + "/data_optimization"
    FILE_READ_K = DIR_OPT + "/K_optimized.npz"
    FILE_READ_META = NAME + "/_meta_data/_removed_segments.txt"
    DIR = NAME + "/data_losstangent"
    FILE_OUT_LT = DIR + "/losstangent_matrix.npz"
    os.makedirs(DIR, exist_ok=True)
    # -------------------------------------------------------------------------
    K, N = read_K(FILE_READ_K)
    L = transform_K_into_L(K)
    lam, Q = np.linalg.eigh(L)
    lam1 = lam[1:]
    Q1   = Q[:, 1:]
    Qsq  = Q1 ** 2
    # tau_p = 1 / (3 * lam_p); the omega window is the inverse range.
    # LOWER uses 1/(3*tau_max) so the low-frequency tail captures equilibration.
    tau1 = 1.0 / (3.0 * lam1)
    tau = np.concatenate(([np.nan], tau1))
    tau_min = tau1.min()
    tau_max = tau1.max()
    LOWER = int(np.floor(np.log10(1.0 / (3.0 * tau_max))))
    UPPER = int(np.ceil(np.log10(1.0 / tau_min)))
    click.echo(f"Relaxation times: tau_min={tau_min:.3e}, tau_max={tau_max:.3e}")
    click.echo(f"Auto omega range from eigenvalues: LOWER={LOWER}, UPPER={UPPER}")
    M = POINTS_PER_DECADE * (UPPER - LOWER)
    exponents = LOWER + np.arange(M + 1) / POINTS_PER_DECADE
    omega = 10.0 ** exponents
    # J(omega, n) = sum_p Q[n,p]^2 * tau[p] / (1 + i * omega * tau[p])
    #   Jp  = Re(J) = sum_p Q^2 * tau / (1 + (omega*tau)^2)
    #   Jpp = Im(J) = sum_p Q^2 * tau * (omega*tau) / (1 + (omega*tau)^2)
    x = tau1[:, None] * omega[None, :]                # (N-1, M+1)
    kernel = tau1[:, None] / (1.0 + x ** 2)           # tau / (1 + (omega*tau)^2)
    Jp  = (Qsq @ kernel).T                            # (M+1, N)
    Jpp = (Qsq @ (kernel * x)).T                      # (M+1, N)
    tan_delta = Jpp / Jp
    # -------------------------------------------------------------------------
    df_meta = pd.read_csv(FILE_READ_META, usecols=[0])
    if df_meta.shape[0] > 0:
        nan_indices = df_meta.iloc[:, 0].tolist()
        for idx in nan_indices:
            tan_delta = np.insert(tan_delta, idx, np.full(M + 1, np.nan), axis=1)
    # -------------------------------------------------------------------------
    np.savez_compressed(FILE_OUT_LT, losstangent=tan_delta, omega=omega, tau=tau)
    # -------------------------------------------------------------------------
    if WRITE_JSON:
        try:
            elapsed_sec = round(time.monotonic() - t0, 3)
            completed_at = _now_iso()
            _phic_data_json = _load_or_init_phic_json(JSON_PATH)
            _profile_id_json = _upsert_runtime_profile(_phic_data_json, _runtime_json)
            _run_json = _get_or_init_run(_phic_data_json, NAME, NAME + "/", run_uuid=RUN_UUID)
            _run_json["losstangent"] = {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "runtime_profile_id": _profile_id_json,
                "parameters": {
                    "upper": UPPER,
                    "lower": LOWER,
                },
                "files": {
                    "losstangent_matrix": FILE_OUT_LT,
                },
            }
            _run_json["updated_at"] = completed_at
            _phic_data_json["updated_at"] = completed_at
            _atomic_write_json(JSON_PATH, _phic_data_json)
            click.echo(f"Written losstangent block to {JSON_PATH}")
        except Exception as _e:
            click.echo(f"Warning: could not write losstangent block to {JSON_PATH}: {_e}", err=True)
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--name", "NAME", required=True,
              help="Target directory name")
@click.option("--plt-upper", "PLT_UPPER", type=int, required=True,
              help="Upper value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-lower", "PLT_LOWER", type=int, required=True,
              help="Lower value of the exponent of the angular frequency in the spectrum")
@click.option("--plt-max-log", "PLT_MAX_LOG", type=float, required=True,
              help="Maximum value of log10 tanδ")
@click.option("--aspect", "ASPECT", type=float, default=0.8,
              help="Aspect ratio of the spectrum  [default=0.8]")
def plot_losstangent(NAME, PLT_UPPER, PLT_LOWER, PLT_MAX_LOG, ASPECT):
    DIR = NAME + "/data_losstangent"
    FILE_READ_LOSSTANGENT = DIR + "/losstangent_matrix.npz"
    FILE_FIG_SPECTRUM = DIR + "/fig_losstangent_spectrum.svg"
    # -------------------------------------------------------------------------
    data  = np.load(FILE_READ_LOSSTANGENT)
    tan   = data["losstangent"]
    omega = data["omega"]
    LOWER = int(np.round(np.log10(omega[0])))
    UPPER = int(np.round(np.log10(omega[-1])))
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 24
    cmap_for_tan = plt.get_cmap("coolwarm")
    cmap_for_tan.set_bad(color=(0.5, 0.5, 0.5))
    # -------------------------------------------------------------------------
    START = POINTS_PER_DECADE * (PLT_LOWER - LOWER)
    END = POINTS_PER_DECADE * (PLT_UPPER - LOWER) + 1
    YTICKS_LABELS = list(range(PLT_LOWER, PLT_UPPER + 1))
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
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

