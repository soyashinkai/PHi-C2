# PHi-C2
PHi-C2 allows for a physical interpretation of a Hi-C contact matrix.
The `phic` package includes a suite of command line tools.

<img src="/img/fig0.svg">

### Installation (with Conda environment)

You can install `phic` in a clean environment as follows:

    conda create -n phic python=3.12
    conda activate phic
    pip install phic

![PyPI Downloads](https://static.pepy.tech/badge/phic)
![PyPI Downloads](https://static.pepy.tech/badge/phic/month)
![PyPI Downloads](https://static.pepy.tech/badge/phic/week)

Without preparing a Python environment, PHi-C2 (=<2.0.13) runs on [Google Colab](https://bit.ly/3rlptGI).

### Requirements
- PHi-C2 is based on `python3`.
- Python packages `numpy`, `matplotlib`, `scipy`, `click`, `pandas`, `hic-straw`, `cooler`, `h5py`, `MDAnalysis`, `tqdm`, `psutil`, `hictkpy`.

To visualize the simulated polymer dynamics and conformations, [VMD](https://www.ks.uiuc.edu/Research/vmd/) can be used. Alternatively, the output PSF and DCD files can be viewed directly in a web browser using [Mol*](https://molstar.org/viewer/) without any local installation.


### Citation

If you use PHi-C2, please cite:

Soya Shinkai, Hiroya Itoga, Koji Kyoda, and Shuichi Onami. (2022).
**PHi-C2: interpreting Hi-C data as the dynamic 3D genome state.**
[_Bioinformatics_ **38**(21) 4984–4986](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac613/6695219).

<!--
We will submit a manuscript on PHi-C2, in which we dramatically updated the algorithm of the optimization procedure.
But, the basic framework remains the same in the following papers:

- Soya Shinkai, Masaki Nakagawa, Takeshi Sugawara, Yuichi Togashi, Hiroshi Ochiai, Ryuichiro Nakato, Yuichi Taniguchi, and Shuichi Onami. (2020). **PHi-C: deciphering Hi-C data into polymer dynamics.** [_NAR Genomics and Bioinformatics_ **2** (2) lqaa020](https://doi.org/10.1093/nargab/lqaa020).

- Soya Shinkai, Takeshi Sugawara, Hisashi Miura, Ichiro Hiratani, and Shuichi Onami. (2020). **Microrheology for Hi-C Data Reveals the Spectrum of the Dynamic 3D Genome Organization.** [_Biophysical Journal_ **118** 2220–2228](https://doi.org/10.1016/j.bpj.2020.02.020). -->

### Quick Start

After the installation of `phic` and downloading of the directory [_demo_](/demo), move to the directory [_demo_](/demo):

    demo/
      run.sh

Then, execute the following script:

    ./run.sh

This process may take a few minutes.

The demo uses Hi-C data of mouse embryonic stem cells (chr2: 40–65 Mb, 25-kb resolution, KR normalization) by [Bonev et al.](https://doi.org/10.1016/j.cell.2017.09.043).

* * *

### Usage

`phic` needs a subcommand on the command line interface:

    phic SUBCOMMAND [OPTIONS]

    Subcommands:
    fetch-fileinfo
          ↓
    preprocessing
          ↓
    optimization
          ├──> plot-optimization
          ├──> dynamics
          ├──> sampling
          ├──> msd
          │    └──> plot-msd
          └──> losstangent
               └──> plot-losstangent

As of version 2.2.1, most subcommands accept experimental `--json` / `--json-path` / `--run-uuid` options that append a structured analysis log to `phic.json` in the workspace. These options are provided for internal pipeline integration; the schema and detailed usage will be documented in a future release.

#### 0. fetch-fileinfo

    phic fetch-fileinfo [OPTIONS]

    Options:
      --input               TEXT     Input Hi-C file (.hic or .mcool format)  [required]

The `fetch-fileinfo` subcommand is used to inspect the basic metadata of a Hi-C data file.
As of version 2.1.1, `phic` supports both `.hic` and `.mcool` formats as input.

Use this command to check available chromosomes, resolution levels, and indexing details in the input file before proceeding with further analysis. This ensures that downstream subcommands reference the correct chromosome names and binning resolutions.

This is a recommended first step when working with new input files.

Example:

    phic fetch-fileinfo --input FILENAME.hic

#### 1. preprocessing

    phic preprocessing [OPTIONS]

    Options:
      --input               TEXT     Input Hi-C file (.hic or .mcool format)  [required]
      --res                 INTEGER  Resolution of the bin size  [required]
      --plt-max-c           FLOAT    Maximum value of contact map  [required]
      --for-high-resolution FLAG     Normalization of contact map for high-resolution case (ex. 1-kb, 500-bp, 200-bp)  [default=False]
      --chr                 TEXT     Target chromosome  [required]
      --grs                 INTEGER  Start position of the target genomic region
      --gre                 INTEGER  End position of the target genomic region
      --norm                TEXT     Type of normalization to apply
      --tolerance           FLOAT    Threshold used to remove segments containing NaN values  [required]
      --help                         Show this message and exit.

In version 2.1.1 and later, the input data format has been changed to `.hic` or `.mcool`. Additionally, it is now possible to exclude rows and columns containing NaN values from the analysis by specifying their allowed proportion (ranging from 0 to 1) using the `tolerance` parameter.

When using the `preprocessing` subcommand, a directory will be automatically created based on the input Hi-C file name, chromosome number, genomic region of interest (optional), resolution, and normalization method. All subsequent analysis results will be stored in this directory. In the following explanations, we refer to this directory as _NAME_.

The outputs are as follows:

    NAME/
    ├── C_normalized.npz
    ├── C_normalized.svg
    ├── P_normalized.npz
    ├── P_normalized.svg
    └── _meta_data/

The two `.npz` files can be loaded with `numpy.load` using the keys below:

| File | Key | Shape | Description |
|---|---|---|---|
| `C_normalized.npz` | `C_normalized` | `(N, N)` | Normalized contact matrix (diagonal = 1; NaN-marked rows/columns are preserved) |
| `P_normalized.npz` | `P_normalized` | `(N, 2)` | Column 0: genomic distance [bp]; column 1: averaged normalized contact probability |

Example:

    phic preprocessing --input FILENAME.hic --res 25000 --plt-max-c 0.05 --chr 2 --grs 40000000 --gre 65000000 --norm KR --tolerance 0.4
    phic preprocessing --input FILENAME.hic --res 100000 --plt-max-c 0.05 --chr 2 --norm KR --tolerance 0.8

<!-- <img src="/img/fig1.svg" height="250"> -->

#### 2. optimization

    phic optimization [OPTIONS]

    Options:
      --name                      TEXT   Target directory name  [required]
      --init-k-backbone           FLOAT  Initial parameter of K_i,i+1  [default=0.5]
      --stop-condition-parameter  FLOAT  Parameter for the stop condition  [default=1e-7]
      --backtracking-factor       FLOAT  Backtracking factor  [default=0.7]
      --gradient-degree           INT    Gradient used for optimizing of K  [default=2]
      --help                             Show this message and exit.


The outputs are the followings:

    NAME/
    └── data_optimization/
        ├── K_optimized.npz
        ├── C_optimized.npz
        ├── P_optimized.npz
        └── optimization.log

As of version 2.2.1, `optimization` also produces `C_optimized.npz` and `P_optimized.npz` (previously generated by `plot-optimization`), so the downstream analyses no longer require running `plot-optimization` first. The three `.npz` files can be loaded with `numpy.load` using the keys below:

| File | Key | Shape | Description |
|---|---|---|---|
| `K_optimized.npz` | `K_optimized` | `(N, N)` | Optimized polymer network interaction matrix |
| `C_optimized.npz` | `C_optimized` | `(N, N)` | Contact matrix reconstructed from the optimized `K` (NaN positions of `C_normalized` are preserved) |
| `P_optimized.npz` | `P_optimized` | `(N, 2)` | Column 0: genomic distance [bp]; column 1: averaged contact probability from `C_optimized` |

Example:

    phic optimization --name NAME


#### 3-1. plot-optimization

    phic plot-optimization [OPTIONS]

    Options:
      --name        TEXT      Target directory name  [required]
      --plt-max-c   FLOAT     Maximum value of contact map  [required]
      --plt-max-k   FLOAT     Maximum and minimum values of optimized K map  [required]
      --help                  Show this message and exit.

As of version 2.2.1, `plot-optimization` reads the arrays pre-computed by `optimization` and only renders figures; it no longer outputs any `.npz` files. The `--res` option has been removed (the resolution is taken from `P_normalized.npz`).

The outputs are the followings:

    NAME/
    └── data_optimization/
        ├── C.svg
        ├── Correlation.png
        ├── Correlation_distance_corrected.png
        ├── Cost.svg
        ├── Eta.svg
        ├── K.svg
        └── P.svg

Example:

    phic plot-optimization --name NAME --plt-max-c 0.05 --plt-max-k 0.01

<!-- <img src="/img/fig2.svg" height="500"> -->


#### 3-2. dynamics

    phic dynamics [OPTIONS]

    Options:
      --name      TEXT      Target directory name  [required]
      --eps       FLOAT     Stepsize in the Langevin dynamics  [default=1e-3]
      --interval  INTEGER   The number of steps between output frames  [required]
      --frame     INTEGER   The number of output frames  [required]
      --sample    INTEGER   The number of output dynamics  [default=1]
      --seed      INTEGER   Seed of the random numbers  [default=12345678]
      --help                Show this message and exit.

The outputs are the followings:

    NAME/
    └── data_dynamics/
        ├── polymer_N{NUMBER-OF-BEADS}.psf
        ├── sample{SAMPLE-NUMBER}.dcd
        └── sample{SAMPLE-NUMBER}.xyz

Example:

    phic dynamics --name NAME --interval 10 --frame 100

#### 3-3. sampling

    phic sampling [OPTIONS]

    Options:
      --name    TEXT      Target directory name  [required]
      --sample  INTEGER   The number of output conformations  [required]
      --seed    INTEGER   Seed of the random numbers  [default=12345678]
      --help              Show this message and exit.

The outputs are the followings:

    NAME/
    └── data_sampling/
        ├── polymer_N{NUMBER-OF-BEADS}.psf
        ├── conformations.dcd
        └── conformations.xyz

Example:

    phic sampling --name NAME --sample 100

#### 3-4-1. msd

    phic msd [OPTIONS]

    Options:
      --name  TEXT     Target directory name  [required]
      --help           Show this message and exit.

As of version 2.2.1, the exponent range of the normalized time is automatically determined from the eigenvalues of the Laplacian matrix induced from the optimized polymer network interaction matrix, so the `--upper` and `--lower` options have been removed.

The output is the following:

    NAME/
    └── data_MSD/
        └── MSD_matrix.npz

`MSD_matrix.npz` contains three arrays and can be loaded with `numpy.load` using the keys below:

| File | Key | Shape | Description |
|---|---|---|---|
| `MSD_matrix.npz` | `MSD` | `(M+1, N)` | MSD of each bead `n` at each normalized time `t[m]` |
| | `t` | `(M+1,)` | Normalized time points (log-spaced) |
| | `tau` | `(N,)` | Relaxation times of the normal modes; `tau[0]` is `NaN` (center-of-mass mode) |

Example:

    phic msd --name NAME

#### 3-4-2. plot-msd

    phic plot-msd [OPTIONS]

    Options:
      --name        TEXT     Target directory name  [required]
      --plt-upper   INTEGER  Upper value of the exponent of the normalized time in the spectrum  [required]
      --plt-lower   INTEGER  Lower value of the exponent of the normalized time in the spectrum  [required]
      --plt-max-log FLOAT    Maximum value of log10 MSD  [required]
      --plt-min-log FLOAT    Minimum value of log10 MSD  [required]
      --aspect      FLOAT    Aspect ratio of the spectrum  [default=0.8]
      --help                 Show this message and exit.

The outputs are the followings:

    NAME/
    └── data_MSD/
        ├── fig_MSD_curves.png
        └── fig_MSD_spectrum.svg

Example:

    phic plot-msd --name NAME --plt-upper 3 --plt-lower 0 --plt-max-log 2.0 --plt-min-log 0.5 --aspect 1.5

#### 3-5-1. losstangent

    phic losstangent [OPTIONS]

    Options:
      --name    TEXT      Target directory name  [required]
      --help              Show this message and exit.

As of version 2.2.1, the exponent range of the angular frequency is automatically determined from the eigenvalues of the Laplacian matrix induced from the optimized polymer network interaction matrix, so the `--upper` and `--lower` options have been removed.

The output is the following:

    NAME/
    └── data_losstangent/
        └── losstangent_matrix.npz

`losstangent_matrix.npz` contains three arrays and can be loaded with `numpy.load` using the keys below:

| File | Key | Shape | Description |
|---|---|---|---|
| `losstangent_matrix.npz` | `losstangent` | `(M+1, N)` | Loss tangent tan δ of each bead `n` at each angular frequency `omega[m]` |
| | `omega` | `(M+1,)` | Normalized angular frequency points (log-spaced) |
| | `tau` | `(N,)` | Relaxation times of the normal modes; `tau[0]` is `NaN` (center-of-mass mode) |

Example:

    phic losstangent --name NAME

#### 3-5-2. plot-losstangent

    phic plot-losstangent [OPTIONS]

    Options:
      --name          TEXT      Target directory name  [required]
      --plt-upper     INTEGER   Upper value of the exponent of the angular frequency in the spectrum  [required]
      --plt-lower     INTEGER   Lower value of the exponent of the angular frequency in the spectrum  [required]
      --plt-max-log   FLOAT     Maximum value of log10 tanδ  [required]
      --aspect        FLOAT     Aspect ratio of the spectrum  [default=0.8]
      --help                    Show this message and exit.

The output is the following:

    NAME/
    └── data_losstangent/
        └── fig_losstangent_spectrum.svg

Example:

    phic plot-losstangent --name NAME --plt-upper 0 --plt-lower -3 --plt-max-log 0.3 --aspect 1.5
