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

Without preparing a Python environment, PHi-C2 (=<2.0.13) rus on [Google Colab](https://bit.ly/3rlptGI).

### Requirements
- PHi-C2 is based on `python3`.
- Python packages `numpy`, `matplotlib`, `scipy`, `click`, `pandas`, `hic-straw`, `cooler`, `h5py`.

To visualize the simulated polymer dynamics and conformations, [VMD](https://www.ks.uiuc.edu/Research/vmd/) is needed.


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

The demo uses Hi-C data of mouse embryonic stem cells (chr8: 42,100–44,525 kb, 25-kb resolution, KR normalization) by [Bonev et al.](https://doi.org/10.1016/j.cell.2017.09.043).

* * *

### Usage

`phic` needs a subcommand on the command line interface:

    phic SUBCOMMAND [OPTIONS]

    Subcommands:
    fetch-fileinfo
      |
    preprocessing
      |
    optimization
      |-->  plot-optimization
      |-->  dynamics
      |-->  sampling
      |-->  msd
      |       |--> plot-msd
      |
      |-->  rheology
              |--> plot-tangent
              |--> plot-compliance
              |--> plot-modulus

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
      --for-high-resolution INTEGER  Normalization of contact map for high-resolution case (ex. 1-kb, 500-bp, 200-bp)  [default=0]
      --chr                 TEXT     Target chromosome  [required]
      --grs                 INTEGER  Start position of the target genomic region
      --gre                 INTEGER  End position of the target genomic region
      --norm                TEXT     Type of normalization to apply
      --tolerance           FLOAT    Threshold used to remove segments containing NaN values [required]
      --help                         Show this message and exit.

In version 2.1.1 and later, the input data format has been changed to `.hic` or `.mcool`. Additionally, it is now possible to exclude rows and columns containing NaN values from the analysis by specifying their allowed proportion (ranging from 0 to 1) using the `tolerance` parameter.

When using the `preprocessing` subcommand, a directory will be automatically created based on the input Hi-C file name, chromosome number, genomic region of interest (optional), resolution, and normalization method. All subsequent analysis results will be stored in this directory. In the following explanations, we refer to this directory as _NAME_.

The outputs are as follows:

    NAME/
      C_normalized.svg
      C_normalized.txt
      P_normalized.svg
      P_normalized.txt
      _meta_data/

Example:

    phic preprocessing --input FILENAME.hic --res 25000 --plt-max-c 0.1 --chr 8 --grs 42100000 --gre 44525000 --norm KR --tolerance 0.6
    phic preprocessing --input FILENAME.hic --res 250000 --plt-max-c 0.1 --chr 8 --norm KR --tolerance 0.6

<img src="/img/fig1.svg" height="250">

#### 2. optimization

    phic optimization [OPTIONS]

    Options:
      --name                      TEXT   Target directory name  [required]
      --init-k-backbone           FLOAT  Initial parameter of K_i,i+1  [default=0.5]
      --learning-rate             FLOAT  Learning rate  [default=1e-4]
      --stop-condition-parameter  FLOAT  Parameter for the stop condition  [default=1e-4]
      --help                             Show this message and exit.


The outputs are the followings:

    NAME/data_optimization/
      K_optimized.txt
      optimization.log

Example:

    phic optimization --name NAME


#### 3-1. plot-optimization

    phic plot-optimization [OPTIONS]

    Options:
      --name        TEXT      Target directory name  [required]
      --res         INTEGER   Resolution of the bin size  [required]
      --plt-max-c   FLOAT     Maximum value of contact map  [required]
      --plt-max-k   FLOAT     Maximum and minimum values of optimized K map  [required]
      --help                  Show this message and exit.

The outputs are the followings:

    NAME/data_optimization/
      C.svg
      C_optimized.txt
      Correlation.png
      Correlation_distance_corrected.png
      Cost.svg
      K.svg
      P.svg

Example:

    phic plot-optimization --name NAME --res 25000 --plt-max-c 0.1 --plt-max-k 0.1

<img src="/img/fig2.svg" height="500">


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

    NAME/data_dynamics/
      polymer_N{NUMBER-OF-BEADS}.psf
      sample{SAMPLE-NUMBER}.xyz

Example:

    phic dynamics --name NAME --interval 100 --frame 1000

#### 3-3. sampling

    phic sampling [OPTIONS]

    Options:
      --name    TEXT      Target directory name  [required]
      --sample  INTEGER   The number of output conformations  [required]
      --seed    INTEGER   Seed of the random numbers  [default=12345678]
      --help              Show this message and exit.

The outputs are the followings:

    NAME/data_sampling/
      polymer_N{NUMBER-OF-BEADS}.psf
      conformations.xyz

Example:

    phic sampling --name NAME --sample 1000

#### 3-4-1. msd

    phic msd [OPTIONS]

    Options:
      --name  TEXT     Target directory name  [required]
      --upper INTEGER  Upper value of the exponent of the normalized time [default=5]
      --lower INTEGER  Lower value of the exponent of the normalized time [default=-1]
      --help           Show this message and exit.

The outputs are the followings:

    NAME/data_MSD/
      n{BEAD-NUMBER}.txt

Example:

    phic msd --name NAME

#### 3-4-2. plot-msd

    phic plot-msd [OPTIONS]

    Options:
      --name        TEXT     Target directory name  [required]
      --upper       INTEGER  Upper value of the exponent of the normalized time  [default=5]
      --lower       INTEGER  Lower value of the exponent of the normalized time  [default=-1]
      --plt-upper   INTEGER  Upper value of the exponent of the normalized time in the spectrum  [required]
      --plt-lower   INTEGER  Lower value of the exponent of the normalized time in the spectrum  [required]
      --plt-max-log FLOAT    Maximum value of log10 MSD  [required]
      --plt-min-log FLOAT    Minimum value of log10 MSD  [required]
      --aspect      FLOAT    Aspect ratio of the spectrum  [default=0.8]
      --help                 Show this message and exit.
    
The output is the following:

    NAME/data_MSD/
      data_MSD_spectrum.txt
    NAME/data_MSD/figs/
      MSD_spectrum.svg
      MSD_curves.png

Example:

    phic plot-msd --name NAME --plt-upper 3 --plt-lower 0 --plt-max-log 2.0 --plt-min-log 0.5 --aspect 0.2

#### 3-5-1. rheology

    phic rheology [OPTIONS]

    Options:
      --name    TEXT      Target directory name  [required]
      --upper   INTEGER   Upper value of the exponent of the angular frequency  [default=1]
      --lower   INTEGER   Lower value of the exponent of the angular frequency  [default=-5]
      --help              Show this message and exit.

The outputs are the followings:

    NAME/data_rheology/
      data_normalized_omega1.txt
      n{BEAD-NUMBER}.txt

Example:

    phic rheology --name NAME

#### 3-5-2. plot-tangent

    phic plot-tangent [OPTIONS]

    Options:
      --name          TEXT      Target directory name  [required]
      --upper         INTEGER   Upper value of the exponent of the angular frequency  [default=1]
      --lower         INTEGER   Lower value of the exponent of the angular frequency  [default=-5]
      --plt-upper     INTEGER   Upper value of the exponent of the angular frequency in the spectrum  [required]
      --plt-lower     INTEGER   Lower value of the exponent of the angular frequency in the spectrum  [required]
      --plt-max-log   FLOAT     Maximum value of log10 tanδ  [required]
      --aspect        FLOAT     Aspect ratio of the spectrum  [default=0.8]
      --help                    Show this message and exit.

The output is the following:

    NAME/data_rheology/
      data_tan_spectrum.txt
    NAME/data_rheology/figs/
      tan_spectrum.svg

Example:

    phic plot-tangent --name NAME --plt-upper 0 --plt-lower -3 --plt-max-log 0.2

<img src="/img/fig5.svg" height="250">

#### 3-5-3. plot-compliance

    phic plot-compliance [OPTIONS]

    Options:
      --name          TEXT      Target directory name  [required]
      --upper         INTEGER   Upper value of the exponent of the angular frequency  [default=1]
      --lower         INTEGER   Lower value of the exponent of the angular frequency  [default=-5]
      --plt-upper     INTEGER   Upper value of the exponent of the angular frequency in the spectrum  [required]
      --plt-lower     INTEGER   Lower value of the exponent of the angular frequency in the spectrum  [required]
      --plt-max-log   FLOAT     Maximum value of log10 |J*|  [required]
      --plt-min-log   FLOAT     Minimum value of log10 |J*|  [required]
      --aspect        FLOAT     Aspect ratio of the spectrum  [default=0.8]
      --help                    Show this message and exit.

The outputs are the followings:

    NAME/data_rheology/
      data_J_storage_spectrum.txt
      data_J_loss_spectrum.txt
      data_J_abs_spectrum.txt
    NAME/data_rheology/figs/
      J_storage_spectrum.svg
      J_loss_spectrum.svg
      J_abs_spectrum.svg
      J_curves.png

Example:

    phic plot-compliance --name NAME --plt-upper 0 --plt-lower -3 --plt-max-log 1.3 --plt-min-log -0.3

<img src="/img/fig3.svg" height="250">

#### 3-5-4. plot-modulus

    phic plot-modulus [OPTIONS]

    Options:
      --name          TEXT      Target directory name  [required]
      --upper         INTEGER   Upper value of the exponent of the angular frequency  [default=1]
      --lower         INTEGER   Lower value of the exponent of the angular frequency  [default=-5]
      --plt-upper     INTEGER   Upper value of the exponent of the angular frequency in the spectrum  [required]
      --plt-lower     INTEGER   Lower value of the exponent of the angular frequency in the spectrum  [required]
      --plt-max-log   FLOAT     Maximum value of log10 |G*|  [required]
      --plt-min-log   FLOAT     Minimum value of log10 |G*|  [required]
      --aspect        FLOAT     Aspect ratio of the spectrum  [default=0.8]
      --help                    Show this message and exit.

The outputs are the followings:

    NAME/data_rheology/
      data_G_storage_spectrum.txt
      data_G_loss_spectrum.txt
      data_G_abs_spectrum.txt
    NAME/data_rheology/figs/
      G_storage_spectrum.svg
      G_loss_spectrum.svg
      G_abs_spectrum.svg
      G_curves.png

Example:

    phic plot-modulus --name NAME --plt-upper 0 --plt-lower -3 --plt-max-log 0.4 --plt-min-log -1.2

<img src="/img/fig4.svg" height="250">

