# PHi-C2
PHi-C2 allows for a physical interpretation of a Hi-C contact matrix.
The `phic` package includes a suite of command line tools.

### Installation

Install `phic` from GitHub using pip:

    pip install git+https://github.com/soyashinkai/PHi-C2


### Requirements
- PHi-C2 is based on `python3`.
- Python packages `numpy`, `matplotlib`, `scipy`, `numba`, `click`.

To visualize the simulated polymer dynamics and conformations, [VMD](https://www.ks.uiuc.edu/Research/vmd/) is needed.


### Citation

We will submit a manuscript on PHi-C2, in which we dramatically updated the algorithm of the optimization procedure.
But, the basic framework remains the same in the following papers:

- Soya Shinkai, Masaki Nakagawa, Takeshi Sugawara, Yuichi Togashi, Hiroshi Ochiai, Ryuichiro Nakato, Yuichi Taniguchi, and Shuichi Onami. (2020). **PHi-C: deciphering Hi-C data into polymer dynamics.** [_NAR Genomics and Bioinformatics_ **2** (2) lqaa020](https://doi.org/10.1093/nargab/lqaa020).

- Soya Shinkai, Takeshi Sugawara, Hisashi Miura, Ichiro Hiratani, and Shuichi Onami. (2020). **Microrheology for Hi-C Data Reveals the Spectrum of the Dynamic 3D Genome Organization.** [_Biophysical Journal_ **118** 2220–2228](https://doi.org/10.1016/j.bpj.2020.02.020).

### Quick Start

After the installation of `phic` and downloading of the directory [_demo_](/demo), move to the directory [_demo_](/demo):

    demo/
      Bonev_ES_observed_KR_chr8_42100-44500kb_res25kb.txt
      run.sh

Then, run the following scripts:

    ./run.sh

It will take a few minutes.

Here, `Bonev_ES_observed_KR_chr8_42100-44500kb_res25kb.txt` is an input file dumped by [Juicertools](https://github.com/aidenlab/juicer/wiki/Data-Extraction) with KR normalization for Hi-C data of mouse embryo stem cells (chr8: 42,100-44,525 kb, 25-kb resolution) by [Bonev et al.](https://doi.org/10.1016/j.cell.2017.09.043).

* * *

### Usage

`phic` needs a subcommand on the command line interface:

    phic SUBCOMMAND [OPTIONS]

    Subcommands:
    preprocessing
      |
    optimization
      |-->  plot-optimization
      |-->  dynamics
      |-->  sampling
      |-->  rheology
              |--> plot-compliance
              |--> plot-modulus
              |--> plot-tangent


#### 1. preprocessing

    phic preprocessing [OPTIONS]

    Options:
      --input     TEXT      Input file dumped by Juicertools for a hic file  [required]
      --res       INTEGER   Resolution of the bin size  [required]
      --plt-max-c FLOAT     Maximum value of contact map  [required]
      --help                Show this message and exit.

Here, _NAME.txt_ as an ipunt is in sparse matrix format produced from [“dump” command of Juicebox](https://github.com/aidenlab/juicer/wiki/Data-Extraction):

    42100000	42100000	12899.836
    42100000	42125000	2076.9636
    42125000	42125000	11072.94
    42100000	42150000	1264.3281
    .............................
    .............................
    44475000	44500000	3374.337
    44500000	44500000	10828.436


The outputs are the followings:

    NAME/
      C.txt
      C_normalized.svg
      C_normalized.txt
      P_normalized.svg
      P_normalized.txt

![C_normalized](/img/C_normalized.svg)

#### 2. optimization

    phic optimization [OPTIONS]

    Options:
      --name                      TEXT    Target directory name  [required]
      --init-k-backbone           FLOAT   Initial parameter of K_i,i+1  [default=0.5]
      --larning-rate              FLOAT   Larning rate  [default=1e-4]
      --stop-condition-parameter  FLOAT   Parameter for the stop condition  [default=1e-4]
      --help                              Show this message and exit.

The outputs are the followings:

    NAME/data_optimization/
      K_optimized.txt
      optimization.log


#### 3-1. plot-optimization

    phic plot-optimization [OPTIONS]

    Options:
      --name                TEXT      Target directory name  [required]
      --res                 INTEGER   Resolution of the bin size  [required]
      --plt-max-c           FLOAT     Maximum value of contact map  [required]
      --plt-max-k-backbone  FLOAT     Maximum value of K_i,i+1 profile  [required]
      --plt-max-k           FLOAT     Maximum and minimum values of optimized K map  [required]
      --plt-k-dis-bins      INTEGER   The number of bins of distribution of optimized K values  [required]
      --plt-max-k-dis       FLOAT     Maximum value of the K distributioin  [required]
      --help                          Show this message and exit.

The outputs are the followings:

    NAME/data_optimization/
      C.svg
      Correlation.png
      Cost.svg
      K.svg
      K_backbone.svg
      K_backbone.txt
      K_distribution.svg
      P.svg

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

#### 3-4-1. rheology

    phic rheology [OPTIONS]

    Options:
      --name    TEXT      Target directory name  [required]
      --upper   INTEGER   Upper value of the exponent of the angular frequency  [default=1]
      --lower   INTEGER   Lower value of the exponent of the angular frequency  [default=-5]
      --help              Show this message and exit.

The outputs are the followings:

    NAME/data_rheology/
      n{BEAD-NUMBER}.txt

#### 3-4-2. plot-compliance

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

    NAME/data_rheology/figs/
      J_abs_spectrum.svg
      J_curves.png

#### 3-4-2. plot-modulus

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

    NAME/data_rheology/figs/
      G_abs_spectrum.svg
      G_curves.png

#### 3-4-3. plot-tangent

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

    NAME/data_rheology/figs/
      tan_spectrum.svg
