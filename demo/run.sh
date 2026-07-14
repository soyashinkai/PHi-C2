#!/bin/bash

# Set Hi-C file (.hic format)
HIC="Kakui_etal_Nature_Genetics_2017_S_pombe_WT_interphase_MAPQ30_GW_normalization.hic"

CHR="I"
RES=25000
NORM="GW_KR"
PLT_MAX_C=0.04
TOL=0.6

NAME="Kakui_etal_Nature_Genetics_2017_S_pombe_WT_interphase_MAPQ30_GW_normalization_${NORM}_chr${CHR}_res${RES}bp"

# Fetch the input Hi-C file
python ${CODE} fetch-fileinfo --input ${HIC}

# Run the preprocessing
python ${CODE} preprocessing --input ${HIC} --res ${RES} --plt-max-c ${PLT_MAX_C} --chr ${CHR} --norm ${NORM} --tolerance ${TOL}

# Run the optimization
python ${CODE} optimization --name ${NAME}

# Plot the optimized results
python ${CODE} plot-optimization --name ${NAME} --plt-max-c ${PLT_MAX_C} --plt-max-k 0.01

# Run the 4D dynamics simulation
python ${CODE} dynamics --name ${NAME} --eps 1e-1 --frame 1000 --seed 1234

# Run the 3D conformation sampling
python ${CODE} sampling --name ${NAME} --sample 100 --seed 1234

# Calculate the MSDs
python ${CODE} msd --name ${NAME}

# Plot the spectrum of the MSDs
python ${CODE} plot-msd --name ${NAME} --plt-upper 4 --plt-lower -1 --plt-max-log 2.0 --plt-min-log 0 --aspect 0.2

# Calculate the loss tangent
python ${CODE} losstangent --name ${NAME}

# Plot the spectrum of the loss tangent
python ${CODE} plot-losstangent --name ${NAME} --plt-upper 1 --plt-lower -4 --plt-max-log 0.5 --aspect 0.2

