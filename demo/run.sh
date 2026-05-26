#!/bin/bash

# Set Hi-C file (.hic format)
HIC="http://hicfiles.s3.amazonaws.com/external/bonev/ES_mapq30.hic"

CHR="2"
START=40000000
END=65000000
RES=25000
PLT_MAX_C=0.05
TOL=0.4

# Fetch the input Hi-C file
phic fetch-fileinfo --input ${HIC}

# Run the preprocessing
phic preprocessing --input ${HIC} --res ${RES} --plt-max-c ${PLT_MAX_C} --chr ${CHR} --grs ${START} --gre ${END} --norm KR --tolerance ${TOL}

NAME="ES_mapq30_KR_chr${CHR}_${START}-${END}_res${RES}bp"

# Run the optimization
phic optimization --name ${NAME}

# Plot the optimized results
phic plot-optimization --name ${NAME} --plt-max-c ${PLT_MAX_C} --plt-max-k 0.01

# Run the 4D dynamics simulation
phic dynamics --name ${NAME} --interval 10 --frame 100

# Run the 3D conformation sampling
phic sampling --name ${NAME} --sample 100

# Calculate the MSDs
phic msd --name ${NAME}

# Plot the spectrum of the MSDs
phic plot-msd --name ${NAME} --plt-upper 3 --plt-lower 0 --plt-max-log 2.0 --plt-min-log 0.5 --aspect 1.5

# Calculate the loss tangent
phic losstangent --name ${NAME}

# Plot the spectrum of the loss tangent
phic plot-losstangent --name ${NAME} --plt-upper 0 --plt-lower -3 --plt-max-log 0.3 --aspect 1.5

