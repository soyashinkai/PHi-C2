#!/bin/bash

# Set Hi-C file (.hic format)
HIC="http://hicfiles.s3.amazonaws.com/external/bonev/ES_mapq30.hic"

CHR="8"
START=42100000
END=44525000
RES=25000
PLT_MAX_C=0.1

# Fetch the input Hi-C file
phic fetch-fileinfo --input ${HIC}

# Run the preprocessing
phic preprocessing --input ${HIC} --res ${RES} --plt-max-c ${PLT_MAX_C} --chr ${CHR} --grs ${START} --gre ${END} --norm KR --tolerance 0.6

NAME="ES_mapq30_KR_chr8_42100000-44525000_res25000bp"

# Run the optimization
phic optimization --name ${NAME}

# Plot the optimized results
phic plot-optimization --name ${NAME} --res ${RES} --plt-max-c ${PLT_MAX_C} --plt-max-k 0.1

# Run the 4D dynamics simulation
phic dynamics --name ${NAME} --interval 100 --frame 1000

# Run the 3D conformation sampleing
phic sampling --name ${NAME} --sample 1000

# Calculate the MSDs
phic msd --name ${NAME}

# Plot the spectrum of the MSDs
phic plot-msd --name ${NAME} --plt-upper 3 --plt-lower 0 --plt-max-log 2.0 --plt-min-log 0.5 --aspect 0.2

# Calculate the rheology features
phic rheology --name ${NAME}

# Plot curves and spectrum of the complex comliance J
phic plot-compliance --name ${NAME} --plt-upper 0 --plt-lower -3 --plt-max-log 1.3 --plt-min-log -0.3 --aspect 0.2

# Plot curves and spectrum of the complex modulus G
phic plot-modulus --name ${NAME} --plt-upper 0 --plt-lower -3 --plt-max-log 0.4 --plt-min-log -1.2 --aspect 0.2

# Plot the spectrum of the loss tangent
phic plot-tangent --name ${NAME} --plt-upper 0 --plt-lower -3 --plt-max-log 0.2 --aspect 0.2
