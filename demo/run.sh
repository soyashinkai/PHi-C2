#!/bin/bash

# Set input name
NAME=Bonev_ES_observed_KR_chr8_42100-44500kb_res25kb

RES=25000
PLT_MAX_C=0.1

# Run the preprocessing
phic preprocessing --input ${NAME}.txt --res ${RES} --plt-max-c ${PLT_MAX_C}

# Run the optimization
phic optimization --name ${NAME}

# Plot the optimized results
phic plot-optimization --name ${NAME} --res ${RES} --plt-max-c ${PLT_MAX_C} \
  --plt-max-k-backbone 1.0 \
  --plt-max-k 0.1 \
  --plt-k-dis-bins 200 \
  --plt-max-k-dis 100

# Run the 4D dynamics simulation
phic dynamics --name ${NAME} \
  --interval 100 \
  --frame 1000

# Run the 3D conformation sampleing
phic sampling --name ${NAME} \
  --sample 1000

# Calculate the rheology features
phic rheology --name ${NAME}

# Plot curves and spectrum of the complex comliance J
phic plot-compliance --name ${NAME} \
  --plt-upper 0 \
  --plt-lower -3 \
  --plt-max-log 1.3 \
  --plt-min-log -0.3 \
  --aspect 0.2

# Plot curves and spectrum of the complex modulus G
phic plot-modulus --name ${NAME} \
  --plt-upper 0 \
  --plt-lower -3 \
  --plt-max-log 0.4 \
  --plt-min-log -1.2 \
  --aspect 0.2

# Plot the spectrum of the loss tangent
phic plot-tangent --name ${NAME} \
  --plt-upper 0 \
  --plt-lower -3 \
  --plt-max-log 0.2 \
  --aspect 0.2
