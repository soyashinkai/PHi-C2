import numpy as np
import hicstraw

HIC_FILE = "http://hicfiles.s3.amazonaws.com/external/bonev/ES_mapq30.hic"

RES = 25000
CHR = "8"
START = 42100000
END = 44500000
NAME = "Bonev_ES_observed_KR_chr{0:s}_{1:d}-{2:d}_res{3:d}bp".format(CHR, START, END, RES)

hic = hicstraw.HiCFile(HIC_FILE)
mzd = hic.getMatrixZoomData(CHR, CHR, "observed", "KR", "BP", RES)
input_matrix = mzd.getRecordsAsMatrix(START, END, START, END)

np.savetxt("{0:s}.txt".format(NAME), input_matrix, fmt="%e")
