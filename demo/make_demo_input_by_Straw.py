import numpy as np
import hicstraw
# --------------------------------------------------------------------------------------------------
HIC_FILE = "http://hicfiles.s3.amazonaws.com/external/bonev/ES_mapq30.hic"
SUBJECT = "Bonev_mESCs_observed_KR"
CHR = "8"
START = 42100000
END = 44525000
RES = 25000
NAME = "{0:s}_chr{1:s}_{2:d}-{3:d}_res{4:d}bp".format(SUBJECT, CHR, START, END, RES)
# --------------------------------------------------------------------------------------------------
hic = hicstraw.HiCFile(HIC_FILE)
mzd = hic.getMatrixZoomData(CHR, CHR, "observed", "KR", "BP", RES)
input_matrix = mzd.getRecordsAsMatrix(START, END - RES, START, END - RES)
records_list = mzd.getRecords(START, END - RES, START, END - RES)

np.savetxt("{0:s}.txt".format(NAME), input_matrix, fmt="%e")
print("Contact matrix size is {0:d}x{0:d}".format(input_matrix.shape[0]))

fp = open("{0:s}_list.txt".format(NAME), "w")
for i in range(len(records_list)):
    print("{0:d}\t{1:d}\t{2:f}".format(records_list[i].binX, records_list[i].binY, records_list[i].counts),
          file=fp)
fp.close()
