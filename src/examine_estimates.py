import numpy as np
import pandas as pd

i = 6
data_zero_l = np.load("estimates_{}.npy".format(i), allow_pickle=True).item()
f = open("metrics_{}.txt".format(i), "w")
f.write("K,ARI,ASW,NMI,Batch ASW,kBET\n")
for element in data_zero_l:
    f.write(
        "$${0:.0f}$$,$${1:.3f}\pm {2:.3f}$$,$${3:.3f}\pm {4:.3f}$$,$${5:.3f}\pm {6:.3f}$$,$${7:.3f}\pm {8:.3f}$$,$${9:.3f}\pm {10:.3f}$$\n".format(
            element,
            np.mean(data_zero_l[element], 0)[0],
            np.std(data_zero_l[element], 0)[0],
            np.mean(data_zero_l[element], 0)[1],
            np.std(data_zero_l[element], 0)[1],
            np.mean(data_zero_l[element], 0)[2],
            np.std(data_zero_l[element], 0)[2],
            np.mean(data_zero_l[element], 0)[3],
            np.std(data_zero_l[element], 0)[3],
            np.mean(data_zero_l[element], 0)[4],
            np.std(data_zero_l[element], 0)[4],
        )
    )

f.close()

