from train import train
import argparse
import numpy as np
import os
parser=argparse.ArgumentParser(description="GMM VAE for microseq clustering")
parser.add_argument("-b","--batch-size",default=128,type=int)
parser.add_argument("-n","--num-epoche",default=40,type=int)
parser.add_argument("-lr","--learning-rate",default=1e-4,type=float)
parser.add_argument("-c","--cuda",default=True,type=bool)
parser.add_argument("-x","--dimx",default=80,type=int)
parser.add_argument("-w","--dimw",default=60,type=int)
parser.add_argument("-k","--K",default=22,type=int)
parser.add_argument("-m","--M",default=10,type=int)
parser.add_argument("-l","--L",default=0,type=float)
parser.add_argument("-s","--speed-compile",default=True,type=bool)
args=parser.parse_args()
l_tuple=(2,2.25,2.5,2.75,3)
odp={}
n=4
for l in l_tuple:
    args.L=l
    values=np.zeros([n,5])
    for i in range(n):
        values[i,:]=train(args)
    odp[l]=values
np.save("estimates_change_l",odp)
