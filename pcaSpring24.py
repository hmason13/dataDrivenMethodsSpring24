
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import netCDF4 as nc

# parms
loc0 = "/scratch/hcm7920/pyqgOutput/"
loc2 = "fiveLayerFreeslip3/"
loc3 = "fiveLayerDrag3/"
loc4 = "fiveLayerExtradrag3/"
nx = 256
ny = 256
nz = 5
nt = 96
numNC = 1
maxNC = 35
numData1NC = (1+nt//10)*(ny//32)*(nx//32)

# setup the data holders
pv = np.zeros(shape=(5,numData1NC*numNC))
uVel = np.zeros(shape=(5,numData1NC*numNC))
vVel = np.zeros(shape=(5,numData1NC*numNC))

# load the data
print('loading data')
for i in range(numNC):
  data = nc.Dataset(loc0+loc4+f"rawOutput/model_output_{maxNC-i}.nc")
  indices = np.array(range(numData1NC))
  indices += int(i*numData1NC)
  
  for iz in range(nz):
    pv[iz,indices] = data["q"][::10,iz,::32,::32].flatten()
    vVel[iz,indices] = data["v"][::10,iz,::32,::32].flatten()
    uVel[iz,indices] = data["u"][::10,iz,::32,::32].flatten()
    
# # center the data
# print('centering data')
# pv   -= pv.mean((1,))[:,np.newaxis]
# vVel -= vVel.mean((1,))[:,np.newaxis]
# uVel -= uVel.mean((1,))[:,np.newaxis]

# do the linear algebra
print('doing linear algebra')
pv2 = pv@pv.T
vVel2 = vVel@vVel.T
uVel2 = uVel@uVel.T
Qevals, Qevecs = sp.linalg.eig(pv2)
Vevals, Vevecs = sp.linalg.eig(vVel2)
Uevals, Uevecs = sp.linalg.eig(uVel2)

# sort in decending order
print('reordering results')
dx = np.argsort(Qevals)[::-1]
Qevecs = Qevecs[:,dx]
Qevals = Qevals[dx]
dx = np.argsort(Vevals)[::-1]
Vevecs = Vevecs[:,dx]
Vevals = Vevals[dx]
dx = np.argsort(Uevals)[::-1]
Uevecs = Uevecs[:,dx]
Uevals = Uevals[dx]

# root the eigvals
Qevals = np.sqrt(Qevals.real)
Vevals = np.sqrt(Vevals.real)
Uevals = np.sqrt(Uevals.real)

# done!
print('done')
