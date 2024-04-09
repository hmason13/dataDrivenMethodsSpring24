# cartEnv


import netCDF4 as nc
import numpy as np


# params
nx = 500
ny = 500
loadLoc = "/scratch/hcm7920/amb0/data/"
saveLoc = "/scratch/hcm7920/class/mitgcmData/"
blockSize = 50
loopSizeX = nx//blockSize
loopSizeY = ny//blockSize

# load data
data = nc.Dataset(loadLoc+"state.nc")
u = data["U"]
v = data["V"]
eta = data["Eta"]
nt = 10

# process data
counter = 0
for tStep in range(0,700,50):
  for bx in range(loopSizeX):
    for by in range(loopSizeY):
      
      xRange = np.arange(bx*blockSize,(bx+1)*blockSize,1)
      yRange = np.arange(by*blockSize,(by+1)*blockSize,1)
      
      myU = u[tStep,0,yRange,xRange].data.flatten()
      myV = v[tStep,0,yRange,xRange].data.flatten()
      myEta = eta[tStep,yRange,xRange].data.flatten()
      
      np.save(saveLoc+f"u/dataU-{counter:05}.npy", myU)
      np.save(saveLoc+f"v/dataV-{counter:05}.npy", myV)
      np.save(saveLoc+f"eta/dataEta-{counter:05}.npy", myEta)
      
      counter += 1
      print(f"onto # {counter}")