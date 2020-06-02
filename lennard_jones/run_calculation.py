import ljmd

N    = 125
T    = 0.728
rho  = 0.8442
rcut = 2.5

delt   = 0.001
tmax   = 0.01
tequil = 0.0
Nstep  = round(tmax/delt)
Nprint = 1

lj_calc = ljmd.ljmd( N, delt, Nstep, tequil, Nprint, T, rho, rcut )

lj_calc.kernel()
