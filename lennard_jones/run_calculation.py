import ljmd

N    = 125
T    = 0.728
rho  = 0.8442
rcut = 2.5

delt   = 0.001
tmax   = 100.0
tequil = 2.0
Nstep  = round(tmax/delt)
Nprint = 100

lj_calc = ljmd.ljmd( N, delt, Nstep, tequil, Nprint, T, rho, rcut )

lj_calc.kernel()
