import rpmd

beta   = 1.0
m      = 1.0
nbeads = 4

delt   = 0.001
tmax   = 30.0
Nstep  = round(tmax/delt)
Nprint = 100
Ntraj  = 3000
Nequil = 10000
Ntemp  = 100

systype = 'quartic'

rpmd_calc = rpmd.rpmd( beta, m, nbeads, delt, systype, Nstep, Nprint, Ntraj, Nequil, Ntemp )

rpmd_calc.kernel()
