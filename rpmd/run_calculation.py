import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims/rpmd')
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
integ   = 'cayley'

rpmd_calc = rpmd.rpmd( beta, m, nbeads, delt, systype, Nstep, Nprint, Ntraj, Nequil, Ntemp, integ )

rpmd_calc.kernel()
