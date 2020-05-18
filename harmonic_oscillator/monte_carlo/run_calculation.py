import ho_mc

m      = 1.0
k      = 1.0
xi     = 0.0
temp   = 300
delx   = 0.1
Nstep  = 1000000 - 1

ho_calc = ho_mc.harm_osc_mc( m, k, delx, Nstep, xi, temp )

ho_calc.kernel()
