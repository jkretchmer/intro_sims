import mb_mc

m      = 1.0
beta   = 1.0
delp   = 0.1
Nprint = 10
Nstep  = 10000000 - Nprint
mb_calc = mb_mc.mb_mc( m, beta, delp, Nstep, Nprint )

mb_calc.kernel()
