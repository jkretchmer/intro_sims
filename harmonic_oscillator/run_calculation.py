import harmonic_oscillator

m      = 1.0
k      = 1.0
xi     = 1.0

delt   = 0.001
tmax   = 30.0
Nstep  = round(tmax/delt)
Nprint = 100

ho_calc = harmonic_oscillator.harm_osc( m, k, delt, Nstep, Nprint, xi )

ho_calc.kernel()
