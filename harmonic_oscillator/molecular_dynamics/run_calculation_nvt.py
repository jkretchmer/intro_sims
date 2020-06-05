import harmonic_oscillator

m      = 1.0
k      = 1.0
xi     = 0.0

delt   = 0.001
tmax   = 99999.9
Nstep  = round(tmax/delt)
Nprint = 100

vi       = 0.0
temp     = 300
resample = True
Ntemp    = 1000

ho_calc = harmonic_oscillator.harm_osc( m, k, delt, Nstep, Nprint, xi, vi, temp, resample, Ntemp )

ho_calc.kernel()
