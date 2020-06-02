import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class rpmd():

    #class to integrate rpmd equations of motion

    def __init__( self, beta, m, nbeads, delt, systype, Nstep, Nprint, Ntraj, Nequil, Ntemp, integ='vv' ):


        #delt   - time-step
        #Nstep  - total number of MD steps per trajectory
        #Nprint - how often output data is calculated/printed
        #Ntraj  - number of dynamical trajectories
        #Nequil - number of equilibration steps prior to calculating correlation functions
        #Ntemp  - number of steps before resampling velocities during equilibration
        #beta   - inverse temperature
        #m      - physical mass of particle
        #nbeads - number of rpmd beads
        #systype - string defining the type of system ie the potential
        #integ   - string defining how the system is integrated

        #note that hbar = 1

        self.m       = m
        self.nbeads  = nbeads
        self.beta_n  = beta / nbeads
        self.omega_n = 1.0 / self.beta_n

        self.delt    = delt
        self.systype = systype
        self.Nstep   = Nstep
        self.Nprint  = Nprint
        self.Ntraj   = Ntraj
        self.Nequil  = Nequil
        self.Ntemp   = Ntemp
        self.integ   = integ

        #Check inputs are defined correctly
        self.check_input()
 
        #initialize system - position array xxx, velocity array vvv, and force array fff
        self.init()

        #initialize correlation function array
        self.corrfcn = np.zeros( [ round(Nstep/Nprint)+1, 2 ] )

        #Define output files
        self.file_output = open( 'output.dat', 'w' )

###############################################################

    def kernel( self ):

        print('******* RUNNING RPMD CALCULATION ********')
        print('Running ',self.Ntraj,'trajectories, each for ',self.Nstep,'steps')
        print('*****************************************')
        print()

        #equilibrate system in NVT ensemble
        self.equilibrate()

        #loop over number of trajectories to calculate correlation function
        for itraj in range( self.Ntraj ):

            #re-sample velocities at beginning of each trajectory
            self.sample_vel( self.beta_n )

            #save initial position com for correlation function
            initxcom = self.get_xcom()

            #MD loop for a given trajectory
            for step in range( self.Nstep ):

                currtime = step*self.delt

                #Calculate and print data of interest
                if( np.mod( step, self.Nprint ) == 0 ):
                    print( 'Writing data at step ', step, 'and time', currtime, 'for trajectory ',itraj )
                    self.calc_data( itraj, step, currtime, initxcom )

                #integrate eom
                self.integrate()

            #Calculate and print data of interest at last time-step for each trajectory
            step += 1
            currtime = step*self.delt
            print( 'Writing data at step ', step, 'and time', currtime, 'for trajectory ',itraj )
            self.calc_data( itraj, step, currtime, initxcom )

        #Finish calculation of correlation function
        self.corrfcn[:,1] = self.corrfcn[:,1] / self.Ntraj
        utils.printarray( self.corrfcn, 'corrfcn.dat' )

###############################################################

    def equilibrate( self ):

        #equilibrate system in NVT ensemble
        for step in range(self.Nequil):

            #integrate eom
            self.integrate()

            #re-sample velocities every Ntemp steps (note that this resamples velocities again at t=0)
            if( np.mod( step, self.Ntemp ) == 0 ):
                self.sample_vel( self.beta_n )

###############################################################

    def integrate( self ):

        #subroutine to integrate equations of motion
        #note switched from conventional velocity-verlet algorithm by integrating velocities first and then positions
        #this is more efficient and also interfaces with analytical and cayley integrations of internal modes

        self.get_velocities()   

        if( self.integ == 'vv' ):
            self.get_vv_positions()
        elif( self.integ == 'analyt' ):
            self.get_analyt_positions()
        elif( self.integ == 'cayley' ):
            self.get_cayley_positions()

        self.get_forces()
        self.get_velocities()

###############################################################

    def get_vv_positions( self ): 

        #integrate positions by a full time-step using velocity-verlet
        self.xxx += self.vvv * self.delt

###############################################################

    def get_analyt_positions( self ):

        #integrate positions using analytical result for internal modes of ring-polymer

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        xnm = self.real_to_normal_mode( self.xxx )
        vnm = self.real_to_normal_mode( self.vvv )

        #evolve position of the zero-freq mode using velocity-verlet
        #this is the force on the centroid and accounts for the external force on the position
        #external force on velocity accounted for in velocity call of velocity-verlet algorithm
        xnm[0] += vnm[0]*self.delt

        #evolve position/velocity of all other modes using analytical result for harmonic oscillators

        c1 = np.copy( vnm[1:] / self.nm_freq[1:] )
        c2 = np.copy( xnm[1:] )
        freq_dt = self.nm_freq[1:] * self.delt

        xnm[1:] = c1 * np.sin( freq_dt ) + c2 * np.cos( freq_dt )

        vnm[1:] = self.nm_freq[1:] * ( c1 * np.cos( freq_dt ) - c2 * np.sin( freq_dt ) )

        #Inverse transform back to real space
        self.xxx = self.normal_mode_to_real( xnm )
        self.vvv = self.normal_mode_to_real( vnm )

###############################################################

    def get_cayley_positions( self ):

        #integrate positions using analytical result for internal modes of ring-polymer

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        xnm = self.real_to_normal_mode( self.xxx )
        vnm = self.real_to_normal_mode( self.vvv )

        #evolve position/velocity of all modes using cayley transform
        xnm_copy = np.copy( xnm )
        vnm_copy = np.copy( vnm )

        xnm = ( self.nm_freq_dif * xnm_copy + self.delt * vnm_copy ) / self.nm_freq_sum

        vnm = ( -self.nm_freq_prod * xnm_copy + self.nm_freq_dif * vnm_copy ) / self.nm_freq_sum

        #Inverse transform back to real space
        self.xxx = self.normal_mode_to_real( xnm )
        self.vvv = self.normal_mode_to_real( vnm )

###############################################################

    def get_velocities( self ): 

        #integrate velocities by half a time-step
        #this update always includes the external force,
        #but only internal ring-polymer force if doing velocity-verlet integrator
        self.vvv += 0.5 * self.fff * self.delt / self.m

###############################################################

    def get_forces( self ): 

        self.fff = np.zeros(self.nbeads)
    
        #calculate forces due to internal modes of ring polymer only if using velocity-verlet for everything
        if( self.integ == 'vv' ):
            self.rp_force()
    
        #forces due to external potential
        if( self.systype == 'harmonic' ):
    
            k   = 1.0
            self.fff += -k * self.xxx
    
        elif( self.systype == 'anharmonic' ):
    
            self.fff += -self.xxx - 3.0/10.0 * self.xxx**2 - 1.0/25.0 * self.xxx**3
    
        elif( self.systype == 'quartic' ):
    
            self.fff += -self.xxx**3

###############################################################

    def rp_force( self ):

        #calculate forces due to internal modes of ring polymer

        nbeads = self.nbeads

        for i in range(nbeads):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[nbeads-1] - self.xxx[i+1] )
            elif( i == nbeads-1 ):
                #periodic boundary conditions for the last bead
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[i-1] - self.xxx[0] )
            else:
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[i-1] - self.xxx[i+1] )
    
###############################################################

    def init( self ):
    
        #initial position
        self.xxx = np.zeros( self.nbeads )
    
        #initial velocity from Maxwell-Boltzmann distribution
        self.sample_vel( self.beta_n )

        #calculate initial forces
        self.get_forces()   

        #initialize normal mode frequencies of ring polymer if doing analytical or cayley integration
        if( self.integ == 'analyt' or self.integ == 'cayley' ):
            self.normal_mode_freq()

###############################################################

    def normal_mode_freq( self ):

        #calculate frequencies of normal modes of ring-polymer
        self.nm_freq = 2.0 / self.beta_n * np.sin( np.arange(self.nbeads) * np.pi / self.nbeads )

        #if using cayley integrator make additional frequency arrays
        if( self.integ == 'cayley' ):
            self.nm_freq_prod = self.delt * self.nm_freq**2
            self.nm_freq_sum  = 1 + 0.25 * self.delt * self.nm_freq_prod
            self.nm_freq_dif  = 1 - 0.25 * self.delt * self.nm_freq_prod

###############################################################

    def real_to_normal_mode( self, real_space ):

        #Takes input array and calculates normal-modes using fourier-transform
        #Assumes input array is real and transforms complex output from FT to
        #real-valued normal modes by taking sums of real and imaginary components of degenerate frequency modes

        #discrete fourier transform, which gives complex results
        nm_cmplx = np.fft.rfft( real_space, norm='ortho' )

        #intialize terms for real valued normal mode array
        sz    = real_space.shape[0]
        midpt = round( sz/2 )
        nm    = np.zeros(sz)

        #define real valued normal modes
        nm[0]        = np.real( nm_cmplx[0] )
        nm[midpt]    = np.real( nm_cmplx[midpt] )
        nm[1:midpt]  = np.real( np.sqrt(0.5) * ( nm_cmplx[1:midpt] + np.conjugate(nm_cmplx[1:midpt]) ) )
        nm[midpt+1:] = np.flip( np.imag( np.sqrt(0.5) * ( nm_cmplx[1:midpt] - np.conjugate(nm_cmplx[1:midpt]) ) ) )

        return nm

###############################################################

    def normal_mode_to_real( self, nm ):

        #Takes input array of real-valued normal modes and calculates inverse FT
        #to obtain array in real-space

        #initialize terms for complex valued normal mode array
        sz       = nm.shape[0]
        midpt    = round( sz/2 )
        nm_cmplx = np.zeros( midpt+1, dtype=complex )

        #convert real valued to complex valued normal modes
        nm_cmplx[0]       = nm[0]
        nm_cmplx[midpt]   = nm[midpt]
        nm_cmplx[1:midpt] = np.sqrt(0.5) * ( nm[1:midpt] + 1j * np.flip(nm[midpt+1:]) )

        #inverse fourier transform back to real space
        return np.fft.irfft( nm_cmplx, norm='ortho' )

###############################################################

    def sample_vel( self, beta ):
    
        #generate new velocities from m-b distribution
    
        #standard-deviation, same for all beads
        #distribution defined as e^(-1/2*x^2/sigma^2)
        sigma = np.sqrt( 1.0 / ( beta * self.m ) )
    
        self.vvv = np.random.normal( 0.0, sigma, self.nbeads )

###############################################################

    def get_xcom( self ):

        return np.sum( self.xxx ) / self.nbeads

###############################################################

    def get_vcom( self ):

        return np.sum( self.vvv ) / self.nbeads

###############################################################

    def get_pe( self ):

        #calculate potential energy
        
        #contribution from internal modes
        engpe = 0.0
        for i in range(self.nbeads):
           if( i == 0 ):
               #periodic boundary conditions for the first bead
               engpe += 0.5 * self.m * self.omega_n**2 * ( self.xxx[i] - self.xxx[self.nbeads-1] )**2
           else:
               engpe += 0.5 * self.m * self.omega_n**2 * ( self.xxx[i] - self.xxx[i-1] )**2

        #contribution from external potential
        if( self.systype == 'harmonic' ):
    
            k   = 1.0
            engpe += 0.5*k * np.sum( self.xxx**2 )
    
        elif( self.systype == 'anharmonic' ):

            engpe += 0.5 * np.sum( self.xxx**2 ) + 0.1 * np.sum( self.xxx**3) + 0.01 * np.sum( self.xxx**4 )
    
        elif( self.systype == 'quartic' ):
    
            engpe += 0.25 * np.sum( self.xxx**4 )

        return engpe

###############################################################

    def get_ke( self ):

        #calculate kinetic energy
        return 0.5 * self.m * np.sum( self.vvv**2 )

###############################################################

    def calc_data( self, itraj, step, currtime, initxcom ):

        #output desired data

        fmt_str = '%20.8e'

        #calculate energy terms
        engpe = self.get_pe()
        engke = self.get_ke()
        etot  = engpe + engke

        #calculate com terms
        xcom = self.get_xcom()
        vcom = self.get_vcom()

        #Print output data
        output = np.zeros(7)
        output[0] = itraj
        output[1] = currtime
        output[2] = etot
        output[3] = engpe
        output[4] = engke
        output[5] = xcom
        output[6] = vcom
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Accumulate contributions to correlation function

        self.corrfcn[ round(step/self.Nprint), 0 ] = currtime #this doesn't actually change between trajectories
        self.corrfcn[ round(step/self.Nprint), 1 ] += initxcom*xcom

###############################################################

    def check_input( self ):

        #Subroutine to check that inputs are defined correctly

        if( self.systype != 'harmonic' and self.systype != 'anharmonic' and self.systype != 'quartic' ):
            print( 'ERROR: Incorrect option specified for systpe' )
            print( 'Possible options are: harmonic, anharmonic, or quartic' ) 
            exit()

        if( self.integ != 'vv' and self.integ != 'analyt' and self.integ != 'cayley' ):
            print( 'ERROR: Incorrect option specified for integ' )
            print( 'Possible options are: vv, analyt, cayley' )
            exit()


###############################################################

