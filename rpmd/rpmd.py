import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class rpmd():

    #class to integrate rpmd equations of motion

    def __init__( self, beta, m, nbeads, delt, systype, Nstep, Nprint, Ntraj, Nequil, Ntemp ):


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

                #integrate using velocity verlet
                self.velocity_verlet()

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

            #integrate using velocity verlet
            self.velocity_verlet()

            #re-sample velocities every Ntemp steps (note that this resamples velocities again at t=0)
            if( np.mod( step, self.Ntemp ) == 0 ):
                self.sample_vel( self.beta_n )

###############################################################

    def velocity_verlet( self ):

        #velocity-verlet algorithm
        #note switched from conventional algorithm by integrating velocities first and then positions

        self.get_velocities()   
        self.get_positions()
        self.get_forces()
        self.get_velocities()

###############################################################

    def get_positions( self ): 

        #integrate positions by a full time-step
        self.xxx += self.vvv * self.delt

###############################################################

    def get_velocities( self ): 

        #integrate velocities by half a time-step
        self.vvv += 0.5 * self.fff * self.delt / self.m

###############################################################

    def get_forces( self ): 

        nbeads = self.nbeads
    
        #calculate forces due to internal modes
        self.fff = np.zeros(nbeads)
        for i in range(nbeads):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[nbeads-1] - self.xxx[i+1] )
            elif( i == nbeads-1 ):
                #periodic boundary conditions for the last bead
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[i-1] - self.xxx[0] )
            else:
                self.fff[i] = -self.m * self.omega_n**2 * ( 2.0*self.xxx[i] - self.xxx[i-1] - self.xxx[i+1] )
    
        #forces due to external potential
        if( self.systype == 'harmonic' ):
    
            k   = 1.0
            self.fff += -k * self.xxx
    
        elif( self.systype == 'anharmonic' ):
    
            self.fff += -self.xxx - 3.0/10.0 * self.xxx**2 - 1.0/25.0 * self.xxx**3
    
        elif( self.systype == 'quartic' ):
    
            self.fff += -self.xxx**3
    
        else:
            print( 'ERROR: Incorrect option specified for systpe' )
            print( 'Possible options are: harmonic, anharmonic, or quartic' ) 
            exit()

###############################################################

    def init( self ):
    
        #initial position
        self.xxx = np.zeros( self.nbeads )
    
        #initial velocity from Maxwell-Boltzmann distribution
        self.sample_vel( self.beta_n )

        #calculate initial forces
        self.get_forces()   

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



