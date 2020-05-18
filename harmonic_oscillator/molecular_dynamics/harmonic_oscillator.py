import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class harm_osc():

    #class to integrate 1d harmonic oscillator

    def __init__( self, m, k, delt, Nstep, Nprint, xi, vi=0.0, temp=300,  resample=False, Ntemp=100 ):


        #delt   - time-step
        #Nstep  - total number of MD steps
        #Nprint - how often output data is calculated/printed
        #Ntemp  - number of steps before resampling velocities during equilibration
        #temp   - temperature in kelvin
        #resample - boolean to state whether to resample velocities or not
        #m      - physical mass of particle
        #xi     - initial position
        #vi     - initial velocity

        #work in atomic units

        self.m       = m
        self.k       = k

        self.resample = resample
        self.beta    = 1.0 / ( temp / 3.1577465e5 ) #inverse temp converting from K to a.u.

        self.delt    = delt
        self.Nstep   = Nstep
        self.Nprint  = Nprint
        self.Ntemp   = Ntemp
    
        #initialize system
        self.xxx = xi
        self.vvv = vi
        self.init()

        #Define output files
        self.file_output = open( 'output.dat', 'w' )

###############################################################

    def kernel( self ):

        print('******* RUNNING MD CALCULATION ********')
        print('Running MD for 1D Harmonic Oscillator for ',self.Nstep,'steps')
        print('*****************************************')
        print()

        #MD loop for a given trajectory
        for step in range( self.Nstep ):

            currtime = step*self.delt

            #Calculate and print data of interest
            if( np.mod( step, self.Nprint ) == 0 ):
                print( 'Writing data at step ', step, 'and time', currtime )
                self.calc_data( step, currtime )

            #integrate using velocity verlet
            self.velocity_verlet()

        #Calculate and print data of interest at last time-step
        step += 1
        currtime = step*self.delt
        print( 'Writing data at step ', step, 'and time', currtime )
        self.calc_data( step, currtime )

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

        self.fff = -self.k * self.xxx

###############################################################

    def init( self ):
    
        #initial velocity from Maxwell-Boltzmann distribution
        if( self.resample ):
            self.sample_vel( self.beta )

        #calculate initial forces
        self.get_forces()   

###############################################################

    def sample_vel( self, beta ):
    
        #generate new velocities from m-b distribution
    
        #standard-deviation
        #distribution defined as e^(-1/2*x^2/sigma^2)
        sigma = np.sqrt( 1.0 / ( beta * self.m ) )
    
        self.vvv = np.random.normal( 0.0, sigma )

###############################################################

    def get_pe( self ):

        #calculate potential energy
        return 0.5 * self.k * self.xxx**2

###############################################################

    def get_ke( self ):

        #calculate kinetic energy
        return 0.5 * self.m * self.vvv**2

###############################################################

    def calc_data( self, step, currtime ):

        #output desired data

        fmt_str = '%20.8e'

        #calculate energy terms
        engpe = self.get_pe()
        engke = self.get_ke()
        etot  = engpe + engke

        #Print output data
        output = np.zeros(6)
        output[0] = currtime
        output[1] = etot
        output[2] = engpe
        output[3] = engke
        output[4] = self.xxx
        output[5] = self.vvv
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

###############################################################



