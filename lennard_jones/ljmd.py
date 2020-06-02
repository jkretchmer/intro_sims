import numpy as np
import gr
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class ljmd():

    #class to integrate lennard-jones fluid in 3d
    #calculates radial distribution function and diffusion coefficient
    #work in reduced units with epsilon, sigma, and mass of the particles all set to 1

    def __init__( self, N, delt, Nstep, tequil, Nprint, temp, density, rcut, resample=False, Ntemp=100 ):

        #N      - number of particles in system
        #delt   - time-step
        #Nstep  - total number of MD steps
        #tequil - time for equilibration after which start sampling
        #Nprint - how often output data is calculated/printed
        #Ndim   - number of dimensions (set to 3)
        #temp   - temperature in reduced units
        #density - density in reduced units
        #rcut   - cut- off distance for LJ interaction
        #resample - boolean to state whether to resample velocities or not
        #Ntemp  - number of steps before resampling velocities during equilibration

        self.resample = resample
        self.density = density
        self.temp    = temp
        self.beta    = 1.0 / ( temp ) #inverse temp

        self.delt    = delt
        self.Nstep   = Nstep
        self.tequil  = tequil
        self.Nprint  = Nprint
        self.Ntemp   = Ntemp
  
        self.N       = N 
        self.Ndim    = 3
        self.rcut2   = rcut**2

        #calculate box length from
        self.boxL = ( N/density ) ** (1/self.Ndim)

        #initialize system
        self.init()

        #Define output files
        self.file_output = open( 'output.dat', 'w' )


###############################################################

    def kernel( self ):

        print('******* RUNNING LJ CALCULATION ********')
        print('Running MD for Lennard-Jones Fluid for ',self.Nstep,'steps')
        print('*****************************************')
        print()

        #MD loop
        for step in range( self.Nstep ):

            currtime = step*self.delt

            #resample velocities
            if( self.resample ):
                if( np.mod( step, self.Ntemp ) == 0 ):
                    self.sample_vel( self.beta ) 

            #Calculate and print data of interest
            if( np.mod( step, self.Nprint ) == 0 and currtime >= self.tequil ):
                print( 'Writing data at step ', step, 'and time', currtime )
                self.calc_data( step, currtime )

            #integrate using velocity verlet
            self.velocity_verlet()


        #Calculate and print data of interest at last time-step
        step += 1
        currtime = step*self.delt
        print( 'Writing data at step ', step, 'and time', currtime )
        self.calc_data( step, currtime )

        #finish calculation of rdf
        self.grcalc.calc_gr( self.N, self.density )

        #Close output file
        self.file_output.close()

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
        self.vvv += 0.5 * self.fff * self.delt

###############################################################

    def get_forces( self ): 

        #subroutine to calculate LJ forces

        self.fff = np.zeros( [ self.N, self.Ndim ] )
        for i in range( self.N-1 ):
            for j in range( i+1, self.N ):

                #distance between particle i and j in each dimension
                xij = self.xxx[i] - self.xxx[j]

                #periodic boundary conditions in each dimension
                xij = xij - self.boxL * np.rint( xij/self.boxL )

                #absolute distance squared between particles
                r2 = np.sum( xij**2 )

                #add contribution to force on particle if below cut-off distance
                if( r2 < self.rcut2 ):
                    r2i      = 1.0/r2
                    r6i      = r2i**3
                    ff       = 48.0 * r2i* r6i * ( r6i - 0.5 )

                    self.fff[i,:] += ff*xij
                    self.fff[j,:] -= ff*xij

###############################################################

    def init( self ):

        ### initialize positions evenly on linear/square/cubic lattice ###

        self.xxx = np.zeros( [ self.N, self.Ndim ] )

        N1d = self.N**(1/self.Ndim) #number of particles along one dimension
        N1d = int(round(N1d))

        delL = self.boxL / N1d #spacing in one dimension

        cnt = 0
        if( self.Ndim == 2 ):
            for i in range(N1d):
                for j in range(N1d):
                    self.xxx[ cnt, 0 ] = i*delL
                    self.xxx[ cnt, 1 ] = j*delL
                    cnt += 1
        else:
            for i in range(N1d):
                for j in range(N1d):
                    for k in range(N1d):
                        self.xxx[ cnt, 0 ] = i*delL
                        self.xxx[ cnt, 1 ] = j*delL
                        self.xxx[ cnt, 2 ] = k*delL
                        cnt += 1
        ##########

        #initial velocity from Maxwell-Boltzmann distribution
        self.sample_vel( self.beta )

        #calculate initial forces
        self.get_forces()   

        #calculate energy at cut-off distance
        self.ecut = 4.0 * ( 1.0/self.rcut2**6 - 1.0/self.rcut2**3 )

        #initialize radial distribution function calculation
        self.grcalc = gr.gr( self.boxL )

###############################################################

    def sample_vel( self, beta ):
    
        #generate new velocities from m-b distribution
    
        #standard-deviation
        #distribution defined as e^(-1/2*x^2/sigma^2)
        sigma = np.sqrt( 1.0 / beta )
    
        self.vvv = np.random.normal( 0.0, sigma, self.N*self.Ndim ).reshape( self.N, self.Ndim )

        #remove center of mass motion
        vcom      = np.sum( self.vvv, axis=0 ) / self.N
        self.vvv -= vcom

        #re-scale velocities to proper temperature
        scale = np.sqrt( self.temp / ( np.sum( self.vvv**2 ) / ( self.Ndim*self.N ) ) )
        self.vvv *= scale

################################################################

    def get_pe( self ):

        #calculate potential energy

        engpe = 0.0
        for i in range( self.N-1 ):
            for j in range( i+1, self.N ):

                #distance between particle i and j in each dimension
                xij = self.xxx[i] - self.xxx[j]

                #periodic boundary conditions in each dimension
                xij = xij - self.boxL * np.rint( xij/self.boxL )

                #absolute distance squared between particles
                r2 = np.sum( xij**2 )

                #add contribution to energy if below cut-off distance
                if( r2 < self.rcut2 ):
                    r6i    = 1.0/r2**3
                    engpe += 4.0 * r6i * ( r6i - 1.0 ) - self.ecut

        return engpe

################################################################

    def get_ke( self ):
    
        #calculate kinetic energy
        return 0.5 * np.sum( self.vvv**2 )

###############################################################

    def calc_data( self, step, currtime ):

        #update rdf calculation
        self.grcalc.sample( self.N, self.boxL, self.xxx )

        #output desired data

        fmt_str = '%20.8e'

        ##calculate energy terms
        engpe = self.get_pe()
        engke = self.get_ke()
        etot  = engpe + engke

        ##Print output data
        output = np.zeros(4)
        output[0] = currtime
        output[1] = etot
        output[2] = engpe
        output[3] = engke
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

###############################################################

