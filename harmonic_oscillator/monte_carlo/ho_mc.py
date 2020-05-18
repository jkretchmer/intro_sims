import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class harm_osc_mc():

    #class to perform monte-carlo simulation of 1d harmonic oscillator

    def __init__( self, m, k, delx, Nstep, xi, temp=300 ):


        #delt   - time-step
        #Nstep  - total number of MD steps
        #temp   - temperature in kelvin
        #m      - physical mass of particle
        #xi     - initial position
        #delx   - max displacement distance

        #work in atomic units

        self.m       = m
        self.k       = k

        self.beta    = 1.0 / ( temp / 3.1577465e5 ) #inverse temp converting from K to a.u.

        self.delx    = delx
        self.Nstep   = Nstep
    
        #initialize system
        self.pos_array    = np.zeros(self.Nstep+1)
        self.pos_array[0] = xi
        self.xi           = xi

###############################################################

    def kernel( self ):

        print()
        print('******* RUNNING MONTE-CARLO CALCULATION ********')
        print('Running MC for 1D Harmonic Oscillator for ',self.Nstep,'steps')
        print('*****************************************')
        print()

        #set initial position and energy to old position and energy
        xo   = self.xi
        pe_o = self.get_pe( xo )

        #MC loop
        for step in range( self.Nstep ):

            #randomly displace particle position
            xn = xo + ( np.random.rand() - 0.5 ) * self.delx

            #potential energy of new position
            pe_n = self.get_pe( xn )

            #check if new position is accepted using metropolis algorithm

            chk = np.exp( - self.beta * ( pe_n - pe_o ) )

            if( np.random.rand() < chk ):

                #new position is accepted
                #replace old position and energy with the new position and energy
                xo   = xn
                pe_o = pe_n

            #update position array with current position
            #(note that this updates the position array with the old position again
            #if the new position is rejected )
            self.pos_array[ step+1 ] = xo

            if( np.mod( step, 100 ) == 0 ):
                print( 'Currently at step ', step )

        print()
        print('Calculating final data')
        print()

        #Calculate final data
        self.calc_data()

###############################################################

    def get_pe( self, xxx ):

        #calculate potential energy
        return 0.5 * self.k * xxx**2

###############################################################

    def calc_data( self ):

        #split data into chunks and calculate P(x) for each chunk
        Nchnk = 5
        Nbins = 100
        histo = np.zeros( [Nbins,Nchnk+1] )
        split_pos_array = np.split( self.pos_array, Nchnk )

        for i in range(Nchnk):

            if( i == 0 ):
                #calculate midpoint of each bin
                bin_edges = np.histogram( split_pos_array[i], bins=Nbins, range=( self.pos_array.min(), self.pos_array.max() ), density=True )[1]
                for j in range( Nbins ):
                    histo[j,0] = ( bin_edges[j] + bin_edges[j+1] ) / 2.0

            #calculate normalized histogram for each chunk, note that all histograms have the same range
            histo[:,i+1]   = np.histogram( split_pos_array[i], bins=Nbins, range=( self.pos_array.min(), self.pos_array.max() ), density=True )[0]

        #calculate average and error over all instances of P(X)
        avg_histo = np.zeros( [Nbins,3] )
        avg_histo[:,0] = np.copy( histo[:,0] )
        avg_histo[:,1] = np.mean( histo[:,1:], axis=1 )
        avg_histo[:,2] = np.std( histo[:,1:], axis=1 ) / np.sqrt(Nchnk-1)

        #print out P(x)
        utils.printarray(avg_histo,'prob_x.dat')

###############################################################



