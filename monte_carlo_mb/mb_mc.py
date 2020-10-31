import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class mb_mc():

    #class to perform monte-carlo simulation of maxwell-boltzmann distribution of momentum

    def __init__( self, m, beta, delp, Nstep, Nprint ):


        #Nstep  - total number of MC steps
        #Nprint - how often to save configurations
        #beta   - inverse temperature
        #m      - physical mass of particle
        #delp   - max displacement distance

        #work in atomic units

        self.m       = m

        self.beta    = beta

        self.delp    = delp
        self.Nstep   = Nstep
        self.Nprint  = Nprint   
        self.Ndata   = int(Nstep/Nprint)

        #initialize system
        self.p_array    = np.zeros(self.Ndata+1)
        self.p_array[0] = 0.0
        self.pi         = 0.0

###############################################################

    def kernel( self ):

        print()
        print('******* RUNNING MONTE-CARLO CALCULATION ********')
        print('Running MC for Maxwell-Boltzmann Distribution for ',self.Nstep,'steps')
        print('*****************************************')
        print()

        #set initial position and energy to old position and energy
        po   = self.pi
        ke_o = self.get_ke( po )

        #MC loop
        for step in range( self.Nstep ):

            #randomly displace particle momentum
            pn = po + ( np.random.rand() - 0.5 ) * self.delp

            #kinetic energy of new momenta
            ke_n = self.get_ke( pn )

            #check if new momenta is accepted using metropolis algorithm

            chk = np.exp( - self.beta * ( ke_n - ke_o ) )

            if( np.random.rand() < chk ):

                #new momenta is accepted
                #replace old position and energy with the new position and energy
                po   = pn
                ke_o = ke_n

            #update momentum array with current momenta
            #(note that this updates the momentum array with the old momenta again
            #if the new momenta is rejected )
            if( np.mod( step, self.Nprint ) == 0 ):
                self.p_array[ int(step/self.Nprint)+1 ] = po
                print( 'Currently at step ', step )

        print()
        print('Calculating final data')
        print()

        #Calculate final data
        self.calc_data()

###############################################################

    def get_ke( self, ppp ):

        #calculate potential energy
        return 0.5 * ppp**2 / self.m

###############################################################

    def calc_data( self ):

        #split data into chunks and calculate P(p) for each chunk
        Nchnk = 5
        Nbins = 100
        histo = np.zeros( [Nbins,Nchnk+1] )
        split_p_array = np.split( self.p_array, Nchnk )

        for i in range(Nchnk):

            if( i == 0 ):
                #calculate midpoint of each bin
                bin_edges = np.histogram( split_p_array[i], bins=Nbins, range=( self.p_array.min(), self.p_array.max() ), density=True )[1]
                for j in range( Nbins ):
                    histo[j,0] = ( bin_edges[j] + bin_edges[j+1] ) / 2.0

            #calculate normalized histogram for each chunk, note that all histograms have the same range
            histo[:,i+1]   = np.histogram( split_p_array[i], bins=Nbins, range=( self.p_array.min(), self.p_array.max() ), density=True )[0]

        #calculate average and error over all instances of P(p)
        avg_histo = np.zeros( [Nbins,3] )
        avg_histo[:,0] = np.copy( histo[:,0] )
        avg_histo[:,1] = np.mean( histo[:,1:], axis=1 )
        avg_histo[:,2] = np.std( histo[:,1:], axis=1 ) / np.sqrt(Nchnk-1)

        #print out P(p)
        utils.printarray(avg_histo,'prob_p.dat')

###############################################################



