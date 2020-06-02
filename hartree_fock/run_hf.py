import make_hams
import hartreefock as hf

def main():

    #STO-3G RHF calculation on H2
    #S, Hcore, g, Enuc = make_hams.make_ham_diatomic_sto3g('h2',1.4)
    #E_HF, evalsa, orbsa, evalsb, orbsb = hf.hf_calc('rhf',2,1,1,S,Hcore,g)

    #STO-3G RHF calculation on HeH+
    S, Hcore, g, Enuc = make_hams.make_ham_diatomic_sto3g('hehp',1.4632)
    E_HF, evalsa, orbsa, evalsb, orbsb = hf.hf_calc('rhf',2,1,1,S,Hcore,g)

    print( "HeH+: " )
    print( "Energy= ", E_HF )
    print( "Evalsa= " )
    print( evalsa )
    print( "Evalsb= " )
    print( evalsb )
    print( "Orbsa= " )
    print( orbsa )
    print( "Orbsb= " )
    print( orbsb )

if __name__ == '__main__':
    main()
