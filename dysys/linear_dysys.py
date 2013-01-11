from dysys import DySys

class LinearDySys(DySys):
    
    def __init__(self, M, D, f):
        '''a DySys defined by mass and damping operators and
        a time-dependent forcing function, according to (something like)

            M * x' + D * x = f (t)

        though this class is still virtual since it depends on:

          . the implementation of the M & D operators (e.g. as sparse)

          . the discretization of the temporal derivative.

        '''

        self.M, self.D, self.f = M, D, f

