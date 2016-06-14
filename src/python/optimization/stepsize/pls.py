class ProbabilisticLineSearcher:

    def __init__(self, c1, c2):

        if c1 < 0 or c1 >= c2:
            raise ValueError(
                'Parameter c1 must be in interval [0, c2).')

        if c2 <= c1 or c2 > 1:
            raise ValueError(
                'Parameter c2 must b in interval (c1,1].')

        self.c1 = c1
        self.c2 = c2

    def get_step_size(self, f, f_prime):

        print "Stuff"

        return 0

    def wolfe_condition1(self, t, f, f_prime):

        slope = self.c1 * t
        lhs = f(t)
        rhs = f(0) + slope * f_prime(0)

        return lhs <= rhs

    def wolfe_condition2(self, t, f_prime):

        lhs = f_prime(t) 
        rhs = self.c2 * f_prime(0)

        return lhs >= rhs
