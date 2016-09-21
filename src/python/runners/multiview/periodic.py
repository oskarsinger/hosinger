from wavelets import dtcwt

class MultiviewDTCWTCCAAnalysisRunner:

    def __init__(self, 
        get_biort,
        get_qshift,
        nlevels,
        servers, 
        period, 
        max_iter):

        self.model = model
        self.servers = servers
        self.period = period
        self.max_iter = max_iter

        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        data = [ds.get_data() for ds in self.servers]
        min_length = min(
            [ds.rows() for ds in self.servers])
        Yls = []
        Yhs = []
        k = 0

        while (k + 1) * period < min_length:
            begin = k * period
            end = begin + period
            current_data = [view[begin:end,:] for view in data]

            for view in current_data:
                (Yl, Yh, Y_scale) = dtcwt.twod.dtwavexfm(
                    view, 
                    self.nlevels, 
                    self.get_biort, 
                    self.get_qshift)
                     
                Yls.append(Yl)
                Yhs.append(Yh)

            k += 1
