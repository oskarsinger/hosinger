class Minibatch2Periodic:

    def __init__(self, 
        data_server, 
        period_length, 
        num_periods, 
        which_period):

        self.ds = data_server
        self.pl = period_length
        self.num_periods = num_periods

        if which_period < 1 or which_period > num_periods:
            raise ValueError(
                'Parameter which_period must be in range 1,...,num_periods.')

        self.wp = which_period

        # Full epoch length
        self.el = self.pl * self.num_periods

        # Head length
        self.hl = (self.wp - 1) * self.pl

        # Tail length
        self.tl = (self.num_periods - self.wp) * self.pl

        self.num_rounds = 0
        self.num_requests = 0

    def get_data(self):

        remainder = self.num_requests % self.el

        if remainder < self.hl:
            num_skips = self.hl - remainder

            for i in xrange(num_skips):
                self._update_data_server()
        elif remainder > self.hl + self.pl:
            num_overflow = remainder - (self.hl + self.pl)
            num_skips = self.tl - num_overflow

            for i in xrange(num_skips):
                self._update_data_server()

        self.num_rounds += 1

        return self._update_data_server()

    def _update_data_server(self):

        self.num_requests += 1

        return self.ds.get_data()

    def rows(self):

        return self.num_rounds

    def cols(self):

        return self.ds.cols()

    def refresh(self):

        self.ds.refresh()

        self.num_rounds = 0

    def get_status(self):

        periodic_items = {
            'data_server': self.ds,
            'period_length': self.pl,
            'num_periods': self.num_periods,
            'which_period': self.wp,
            'num_rounds': self.num_rounds}.items()
        ds_dict = self.ds.get_status()

        # ds's num_rounds field does not account for skipped data
        del ds_dict['num_rounds']

        return dict(periodic_items + ds_dict.items())
