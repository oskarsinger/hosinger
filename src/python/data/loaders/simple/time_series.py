class BatchPhysiologicalTimeSeriesLoader:

    def __init__(self,
        TS,
        subject,
        measurement_type,
        hertz,
        period,
        num_periods):

        self.TS = TS
        self.subject = subject
        self.measurement_type = measurement_type
        self.hertz = hertz
        self.period = period
        self.num_periods = num_periods

        self.class_name = 'SimpleBatchPhysiologicalTimeSeries'
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1

        return self.TS

    def finished(self):

        return self.num_rounds > 0

    def name(self):

        return self.measurement_type

    def rows(self):

        return self.TS.shape[0]

    def cols(self):

        return self.TS.shape[1]

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'subject': self.subject,
            'measurement_type': self.measurement_type,
            'hertz': self.hertz,
            'period': self.period,
            'num_periods': self.num_periods}

