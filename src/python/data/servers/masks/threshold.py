import numpy as np

from data.pseudodata import ThresholdEvent

class ThresholdEventMask:

    def __init__(self,
        data_server,
        min_duration,
        threshold,
        g_or_l='g',
        eq=False):

        self.ds = data_server
        self.min_duration = min_duration
        self.threshold = threshold
        self.g_or_l = g_or_l,
        self.eq = eq

        self.checker = None

        if self.g_or_l == 'g':
            if self.eq:
                checker = lambda x: x >= self.threshold
            else:
                checker = lambda x: x > self.threshold
        else:
            if self.eq:
                checker = lambda x: x <= self.threshold
            else:
                checker = lambda x: x < self.threshold

        self.checker = lambda x: np.all(checker(x))
        self.current_duration = 0
        self.event_count = 0

    def get_data(self):

        data = self.ds.get_data()

        if self.checker(data):
            self.current_duration += 1
        else:
            self.current_duration = 0

        if self.current_duration >= self.min_duration:
            self.event_count += 1

            info = 'poop'
            data = ThresholdEvent(info, data)

        return data

    def refresh(self):

        self.ds.refresh()
        self.current_duration = 0
        self.event_count = 0

    def get_status(self):

        threshold_items = {
            'data_server': self.ds,
            'min_duration': self.min_duration,
            'current_duration': self.current_duration,
            'event_count': self.event_count,
            'threshold': self.threshold,
            'g_or_l': self.g_or_l,
            'eq': self.eq,
            'checker': self.checker}.items()
        ds_items = self.ds.get_status().items()

        return dict(threshold_items + ds_items)

    def cols(self):
        
        return self.ds.cols()

    def rows(self):

        return self.ds.rows()
