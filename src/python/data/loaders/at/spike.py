import numpy as np

class AlTestSpikeLoader:

    def __init__(self, 
        data_path,
        online=False):

        self.data_path = data_path
        self.online = online
        sub_and_label = data_path.split('/')[-1][:-4]
        (self.subject, self.label) = sub_and_label.split('_')
        self.hertz = 1.0/60
        self.window = 1
        self.data = None
        self.num_rounds = 0

        self._init_data()

    def _init_data(self):

        with open(self.data_path) as f:
            line = f.readline().strip().split('\t')

            self.data = np.array(
                [float(num) for num in line])[:,np.newaxis]

    def get_data(self):

        data = None
        
        if self.online:
            data = self.data[self.num_rounds][:,np.newaxis]
        else:
            data = self.data

        self.num_rounds += 1

        return data

    def finished(self):

        len_data = None

        if self.data is None:
            len_data = float('inf')
        else:
            len_data = self.data.shape[0]

        return self.num_rounds == len_data
            
    def name(self):

        return 'AlTestSpikeLoader' + self.label

    def rows(self):

        rows = None

        if self.online:
            rows = self.num_rounds
        elif self.data is None:
            rows = 0
        else:
            rows = self.data.shape[0]

        return rows

    def cols(self):

        return 1

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'data_path': self.data_path,
            'subject': self.subject,
            'hertz': self.hertz,
            'window': self.window,
            'num_rounds': self.num_rounds,
            'data': self.data,
            'online': self.online}
