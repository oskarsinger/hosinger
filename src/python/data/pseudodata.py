class MissingData:

    def __init__(self, num_missing_rows):

        self.num_missing_rows = num_missing_rows

    def get_status(self):

        return {
            'num_missing_rows': self.num_missing_rows}

class ThresholdEvent:

    def __init__(self, info, data):

        self.data = data
        self.info = info

    def get_status(self):

        return {
            'info': self.info,
            'data': self.data}
