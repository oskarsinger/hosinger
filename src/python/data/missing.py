class MissingData:

    def __init__(self, num_missing_rows):

        self.num_missing_rows = num_missing_rows

    def get_status(self):

        return {
            'num_missing_rows': self.num_missing_rows}
