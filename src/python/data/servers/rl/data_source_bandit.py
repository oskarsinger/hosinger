class DataSourceBanditServer:

    def __init__(self, dl_list):

        self.dl_list = dl_list

        self.action_counts = [0] * len(self.dl_list)

    def get_outcome(self, action):

        self.acount_counts[action] += 1

        return dl_list[action].get_data()

    def get_status(self, action):

        return {
            'ds_list': self.ds_list,
            'action_counts': self.action_counts}
