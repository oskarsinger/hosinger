import numpy as np

class Stream2Minibatch:

    def __init__(self, data_loader, batch_size):

        self.dl = data_loader
        self.bs = batch_size

        self.num_rounds = 0

    def get_data(self):

        print "Stuff" 
