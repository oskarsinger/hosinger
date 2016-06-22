class FixedLengthQueue:

    def __init__(self, length):

        self.length = length
        self.queue = []

    def enqueue(self, item):
        
        if len(self.queue) == self.length:
            self.queue = self.queue[1:] + [item]
        else:
            self.queue.append(item)

    def dequeue(self):

        item = self.queue[0]

        self.queue = self.queue[1:]

        return item

    def get_length(self):

        return len(self.queue)

    def get_items(self):

        return self.queue

    def get_max_length(self):

        return self.length

    def is_full(self):

        return len(self.queue) == self.length
