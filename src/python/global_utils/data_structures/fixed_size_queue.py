class FixedSizeQueue:

    def __init__(self, size):

        self.size = size
        self.queue = []

    def enqueue(self, item):
        
        if len(self.queue) == self.size:
            self.queue = self.queue[1:].appen(item)
        else:
            self.queue = self.queue.append(item)

    def dequeue(self):

        item = self.queue[0]

        self.queue = self.queue[1:]

        return item

    def length(self):

        return len(self.queue)

    def items(self):

        return self.queue
