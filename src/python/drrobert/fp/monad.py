# Think about whether just using a list comprehension might be more useful here.

class SingleItemMonad:

    def __init__(self, thing): 

        self.thing = thing

    def get_status(self):

        # Trolololol a dict of a dict
        return {
            'thing': self.thing}
