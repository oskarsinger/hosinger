class SuffixTree():

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.tree = {}

        self._populate(self.vocabulary)

    def _populate(self, vocabulary):

        for word in vocabulary:
            self._insert(word)

    def _insert(self, word):

        self._recursive_insert(self.tree, word)
        
    def _recursive_insert(self, subtree, subword):
        letter = subword[0]
        is_terminal = len(subword) == 1

        if letter in subtree:
            if is_terminal:
                subtree[letter].is_terminal = True
            else:
                self._recursive_insert(
                    subtree[letter].children, 
                    subword[1:])
        else:
            subtree[letter] = Node(is_terminal)

            if not is_terminal:
                self._recursive_insert(
                    subtree[letter].children, 
                    subword[1:])

    def has_word(self, word):
        
        return self._recursive_has_word(self.tree, word)
    
    def _recursive_has_word(self, subtree, subword):
        letter = subword[0]

        if letter in subtree:
            if len(subword) == 1:
                if subtree[letter].is_terminal:
                    return True
                else:
                    return False
            else:
                return self._recursive_has_word(
                    subtree[letter].children,
                    subword[1:])
        else:
            return False

class Node():

    def __init__(self, is_terminal):
        self.is_terminal = is_terminal
        self.children = {}
