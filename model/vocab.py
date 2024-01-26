

class Vocabulary:

    # class to load and store vocabulary

    def __init__(self, vocab_file_path):

        self.path = vocab_file_path
        self.load_vocab()

    
    def load_vocab(self):

        self.tok_to_id = dict()
        self.id_to_tok = []
        with open(self.path) as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.tok_to_id[token] = idx
                self.id_to_tok.append(token)

        self.n_tokens = len(self.tok_to_id)
        print("Vocabulary loaded. {} tokens".format(self.n_tokens))

