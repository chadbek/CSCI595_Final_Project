import spacy

class TextProcessor:

    def __init__(self, init_token="<start>", eos_token="<end>"):
        self.nlp = spacy.load("en_core_web_sm")
        self.init_token = init_token
        self.eos_token = eos_token

        self.token_to_ix = {
            "<pad>": 0,
            "<unk>": 1,
            init_token: 2,
            eos_token: 3
        }

        self.ix_to_token = {v: k for k, v in self.token_to_ix.items()}

    def tokenize(self, text):
        return [tok.text.lower() for tok in self.nlp(text)]

    def build_vocab(self, texts):

        idx = len(self.token_to_ix)

        for text in texts:
            tokens = self.tokenize(text)

            for token in tokens:
                if token not in self.token_to_ix:
                    self.token_to_ix[token] = idx
                    self.ix_to_token[idx] = token
                    idx += 1

    def numericalize(self, text):

        tokens = [self.init_token] + self.tokenize(text) + [self.eos_token]

        return [
            self.token_to_ix.get(tok, self.token_to_ix["<unk>"])
            for tok in tokens
        ]