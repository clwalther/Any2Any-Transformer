class Default_PT_EN_BatchConfig():
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, pt, en):
        pt = self.tokenizer.pt.tokenize(pt)
        pt = pt[:, :self.max_tokens]
        pt = pt.to_tensor()

        en = self.tokenizer.en.tokenize(en)
        en = en[:, :(self.max_tokens+1)]
        en_inputs = en[:, :-1].to_tensor()
        en_labels = en[:, 1:].to_tensor()

        return (pt, en_inputs), en_labels
