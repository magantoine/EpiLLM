from transformers import AutoModel, AutoTokenizer



class EmbSimModel():

    def __init__(self, model_name) -> None:
        self.model = None
        self.tok = None
        self.model_name = model_name
        self.max_length = None

    def get_emb(self, sentence) -> float:
        if(self.model is None or self.tok is None):
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.max_length = self.model.model_max_length
        return self.model(
            **self.tok(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        ).last_hidden_state[:, 0, :]

