import torch
import pandas as pd
import numpy as np

from transformers import pipeline

class RobertaHuggingface:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    
    def __init__(self, model_url, label_mapping, *args, **kwargs):
        self.pipeline = pipeline("sentiment-analysis",
            model_url,
            tokenizer=model_url,
            max_length=512,
            truncation=True,
            return_all_scores=True,
            *args,
            **kwargs
        )
        self.label_mapping = label_mapping
    
    def predict(self, texts):
        texts = list(texts)
        results = self.pipeline(texts)
        
        results = [
            {
                self.label_mapping[label_dict["label"]] : label_dict["score"]
                for label_dict in list_of_dicts
            }
            for list_of_dicts in results
        ]
        return pd.DataFrame(results)
        
def create_twitter_model(*args, **kwargs):
    label_mapping = {
        "LABEL_0" : RobertaHuggingface.NEGATIVE,
        "LABEL_1" : RobertaHuggingface.NEUTRAL,
        "LABEL_2" : RobertaHuggingface.POSITIVE
    }
    model = RobertaHuggingface("cardiffnlp/twitter-roberta-base-sentiment", label_mapping, *args, **kwargs)
    return model

def create_large_english_model(*args, **kwargs):
    label_mapping = {
        "NEGATIVE" : RobertaHuggingface.NEGATIVE,
        "POSITIVE" : RobertaHuggingface.POSITIVE
    }
    model = RobertaHuggingface("siebert/sentiment-roberta-large-english", label_mapping, *args, **kwargs)
    return model

def create_financial_model(*args, **kwargs):
    label_mapping = {
        "negative" : RobertaHuggingface.NEGATIVE,
        "positive" : RobertaHuggingface.POSITIVE,
        "neutral" : RobertaHuggingface.NEUTRAL
    }
    model = RobertaHuggingface("soleimanian/financial-roberta-large-sentiment", label_mapping, *args, **kwargs)
    return model

# EXAMPLE USE
# model = create_financial_model()
# res = model.predict(["I am neutral", "I hate cheese", "kill all trumps", "I am positive"])