from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="savasy/bert-base-turkish-sentiment"
        )
    
    def analyze(self, text: str) -> dict:
        result = self.analyzer(text)[0]
        return {
            "sentiment": result["label"],
            "confidence": result["score"]
        } 