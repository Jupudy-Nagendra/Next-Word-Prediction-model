import joblib

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, context: str):
        prediction = self.model.predict([context])[0]
        probability = self.model.predict_proba([context])[0]
        return {
            "next_word": prediction,
            "probabilities": probability.tolist()
        }

if __name__ == "__main__":
    predictor = ModelPredictor(r"model\next_word_svm.pkl")
    result = predictor.predict("my name")
    print(result)