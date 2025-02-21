import joblib
from momentum_ai_trading.utils.config import MODEL_PATH

def main():
    try:
        features_path = f"{MODEL_PATH}_features.pkl"
        features = joblib.load(features_path)
        print("Saved Features:")
        for feature in features:
            print(feature)
    except Exception as e:
        print(f"Error loading saved_features.pkl: {e}")

if __name__ == "__main__":
    main()
