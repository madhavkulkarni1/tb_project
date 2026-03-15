from services.predict import TBPredictor

predictor = TBPredictor()

print("\nTesting normal image:")
print(predictor.predict(r"C:\Users\Madhav\OneDrive\Desktop\TB_Hospital_AI\others (126).jpg"))

print("\nTesting tb image:")
print(predictor.predict(r"C:\Users\Madhav\OneDrive\Desktop\TB_Hospital_AI\TB.504.jpg"))