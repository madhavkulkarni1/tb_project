from services.predict import TBPredictor

predictor = TBPredictor()

image_path = r"C:\Users\Madhav\OneDrive\Desktop\TB_Hospital_AI\tb.504.jpg"

result = predictor.predict(image_path)

print(result)