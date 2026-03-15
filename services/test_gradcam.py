from services.gradcam import GradCAM, overlay_heatmap

gradcam = GradCAM()

image = r"C:\Users\Madhav\OneDrive\Desktop\TB_Hospital_AI\TB.504.jpg"

heatmap = gradcam.generate(image)

output = overlay_heatmap(image, heatmap)

print("GradCAM saved to:", output)