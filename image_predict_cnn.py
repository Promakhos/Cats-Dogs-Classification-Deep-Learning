from PIL import Image
import torch
import torchvision.transforms as transforms
from model_Cnn import Cnn  



model = Cnn()
#Load weights from the .pth file
checkpoint = torch.load('CNN.pth')


model.load_state_dict(checkpoint['model_state_dict'])


model.eval()


image_path = ''  
image = Image.open(image_path)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Apply transformations
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

#predictions
with torch.no_grad():
    output = model(input_batch)

#softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

predicted_class = torch.argmax(probabilities).item()
print(f"Predicted Class: {predicted_class}")

predicted_probability = probabilities[predicted_class].item()
print(f"Predicted Probability: {predicted_probability}")

class_names = ["Cat", "Dog"]  


actual_class = class_names[predicted_class]
print(f"Class: {actual_class}")