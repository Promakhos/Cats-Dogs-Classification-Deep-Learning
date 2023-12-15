from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models


#Load weights from the .pth file
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  



# Loading the checkpoint without the model's weights
checkpoint = torch.load('ResNet18.pth', map_location=torch.device('cpu'))
model_dict = model.state_dict()

# Loading the common weights
common_weights = {k: v for k, v in checkpoint.items() if 'fc' not in k}
model_dict.update(common_weights)
model.load_state_dict(model_dict, strict=False)

model.eval()

# load image
image_path = ''
image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

# predictions
with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)

predicted_class = torch.argmax(probabilities).item()
print(f"Predicted Class: {predicted_class}")

predicted_probability = probabilities[predicted_class].item()
print(f"Predicted Probability: {predicted_probability}")

class_names = ["Cat", "Dog"]  

actual_class = class_names[predicted_class]
print(f"Class: {actual_class}")
