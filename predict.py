import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from pytorch_model_resnet import ResNet9, get_default_device, to_device
import pickle
import torchvision.transforms as transforms

# Dictionary to store benefits and properties of different plants
plant_info = {
    "Insulin": {
        "benefits": [
            "Helps regulate blood sugar levels",
            "Improves insulin sensitivity",
            "Supports diabetes management"
        ],
        "properties": [
            "Anti-diabetic",
            "Anti-inflammatory",
            "Antioxidant"
        ]
    },
    "Tulasi": {
        "benefits": [
            "Relieves stress",
            "Boosts immunity",
            "Improves respiratory health"
        ],
        "properties": [
            "Adaptogen",
            "Antioxidant",
            "Anti-inflammatory"
        ]
    },
    "Raktachandini": {
        "benefits": [
            "Promotes relaxation and stress relief",
            "Improves sleep quality",
            "Soothes skin irritation"
        ],
        "properties": [
            "Sedative",
            "Antiseptic",
            "Anti-inflammatory"
        ]
    }
}

def predict_image(img_path, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    img = Image.open(img_path).convert('RGB')

    # Apply transformations to the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust the size as needed
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    xb = to_device(img_tensor.unsqueeze(0), torch.device("cpu"))
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return train_classes[preds[0].item()]

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((700, 600))
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img
        
        predicted_label = predict_image(file_path, loaded_model)
        label_prediction.config(text=f'Predicted plant: {predicted_label}')
        
        # Display benefits and properties if available
        if predicted_label in plant_info:
            benefits = "\n".join(plant_info[predicted_label]["benefits"])
            properties = "\n".join(plant_info[predicted_label]["properties"])
            
            # Update labels for benefits and properties
            label_benefits.config(text=f"Benefits:\n{benefits}")
            label_properties.config(text=f"Properties:\n{properties}")
        else:
            label_benefits.config(text="Information not available for this plant.")
            label_properties.config(text="")

# Load the trained model
with open('C:/Users/PRANAY/Downloads/medicinal/medicinal/train_classes.pkl', 'rb') as file:
    train_classes = pickle.load(file)

loaded_model = ResNet9(in_channels=3, num_diseases=3)
loaded_model.load_state_dict(torch.load("C:/Users/PRANAY/Downloads/medicinal/medicinal/medicine-plants-model.pth", map_location=torch.device("cpu")))
loaded_model.eval()
device = get_default_device()

# Create tkinter GUI
root = tk.Tk()
root.title("Plant Classifier")

# Create Open Image button
button_open_image = tk.Button(root, text="Open Image", command=open_image)
button_open_image.pack(pady=10)

# Create label for displaying the image
label_image = tk.Label(root)
label_image.pack()

# Create label for displaying the predicted plant
label_prediction = tk.Label(root, text="")
label_prediction.pack()

# Create labels for benefits and properties
label_benefits = tk.Label(root, text="")
label_benefits.pack()

label_properties = tk.Label(root, text="")
label_properties.pack()

root.mainloop()
