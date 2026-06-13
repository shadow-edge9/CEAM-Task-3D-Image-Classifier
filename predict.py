import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import datasets


class CEAMShapeNet(nn.Module):
    def __init__(self, use_dropout=False):
        super(CEAMShapeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        #Classifier (Dense Layers)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict_custom_image(image_path, weights_path="ceam_shapenet_weights.pth"):
    # Precise pipeline transformation mirroring training normalization values
    predict_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop((64, 64)),  #Ensures the square geometry matches
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Validation and Prediction pipelines must act as an exact mirror to training env.
    #You must ensure visual parity.

    model = CEAMShapeNet(use_dropout=True) #Dropout is turned off by default, but I am explicitly switching it on here.
    if os.path.exists(weights_path):
        try:
            # strict=True to guarantee layer matches
            model.load_state_dict(torch.load(weights_path), strict=True)
            print(f"[INFO]: {weights_path} mapped and validated successfully.")
        except RuntimeError as e:
            print(
                f"\n[CRITICAL MISMATCH DETECTED]: Your saved weight file structure does not match this script's layers!")
            print(f"Error Details: {e}")
            return
    else:
        print(f"[ERROR]: {weights_path} not found on disk.")
        return

    model.eval() # ---> this command here makes my usage of Dropout null and void.

    # Open image and drop alpha channel context safely
    raw_image = Image.open(image_path).convert('RGB')
    #Because modern test .png files have a fourth channel called Alpha (RGBA) to handle transparency
    input_tensor = predict_transform(raw_image).unsqueeze(0) #converts 3D Tensor to 4D


    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #normalizes the scores from fc2 to a value between [0.0, 1.0]
        _, predicted_class_idx = torch.max(outputs, 1)

    # Clean label indexing array
    data_dir = "./Dataset/train"
    train_dataset = datasets.ImageFolder(root=data_dir)
    class_map = train_dataset.class_to_idx  # This is the source of truth
    idx_to_label = {v: k for k, v in class_map.items()}  # Invert the dictionary

    print(f"\n[VERIFIED MAPPING]: {idx_to_label}")
    predicted_label = idx_to_label[predicted_class_idx.item()]
    confidence = probabilities[0][predicted_class_idx.item()].item() * 100

    print(f"\n📁 File Checked: {image_path}")
    print(f"🎯 Inference Output: {predicted_label.upper()} ({confidence:.2f}% Confidence)")

    print("\n📊 Class Probabilities Breakdown:")
    for idx, name in idx_to_label.items():
        print(f"   -> {name}: {probabilities[0][idx].item() * 100:.1f}%")


if __name__ == "__main__":
    TARGET_IMAGE = ("Test/image.png") #Try out any image from the 'Test' directory. image.png is a placeholder here
    predict_custom_image(TARGET_IMAGE)