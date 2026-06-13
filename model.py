import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

# torchvision is a specialized library for computer vision
# datasets - prebuilt code to read images off hard drive
# transforms - to give instructions to chop change and resize images, greyscale, etc.
# Data Loader - handles shuffling and batching them and loading into memory.

class CEAMShapeNet(nn.Module):
    #basically like in Java class CEAMShapeNet extends nn.Module.
    def __init__(self, use_dropout=False):
        #self is equivalent to this in Java and this line is like a constructor for the class
        super(CEAMShapeNet, self).__init__()
        #constructor call to super, calling the nn.Module class from torch.nn

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        #2D CNN with 3 Input Channels (RGB), 32 is the low-level feature maps (edges, lines, etc.)
        #kernel_size means a 3x3 'pixel' scanner with padding=1 is sweeping across my image to 'read' it.
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        #Downsampling: Here we take the highest value in a grid of four pixels. Think of a 2x2 matrix and select the highest value.
        #That is what MaxPool2D is doing here, saving processing.

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #32x32 -> 64X64 (double the number of features channels).
        #You ended with 32 out_channels in the previous round, and you push out 64.
        #It will analyze more features like vertex combinations

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #This #rd layer maps out complete 3D surface geometry profiles.

        #Classifier (Dense Layers)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        #--->128 out channels, Input is 64*64. Half 32*32 (conv1). Half 16*16(conv2) Half 8*8(conv3)
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        #Means if I declare use_dropout = true, the Dropout is 0.5. That means 50% of the neurons are randomly severed on every step.
        #Thsi forces my model to distribute its training evenly instead of relying on dominant neurons.
        self.fc2 = nn.Linear(128, 3)
        #I have three shapes : cone, cube and sphere

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten #-1 mean keep batch size intact
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":

    #lists for data gen graphs
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    # --- TRAINING TRANSFORMS (With added chaos/augmentation) ---
    train_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop((64, 64)),  # Prevents aspect ratio warping!
        transforms.RandomHorizontalFlip(p=0.5), #There is 50% chance the image my model gets is flipped during training, doubling my effective dataset!
        transforms.RandomRotation(degrees=20),  # Teaches the model to expect skewed shapes
        transforms.Grayscale(num_output_channels=3), #Converts figure to greyscale. It was done to prevent any cheating from the model based on colour.
        transforms.ToTensor(), #Converts pixel data to Tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Caps to pixel value range between [-1.0, +1.0]
    ])

    # --- THE 'EXAM' (VALIDATION) TRANSFORMS (Pristine & Static) ---
    val_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop((64, 64)),  # Must match the training resolution exactly
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    DATA_DIR = "./Dataset"
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Feed 32 images to the network and get them shuffled.
    # If I don't shuffle, the network will see all cones first, then all cubes, then all 10 spheres.
    # It will quickly learn a cheat rule: the first few guesses are always cones.
    # Shuffling forces the network to actually learn the geometry rather than memorizing the order of the presentation.
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    #No need to shuffle here. This is the 'exam' for my network.

    print("--- DATASET DIAGNOSTIC REPORT ---\n")
    print(f" Total Training Images: {len(train_dataset)}")
    print(f" Total Validation Images: {len(val_dataset)}")
    print(f" PyTorch Label Mapping: {train_dataset.class_to_idx}")

    model = CEAMShapeNet(use_dropout=True) #switched on Dropout here!
    criterion = nn.CrossEntropyLoss()
    #LogSoftMax function applied here. Will punish the model for incorrect answers
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Learning Rate = 0.001 here

    print("Model Brain initialized in PyCharm.")
    print("Optimizer : Adam")

    training_epochs = 25 #Number of passes through my entire dataset
    print(f"Training for {training_epochs} Epochs...")

    for epoch in range(training_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad() #clears any old gradients from earlier use
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() #backpropagation triggered here
            optimizer.step() #adjusts layer parameters based on calculated gradients scaled by learing rate
            running_train_loss += loss.item() * images.size(0) #extracts exact number of items in the current batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss /len(train_dataset)
        epoch_train_acc = (correct_train /total_train ) * 100
        train_acc_history.append(epoch_train_acc)
        train_loss_history.append(epoch_train_loss)


        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Turn off gradient calculation to save memory and processing speed
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        val_acc_history.append(epoch_val_acc)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1:02d}/{training_epochs}] -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%")

    print("\n🏁 --- EXPERIMENT COMPLETED CLEANLY --- 🏁")

    # Save the model parameters to a persistent file
    torch.save(model.state_dict(), "ceam_shapenet_weights.pth")
    print("Model memory safely saved as a permanent checkpoint file!")

    #Graphs
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    epochs_range = list(range(1, training_epochs+1))

    sns.lineplot(x=epochs_range, y=train_loss_history, label='Training Loss', color='red', linewidth=2.5)
    sns.lineplot(x=epochs_range, y=val_loss_history, label='Validation Loss', color='blue', linewidth=2.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Loss Analysis : Training vs Validation")
    plt.xticks(epochs_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_analysis_plot.png")
    plt.show()



    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs_range, y=train_acc_history, label='Training Accuracy', color='red', linewidth=2.5)
    sns.lineplot(x=epochs_range, y=val_acc_history, label='Validation Accuracy', color='blue', linewidth=2.5)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.title("Accuracy Analysis : Training vs Validation")
    plt.xticks(epochs_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_analysis_plot.png")
    plt.show()





   
