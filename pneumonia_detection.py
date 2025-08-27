# Install required libraries
!pip install pydicom  # For reading DICOM files
!pip install -Uqq ipdb  # For debugging with ipdb
!pip install torchmetrics  # For computing evaluation metrics
!pip install pytorch_lightning  # For building and training deep learning models

# Import required libraries
from pathlib import Path  
import pydicom  
import numpy as np  
import cv2  
import pandas as pd  
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
import os  
import glob  
import ipdb  
import torch  
import torchvision  
from torchvision import transforms 
import torchmetrics  
import pytorch_lightning as pl  
from pytorch_lightning.callbacks import ModelCheckpoint  
from pytorch_lightning.loggers import TensorBoardLogger  
import numpy as np  

# Enable interactive debugging
%pdb = on

# Define root directory for the dataset
root = "/data/kaggle-pneumonia/"

# Load and preprocess the labels CSV file, removing duplicate patient IDs
labels = pd.read_csv(root + "stage_2_train_labels.csv")
labels = labels.drop_duplicates("patientId")
labels.head(10)  # Display the first 10 rows of the labels

# Define paths for input DICOM files and processed output
root_path = Path(root + "/stage_2_train_images/")
save_path = Path(root + "proc/")

# Create a 3x3 subplot to visualize sample DICOM images with their labels
fig, ax = plt.subplots(3, 3, figsize=(9, 9))
counter = 0
for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[counter]
        dcm_path = root_path / patient_id
        dcm_path = dcm_path.with_suffix(".dcm")  # Append .dcm extension
        print(dcm_path)
        dcm = pydicom.read_file(dcm_path).pixel_array  # Read DICOM pixel data
        label = labels["Target"].iloc[counter]  # Get the corresponding label
        ax[i][j].imshow(dcm, cmap='bone')  # Display image in bone colormap
        ax[i][j].set_title(label)  # Set title as the label
        counter += 1

# Calculate the number of training and test images
lentrain = len(os.listdir(root + "/stage_2_train_images/"))
lentest = len(os.listdir(root + "/stage_2_test_images/"))

# Initialize variables for computing mean and standard deviation
sums = 0
sums_sqr = 0
x = round(lentrain * 0.9)  # Split point for train/validation (90% train)

# Process DICOM files, resize, and save as NumPy arrays
for counter, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = root_path / patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    if os.path.exists(dcm_path):
        dcm = pydicom.read_file(dcm_path).pixel_array / 255  # Normalize pixel values to [0, 1]
        dcm_resized = cv2.resize(dcm, (224, 224)).astype(np.float16)  # Resize to 224x224
        label = labels.Target.iloc[counter]  # Get the label
        train_val = "train" if counter < x else "val"  # Assign to train or validation set
        my_save_path = save_path / train_val / str(label)  # Create save path
        my_save_path.mkdir(parents=True, exist_ok=True)  # Create directories if needed
        np.save(my_save_path / patient_id, dcm_resized)  # Save resized image
        normalizer = dcm_resized.shape[0] * dcm_resized.shape[1]  # Pixel count for normalization
        if train_val == "train":
            sums += np.sum(dcm_resized) / normalizer  # Accumulate mean
            sums_sqr += (np.power(dcm_resized, 2).sum()) / normalizer  # Accumulate squared sum

# Calculate mean and standard deviation for training data
sums = 0
sums_sqr = 0
for files in glob.glob("/data/kaggle-pneumonia/processed/train/**/*.npy"):
    dcm = np.load(files)  # Load processed NumPy array
    normalizer = dcm.shape[0] * dcm.shape[1]  # Pixel count for normalization
    sums += np.sum(dcm) / normalizer  # Accumulate mean
    sums_sqr += (np.power(dcm, 2).sum()) / normalizer  # Accumulate squared sum

mean = sums / lentrain  # Compute mean
std = np.sqrt(sums_sqr / lentrain - (mean ** 2))  # Compute standard deviation
mean, std  # Print mean and standard deviation

# Define function to load NumPy files
def load_file(path):
    return np.load(path).astype(np.float32)

# Define data transformations for training dataset
train_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(0.44, 0.27),  # Normalize with precomputed mean and std
    transforms.RandomAffine(degrees=(-5, +5), translate=(0, 0.05), scale=(0.9, 1.1)),  # Random affine transformations
    transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))  # Random crop and resize
])

# Define data transformations for validation dataset
val_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.44], [0.27])  # Normalize with precomputed mean and std
])

# Create training and validation datasets
train_dataset = torchvision.datasets.DatasetFolder(
    root + "processed/train/", loader=load_file, extensions="npy", transform=train_transforms
)
val_dataset = torchvision.datasets.DatasetFolder(
    root + "processed/val/", loader=load_file, extensions="npy", transform=val_transforms
)

# Visualize random samples from the training dataset
fig, ax = plt.subplots(2, 2, figsize=(9, 9))
for i in range(2):
    for j in range(2):
        ind = np.random.randint(0, 20000)  # Random index
        x_ray, label = train_dataset[ind]  # Get image and label
        ax[i][j].imshow(x_ray[0], cmap="bone")  # Display image in bone colormap
        ax[i][j].set_title(label)  # Set title as the label

# Define batch size and number of workers for data loading
batch_size = 64
num_workers = 4

# Create data loaders for training and validation
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# Load a pre-trained ResNet18 model
model = torchvision.models.resnet18()
model

# Define a custom PyTorch Lightning module for pneumonia detection
class PnuModel(pl.LightningModule):
    def __init__(self, weight=1):
        super().__init__()
        self.model = torchvision.models.resnet18()  # Initialize ResNet18
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Modify input layer for grayscale
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)  # Modify output layer for binary classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)  # Adam optimizer
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))  # Binary cross-entropy loss with class weighting
        self.train_acc = torchmetrics.Accuracy()  # Accuracy metric for training
        self.val_acc = torchmetrics.Accuracy()  # Accuracy metric for validation

    def forward(self, data):
        pred = self.model(data)  # Forward pass
        return pred

    def training_step(self, batch, batch_idx):
        x_ray, label = batch  # Unpack batch
        label = label.float()  # Convert label to float
        pred = self(x_ray)[:, 0]  # Get predictions
        loss = self.loss_fn(pred, label)  # Compute loss
        self.log("train loss", loss)  # Log training loss
        self.log("step train acc", self.train_acc(torch.sigmoid(pred), label.int()))  # Log training accuracy
        return loss

    def training_epoch_end(self, outputs):
        self.log("train_acc", self.train_acc.compute())  # Log epoch-level training accuracy

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch  # Unpack batch
        label = label.float()  # Convert label to float
        pred = self(x_ray)[:, 0]  # Get predictions
        loss = self.loss_fn(pred, label)  # Compute loss
        self.log("val loss", loss)  # Log validation loss
        self.log("step val acc", self.val_acc(torch.sigmoid(pred), label.int()))  # Log validation accuracy
        return loss

    def validation_epoch_end(self, outputs):
        self.log("val_acc", self.val_acc.compute())  # Log epoch-level validation accuracy

    def configure_optimizers(self):
        return [self.optimizer]  # Return optimizer

# Instantiate the model
model = PnuModel()
model

# Define a checkpoint callback to save the best models based on validation accuracy
checkpoint_callback = ModelCheckpoint(
    monitor="Val Acc", save_top_k=10, mode="max"
)

# Initialize PyTorch Lightning trainer with GPU support and TensorBoard logging
gpus = 1
trainer = pl.Trainer(
    logger=TensorBoardLogger(save_dir=root + "logs"),
    gpus=gpus,
    log_every_n_steps=1,
    callbacks=checkpoint_callback,
    max_epochs=35
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Set device for inference (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the best checkpoint and set model to evaluation mode
model = PnuModel.load_from_checkpoint(
    "/data/kaggle-pneumonia/logs/lightning_logs/version_0/checkpoints/epoch=32-step=12408.ckpt"
)
model.eval()
model.to(device)

# Initialize lists to store predictions and labels
preds = []
labels = []

# Perform inference on the validation dataset
with torch.no_grad():
    for data, label in tqdm(val_dataset):
        data = data.to(device).float().unsqueeze(0)  # Move data to device and add batch dimension
        pred = torch.sigmoid(model(data)[0].cpu())  # Get prediction and apply sigmoid
        preds.append(pred)  # Store prediction
        labels.append(label)  # Store label

# Convert predictions and labels to tensors
preds = torch.tensor(preds)
labels = torch.tensor(labels).int()

# Compute evaluation metrics
acc = torchmetrics.Accuracy()(preds, labels)  # Accuracy
precision = torchmetrics.Precision()(preds, labels)  # Precision
recall = torchmetrics.Recall()(preds, labels)  # Recall
CM = torchmetrics.ConfusionMatrix(num_classes=2)(preds, labels)  # Confusion matrix with default threshold
CM_threshold = torchmetrics.ConfusionMatrix(num_classes=2, threshold=0.25)(preds, labels)  # Confusion matrix with custom threshold

# Print evaluation metrics
print(f"acc: {acc}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"CM: {CM}")
print(f"CM_threshold: {CM_threshold}")

# Visualize random validation predictions
fig, ax = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    for j in range(3):
        random_idx = np.random.randint(0, len(preds))  # Random index
        ax[i][j].imshow(val_dataset[random_idx][0][0], cmap="gray")  # Display image
        ax[i][j].set_title(f"pred: {int(preds[random_idx] > 0.5)}, label: {labels[random_idx]}")  # Show prediction and label
        ax[i][j].axis("off")  # Hide axes