import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image dimensions
IMAGE_SIZE = 128  # Updated image size
CHANNELS = 1  # Grayscale images
BATCH_SIZE = 350
EPOCHS = 500
NOISE_DIM = 100  # Dimension of the noise vector for generator

# Paths to your datasets
train_dir = '/home/groups/comp3710/OASIS/keras_png_slices_train'
test_dir = '/home/groups/comp3710/OASIS/keras_png_slices_test'
validation_dir = '/home/groups/comp3710/OASIS/keras_png_slices_validate'

# Custom dataset loader for a directory without class subfolders
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)  # Read image using torchvision's read_image
        if self.transform:
            image = self.transform(image)
        return image

# Load and preprocess images
def load_images_from_directory(directory):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    dataset = ImageDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

# Load datasets
train_loader = load_images_from_directory(train_dir)
test_loader = load_images_from_directory(test_dir)
validation_loader = load_images_from_directory(validation_dir)

# Generator model
# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),   # 128x128
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # Output: (64, 64, 64) for 128x128 images
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # Output: (128, 32, 32)
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Flatten(),  # Flatten the output from (batch_size, 128, 32, 32) to (batch_size, 131072)

            nn.Linear(128 * 32 * 32, 1)  # Linear layer expects input of size (batch_size, 131072)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
loss_fn = nn.BCEWithLogitsLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# Training step
def train_step(real_images):
    real_images = real_images.to(device)

    # Generate noise and fake images
    noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
    fake_images = generator(noise)

    # Train the discriminator
    real_labels = torch.ones(real_images.size(0), 1, device=device)
    fake_labels = torch.zeros(fake_images.size(0), 1, device=device)

    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images.detach())

    disc_loss_real = loss_fn(real_output, real_labels)
    disc_loss_fake = loss_fn(fake_output, fake_labels)
    disc_loss = disc_loss_real + disc_loss_fake

    discriminator_optimizer.zero_grad()
    disc_loss.backward()
    discriminator_optimizer.step()

    # Train the generator
    fake_output = discriminator(fake_images)
    gen_loss = loss_fn(fake_output, real_labels)

    generator_optimizer.zero_grad()
    gen_loss.backward()
    generator_optimizer.step()

    return gen_loss.item(), disc_loss.item()

# Lists to store losses for plotting
gen_losses = []
disc_losses = []

# Train the GAN
def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            gen_loss, disc_loss = train_step(batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
        print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}", end="\r")

        # Save generated images at certain intervals (e.g., every 10 epochs)
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, epoch + 1)

    # Plot the generator and discriminator losses
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_plot_gan5.png')
    plt.show()

# Generate and save images
def save_generated_images(generator, epoch, num_images=10):
    noise = torch.randn(num_images, NOISE_DIM, device=device)
    generated_images = generator(noise)
    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1] for saving
    for i in range(num_images):
        save_image(generated_images[i], f"generated_image_{i}_epoch_{epoch}_gan5.png")

# Start training with training data
train(train_loader, EPOCHS)

