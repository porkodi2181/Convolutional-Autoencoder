# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9), often used for image processing tasks. The goal of this experiment is image denoising using autoencoders, a neural network designed to learn efficient representations. By introducing noise to images, the model is trained to reconstruct clean versions.

## DESIGN STEPS

# STEP 1:
Load MNIST dataset and convert to tensors.

# STEP 2:
Apply Gaussian noise to images for training.

# STEP 3:
Design encoder-decoder architecture for reconstruction.

# STEP 4:
Use MSE loss to measure reconstruction quality.

# STEP 5:
Train autoencoder using Adam optimizer efficiently.

# STEP 6:
Evaluate model on noisy and clean images.

# STEP 7:
Visualize results comparing original, noisy, denoised versions.

# STEP 8:
Improve performance by tuning hyperparameters carefully.

## PROGRAM
### Name:PORKODI B
### Register Number:212224240114

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)

            # Add noise
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```
```

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=5):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:PORKODI B ")
    print("Register Number:212224240114 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

## OUTPUT

### Model Summary

<img width="974" height="571" alt="image" src="https://github.com/user-attachments/assets/4bcb2f06-11ad-4336-b595-774186d8a2a3" />


### Original vs Noisy Vs Reconstructed Image

<img width="1040" height="691" alt="image" src="https://github.com/user-attachments/assets/c8c38bf2-5b4f-41da-a2b8-49fe281dec64" />




## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
