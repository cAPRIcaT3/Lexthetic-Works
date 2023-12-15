from torchvision import models

# Load a pre-trained ResNet18 model
pretrained_resnet18 = models.resnet18(pretrained=True)
pretrained_resnet18 = nn.Sequential(*list(pretrained_resnet18.children())[:-1])

# Now, load the weights of your trained model
model = AlphabetCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("/content/alphabet_model.pth"))

# Create a new model combining the pre-trained ResNet18 and your trained model
combined_model = nn.Sequential(
    pretrained_resnet18,
    model
).to(device)
