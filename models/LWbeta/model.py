import torch
import torch.nn as nn
import torchvision.models as models

class SharedParameters(nn.Module):
    def __init__(self):
        super(SharedParameters, self).__init__()

        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Embedding(Config.vocab_size, Config.embedding_dim),
            nn.Linear(Config.embedding_dim, Config.text_dim),
            nn.ReLU()
        )

        # ShuffleNetV2-based image processing branch
        # Assuming shufflenet_model is ShuffleNetV2 with proper input channels
        self.shufflenet_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.shufflenet_branch = nn.Sequential(
            self.shufflenet_model,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.latent_dim = Config.latent_dim
        self.text_dim = Config.text_dim
        self.shufflenet_feature_dim = Config.shufflenet_feature_dim


class Generator(nn.Module):
    def __init__(self, shared_params):
        super(Generator, self).__init__()
        self.shared_params = shared_params

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.shared_params.latent_dim + self.shared_params.text_dim + self.shared_params.shufflenet_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, Config.output_dim),
            nn.Tanh()
        )

    def forward(self, z, text, image):
        text_embedding = self.shared_params.text_branch(text)
        shufflenet_features = self.shared_params.shufflenet_branch(image)
        
        # Concatenate text, image, and noise vectors
        combined_input = torch.cat([z, text_embedding, shufflenet_features], dim=1)
        
        # Pass the combined input through the generator's combined_branch
        generated_output = self.combined_branch(combined_input)
        
        # Include the text embedding in the final output
        output_with_text = torch.cat([generated_output, text_embedding], dim=1)
        return output_with_text


class Discriminator(nn.Module):
    def __init__(self, shared_params):
        super(Discriminator, self).__init__()
        self.shared_params = shared_params

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.shared_params.text_dim + self.shared_params.shufflenet_feature_dim + Config.output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text, image, generated_output_with_text):
        text_embedding = self.shared_params.text_branch(text)
        shufflenet_features = self.shared_params.shufflenet_branch(image)
        
        # Concatenate text, image, and generated output with text vectors
        combined_input = torch.cat([text_embedding, shufflenet_features, generated_output_with_text], dim=1)
        return self.combined_branch(combined_input)
