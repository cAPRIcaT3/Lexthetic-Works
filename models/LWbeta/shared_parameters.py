import torch.nn as nn

class SharedParameters(nn.Module):
    def __init__(self, latent_dim, text_dim, shufflenet_feature_dim):
        super(SharedParameters, self).__init__()

        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, text_dim),
            nn.ReLU()
        )

        # ShuffleNetV2-based image processing branch
        self.shufflenet_branch = nn.Sequential(
            shufflenet_model,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.shufflenet_feature_dim = shufflenet_feature_dim

class Generator(nn.Module, SharedParameters):
    def __init__(self, latent_dim, text_dim, shufflenet_feature_dim, output_dim):
        super(Generator, self).__init__()
        SharedParameters.__init__(self, latent_dim, text_dim, shufflenet_feature_dim)

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.latent_dim + self.text_dim + self.shufflenet_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, text, image):
        text_embedding = self.text_branch(text)
        shufflenet_features = self.shufflenet_branch(image)
        
        # Concatenate text, image, and noise vectors
        combined_input = torch.cat([z, text_embedding, shufflenet_features], dim=1)
        return self.combined_branch(combined_input)

class Discriminator(nn.Module, SharedParameters):
    def __init__(self, text_dim, shufflenet_feature_dim):
        super(Discriminator, self).__init__()
        SharedParameters.__init__(self, latent_dim=None, text_dim=text_dim, shufflenet_feature_dim=shufflenet_feature_dim)

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.text_dim + self.shufflenet_feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text, image):
        text_embedding = self.text_branch(text)
        shufflenet_features = self.shufflenet_branch(image)
        
        # Concatenate text and image features
        combined_input = torch.cat([text_embedding, shufflenet_features], dim=1)
        return self.combined_branch(combined_input)
