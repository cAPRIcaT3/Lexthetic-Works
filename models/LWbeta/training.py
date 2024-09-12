import torch
import torch.nn as nn
from config import Config

class SharedParameters(nn.Module):
    def __init__(self):
        super(SharedParameters, self).__init__()

        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Embedding(Config.vocab_size, Config.embedding_dim),
            nn.Linear(Config.embedding_dim, Config.text_dim),
            nn.ReLU()
        )

        self.latent_dim = Config.latent_dim
        self.text_dim = Config.text_dim

class Generator(nn.Module):
    def __init__(self, shared_params):
        super(Generator, self).__init__()
        self.shared_params = shared_params

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.shared_params.latent_dim + self.shared_params.text_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, Config.output_dim),
            nn.Tanh()
        )

    def forward(self, z, text):
        text_embedding = self.shared_params.text_branch(text)
        
        # Concatenate text and noise vectors
        combined_input = torch.cat([z, text_embedding], dim=1)
        
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
            nn.Linear(self.shared_params.text_dim + Config.output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text, generated_output_with_text):
        text_embedding = self.shared_params.text_branch(text)
        
        # Concatenate text and generated output with text vectors
        combined_input = torch.cat([text_embedding, generated_output_with_text], dim=1)
        return self.combined_branch(combined_input)
