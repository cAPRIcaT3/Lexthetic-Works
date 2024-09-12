#Configuration file to make shared parameters work

class Config:
    # Model parameters
    latent_dim = 100
    text_dim = 256
    shufflenet_feature_dim = 1024
    output_dim = 3  # Output dimension for the generator

    # Text processing parameters
    vocab_size = 10000
    embedding_dim = 128

    # Training parameters (you can add more as needed)
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    batch_size = 64
