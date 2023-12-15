class Config:
    # Dataset
    CSV_FILE = 'your_dataset.csv'
    IMAGE_SIZE = (64, 64)  # Adjust the image size based on your requirements

    # Model
    EMBEDDING_SIZE = 256  # Size of the text embedding
    NOISE_DIM = 100  # Size of the random noise vector for generator input

    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA1 = 0.5

    # Paths
    CHECKPOINT_DIR = 'checkpoints/'
    RESULT_DIR = 'results/'
    LOG_DIR = 'logs/'

    # Other options
    NUM_WORKERS = 2  # Number of parallel data loading processes
