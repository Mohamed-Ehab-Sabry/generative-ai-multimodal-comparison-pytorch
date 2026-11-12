import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset, Audio
from collections import Counter
import re
import io
import wave


def _load_wav_without_torchcodec(source, format=None):
    """Bypasses PyTorch's C++ audio decoders to read WAV files natively in Python."""
    with wave.open(source, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    data = data.reshape(-1, channels).T
    return torch.from_numpy(data), 8000

torchaudio.load = _load_wav_without_torchcodec

def get_mnist_data() -> tuple[torch.Tensor, torch.Tensor]:
    """"
    Load and preprocesse the MINIST dataset for generative modeling.

    Download the MNIST dataset from Hugging Face, scale pixel values to [0,1], flatten the 28 * 28 images into 784-dimensional vectors,
    and binarize the values using a > 0.5 threshold.

    Returns:
    tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing:
        - x_train (torch.FloatTensor): Training data tensor of shape (N, 784)
            containing binarized float values.
        - x_test (torch.FloatTensor): Testing data tensor of shape (N, 784)
            containing binarized float values.

    """

    dataset = load_dataset("ylecun/mnist")

    dataset.set_format("torch", columns = ["image"])

    x_train = torch.tensor(np.array(dataset["train"]["image"])).float()
    x_test = torch.tensor(np.array(dataset["test"]["image"])).float()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.view(-1, 784)
    x_test = x_test.view(-1, 784)

    x_train = (x_train > 0.5).float()
    x_test = (x_test > 0.5).float()

    return x_train, x_test



def get_fashion_mnist_data() -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """"
    Load and preprocesse the Fashion MINIST dataset for generative modeling.

    Download the Fashion MNIST dataset from Hugging Face, scale pixel values to [0,1], flatten the 28 * 28 images into 784-dimensional vectors,
    and binarize the values using a > 0.5 threshold.

    Returns:
    tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing:
        - x_train (torch.FloatTensor): Training data tensor of shape (N, 784)
            containing binarized float values.
        - x_test (torch.FloatTensor): Testing data tensor of shape (N, 784)
            containing binarized float values.

    """



    dataset = load_dataset("zalando-datasets/fashion_mnist")

    dataset.set_format("torch", columns = ["image"])

    x_train = torch.tensor(np.array(dataset["train"]["image"])).float()
    x_test = torch.tensor(np.array(dataset["test"]["image"])).float()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.view(-1, 784)
    x_test = x_test.view(-1, 784)

    x_train = (x_train > 0.5).float()
    x_test = (x_test > 0.5).float()



    return x_train, x_test



def calculate_sparsity_metrics(dataset_tensor: torch.FloatTensor) -> tuple[float, float]:
    """Calculates both the density (fraction of 1s) and sparsity (fraction of 0s).
    
    Args:
        dataset_tensor (torch.FloatTensor): A binary PyTorch float tensor.
            
    Returns:
        tuple[float, float]: A tuple containing (density, sparsity).
    """
    
    density = dataset_tensor.mean().item()
    sparsity = 1.0 - density
    return round(density, 4), round(sparsity, 4)
     

def get_ag_news_data() -> tuple[torch.Tensor, torch.Tensor]:
    dataset_train = load_dataset("fancyzhx/ag_news", split="train[:20000]")
    dataset_test = load_dataset("fancyzhx/ag_news", split="test[:5000]")
    
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    train_texts = [tokenize(text) for text in dataset_train["text"]]
    test_texts = [tokenize(text) for text in dataset_test["text"]]

    word_counts = Counter(word for text in train_texts for word in text)
    top_784 = [word for word, _ in word_counts.most_common(784)]
    word_to_idx = {word: idx for idx, word in enumerate(top_784)}

    x_train = torch.zeros((len(train_texts), 784), dtype=torch.float32)
    x_test = torch.zeros((len(test_texts), 784), dtype=torch.float32)

    for i, text in enumerate(train_texts):
        for word in text:
            if word in word_to_idx:
                x_train[i, word_to_idx[word]] = 1.0

    for i, text in enumerate(test_texts):
        for word in text:
            if word in word_to_idx:
                x_test[i, word_to_idx[word]] = 1.0

    return x_train, x_test

def get_spoken_digit_data() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    mel_transform = T.MelSpectrogram(sample_rate=8000, n_mels=28)
    
    def process_audio(audio_list):
        tensors = []
        for audio_dict in audio_list:
            # 2. Extract the raw bytes
            audio_bytes = audio_dict["bytes"]
            
            # 3. Manually decode the bytes natively using torchaudio
            waveform, _ = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
            
            # torchaudio loads as (Channels, Time), so waveform is (1, Time)
            spectrogram = mel_transform(waveform)
            log_spectrogram = torch.log(spectrogram + 1e-9)
            
            # Because spectrogram is already 3D (1, 28, Time), we only unsqueeze once for Batch
            spec_4d = log_spectrogram.unsqueeze(0)
            spec_resized = F.interpolate(spec_4d, size=(28, 28), mode='bilinear', align_corners=False)
            spec_2d = spec_resized.squeeze()
            
            spec_min = spec_2d.min()
            spec_max = spec_2d.max()
            spec_scaled = (spec_2d - spec_min) / (spec_max - spec_min)
            
            spec_binarized = (spec_scaled > 0.5).float()
            tensors.append(spec_binarized.view(-1))
            
        return torch.stack(tensors)

    x_train = process_audio(dataset["train"]["audio"])
    x_test = process_audio(dataset["test"]["audio"])
    
    return x_train, x_test