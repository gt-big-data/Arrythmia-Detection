import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Define the Fully Connected Neural Network (FCNN)
class FCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FCNNClassifier, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.5))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(0.5))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Function to extract frequencies
def extract_frequencies(filename, delta_time, method='dominant'):
    y, sr = librosa.load(filename, sr=None)
    frame_length = int(delta_time * sr)
    frame_length = 2 ** int(np.floor(np.log2(frame_length)))
    hop_length = frame_length // 4
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    amplitudes = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    if amplitudes.shape[0] != len(freqs):
        raise ValueError(f"Mismatch between frequency bins and amplitude frames.")
    
    frequency_amplitude_pairs = []
    for frame_amplitudes in amplitudes.T:
        if method == 'dominant':
            max_amp_index = np.argmax(frame_amplitudes)
            representative_freq = float(freqs[max_amp_index])
            representative_amp = float(frame_amplitudes[max_amp_index])        
        elif method == 'average':
            total_amp = np.sum(frame_amplitudes)
            if total_amp == 0:
                representative_freq = 0
                representative_amp = 0
            else:
                representative_freq = float(np.sum(freqs * frame_amplitudes) / total_amp)
                representative_amp = float(np.mean(frame_amplitudes))        
        elif method == 'median':
            sorted_indices = np.argsort(frame_amplitudes)
            median_index = sorted_indices[len(sorted_indices) // 2]
            representative_freq = float(freqs[median_index])
            representative_amp = float(np.median(frame_amplitudes))
        else:
            raise ValueError(f"Invalid method: {method}. Choose from 'dominant', 'average', or 'median'.")        
        
        frequency_amplitude_pairs.append((representative_freq, representative_amp))
    
    return frequency_amplitude_pairs

# Function to align and pad sequences
def align_and_pad_sequence(current_seq, max_len_seq, pad_mode='both'):
    if len(current_seq) > len(max_len_seq):
        current_seq = current_seq[:len(max_len_seq)]
    if len(current_seq) == len(max_len_seq):
        return current_seq
    
    curr_array = np.array(current_seq)
    max_array = np.array(max_len_seq)
    min_euclidean_distance = float('inf')
    best_start_idx = 0
    
    for i in range(len(max_len_seq) - len(current_seq) + 1):
        window = max_array[i:i+len(current_seq)]
        distance = np.linalg.norm(curr_array - window)
        if distance < min_euclidean_distance:
            min_euclidean_distance = distance
            best_start_idx = i
    
    total_pad = len(max_len_seq) - len(current_seq)
    if pad_mode == 'both':
        front_pad = best_start_idx
        back_pad = total_pad - front_pad
    elif pad_mode == 'front':
        front_pad = total_pad
        back_pad = 0
    else:  # pad_mode == 'back'
        front_pad = 0
        back_pad = total_pad
    
    padded_seq = ([[0, 0]] * front_pad + 
                  current_seq + 
                  [[0, 0]] * back_pad)
    
    return padded_seq

# Function to process and normalize frequencies
def process_and_normalize(freq, alignment_reference, freq_scaler, amp_scaler):
    aligned_freq = align_and_pad_sequence(freq, alignment_reference)
    aligned_freq = np.array([aligned_freq])
    normalized = np.zeros_like(aligned_freq)
    normalized[:, :, 0] = freq_scaler.transform(aligned_freq[:, :, 0])  # frequencies
    normalized[:, :, 1] = amp_scaler.transform(aligned_freq[:, :, 1])   # amplitudes
    flattened_data = normalized.reshape(normalized.shape[0], -1)
    return flattened_data

# Main function
def run_audio_model(filename_to_check, delta_time):
    encoding = {0 : "normal", 1 : "extrahls", 2 : "murmur", 4 : "extrasole"}
    try:
        # Load reference data and preprocessors
        with open("alignment_reference.pkl", "rb") as f:
            alignment_reference = pickle.load(f)
        with open("freq_scaler.pkl", "rb") as f:
            freq_scaler = pickle.load(f)
        with open("amp_scaler.pkl", "rb") as f:
            amp_scaler = pickle.load(f)
        
        # Dynamically calculate input size
        ext_freq = extract_frequencies(filename_to_check, delta_time)
        processed_data = process_and_normalize(ext_freq, alignment_reference, freq_scaler, amp_scaler)
        input_size = processed_data.shape[1]  # Flattened size
        
        # Define the model architecture
        hidden_sizes = [16, 8]
        num_classes = 5  # Ensure this matches the trained model
        model = FCNNClassifier(input_size, hidden_sizes, num_classes)
        
        # Check for GPU availability and move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move model to device (GPU or CPU)
        
        # Load the weights
        model.load_state_dict(torch.load("final_model_fcnn_classifier_16_8_7368.pth", weights_only=False, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        
        # Process input file
        freq = extract_frequencies(filename_to_check, delta_time)
        final_processed_data = process_and_normalize(freq, alignment_reference, freq_scaler, amp_scaler)
        
        # Convert to torch tensor for model input and move input tensor to the same device as the model
        input_tensor = torch.tensor(final_processed_data, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Get the predicted label
        predicted_label = prediction.argmax(dim=1).item()
        return encoding[predicted_label]
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

# Normal = 0
# Extrahls = 1
# Murmur = 2
# Extrastole = 4