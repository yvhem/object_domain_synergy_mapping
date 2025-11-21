import socket
import argparse
import ast
import joblib
import numpy as np
import torch
from src import HandPoseTransformer, HandPoseFCNN

# configuration constants
HOST = '127.0.0.1'
PORT = 65432
BUFFER_SIZE_IN = 16     # 4 floats * 4 bytes
BUFFER_SIZE_OUT = 180   # 45 floats * 4 bytes

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hand synergy decoding server')
    parser.add_argument('--config', type=str, default='models/15/training_info.txt', help='Path to configuration file')
    return parser.parse_args()

def load_configuration(path):
    config = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip()
                    try: 
                        config[key] = ast.literal_eval(val.strip())
                    except (ValueError, SyntaxError):
                        config[key] = val.strip()
    except FileNotFoundError:
        print(f'Error: configuration file not found at {path}')
        exit(1)
    return config

def post_process_angles(raw_output, fixed_indices, target_dim=45):
    """
    Converts model output to joint angles.
    Handles conversion of sin/cos pairs back to degrees for fixed indices.
    """
    reconstructed = np.zeros(target_dim)
    read_idx = 0

    for i in range(target_dim):
        if i in fixed_indices:
            # fixed joints are represented as sin/cos pairs in the model output
            sin_val = raw_output[read_idx]
            cos_val = raw_output[read_idx + 1]
            reconstructed[i] = np.rad2deg(np.arctan2(sin_val, cos_val))
            read_idx += 2
        else:
            reconstructed[i] = raw_output[read_idx]
            read_idx += 1

    return reconstructed

def main():
    args = parse_arguments()
    config = load_configuration(args.config)

    # extract params
    pca_dim = config.get('PCA Components', 0)
    fixed_indices = config.get('Fixed Indices', [])
    model_arch = config.get('Model', 'FCNN')
    weights_path = config.get('Model Save Path', 'model.pth')
    scaler_path = config.get('Scaler Path', 'scaler.save')
    pca_path = config.get('PCA Path', None)

    # the network output dimension depends on whether fixed joints use sin/cos (2 vals) or angle (1 val)
    net_output_dim = config.get('Final Output Dimension', 45 + len(fixed_indices))
    
    print(f'Loading {model_arch} | Output dim: {net_output_dim} | PCA: {pca_dim > 0}')

    # init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_arch == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=net_output_dim)
    elif model_arch == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fixed_indices, pca_dim=pca_dim if pca_path else 0)
    else:
        raise ValueError(f'Unknown model architecture: {model_arch}')

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except:
        print(f'Error: model weights not found at {weights_path}')
        exit(1)

    model.to(device)
    model.eval()

    # load preprocessing objects
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path) if pca_path else None

    # network loop
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print(f'Server listening on {HOST}:{PORT}...')

        while True:
            print('Waiting for Unity client...')
            conn, addr = server.accept()
            print(f'Connected: {addr}')

            with conn:
                try:
                    while True:
                        data = conn.recv(BUFFER_SIZE_IN)
                        if not data:
                            break

                        # process input
                        input_array = np.frombuffer(data, dtype=np.float32)
                        input_tensor = torch.FloatTensor(input_array.reshape(1, -1)).to(device)

                        # inference
                        with torch.no_grad():
                            prediction = model(input_tensor).cpu().numpy()

                        # inverse transform (PCA -> scaler)
                        if pca:
                            prediction = pca.inverse_transform(prediction.reshape(1, -1))
                        prediction = scaler.inverse_transform(prediction).flatten()

                        # reconstruction (handle sin/cos constraints)
                        final_angles = post_process_angles(prediction, fixed_indices)

                        # send back to Unity
                        conn.sendall(final_angles.astype(np.float32).tobytes())

                except (ConnectionResetError, BrokenPipeError):
                    print('Connection reset by client')
                finally:
                    print('Client disconnected.')

if __name__ == '__main__':
    main()
