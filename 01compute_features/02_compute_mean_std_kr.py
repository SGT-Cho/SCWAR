# -*- coding: utf-8 -*-
import os
import numpy as np

def compute_global_mean_std(feature_dir):
    """
    Compute the global mean and standard deviation across all feature files in a directory.

    Args:
        feature_dir (str): Path to the directory containing .bin feature files.
    
    Returns:
        global_mean (np.ndarray): Global mean vector.
        global_std (np.ndarray): Global standard deviation vector.
    """
    # Collect all feature files in the directory
    feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.bin')]

    if not feature_files:
        raise FileNotFoundError("No feature files found in the directory.")
    
    print(f"Found {len(feature_files)} feature files.")

    # Initialize lists to store data
    all_features = []

    for file_path in feature_files:
        # Load the binary data
        data = np.fromfile(file_path, dtype=np.float32)
        
        #######################
        # Assuming 13 features per frame (MFCC format)
        #######################
        feature_dim = 13
        
        try:
            # Reshape the data based on the assumed feature dimension
            data = data.reshape(-1, feature_dim)
        except ValueError:
            print(f"Error reshaping data in file: {file_path}. Skipping this file.")
            continue
        
        all_features.append(data)

    # Concatenate all features into a single array
    all_features = np.vstack(all_features)

    # Compute global mean and standard deviation
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)

    return global_mean, global_std

if __name__ == "__main__":
    #######################
    # Path to the directory containing feature .bin files
    # 변경된 부분: feature_dir 경로를 구조에 맞게 설정
    feature_dir = 'D:/Dataset/scwar/features/test'  
    #######################
    
    global_mean, global_std = compute_global_mean_std(feature_dir)

    print("Global Mean:", global_mean)
    print("Global Std:", global_std)

    #######################
    # Save the results for later use
    # 변경된 부분: 결과 파일 저장 경로를 맞게 설정
    np.save(os.path.join(feature_dir, 'global_mean.npy'), global_mean)
    np.save(os.path.join(feature_dir, 'global_std.npy'), global_std)
    #######################
    
    print(f"Global mean and std saved in: {feature_dir}")
