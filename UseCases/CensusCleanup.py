import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from sitn_prov import SITNProv

def main():
    input_path = './Datasets/census.csv'
    filename_ext = os.path.basename(input_path)
    filename, ext = os.path.splitext(filename_ext)
    output_path = 'processed_results'
    
    # Specify where to save the processed files
    savepath = os.path.join(output_path, filename)
    os.makedirs(savepath, exist_ok=True)

    df = pd.read_csv(input_path)

    # Assign names to columns
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
             'native-country', 'label']
    df.columns = names
    
    print("Initial dataset preview:")
    print(df.head())
    
    print('[' + time.strftime("%d/%m-%H:%M:%S") + '] Processing started')
    
    # Initialize provenance tracking
    replace_with_prov = SITNProv(pd.DataFrame.replace)
    
    # Store provenance tensors
    tensors = []
    start_time = time.time()
    tensor_sizes = []
    
    # OPERATION 0: Cleanup names from spaces
    original_df = df.copy()
    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'label']
    for c in col:
        df[c] = df[c].map(str.strip)
    # Since map preserves row correspondence
    tensor0 = np.eye(len(df), dtype=int)
    tensors.append(tensor0)
    tensor_sizes.append(tensor0.nbytes)

    # OPERATION 1: Replace ? character with NaN
    df, tensor1 = replace_with_prov(df, to_replace='?', value=np.nan)
    tensors.append(tensor1)
    tensor_sizes.append(tensor1.nbytes)
    
    # OPERATION 2-3: One-hot encode categorical variables
    original_df = df.copy()
    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    df = pd.get_dummies(df, columns=col)
    # One-hot encoding preserves row correspondence
    tensor2 = np.eye(len(df), dtype=int)
    tensors.append(tensor2)
    tensor_sizes.append(tensor2.nbytes)
    
    # OPERATION 4: Assign sex and label binary values 0 and 1
    # First replace sex values
    df, tensor4a = replace_with_prov(df, to_replace={'sex': 'Male'}, value={'sex': 1})
    tensors.append(tensor4a)
    tensor_sizes.append(tensor4a.nbytes)
    
    # Then replace Female values
    df, tensor4b = replace_with_prov(df, to_replace={'sex': 'Female'}, value={'sex': 0})
    tensors.append(tensor4b)
    tensor_sizes.append(tensor4b.nbytes)
    
    # Then replace label values
    df, tensor4c = replace_with_prov(df, to_replace={'label': '<=50K'}, value={'label': 0})
    tensors.append(tensor4c)
    tensor_sizes.append(tensor4c.nbytes)
    
    df, tensor4d = replace_with_prov(df, to_replace={'label': '>50K'}, value={'label': 1})
    tensors.append(tensor4d)
    tensor_sizes.append(tensor4d.nbytes)
    
    # OPERATION 5: Drop fnlwgt variable
    original_df = df.copy()
    df = df.drop(['fnlwgt'], axis=1)
    # Dropping columns preserves row correspondence
    tensor5 = np.eye(len(df), dtype=int)
    tensors.append(tensor5)
    tensor_sizes.append(tensor5.nbytes)
    
    print("Processed dataset preview:")
    print(df.head())
    
    # Calculate total provenance capture time
    capture_time = time.time() - start_time
    print(f"\nProvenance capture time: {capture_time:.4f} seconds")
    
    # Calculate total tensor storage
    total_storage = sum(tensor_sizes)
    print(f"Total tensor storage: {total_storage / (1024*1024):.2f} MB")
    
    # Save processed dataset
    df.to_csv(os.path.join(savepath, 'census_processed.csv'), index=False)
    
    # Perform provenance queries
    prov_tracer = SITNProv(lambda x: x)  # Dummy function for tracing
    
    # Example 1: Trace a specific output record back to the original input
    query_start = time.time()
    output_idx = 10  # Example output record
    original_indices = prov_tracer.trace_through_pipeline(tensors, output_idx, direction='backward')
    query_time = time.time() - query_start
    
    print(f"\nProvenance Query Example:")
    print(f"Output record at index {output_idx} (selected columns):")
    print(df.iloc[output_idx][['age', 'sex', 'label']])
    print(f"\nCame from original record at index {original_indices} (selected columns):")
    original_df = pd.read_csv(input_path)
    original_df.columns = names
    print(original_df.iloc[original_indices][['age', 'workclass', 'sex', 'label']])
    print(f"Query execution time: {query_time:.4f} seconds")
    
    # Example 2: Find all output records derived from a specific input record
    query_start = time.time()
    input_idx = 50  # Example input record
    output_indices = prov_tracer.trace_through_pipeline(tensors, input_idx, direction='forward')
    query_time = time.time() - query_start
    
    if output_indices:
        print(f"\nInput record at index {input_idx} contributed to {len(output_indices)} output records")
        print(f"First such output record (selected columns):")
        print(df.iloc[output_indices[0]][['age', 'sex', 'label']])
    else:
        print(f"\nInput record at index {input_idx} did not contribute to any output records")
    print(f"Query execution time: {query_time:.4f} seconds")
    
    # Visualize tensor sizes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(tensor_sizes)), [size/(1024*1024) for size in tensor_sizes])
    plt.xlabel('Operation')
    plt.ylabel('Tensor Size (MB)')
    plt.title('Provenance Tensor Storage Requirements')
    plt.xticks(range(len(tensor_sizes)), [f'Op {i}' for i in range(len(tensor_sizes))])
    plt.savefig('census_tensor_sizes.png')
    
    print('[' + time.strftime("%d/%m-%H:%M:%S") + '] Processing completed')

if __name__ == '__main__':
    main()
