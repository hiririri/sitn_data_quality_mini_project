import sys
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from sitn_prov import SITNProv

def main(opt):
    input_path = './Datasets/compas.csv'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path, header=0)

    # Show initial dataset preview
    print("\nInitial dataset preview:")
    print(df.head())
    
    # Initialize provenance tracking
    select_with_prov = SITNProv(pd.DataFrame.__getitem__)
    dropna_with_prov = SITNProv(pd.DataFrame.dropna)
    
    # Store provenance tensors
    tensors = []
    start_time = time.time()
    tensor_sizes = []
    
    # OPERATION 0: Select relevant columns
    columns = ['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 
             'days_b_screening_arrest', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    df, tensor0 = select_with_prov(df, columns)
    tensors.append(tensor0)
    tensor_sizes.append(tensor0.nbytes)
    
    # OPERATION 1: Remove missing values
    df, tensor1 = dropna_with_prov(df)
    tensors.append(tensor1)
    tensor_sizes.append(tensor1.nbytes)
    
    # OPERATION 2: Make race binary (1 = Caucasian, 0 = Other)
    # For apply operations, we need to track row correspondence manually
    original_indices = df.index.copy()
    df['race'] = df['race'].apply(lambda r: 1 if r == 'Caucasian' else 0)
    # Since apply preserves row order, we use identity matrix
    tensor2 = np.eye(len(df), dtype=int)
    tensors.append(tensor2)
    tensor_sizes.append(tensor2.nbytes)
    
    # OPERATION 3: Make 'two_year_recid' the label and reverse values 
    df = df.rename({'two_year_recid': 'label'}, axis=1)
    df['label'] = df['label'].apply(lambda l: 0 if l == 1 else 1)
    # Renaming and value changes preserve row correspondence
    tensor3 = np.eye(len(df), dtype=int)
    tensors.append(tensor3)
    tensor_sizes.append(tensor3.nbytes)
    
    # OPERATION 4: Convert jail time to days
    df['jailtime'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days
    # Adding a column preserves row correspondence
    tensor4 = np.eye(len(df), dtype=int)
    tensors.append(tensor4)
    tensor_sizes.append(tensor4.nbytes)
    
    # OPERATION 5: Drop jail in and out dates
    original_df = df.copy()
    df.drop(columns=['c_jail_in', 'c_jail_out'], inplace=True)
    # Dropping columns preserves row correspondence
    tensor5 = np.eye(len(df), dtype=int)
    tensors.append(tensor5)
    tensor_sizes.append(tensor5.nbytes)
    
    # OPERATION 6: Convert charge degree to binary (1 = Felony, 0 = Misdemeanor)
    df['c_charge_degree'] = df['c_charge_degree'].apply(lambda s: 1 if s == 'F' else 0)
    # Value changes preserve row correspondence
    tensor6 = np.eye(len(df), dtype=int)
    tensors.append(tensor6)
    tensor_sizes.append(tensor6.nbytes)
    
    # Show processed dataset preview
    print("\nProcessed dataset preview:")
    print(df.head())
    
    # Calculate total provenance capture time
    capture_time = time.time() - start_time
    print(f"\nProvenance capture time: {capture_time:.4f} seconds")
    
    # Calculate total tensor storage
    total_storage = sum(tensor_sizes)
    print(f"Total tensor storage: {total_storage / (1024*1024):.2f} MB")
    
    # Perform provenance queries
    prov_tracer = SITNProv(lambda x: x)  # Dummy function for tracing
    
    # Example 1: Trace a specific output record back to the original input
    query_start = time.time()
    output_idx = 10  # Example output record
    original_indices = prov_tracer.trace_through_pipeline(tensors, output_idx, direction='backward')
    query_time = time.time() - query_start
    
    print(f"\nProvenance Query Example:")
    print(f"Output record at index {output_idx}:")
    print(df.iloc[output_idx])
    print(f"\nCame from original record at index {original_indices}:")
    print(pd.read_csv(input_path, header=0).iloc[original_indices])
    print(f"Query execution time: {query_time:.4f} seconds")
    
    # Example 2: Find all output records derived from a specific input record
    query_start = time.time()
    input_idx = 50  # Example input record
    output_indices = prov_tracer.trace_through_pipeline(tensors, input_idx, direction='forward')
    query_time = time.time() - query_start
    
    if output_indices:
        print(f"\nInput record at index {input_idx} contributed to {len(output_indices)} output records")
        print(f"First such output record:")
        print(df.iloc[output_indices[0]])
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
    plt.savefig('compass_tensor_sizes.png')
    
    print("\nPipeline execution completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', dest='opt', action='store_true', help='Use the optimized capture')
    args = parser.parse_args()
    main(args.opt)
