import sys
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
sys.path.append('..')
from sitn_prov import SITNProv

def main(opt):
    input_path = './Datasets/german.csv'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path, header=0)

    # Show initial dataset preview
    print("\nInitial dataset preview:")
    print(df.head())
    
    # Initialize provenance tracking
    replace_with_prov = SITNProv(pd.DataFrame.replace)
    
    # Store provenance tensors
    tensors = []
    start_time = time.time()
    tensor_sizes = []

    # OPERATION 0: Replace cryptic values with meaningful labels
    df, tensor0 = replace_with_prov(df, to_replace={
        'checking': {'A11': 'check_low', 'A12': 'check_mid', 'A13': 'check_high', 'A14': 'check_none'},
        'credit_history': {'A30': 'debt_none', 'A31': 'debt_noneBank', 'A32': 'debt_onSchedule', 'A33': 'debt_delay', 'A34': 'debt_critical'},
        'purpose': {'A40': 'pur_newCar', 'A41': 'pur_usedCar', 'A42': 'pur_furniture', 'A43': 'pur_tv',
                    'A44': 'pur_appliance', 'A45': 'pur_repairs', 'A46': 'pur_education', 'A47': 'pur_vacation',
                    'A48': 'pur_retraining', 'A49': 'pur_business', 'A410': 'pur_other'},
        'savings': {'A61': 'sav_small', 'A62': 'sav_medium', 'A63': 'sav_large', 'A64': 'sav_xlarge', 'A65': 'sav_none'},
        'employment': {'A71': 'emp_unemployed', 'A72': 'emp_lessOne', 'A73': 'emp_lessFour', 'A74': 'emp_lessSeven', 'A75': 'emp_moreSeven'},
        'other_debtors': {'A101': 'debtor_none', 'A102': 'debtor_coApp', 'A103': 'debtor_guarantor'},
        'property': {'A121': 'prop_realEstate', 'A122': 'prop_agreement', 'A123': 'prop_car', 'A124': 'prop_none'},
        'other_inst': {'A141': 'oi_bank', 'A142': 'oi_stores', 'A143': 'oi_none'},
        'housing': {'A151': 'hous_rent', 'A152': 'hous_own', 'A153': 'hous_free'},
        'job': {'A171': 'job_unskilledNR', 'A172': 'job_unskilledR', 'A173': 'job_skilled', 'A174': 'job_highSkill'},
        'phone': {'A191': 0, 'A192': 1},
        'foreigner': {'A201': 1, 'A202': 0},
        'label': {2: 0}
    }, value=None)
    tensors.append(tensor0)
    tensor_sizes.append(tensor0.nbytes)

    # OPERATION 1: Map gender and marital status
    original_df = df.copy()
    df['status'] = df['personal_status'].map({'A91': 'divorced', 'A92': 'divorced', 'A93': 'single', 'A95': 'single'}).fillna('married')
    df['gender'] = df['personal_status'].map({'A92': 0, 'A95': 0}).fillna(1)  # 0 for female, 1 for male
    # Since map preserves row correspondence
    tensor1 = np.eye(len(df), dtype=int)
    tensors.append(tensor1)
    tensor_sizes.append(tensor1.nbytes)

    # OPERATION 2: Drop the original 'personal_status' column
    original_df = df.copy()
    df.drop(columns=['personal_status'], inplace=True)
    # Dropping columns preserves row correspondence
    tensor2 = np.eye(len(df), dtype=int)
    tensors.append(tensor2)
    tensor_sizes.append(tensor2.nbytes)

    # OPERATION 3-13: One-hot encode categorical columns
    original_df = df.copy()
    categorical_cols = ['checking', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                        'other_inst', 'housing', 'job', 'status']
    
    df = pd.get_dummies(df, columns=categorical_cols)
    # One-hot encoding preserves row correspondence
    tensor3 = np.eye(len(df), dtype=int)
    tensors.append(tensor3)
    tensor_sizes.append(tensor3.nbytes)

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
    output_idx = 5  # Example output record
    original_indices = prov_tracer.trace_through_pipeline(tensors, output_idx, direction='backward')
    query_time = time.time() - query_start
    
    print(f"\nProvenance Query Example:")
    print(f"Output record at index {output_idx} (selected columns):")
    selected_cols = ['duration', 'credit_amount', 'gender', 'label']
    print(df.iloc[output_idx][selected_cols])
    print(f"\nCame from original record at index {original_indices}:")
    original_df = pd.read_csv(input_path, header=0)
    print(original_df.iloc[original_indices])
    print(f"Query execution time: {query_time:.4f} seconds")
    
    # Example 2: Find all output records derived from a specific input record
    query_start = time.time()
    input_idx = 10  # Example input record
    output_indices = prov_tracer.trace_through_pipeline(tensors, input_idx, direction='forward')
    query_time = time.time() - query_start
    
    if output_indices:
        print(f"\nInput record at index {input_idx} contributed to {len(output_indices)} output records")
        print(f"First such output record (selected columns):")
        print(df.iloc[output_indices[0]][selected_cols])
    else:
        print(f"\nInput record at index {input_idx} did not contribute to any output records")
    print(f"Query execution time: {query_time:.4f} seconds")

    print("\nPipeline execution completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', dest='opt', action='store_true', help='Use the optimized capture')
    args = parser.parse_args()
    main(args.opt)
