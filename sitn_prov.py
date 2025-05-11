import pandas as pd
import numpy as np

class SITNProv:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        method_name = self.func.__name__
        if method_name == "filter":
            return self.decorate_filter(*args, **kwargs)
        elif method_name == "merge":
            return self.decorate_merge(*args, **kwargs)
        elif method_name == "replace":
            return self.decorate_identity(*args, **kwargs)
        elif method_name == "__getitem__":
            return self.decorate_select(*args, **kwargs)
        elif method_name == "dropna":
            return self.decorate_dropna(*args, **kwargs)
        else:
            raise NotImplementedError(f"Decoration for '{method_name}' is not implemented.")
        
    def decorate_filter(self, df, condition):
        """Decorates the filter operation by applying a query and tracking provenance.
        
        - The operation uses `df.query(condition)` to filter rows.
        - The provenance tensor is a binary matrix of shape (m, n) where each row corresponds
          to an output row and has a 1 at the column corresponding to its original row position.
        """
        # Perform filtering using DataFrame.query
        filtered_df = df.query(condition)
        
        # Create provenance tensor
        n = len(df)
        m = len(filtered_df)
        T = np.zeros((m, n), dtype=int)
        # For each row in filtered_df, mark the corresponding original row location
        for i, orig_idx in enumerate(filtered_df.index):
            pos = df.index.get_loc(orig_idx)
            T[i, pos] = 1
        
        return filtered_df, T

    def decorate_merge(self, df1, df2, **kwargs):
        """
        Modifies the merge operation (acting as join) by constructing a single 3D provenance tensor T.

        Steps:
        - Copy df1 and df2 and add temporary columns that tag the original row indices.
        - Perform the merge operation using pd.merge (which represents join in pandas).
        - Create T of shape (m, n1, n2) where:
            m = number of rows in the merged result,
            n1 = number of rows in df1,
            n2 = number of rows in df2.
        - For each output row i:
              If the contributing left row is at position j and the right row (if any) is at position k,
              then set T[i, j, k] = 1.
          If there is no matching right row (e.g., in a left join), T[i, j, :] remains zero.
        - Remove the temporary provenance columns before returning.
        """
        # Create copies to preserve original DataFrames
        left_df = df1.copy()
        right_df = df2.copy()
        
        # Add temporary columns for provenance tracking
        left_df["_left_index"] = left_df.index
        right_df["_right_index"] = right_df.index
        
        # Perform the merge (join) operation
        merged_df = pd.merge(left_df, right_df, **kwargs)
        merged_df = merged_df.reset_index(drop=True)
        
        # Dimensions for the 3D provenance tensor
        m = len(merged_df)      # number of rows in the merged output
        n_left = len(df1)        # number of rows in the left DataFrame
        n_right = len(df2)       # number of rows in the right DataFrame
        
        # Initialize 3D provenance tensor T with zeros
        T = np.zeros((m, n_left, n_right), dtype=int)
        
        # For each merged row, record provenance information in T
        for i, row in merged_df.iterrows():
            # Get provenance from the left DataFrame
            left_orig = row["_left_index"]
            j = df1.index.get_loc(left_orig)
            
            # Get provenance from the right DataFrame (may be NaN if no match)
            right_orig = row.get("_right_index", None)
            if pd.notna(right_orig):
                k = df2.index.get_loc(right_orig)
                T[i, j, k] = 1
            # If right_orig is NaN (e.g., left join with no match), T[i, j, :] stays zero.
        
        # Clean up the temporary provenance columns
        merged_df.drop(columns=["_left_index", "_right_index"], inplace=True, errors="ignore")
        
        return merged_df, T
    
    def decorate_identity(self, df, to_replace, value):
        """
        Modifies the replace operation by tracking provenance.
        
        - The operation uses `df.replace(to_replace, value)` to replace values.
        - The provenance tensor is a binary matrix of shape (m, n) where each row corresponds
          to an output row and has a 1 at the column corresponding to its original row position.
        """
        # Perform replacement using DataFrame.replace
        replaced_df = self.func(df, to_replace, value)

        n = len(df)


        # since replace operation maintains row correspondence
        T = np.identity(n, dtype=int)

        # If indexes might be different or reordered, this ensures correct mapping
        if not np.array_equal(replaced_df.index, df.index):
            idx_map = np.array([df.index.get_loc(i) for i in replaced_df.index])
            T = T[idx_map]

        return replaced_df, T

    def decorate_select(self, df, columns):
        """
        Modifies the select operation by tracking provenance.
        
        - The operation uses `df[columns]` to select columns.
        - The provenance tensor is a binary matrix of shape (m, n) where each row corresponds
          to an output row and has a 1 at the column corresponding to its original row position.
        """
        # Perform selection using DataFrame[columns]
        selected_df = df[columns]
        
        n = len(df)
        m = len(selected_df)
        
        # Create provenance tensor
        T = np.zeros((m, n), dtype=int)
        
        # For each row in selected_df, mark the corresponding original row location
        for i, orig_idx in enumerate(selected_df.index):
            pos = df.index.get_loc(orig_idx)
            T[i, pos] = 1
        
        return selected_df, T

    def decorate_dropna(self, df):
        """
        Modifies the dropna operation by tracking provenance.
        
        - The operation uses `df.dropna(axis, how, thresh, subset, inplace)` to drop rows/columns with missing values.
        - The provenance tensor is a binary matrix of shape (m, n) where each row corresponds
          to an output row and has a 1 at the column corresponding to its original row position.
        """
        # Perform dropna using DataFrame.dropna
        dropped_df = self.func(df)
        
        # Create provenance tensor: rows of dropped_df vs. original rows of df
        n = len(df)            # original DataFrame row count
        m = len(dropped_df)    # resulting DataFrame row count
        prov_tensor = np.zeros((m, n), dtype=int)
        
        # For each row in the resulting DataFrame, mark the corresponding original row.
        for i, orig_index in enumerate(dropped_df.index):
            pos = df.index.get_loc(orig_index)
            prov_tensor[i, pos] = 1
        
        return dropped_df, prov_tensor
    
    def trace_output_to_input(self, tensor, output_indices):
        """
        Traces output records back to their contributing input records.
        
        Args:
            tensor: The provenance tensor from an operation
            output_indices: Indices of output records to trace (list or int)
            
        Returns:
            List of input indices that contributed to the specified output records
        """
        if isinstance(output_indices, int):
            output_indices = [output_indices]
            
        # Handle different tensor dimensions based on operation type
        if tensor.ndim == 2:  # For filter, select, dropna, replace
            # For each output index, find the input indices with value 1
            input_indices = []
            for idx in output_indices:
                # Get positions where the tensor has 1s for this output row
                contributors = np.where(tensor[idx] == 1)[0]
                input_indices.extend(contributors.tolist())
            return sorted(list(set(input_indices)))  # Remove duplicates and sort
            
        elif tensor.ndim == 3:  # For merge operations
            # For merge, we need to return indices from both input dataframes
            left_indices = []
            right_indices = []
            for idx in output_indices:
                # Find non-zero positions in the 2D slice for this output row
                contributors = np.where(tensor[idx] == 1)
                left_indices.extend(contributors[0].tolist())
                right_indices.extend(contributors[1].tolist())
            return {
                'left': sorted(list(set(left_indices))),
                'right': sorted(list(set(right_indices)))
            }
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
    
    def trace_input_to_output(self, tensor, input_indices, input_side=None):
        """
        Traces input records forward to their resulting output records.
        
        Args:
            tensor: The provenance tensor from an operation
            input_indices: Indices of input records to trace (list or int)
            input_side: For merge operations, specify 'left' or 'right' dataframe
            
        Returns:
            List of output indices that were derived from the specified input records
        """
        if isinstance(input_indices, int):
            input_indices = [input_indices]
            
        # Handle different tensor dimensions based on operation type
        if tensor.ndim == 2:  # For filter, select, dropna, replace
            # For each input index, find output rows that have a 1 in that position
            output_indices = []
            for idx in input_indices:
                # Get positions where the tensor has 1s in this input column
                dependents = np.where(tensor[:, idx] == 1)[0]
                output_indices.extend(dependents.tolist())
            return sorted(list(set(output_indices)))  # Remove duplicates and sort
            
        elif tensor.ndim == 3:  # For merge operations
            if input_side not in ['left', 'right']:
                raise ValueError("For merge operations, input_side must be 'left' or 'right'")
                
            output_indices = []
            for idx in input_indices:
                if input_side == 'left':
                    # Find output rows where this left input contributed
                    dependents = np.where(np.any(tensor[:, idx, :], axis=1))[0]
                else:  # right
                    # Find output rows where this right input contributed
                    dependents = np.where(np.any(tensor[:, :, idx], axis=1))[0]
                output_indices.extend(dependents.tolist())
            return sorted(list(set(output_indices)))
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
    
    def compose_provenance(self, tensor1, tensor2):
        """
        Composes two provenance tensors to track provenance through a pipeline of operations.
        
        Args:
            tensor1: The provenance tensor from the first operation
            tensor2: The provenance tensor from the second operation
            
        Returns:
            A composed tensor that directly maps from original inputs to final outputs
        """
        # Handle different tensor dimensions based on operation types
        if tensor1.ndim == 2 and tensor2.ndim == 2:
            # For two 2D tensors (e.g., filter followed by select)
            # Matrix multiplication gives us the composed mapping
            return np.matmul(tensor2, tensor1)
            
        # Add more cases for different combinations of operations
        # This is a simplified version - you may need more complex logic
        # for operations like merge that have 3D tensors
        
        else:
            raise NotImplementedError("Composition not implemented for these tensor types")
    
    def trace_through_pipeline(self, tensors, output_indices, direction='backward'):
        """
        Traces records through a pipeline of operations.
        
        Args:
            tensors: List of provenance tensors from a sequence of operations
            output_indices: Indices to trace
            direction: 'backward' to trace from output to input, 'forward' for input to output
            
        Returns:
            Indices of records at the other end of the pipeline
        """
        if direction == 'backward':
            # Start from the final output indices
            current_indices = output_indices
            
            # Work backwards through each operation
            for tensor in reversed(tensors):
                current_indices = self.trace_output_to_input(tensor, current_indices)
                
            return current_indices
            
        elif direction == 'forward':
            # Start from the initial input indices
            current_indices = output_indices
            
            # Work forwards through each operation
            for tensor in tensors:
                current_indices = self.trace_input_to_output(tensor, current_indices)
                
            return current_indices
        
        else:
            raise ValueError("Direction must be 'backward' or 'forward'")

if __name__ == "__main__":
    filter_with_prov = SITNProv(pd.DataFrame.filter) 
    merge_with_prov = SITNProv(pd.merge)
    # Example DataFrames
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    df1 = pd.DataFrame({'key': [1, 2], 'value': ['A', 'B']})
    df2 = pd.DataFrame({'key': [2, 3], 'value': ['C', 'D']})

    # Filter example
    condition = 'A > 2'
    filtered_df, filter_tensor = filter_with_prov(df, condition)
    print("Filtered DataFrame:")
    print(filtered_df)
    print("Filter Tensor:")
    print(filter_tensor)

    # Merge example
    merged_df, merge_tensor = merge_with_prov(df1, df2, on='key', how='inner')
    print("\nMerged DataFrame:")
    print(merged_df)
    print("Merge Tensor:")
    print(merge_tensor)

    # replace example
    replace_with_prov = SITNProv(pd.DataFrame.replace)
    replaced_df, replace_tensor = replace_with_prov(df, to_replace=1, value=5)
    print("\nReplaced DataFrame:")
    print(replaced_df)
    print("Replace Tensor:")
    print(replace_tensor)

    # select example
    select_with_prov = SITNProv(pd.DataFrame.__getitem__)
    selected_df, select_tensor = select_with_prov(df, ['A'])
    print("\nSelected DataFrame:")
    print(selected_df)
    print("Select Tensor:")
    print(select_tensor)

    # dropna example
    dropna_with_prov = SITNProv(pd.DataFrame.dropna)
    df.loc[1, 'A'] = np.nan
    dropped_df, dropna_tensor = dropna_with_prov(df)
    print("\nDropped DataFrame:")
    print(dropped_df)
    print("Dropna Tensor:")
    print(dropna_tensor)

    # Example of tracing through a pipeline
    print("\n--- Provenance Tracing Examples ---")
    
    # Create a simple pipeline: filter then select
    df_original = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    
    # Step 1: Filter rows where A > 2
    filtered_df, filter_tensor = filter_with_prov(df_original, 'A > 2')
    print("\nAfter filtering A > 2:")
    print(filtered_df)
    
    # Step 2: Select column A
    final_df, select_tensor = select_with_prov(filtered_df, ['A'])
    print("\nAfter selecting column A:")
    print(final_df)
    
    # Create provenance tracer
    prov_tracer = SITNProv(lambda x: x)  # Dummy function, not used for tracing
    
    # Trace an output record back to the original input
    output_idx = 0  # First row of the final result
    print(f"\nTracing output row {output_idx} back to original input:")
    
    # First, find which row in filtered_df contributed to this output
    intermediate_indices = prov_tracer.trace_output_to_input(select_tensor, output_idx)
    print(f"Came from filtered_df rows: {intermediate_indices}")
    
    # Then, find which row in the original df contributed to that
    original_indices = prov_tracer.trace_output_to_input(filter_tensor, intermediate_indices)
    print(f"Came from original df rows: {original_indices}")
    
    # Alternatively, use the pipeline tracing function
    print("\nTracing through the entire pipeline:")
    original_indices = prov_tracer.trace_through_pipeline(
        [filter_tensor, select_tensor], 
        output_idx, 
        direction='backward'
    )
    print(f"Output row {output_idx} came from original rows: {original_indices}")
    
    # Compose the tensors to get a direct mapping
    print("\nUsing tensor composition:")
    composed_tensor = prov_tracer.compose_provenance(filter_tensor, select_tensor)
    print("Composed provenance tensor:")
    print(composed_tensor)
    direct_indices = prov_tracer.trace_output_to_input(composed_tensor, output_idx)
    print(f"Output row {output_idx} came from original rows: {direct_indices}")

