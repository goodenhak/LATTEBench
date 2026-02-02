import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

def csv_split(csv_file_path, hdf_file_path, hdf_dataset_name='my_dataset', 
               target_col=None, fillna_method='auto', output_format='hdf', encoding='utf-8',
               handle_categorical='auto', test_size=0.2, seed=42, task_type=1):
    """
    Converts a CSV file to an HDF5 file, performs preprocessing, and splits the
    data into training and testing sets. Only the training data is saved.

    Args:
        csv_file_path (str): The path to the input CSV file.
        hdf_file_path (str): The path where the output HDF5 file will be saved.
        hdf_dataset_name (str): The name of the dataset for features within the HDF5 file.
        target_col (str): The name of the target variable column for train-test split.
                          Required for splitting.
        fillna_method (str): Method to fill missing values. Options: 
                            'auto', 'mean', 'median', 'mode', 'zero', 'ffill', 'none'
        output_format (str): The format to save the output file. Options: 'hdf', 'csv', 'both'
        encoding (str): Encoding for reading CSV file.
        handle_categorical (str): How to handle categorical columns. Options:
                                 'auto', 'label_encode', 'frequency', 'none'
        test_size (float): The proportion of the dataset to include in the test split.
        seed (int): Random state for reproducibility.
        task_type (int): 1 for classification (stratify), 0 for regression (no stratify).
                         Used for stratified splitting based on the target variable.
    """
    try:
        # Check if the CSV file exists
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at '{csv_file_path}'")
            return

        # Read the CSV file
        print(f"Reading CSV file: '{csv_file_path}'...")
        try:
            df = pd.read_csv(csv_file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
        
        print(f"Original data shape: {df.shape}")

        # Separate features and target if a target column is specified
        if target_col and target_col in df.columns:
            print(f"Separating features and target variable: '{target_col}'")
            y = df[target_col]
            X = df.drop(target_col, axis=1)
        else:
            if target_col:
                print(f"Warning: Target column '{target_col}' not found. Skipping train-test split.")
            X = df
            y = None

        # Identify column types on the feature DataFrame (X)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}: {categorical_cols}")
        print(f"Boolean columns: {len(boolean_cols)}: {boolean_cols}")

        # Handle boolean columns - convert to 0 and 1
        if boolean_cols:
            print(f"\nConverting boolean columns to integers (0 and 1)...")
            for col in boolean_cols:
                X[col] = X[col].astype(int)
                print(f"Converted boolean column '{col}' to integers (0/1)")

        # Handle missing values
        print(f"\nHandling missing values with method: {fillna_method}")
        missing_info = X.isnull().sum()
        if missing_info.sum() > 0:
            print("Missing values per column:")
            for col, count in missing_info[missing_info > 0].items():
                print(f"  {col}: {count} missing values")

        if fillna_method != 'none':
            # Handle numerical columns (including converted boolean columns)
            numerical_cols_extended = numerical_cols + boolean_cols
            for col in numerical_cols_extended:
                if X[col].isnull().sum() > 0:
                    if fillna_method == 'auto':
                        fill_value = X[col].median()
                    elif fillna_method == 'mean':
                        fill_value = X[col].mean()
                    elif fillna_method == 'median':
                        fill_value = X[col].median()
                    elif fillna_method == 'zero':
                        fill_value = 0
                    elif fillna_method == 'ffill':
                        X[col] = X[col].fillna(method='ffill')
                        continue
                    else:
                        fill_value = 0
                    
                    X[col] = X[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with {fill_value:.2f}")

            # Handle categorical columns
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    if fillna_method in ['auto', 'mode', 'ffill']:
                        fill_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                    elif fillna_method == 'zero':
                        fill_value = 'Unknown'
                    else:
                        fill_value = 'Unknown'
                    
                    X[col] = X[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with '{fill_value}'")

        # Handle categorical columns - encode all text columns
        if handle_categorical != 'none' and categorical_cols:
            print(f"\nEncoding categorical columns with method: {handle_categorical}")
            
            for col in categorical_cols:
                if handle_categorical == 'auto':
                    # Auto-detect: label encode if few categories, frequency encode if many
                    unique_count = X[col].nunique()
                    if unique_count <= 20:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        print(f"Label encoded '{col}' ({unique_count} categories)")
                    else:
                        freq_map = X[col].value_counts(normalize=True).to_dict()
                        X[col] = X[col].map(freq_map)
                        print(f"Frequency encoded '{col}' ({unique_count} categories)")
                
                elif handle_categorical == 'label_encode':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    print(f"Label encoded '{col}'")
                
                elif handle_categorical == 'frequency':
                    freq_map = X[col].value_counts(normalize=True).to_dict()
                    X[col] = X[col].map(freq_map)
                    print(f"Frequency encoded '{col}'")

        # Final data info
        print(f"\nFinal preprocessed data shape: {X.shape}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        
        # Split the data into training and testing sets
        if y is not None:
            # Check if target column needs encoding
            if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
                print(f"Encoding target variable '{target_col}'...")
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))

            print(f"\nSplitting data into training ({1-test_size:.0%}) and testing ({test_size:.0%}) sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=seed,
                stratify=y if task_type == 1 else None
            )
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
            # Combine X_train and y_train into a single DataFrame
            y_train_df = pd.DataFrame(y_train, columns=[target_col], index=X_train.index)
            y_test_df = pd.DataFrame(y_test, columns=[target_col], index=X_test.index)
            train_df = pd.concat([X_train, y_train_df], axis=1)
            test_df = pd.concat([X_test, y_test_df], axis=1)

            # Write the DataFrame to output file(s)
            if output_format in ['hdf', 'both']:
                hdf_path = './'+hdf_file_path if hdf_file_path.endswith('.hdf') else './'+ hdf_file_path + '.hdf'
                print(f"Writing complete training data to HDF5 file: '{hdf_path}'...")
                test_hdf_path = './'+hdf_file_path+'_test' if hdf_file_path.endswith('.hdf') else './'+ hdf_file_path + '_test.hdf'
                train_df.to_hdf(hdf_path, key=hdf_dataset_name, mode='w', format='table', data_columns=True)
                test_df.to_hdf(test_hdf_path, key=hdf_dataset_name, mode='w', format='table', data_columns=True)

                print("Successfully saved complete training set to HDF5.")
                print("Note: The test set was generated but not saved, as per the request.")
        
        else:
            # If no target column was specified, save the whole preprocessed DataFrame
            print("\nNo target column specified. Saving the entire preprocessed DataFrame.")
            if output_format in ['hdf', 'both']:
                hdf_path = hdf_file_path if hdf_file_path.endswith('.hdf') else hdf_file_path + '.hdf'
                X.to_hdf(hdf_path, key=hdf_dataset_name, mode='w', format='table', data_columns=True)
                print("Successfully saved to HDF5.")

            if output_format in ['csv', 'both']:
                csv_path = hdf_file_path if hdf_file_path.endswith('.csv') else hdf_file_path + '.csv'
                X.to_csv(csv_path, index=False)
                print("Successfully saved to CSV.")

        print("\nPreprocessing and data splitting completed successfully!")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file_path}' is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Simplified Example Usage ---
if __name__ == "__main__":
    # This example requires a 'vehicle.csv' file in the same directory.
    # Make sure to replace 'target_column_name' with the actual column name.
    
    csv_file = "./tabular_data/electricity.csv"
    output_file = "electricity"
    features_dataset = "electricity"
    target_column = "target" # IMPORTANT: CHANGE THIS TO YOUR ACTUAL TARGET COLUMN

    print("Starting simplified CSV conversion and splitting...")
    
    # Example for a classification task (stratify is used)
    csv_split(
        csv_file_path=csv_file,
        hdf_file_path=output_file,
        hdf_dataset_name=features_dataset,
        target_col=target_column,
        fillna_method='auto',
        output_format='hdf',
        handle_categorical='auto',
        test_size=0.2,
        seed=2,
        task_type=1 # Use 1 for classification
    )
    
print("\nConversion and preprocessing process finished!")