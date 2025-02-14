import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from config import FILE_PATHS, ASH_CONFIG, MONITORING_CONFIG, get_pdb_directories, ensure_directories_exist

def preprocess_ash_data(df):
    """Preprocesses the ASH data for anomaly detection"""
    # Ensure all column names are lowercase for consistency
    df.columns = df.columns.str.lower()
    
    # Create time-based aggregations with frequency matching monitoring interval
    freq = f"{MONITORING_CONFIG['interval_seconds']}S"  # Convert seconds to pandas frequency string
    agg_df = df.groupby([
        pd.Grouper(key='sample_time', freq=freq),
        'instance_number',
        'wait_class',
        'session_state'
    ]).agg({
        'session_count': 'sum',
        'sql_id': 'nunique',
        'user_id': 'nunique',
        'blocking_session': lambda x: (x.notna()).sum()
    }).reset_index()

    # Pivot wait classes and session states
    wait_pivot = agg_df.pivot_table(
        index=['sample_time', 'instance_number'],
        columns='wait_class',
        values='session_count',
        fill_value=0
    )

    state_pivot = agg_df.pivot_table(
        index=['sample_time', 'instance_number'],
        columns='session_state',
        values='session_count',
        fill_value=0
    )
    
    # Combine features
    features = pd.concat([
        wait_pivot,
        state_pivot,
        agg_df.groupby(['sample_time', 'instance_number']).agg({
            'sql_id': 'first',
            'user_id': 'first',
            'blocking_session': 'first'
        })
    ], axis=1)

    # Add time-based features
    features['hour'] = features.index.get_level_values('sample_time').hour
    features['day_of_week'] = features.index.get_level_values('sample_time').dayofweek

    return features

def train_model(data_file, model_output_file, scaler_output_file, feature_columns_file):
    """Trains the model and saves it to file"""
    try:
        # First read the CSV headers to check column names
        print(f"Reading data from {data_file}")
        headers = pd.read_csv(f"{data_file}.gz", compression='gzip', nrows=0).columns.tolist()
        print("Available columns:", headers)
        
        # Load historical data from CSV
        df = pd.read_csv(f"{data_file}.gz", compression='gzip')
        
        if df.empty:
            raise ValueError("No data found in the CSV file")
            
        print(f"Loaded {len(df)} rows of data")
        
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Convert sample_time to datetime
        if 'sample_time' in df.columns:
            df['sample_time'] = pd.to_datetime(df['sample_time'])
        else:
            raise ValueError(f"Required column 'sample_time' not found. Available columns: {df.columns.tolist()}")
        
        # Preprocess data
        print("Preprocessing data...")
        features = preprocess_ash_data(df)
        print(f"Created {len(features.columns)} features")
        
        if len(features) < MONITORING_CONFIG['min_samples']:
            raise ValueError(f"Not enough samples for training. Need at least {MONITORING_CONFIG['min_samples']}, got {len(features)}")
        
        # Initialize and train model
        print("Training model...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Create IsolationForest with only supported parameters
        model = IsolationForest(
            contamination=MONITORING_CONFIG['contamination'],
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        model.fit(scaled_features)
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
        
        # Save model and scaler
        print(f"Saving model to {model_output_file}")
        with open(model_output_file, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Saving scaler to {scaler_output_file}")
        with open(scaler_output_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature columns for reference
        print(f"Saving feature columns to {feature_columns_file}")
        with open(feature_columns_file, 'w') as f:
            f.write('\n'.join(features.columns))
            
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting model training process...")
        
        # Get PDB-specific paths and ensure directories exist
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        print(f"Using data file: {paths['historical_data_file']}")
        
        train_model(
            paths['historical_data_file'],
            paths['model_file'],
            paths['scaler_file'],
            paths['feature_columns_file']
        )
    except Exception as e:
        print(f"Failed to train model: {str(e)}")
        raise