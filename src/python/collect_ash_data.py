import cx_Oracle
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
from config import ORACLE_ENV, ASH_CONFIG, FILE_PATHS, get_pdb_directories, ensure_directories_exist

def setup_oracle_env():
    """Setup Oracle environment variables if not already set"""
    for var, value in ORACLE_ENV.items():
        # Expand any environment variables in the value
        expanded_value = os.path.expandvars(value) if isinstance(value, str) else value
        # Only set if not already set or if different
        if var not in os.environ or os.environ[var] != expanded_value:
            os.environ[var] = str(expanded_value)
            print(f"Setting {var}={expanded_value}")

    # Verify environment setup
    missing_vars = [var for var in ORACLE_ENV if var not in os.environ]
    if missing_vars:
        raise EnvironmentError(f"Missing required Oracle environment variables: {', '.join(missing_vars)}")

def connect_as_sysdba():
    """Establishes connection as SYSDBA using OS authentication"""
    setup_oracle_env()

    try:
        # Connect with '/' which indicates OS authentication
        connection = cx_Oracle.connect(
            '/',
            mode=cx_Oracle.SYSDBA
        )

        # Set container if PDB_NAME is specified
        if ORACLE_ENV['PDB_NAME']:
            cursor = connection.cursor()
            cursor.execute(f"ALTER SESSION SET CONTAINER = {ORACLE_ENV['PDB_NAME']}")
            cursor.close()

        return connection
    except cx_Oracle.Error as e:
        error, = e.args
        print(f"Oracle Error: {error.code} - {error.message}")
        raise

def validate_ash_data(df):
    """Validates the ASH data DataFrame"""
    # Convert all column names to lowercase for consistency
    df.columns = df.columns.str.lower()
    
    required_columns = [
        'instance_number', 'sample_time', 'session_state',
        'wait_class', 'session_count'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        # Print available columns for debugging
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Convert sample_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['sample_time']):
        try:
            df['sample_time'] = pd.to_datetime(df['sample_time'])
        except Exception as e:
            raise ValueError(f"Failed to convert sample_time to datetime: {str(e)}")

    return df

def collect_historical_ash(start_date, end_date, output_file):
    """Collects historical ASH data and saves to file"""
    ash_query = """
    SELECT
        ash.INSTANCE_NUMBER as instance_number,
        ash.SAMPLE_TIME as sample_time,
        ash.SESSION_STATE as session_state,
        ash.WAIT_CLASS as wait_class,
        ash.EVENT as event,
        ash.SQL_ID as sql_id,
        ash.SQL_PLAN_HASH_VALUE as sql_plan_hash_value,
        ash.SESSION_TYPE as session_type,
        ash.BLOCKING_SESSION as blocking_session,
        ash.BLOCKING_SESSION_SERIAL# as blocking_session_serial#,
        ash.USER_ID as user_id,
        ash.PROGRAM as program,
        ash.MODULE as module,
        ash.ACTION as action,
        ash.MACHINE as machine,
        ash.SERVICE_HASH as service_hash,
        COUNT(*) as session_count
    FROM dba_hist_active_sess_history ash
    WHERE ash.SAMPLE_TIME BETWEEN :start_date AND :end_date
    GROUP BY
        ash.INSTANCE_NUMBER,
        ash.SAMPLE_TIME,
        ash.SESSION_STATE,
        ash.WAIT_CLASS,
        ash.EVENT,
        ash.SQL_ID,
        ash.SQL_PLAN_HASH_VALUE,
        ash.SESSION_TYPE,
        ash.BLOCKING_SESSION,
        ash.BLOCKING_SESSION_SERIAL#,
        ash.USER_ID,
        ash.PROGRAM,
        ash.MODULE,
        ash.ACTION,
        ash.MACHINE,
        ash.SERVICE_HASH
    ORDER BY ash.SAMPLE_TIME
    """

    try:
        print(f"Collecting ASH data from {start_date} to {end_date}")
        with connect_as_sysdba() as connection:
            # Read data in batches
            dfs = []
            for offset in range(0, sys.maxsize, ASH_CONFIG['batch_size']):
                batch_query = f"{ash_query} OFFSET {offset} ROWS FETCH NEXT {ASH_CONFIG['batch_size']} ROWS ONLY"
                batch_df = pd.read_sql(batch_query, connection,
                                     params={'start_date': start_date, 'end_date': end_date})
                
                if batch_df.empty:
                    break
                    
                dfs.append(batch_df)
                print(f"Collected {len(dfs) * ASH_CONFIG['batch_size']} records...")

            if not dfs:
                print("Warning: No ASH data found for the specified time range")
                return

            # Combine all batches
            df = pd.concat(dfs, ignore_index=True)
            print(f"Total records collected: {len(df)}")

            # Validate and process the data
            df = validate_ash_data(df)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save data in specified format
            if ASH_CONFIG['data_format'].lower() == 'csv':
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported data format: {ASH_CONFIG['data_format']}")

            print(f"Data saved to {output_file}")

            # Print summary statistics
            print("\nData Summary:")
            try:
                print(f"Time range: {df['sample_time'].min()} to {df['sample_time'].max()}")
                print(f"Unique instances: {df['instance_number'].nunique()}")
                print(f"Unique wait classes: {df['wait_class'].nunique()}")
                print(f"Total session count: {df['session_count'].sum()}")

                # Wait class statistics with severity levels
                print("\nWait Class Statistics:")
                wait_class_stats = df.groupby('wait_class')['session_count'].sum().sort_values(ascending=False)
                for wait_class, count in wait_class_stats.items():
                    if pd.notna(wait_class):
                        severity = ASH_CONFIG['wait_classes'].get(wait_class, {}).get('severity', 'UNKNOWN')
                        print(f"  {wait_class} ({severity}): {count}")

            except Exception as e:
                print(f"Warning: Error generating summary statistics: {str(e)}")

    except Exception as e:
        print(f"Error collecting ASH data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting historical ASH data collection...")
        
        # Get PDB-specific paths and ensure directories exist
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        START_DATE = datetime.now() - timedelta(days=ASH_CONFIG['retention_days'])
        END_DATE = datetime.now()

        collect_historical_ash(
            START_DATE,
            END_DATE,
            paths['historical_data_file']
        )
        print("Data collection completed successfully")

    except Exception as e:
        print(f"Failed to collect historical ASH data: {str(e)}")
        raise
