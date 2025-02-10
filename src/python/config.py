"""Configuration settings for Oracle ASH Analysis

This configuration file contains all settings for the Oracle ASH Analysis tools.
Each section is documented with its purpose and which scripts use the settings.
"""

# Oracle Environment Settings
# Used by: collect_ash_data.py, monitor_ash.py
# Mandatory settings for Oracle connectivity
ORACLE_ENV = {
    'ORACLE_HOME': '/u01/app/oracle/product/19.0.0/dbhome_1',  # Path to Oracle installation
    'ORACLE_SID': 'orcl',                                      # Oracle System Identifier
    'LD_LIBRARY_PATH': '$ORACLE_HOME/lib',                     # Oracle library path
    'PDB_NAME': None                                           # Optional: Pluggable Database name
}

# ASH Collection Settings
# Used by: collect_ash_data.py
# Settings specific to ASH data collection
ASH_CONFIG = {
    'retention_days': 30,          # Number of days of ASH data to collect
    'data_format': 'csv',         # Output format for collected data
    'batch_size': 10000,          # Number of rows to process at once
    'wait_classes': {
        # Documentation: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/wait-events.html
        'Administrative': {
            'description': 'Wait events for administrative operations',
            'severity': 'MEDIUM'
        },
        'Application': {
            'description': 'Wait events related to application code',
            'severity': 'MEDIUM'
        },
        'Cluster': {
            'description': 'Wait events for RAC cluster operations',
            'severity': 'CRITICAL'
        },
        'Commit': {
            'description': 'Wait events for transaction commits',
            'severity': 'MEDIUM'
        },
        'Concurrency': {
            'description': 'Wait events for concurrent access to resources',
            'severity': 'HIGH'
        },
        'Configuration': {
            'description': 'Wait events related to database configuration',
            'severity': 'LOW'
        },
        'Network': {
            'description': 'Wait events for network operations',
            'severity': 'HIGH'
        },
        'Other': {
            'description': 'Miscellaneous wait events',
            'severity': 'LOW'
        },
        'Scheduler': {
            'description': 'Wait events for scheduler operations',
            'severity': 'LOW'
        },
        'System I/O': {
            'description': 'Wait events for system I/O operations',
            'severity': 'CRITICAL'
        },
        'User I/O': {
            'description': 'Wait events for user I/O operations',
            'severity': 'HIGH'
        }
    }
}

# Severity Configuration
# Used by: monitor_ash.py
# Defines severity levels and their visual representation
SEVERITY_CONFIG = {
    'colors': {
        'CRITICAL': '\033[1;31m',  # Bold Red
        'HIGH': '\033[0;31m',      # Red
        'MEDIUM': '\033[0;33m',    # Yellow
        'LOW': '\033[0;32m',       # Green
        'RESET': '\033[0m'         # Reset color
    },
    'thresholds': {
        'blocking_chain_length': 3,  # Length of blocking chain to consider critical
        'score_critical': 1.5,       # Multiplier for critical anomaly score
        'score_high': 1.2,           # Multiplier for high anomaly score
    }
}

# Monitoring Settings
# Used by: monitor_ash.py, train_model.py
# Settings for real-time monitoring and model training
MONITORING_CONFIG = {
    'interval_seconds': 60,         # Monitoring interval
    'alert_threshold': 0.9,         # Anomaly score threshold for alerts
    'lookback_minutes': 2,          # Minutes of data to analyze in each check
    'results_retention_days': 7,    # Days to keep monitoring results
    'max_anomalies_per_file': 1000, # Maximum anomalies per file before rotating
    'contamination': 0.01,          # Expected proportion of outliers (1%)
    'min_samples': 10,              # Minimum samples for model training
    'wait_threshold': 10            # Threshold for sessions in wait class
}

def get_file_paths(pdb_name=None):
    """Generate file paths with optional PDB name included
    
    Args:
        pdb_name: Optional name of the Pluggable Database
    
    Returns:
        Dictionary of file paths for various components
    """
    # Use PDB name in file paths only if provided
    suffix = f"_{pdb_name}" if pdb_name else ""
    
    return {
        'model_file': f'ash_anomaly_model{suffix}.pkl',
        'scaler_file': f'ash_scaler{suffix}.pkl',
        'feature_columns_file': f'feature_columns{suffix}.txt',
        'historical_data_file': f'historical_ash_data{suffix}.csv',
        'monitoring_results_dir': f'monitoring_results{suffix}',
        'current_results_file': f'monitoring_results{suffix}/current.json',
        'summary_index_file': f'monitoring_results{suffix}/summary_index.json'
    }

# Initialize file paths based on PDB configuration
FILE_PATHS = get_file_paths(ORACLE_ENV['PDB_NAME'])
