import cx_Oracle
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import time
import json
from config import ORACLE_ENV, ASH_CONFIG, MONITORING_CONFIG, SEVERITY_CONFIG, get_pdb_directories, ensure_directories_exist
from train_model import preprocess_ash_data

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

def colorize(text, severity):
    """Add color to text based on severity"""
    severity = severity.upper()
    color = SEVERITY_CONFIG['colors'].get(severity, SEVERITY_CONFIG['colors']['RESET'])
    return f"{color}{text}{SEVERITY_CONFIG['colors']['RESET']}"

class ResultsManager:
    def __init__(self, monitoring_dir):
        self.monitoring_dir = monitoring_dir
        self.current_file = None
        self.current_count = 0
        
        # Create hourly directory
        self.hourly_dir = os.path.join(monitoring_dir, 'hourly')
        os.makedirs(self.hourly_dir, exist_ok=True)
        
        # Initialize or load summary index
        self.index_file = os.path.join(monitoring_dir, 'summary_index.json')
        self.load_index()

    def load_index(self):
        """Load or initialize the summary index"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'last_update': datetime.now().isoformat(),
                'files': {}
            }

    def save_index(self):
        """Save the summary index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def get_hourly_filename(self, timestamp):
        """Generate hourly filename based on timestamp"""
        return os.path.join(
            self.hourly_dir,
            f"{timestamp.strftime('%Y%m%d_%H')}.json"
        )

    def cleanup_old_files(self):
        """Remove files older than retention period"""
        retention_days = MONITORING_CONFIG['results_retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Clean up hourly files
        for filename in os.listdir(self.hourly_dir):
            try:
                file_date = datetime.strptime(filename.split('_')[0], '%Y%m%d')
                if file_date < cutoff_date:
                    os.remove(os.path.join(self.hourly_dir, filename))
                    # Remove from index if exists
                    if filename in self.index['files']:
                        del self.index['files'][filename]
            except (ValueError, IndexError):
                continue
        
        self.save_index()

    def save_results(self, results):
        """Save results to appropriate files"""
        timestamp = datetime.now()
        hourly_file = self.get_hourly_filename(timestamp)
        
        # Update hourly file
        try:
            if os.path.exists(hourly_file):
                with open(hourly_file, 'r') as f:
                    hourly_data = json.load(f)
            else:
                hourly_data = {
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'hour': timestamp.strftime('%H'),
                    'results': []
                }
            
            hourly_data['results'].append(results)
            
            with open(hourly_file, 'w') as f:
                json.dump(hourly_data, f, indent=2)
            
            # Update index
            self.index['files'][os.path.basename(hourly_file)] = {
                'date': timestamp.strftime('%Y-%m-%d'),
                'hour': timestamp.strftime('%H'),
                'anomaly_count': sum(len(r.get('anomalous_sessions', [])) for r in hourly_data['results']),
                'last_update': timestamp.isoformat()
            }
            
        except Exception as e:
            print(f"Error saving to hourly file: {e}")
            return
        
        # Save current state
        current_file = os.path.join(self.monitoring_dir, 'current.json')
        try:
            with open(current_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving current results: {e}")
        
        # Save index
        self.save_index()

class RACASHMonitor:
    def __init__(self, model_file, scaler_file, feature_columns_file, monitoring_dir):
        self.results_manager = ResultsManager(monitoring_dir)

        # Load model and scaler
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(feature_columns_file, 'r') as f:
            self.feature_columns = f.read().splitlines()

        # Clean up old files on startup
        self.results_manager.cleanup_old_files()

    def get_blocking_chain(self, connection, session_id, session_serial, instance_number):
        """Get the blocking chain for a session"""
        blocking_chain = []
        seen_sessions = set()  # Prevent infinite loops
        
        current_session = {
            'session_id': session_id,
            'session_serial#': session_serial,
            'instance_number': instance_number
        }
        
        while current_session and tuple(current_session.values()) not in seen_sessions:
            seen_sessions.add(tuple(current_session.values()))
            
            query = """
            SELECT 
                s.blocking_session,
                s.blocking_session_serial#,
                s.blocking_instance,
                u.username,
                s.program,
                s.wait_class,
                s.event,
                s.sql_id,
                q.sql_text,
                s.pga_allocated,
                s.temp_space_allocated
            FROM gv$session s
            LEFT JOIN dba_users u ON s.user# = u.user_id
            LEFT JOIN v$sql q ON s.sql_id = q.sql_id
            WHERE s.sid = :1 
            AND s.serial# = :2 
            AND s.inst_id = :3
            """
            
            try:
                cursor = connection.cursor()
                cursor.execute(query, [
                    current_session['session_id'],
                    current_session['session_serial#'],
                    current_session['instance_number']
                ])
                row = cursor.fetchone()
                cursor.close()
                
                if row and row[0]:  # If there's a blocking session
                    # Convert memory metrics to GB
                    pga_gb = float(row[9]) / 1024 / 1024 / 1024 if row[9] else 0
                    temp_gb = float(row[10]) / 1024 / 1024 / 1024 if row[10] else 0
                    
                    blocking_info = {
                        'session_id': int(row[0]),
                        'session_serial#': int(row[1]) if row[1] else None,
                        'instance_number': int(row[2]) if row[2] else None,
                        'username': str(row[3]) if row[3] else 'UNKNOWN',
                        'program': str(row[4]) if row[4] else None,
                        'wait_class': str(row[5]) if row[5] else None,
                        'event': str(row[6]) if row[6] else None,
                        'sql_id': str(row[7]) if row[7] else None,
                        'sql_text': str(row[8])[:100] if row[8] else None,
                        'memory_usage': {
                            'pga_gb': pga_gb,
                            'temp_space_gb': temp_gb
                        }
                    }
                    blocking_chain.append(blocking_info)
                    current_session = {
                        'session_id': blocking_info['session_id'],
                        'session_serial#': blocking_info['session_serial#'],
                        'instance_number': blocking_info['instance_number']
                    }
                else:
                    break
                    
            except Exception as e:
                print(f"Error getting blocking chain: {e}")
                break
                
        return blocking_chain

    def determine_severity(self, session_details):
        """Determine the severity level of an anomaly"""
        severity = 'LOW'
        reasons = []
        
        # Check anomaly score first
        score = session_details.get('anomaly_score', 0)
        threshold = MONITORING_CONFIG['alert_threshold']
        if score > threshold * SEVERITY_CONFIG['thresholds']['score_critical']:
            severity = 'CRITICAL'
            reasons.append(f'Very high anomaly score ({score:.3f})')
        elif score > threshold * SEVERITY_CONFIG['thresholds']['score_high']:
            severity = max(severity, 'HIGH')
            reasons.append(f'High anomaly score ({score:.3f})')
        else:
            reasons.append(f'Normal anomaly score ({score:.3f})')
        
        # Check memory usage
        memory = session_details.get('memory_usage', {})
        if memory:
            pga_gb = memory.get('pga_gb', 0)
            temp_gb = memory.get('temp_space_gb', 0)
            if pga_gb > 10 or temp_gb > 10:  # More than 10GB of either type
                severity = max(severity, 'CRITICAL')
                reasons.append(f'High memory usage (PGA: {pga_gb:.2f} GB, Temp: {temp_gb:.2f} GB)')
            elif pga_gb > 5 or temp_gb > 5:  # More than 5GB of either type
                severity = max(severity, 'HIGH')
                reasons.append(f'Elevated memory usage (PGA: {pga_gb:.2f} GB, Temp: {temp_gb:.2f} GB)')
            else:
                reasons.append(f'Normal memory usage (PGA: {pga_gb:.2f} GB, Temp: {temp_gb:.2f} GB)')
        
        # Check wait class severity from config
        if session_details.get('wait_class'):
            wait_class = session_details['wait_class']
            wait_info = ASH_CONFIG['wait_classes'].get(wait_class, {})
            wait_severity = wait_info.get('severity', 'LOW')
            reasons.append(f"{wait_class} wait class (typical {wait_severity} impact)")
        
        # Check blocking chain
        if session_details.get('blocking_chain'):
            chain_length = len(session_details['blocking_chain'])
            if chain_length >= SEVERITY_CONFIG['thresholds']['blocking_chain_length']:
                severity = 'CRITICAL'
                reasons.append(f'Long blocking chain ({chain_length} sessions)')
            else:
                severity = max(severity, 'HIGH')
                reasons.append('Part of blocking chain')
        
        return severity, reasons

    def collect_current_ash(self):
        """Collects current ASH metrics from GV$ACTIVE_SESSION_HISTORY"""
        current_ash_query = f"""
        SELECT DISTINCT
            ash.inst_id as instance_number,
            ash.sample_time,
            ash.session_state,
            ash.wait_class,
            ash.event,
            ash.sql_id,
            ash.sql_plan_hash_value,
            ash.session_type,
            ash.blocking_session,
            ash.blocking_session_serial#,
            ash.blocking_inst_id,
            ash.user_id,
            ash.program,
            ash.module,
            ash.action,
            ash.machine,
            ash.service_hash,
            ash.session_id,
            ash.session_serial#,
            ash.pga_allocated,
            ash.temp_space_allocated,
            ash.p1,
            ash.p2,
            ash.p3,
            CASE 
                WHEN ash.event like 'enq%' AND ash.session_state = 'WAITING'
                THEN ash.event || ' [mode='||BITAND(ash.P1, POWER(2,14)-1)||']'
                ELSE NVL(ash.event, ash.session_state)
            END as event_extended,
            u.username,
            s.sql_text,
            COUNT(*) as session_count
        FROM gv$active_session_history ash
        LEFT JOIN dba_users u ON ash.user_id = u.user_id
        LEFT JOIN v$sql s ON ash.sql_id = s.sql_id
        WHERE ash.sample_time >= SYSTIMESTAMP - INTERVAL '{MONITORING_CONFIG['lookback_minutes']}' MINUTE
        GROUP BY
            ash.inst_id,
            ash.sample_time,
            ash.session_state,
            ash.wait_class,
            ash.event,
            ash.sql_id,
            ash.sql_plan_hash_value,
            ash.session_type,
            ash.blocking_session,
            ash.blocking_session_serial#,
            ash.blocking_inst_id,
            ash.user_id,
            ash.program,
            ash.module,
            ash.action,
            ash.machine,
            ash.service_hash,
            ash.session_id,
            ash.session_serial#,
            ash.pga_allocated,
            ash.temp_space_allocated,
            ash.p1,
            ash.p2,
            ash.p3,
            CASE 
                WHEN ash.event like 'enq%' AND ash.session_state = 'WAITING'
                THEN ash.event || ' [mode='||BITAND(ash.P1, POWER(2,14)-1)||']'
                ELSE NVL(ash.event, ash.session_state)
            END,
            u.username,
            s.sql_text
        ORDER BY ash.sample_time
        """

        connection = connect_as_sysdba()
        try:
            df = pd.read_sql(current_ash_query, connection)

            if 'sample_time' in df.columns:
                df['sample_time'] = pd.to_datetime(df['sample_time'])

            return df, connection
        except Exception as e:
            connection.close()
            raise

    def get_session_details(self, df, predictions, scores, connection):
        """Extract detailed information about anomalous sessions"""
        anomalous_indices = np.where(predictions == -1)[0]
        session_details = []
        seen_sessions = set()

        for idx in anomalous_indices:
            row = df.iloc[idx]
            session_key = (int(row['instance_number']),
                         int(row['session_id']) if pd.notna(row['session_id']) else None)

            if session_key in seen_sessions:
                continue

            seen_sessions.add(session_key)

            # Get blocking chain
            blocking_chain = []
            if pd.notna(row['blocking_session']):
                blocking_chain = self.get_blocking_chain(
                    connection,
                    int(row['session_id']),
                    int(row['session_serial#']) if pd.notna(row['session_serial#']) else None,
                    int(row['instance_number'])
                )

            # Convert memory metrics to GB
            pga_gb = float(row['pga_allocated']) / 1024 / 1024 / 1024 if pd.notna(row['pga_allocated']) else 0
            temp_gb = float(row['temp_space_allocated']) / 1024 / 1024 / 1024 if pd.notna(row['temp_space_allocated']) else 0

            detail = {
                'timestamp': row['sample_time'].isoformat() if isinstance(row['sample_time'], pd.Timestamp) else str(row['sample_time']),
                'instance': int(row['instance_number']),
                'session_id': int(row['session_id']) if pd.notna(row['session_id']) else None,
                'session_serial#': int(row['session_serial#']) if pd.notna(row['session_serial#']) else None,
                'username': str(row['username']) if pd.notna(row['username']) else 'UNKNOWN',
                'sql_id': str(row['sql_id']) if pd.notna(row['sql_id']) else None,
                'sql_text': str(row['sql_text']) if pd.notna(row['sql_text']) else None,
                'wait_class': str(row['wait_class']) if pd.notna(row['wait_class']) else None,
                'event': str(row['event_extended']) if pd.notna(row['event_extended']) else str(row['event']) if pd.notna(row['event']) else None,
                'program': str(row['program']) if pd.notna(row['program']) else None,
                'module': str(row['module']) if pd.notna(row['module']) else None,
                'machine': str(row['machine']) if pd.notna(row['machine']) else None,
                'blocking_session': int(row['blocking_session']) if pd.notna(row['blocking_session']) else None,
                'blocking_session_serial#': int(row['blocking_session_serial#']) if pd.notna(row['blocking_session_serial#']) else None,
                'blocking_instance': int(row['blocking_inst_id']) if pd.notna(row['blocking_inst_id']) else None,
                'anomaly_score': float(scores[idx]),
                'memory_usage': {
                    'pga_gb': pga_gb,
                    'temp_space_gb': temp_gb
                },
                'blocking_chain': blocking_chain
            }

            # Determine severity
            severity, reasons = self.determine_severity(detail)
            detail['severity'] = severity
            detail['severity_reasons'] = reasons

            session_details.append(detail)

        return session_details, len(seen_sessions)

    def monitor(self, interval_seconds=None):
        interval = interval_seconds or MONITORING_CONFIG['interval_seconds']

        while True:
            try:
                current_ash, connection = self.collect_current_ash()

                if current_ash.empty:
                    print("No current ASH data available")
                    time.sleep(interval)
                    continue

                print("Preprocessing current ASH data...")
                features = preprocess_ash_data(current_ash)
                features = features.reindex(columns=self.feature_columns, fill_value=0)

                print("Detecting anomalies...")
                scaled_features = self.scaler.transform(features)
                predictions = self.model.predict(scaled_features)
                scores = -self.model.score_samples(scaled_features)

                session_details, unique_anomaly_count = self.get_session_details(
                    current_ash, predictions, scores, connection
                )

                results = {
                    'timestamp': datetime.now().isoformat(),
                    'anomalies': unique_anomaly_count,
                    'alert_threshold': float(MONITORING_CONFIG['alert_threshold']),
                    'anomalous_sessions': session_details
                }

                # Save results using the results manager
                self.results_manager.save_results(results)

                print(f"Monitoring results saved. Found {unique_anomaly_count} unique anomalous sessions.")
                if session_details:
                    print("\nAnomalous Sessions:")
                    for detail in session_details:
                        # Print overall assessment first
                        print(colorize(f"Performance Impact: {detail['severity']}", detail['severity']))
                        
                        # Print analysis details
                        print("Analysis:")
                        for reason in detail['severity_reasons']:
                            print(f"- {reason}")
                        
                        # Session details
                        print(f"\nSession Detail (sid,serial#,@inst): {detail['session_id']},{detail['session_serial#']},@{detail['instance']}")
                        print(f"Timestamp: {detail['timestamp']}")
                        print(f"User: {detail['username']}")
                        print(f"Program: {detail['program']}")
                        
                        # Memory usage
                        mem = detail['memory_usage']
                        print(f"Memory Usage: PGA {mem['pga_gb']:.2f} GB, Temp Space {mem['temp_space_gb']:.2f} GB")
                        
                        # SQL and wait information
                        if detail['sql_id']:
                            print(f"SQL ID: {detail['sql_id']}")
                            if detail['sql_text']:
                                print(f"SQL Text: {detail['sql_text'][:100]}...")
                        if detail['wait_class']:
                            print(f"Wait Class: {detail['wait_class']}")
                        if detail['event']:
                            print(f"Event: {detail['event']}")
                            
                        # Blocking chain information
                        if detail['blocking_chain']:
                            print("\nBlocking Chain:")
                            for i, blocked in enumerate(detail['blocking_chain'], 1):
                                print(f"  {i}. Session {blocked['session_id']},{blocked['session_serial#']}@{blocked['instance_number']}")
                                print(f"     User: {blocked['username']}")
                                if blocked['wait_class']:
                                    print(f"     Wait Class: {blocked['wait_class']}")
                                if blocked['event']:
                                    print(f"     Event: {blocked['event']}")
                                if blocked['sql_id']:
                                    print(f"     SQL ID: {blocked['sql_id']}")
                                    if blocked['sql_text']:
                                        print(f"     SQL Text: {blocked['sql_text']}")
                                mem = blocked['memory_usage']
                                print(f"     Memory Usage: PGA {mem['pga_gb']:.2f} GB, Temp Space {mem['temp_space_gb']:.2f} GB")
                        print()

                connection.close()
                time.sleep(interval)

            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                print()
                if 'connection' in locals():
                    try:
                        connection.close()
                    except:
                        pass
                time.sleep(60)  # Wait longer on error

if __name__ == "__main__":
    try:
        print("Starting ASH monitoring...")
        
        # Get PDB-specific paths and ensure directories exist
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        monitor = RACASHMonitor(
            paths['model_file'],
            paths['scaler_file'],
            paths['feature_columns_file'],
            paths['monitoring_dir']
        )
        print("Monitor initialized. Starting monitoring loop...")
        monitor.monitor()
    except Exception as e:
        print(f"Failed to start monitoring: {str(e)}")
        raise