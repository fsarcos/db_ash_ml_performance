import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from config import get_pdb_directories, ensure_directories_exist

class OutlierAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with ASH data"""
        print("Loading data...")
        self.data_file = data_file
        
        # Setup directories
        paths = get_pdb_directories()
        self.analysis_dir = os.path.join(paths['awr_data_dir'], 'outlier_analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)

    def process_chunk(self, chunk):
        """Process a single chunk of data"""
        # Convert sample_time to datetime and extract components
        chunk['sample_time'] = pd.to_datetime(chunk['sample_time'])
        chunk['hour'] = chunk['sample_time'].dt.hour
        chunk['date'] = chunk['sample_time'].dt.strftime('%Y-%m-%d')
        return chunk

    def analyze_hourly_outliers(self):
        """Analyze outlier sessions by hour"""
        print("Processing data in chunks...")
        
        # Initialize storage for metrics
        hourly_metrics = []
        aggs = {}  # Initialize empty dictionary for aggregations
        
        # Process data in chunks
        chunk_size = 100000  # Adjust based on available memory
        
        for chunk in pd.read_csv(
            f"{self.data_file}.gz",
            compression='gzip',
            usecols=[
                'sample_time', 'session_count', 'instance_number',
                'wait_class', 'event', 'sql_id', 'module',
                'pga_allocated', 'temp_space_allocated'
            ],
            dtype={
                'session_count': 'float32',
                'instance_number': 'int32',
                'wait_class': 'category',
                'event': 'category',
                'sql_id': 'category',
                'module': 'category',
                'pga_allocated': 'float32',
                'temp_space_allocated': 'float32'
            },
            chunksize=chunk_size
        ):
            # Process chunk
            chunk = self.process_chunk(chunk)
            
            # Calculate metrics for this chunk
            chunk_metrics = chunk.groupby(['date', 'hour', 'sample_time']).agg({
                'session_count': 'sum',
                'instance_number': 'nunique',
                'pga_allocated': ['mean', 'max'],
                'temp_space_allocated': ['mean', 'max']
            }).reset_index()
            
            hourly_metrics.append(chunk_metrics)
            
            # Process aggregations by hour
            for hour in range(24):
                hour_data = chunk[chunk['hour'] == hour]
                if not hour_data.empty:
                    # Calculate wait class aggregations
                    wait_aggs = hour_data.groupby(['date', 'sample_time', 'wait_class'])['session_count'].sum()
                    event_aggs = hour_data.groupby(['date', 'sample_time', 'event'])['session_count'].sum()
                    sql_aggs = hour_data.groupby(['date', 'sample_time', 'sql_id'])['session_count'].sum()
                    module_aggs = hour_data.groupby(['date', 'sample_time', 'module'])['session_count'].sum()
                    
                    # Initialize hour dictionary if not exists
                    if hour not in aggs:
                        aggs[hour] = {
                            'wait_class': [wait_aggs],
                            'event': [event_aggs],
                            'sql_id': [sql_aggs],
                            'module': [module_aggs]
                        }
                    else:
                        # Append to existing lists
                        aggs[hour]['wait_class'].append(wait_aggs)
                        aggs[hour]['event'].append(event_aggs)
                        aggs[hour]['sql_id'].append(sql_aggs)
                        aggs[hour]['module'].append(module_aggs)
            
            # Clear chunk from memory
            del chunk
        
        # Combine all hourly metrics
        print("Combining metrics...")
        hourly_metrics = pd.concat(hourly_metrics, ignore_index=True)
        
        # Combine aggregations for each hour
        for hour in aggs:
            for agg_type in ['wait_class', 'event', 'sql_id', 'module']:
                if aggs[hour][agg_type]:
                    aggs[hour][agg_type] = pd.concat(aggs[hour][agg_type])
        
        # Process outliers by hour
        print("Analyzing outliers by hour...")
        outliers = {}
        
        for hour in range(24):
            print(f"Processing hour {hour:02d}...")
            hour_data = hourly_metrics[hourly_metrics['hour'] == hour]
            
            if hour_data.empty:
                continue
            
            # Calculate session counts (AAS)
            hour_data['aas'] = hour_data['session_count']
            
            # Calculate quartiles and IQR
            Q1 = hour_data['aas'].quantile(0.25)
            Q3 = hour_data['aas'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier thresholds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outlier_mask = (hour_data['aas'] < lower_bound) | (hour_data['aas'] > upper_bound)
            outlier_samples = hour_data[outlier_mask]
            
            if not outlier_samples.empty:
                # Convert memory metrics to GB
                gb_convert = 1024 * 1024 * 1024
                outlier_samples['pga_gb'] = outlier_samples[('pga_allocated', 'mean')] / gb_convert
                outlier_samples['temp_gb'] = outlier_samples[('temp_space_allocated', 'mean')] / gb_convert
                
                # Process outlier details
                outlier_details = []
                for _, row in outlier_samples.iterrows():
                    sample_time = pd.to_datetime(row['sample_time'])
                    date_str = row['date']
                    
                    # Get top items for this sample
                    def get_top_items(agg_type):
                        try:
                            if hour in aggs and aggs[hour][agg_type] is not None:
                                items = aggs[hour][agg_type].loc[date_str, sample_time]
                                return dict(items.nlargest(5).items())
                        except (KeyError, AttributeError):
                            pass
                        return {}
                    
                    outlier_details.append({
                        'timestamp': sample_time.isoformat(),
                        'date': date_str,
                        'active_sessions': float(row['aas']),
                        'instances': int(row['instance_number']),
                        'resource_usage': {
                            'pga_gb': float(row['pga_gb']),
                            'temp_gb': float(row['temp_gb'])
                        },
                        'top_wait_classes': get_top_items('wait_class'),
                        'top_events': get_top_items('event'),
                        'top_sql_ids': get_top_items('sql_id'),
                        'top_modules': get_top_items('module')
                    })
                
                outliers[hour] = {
                    'thresholds': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound),
                        'q1': float(Q1),
                        'q3': float(Q3),
                        'iqr': float(IQR)
                    },
                    'outlier_samples': outlier_details
                }
                
            # Clear hour data from memory
            del hour_data
        
        return outliers

    def generate_report(self):
        """Generate comprehensive outlier analysis report"""
        print("Generating report...")
        outliers = self.analyze_hourly_outliers()
        
        # Get analysis period from first chunk
        first_chunk = next(pd.read_csv(
            f"{self.data_file}.gz",
            compression='gzip',
            usecols=['sample_time'],
            chunksize=1
        ))
        first_chunk['sample_time'] = pd.to_datetime(first_chunk['sample_time'])
        start_time = first_chunk['sample_time'].min()
        
        # Get end time from last chunk
        for chunk in pd.read_csv(
            f"{self.data_file}.gz",
            compression='gzip',
            usecols=['sample_time'],
            chunksize=100000
        ):
            chunk['sample_time'] = pd.to_datetime(chunk['sample_time'])
            end_time = chunk['sample_time'].max()
        
        report = {
            'analysis_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'total_days': (end_time - start_time).days + 1
            },
            'hourly_outliers': {
                str(hour): {
                    'thresholds': data['thresholds'],
                    'sample_count': len(data['outlier_samples']),
                    'samples': sorted(
                        data['outlier_samples'],
                        key=lambda x: x['active_sessions'],
                        reverse=True
                    )
                }
                for hour, data in outliers.items()
            }
        }
        
        # Save detailed report
        output_file = os.path.join(self.analysis_dir, 'outlier_analysis.json')
        print(f"Saving report to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

def main():
    try:
        print("Starting outlier analysis...")
        
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        analyzer = OutlierAnalyzer(paths['historical_data_file'])
        report = analyzer.generate_report()
        
        # Print business-focused summary
        print("\nOutlier Analysis Summary")
        print("=" * 50)
        
        print(f"\nAnalysis Period: {report['analysis_period']['total_days']} days")
        print(f"From: {report['analysis_period']['start']}")
        print(f"To: {report['analysis_period']['end']}")
        
        for hour, data in report['hourly_outliers'].items():
            print(f"\nHour {hour}:00")
            print("-" * 20)
            print(f"Number of outlier samples: {data['sample_count']}")
            print(f"Thresholds: ")
            print(f"  - Normal range: {data['thresholds']['lower']:.2f} to {data['thresholds']['upper']:.2f} sessions")
            print(f"  - Q1: {data['thresholds']['q1']:.2f}")
            print(f"  - Q3: {data['thresholds']['q3']:.2f}")
            
            if data['samples']:
                print("\nTop 3 outlier samples:")
                for sample in data['samples'][:3]:
                    print(f"\nDate: {sample['date']}")
                    print(f"Active Sessions: {sample['active_sessions']:.2f}")
                    print(f"Number of Instances: {sample['instances']}")
                    print("Resource Usage:")
                    print(f"  - PGA: {sample['resource_usage']['pga_gb']:.2f} GB")
                    print(f"  - Temp: {sample['resource_usage']['temp_gb']:.2f} GB")
                    
                    if sample['top_wait_classes']:
                        print("Top Wait Classes:")
                        for wait_class, count in sample['top_wait_classes'].items():
                            print(f"  - {wait_class}: {count}")
                    
                    if sample['top_sql_ids']:
                        print("Top SQL IDs:")
                        for sql_id, count in sample['top_sql_ids'].items():
                            print(f"  - {sql_id}: {count}")
        
        print(f"\nDetailed analysis saved to: {os.path.join(analyzer.analysis_dir, 'outlier_analysis.json')}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()