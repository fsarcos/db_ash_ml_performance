import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import re
from config import get_pdb_directories, ensure_directories_exist

class SQLPatternAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with ASH data"""
        self.df = pd.read_csv(f"{data_file}.gz", compression='gzip')
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        
        # Filter out rows with null SQL_IDs
        self.df = self.df[self.df['sql_id'].notna()]
        
        # Setup directories
        paths = get_pdb_directories()
        self.analysis_dir = os.path.join(paths['awr_data_dir'], 'sql_analysis')
        self.viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def save_plot(self, name, dpi=300):
        """Save the current plot with high resolution"""
        plt.savefig(os.path.join(self.viz_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def analyze_sql_patterns(self):
        """Analyze SQL execution patterns"""
        # Calculate time range for AAS calculation
        time_range = (self.df['sample_time'].max() - self.df['sample_time'].min()).total_seconds()
        
        # Group by SQL_ID and calculate metrics
        sql_metrics = self.df.groupby('sql_id').agg({
            'session_count': 'sum',
            'sample_time': ['min', 'max', 'count'],
            'wait_class': lambda x: x.value_counts().to_dict(),
            'event_extended': lambda x: x.value_counts().to_dict(),
            'module': lambda x: x.value_counts().to_dict(),
            'program': lambda x: x.value_counts().to_dict(),
            'pga_allocated': ['mean', 'max'],  # Memory usage
            'temp_space_allocated': ['mean', 'max']  # Disk usage
        }).reset_index()

        # Rename columns for clarity
        sql_metrics.columns = [
            'sql_id', 'total_sessions', 'first_seen', 'last_seen',
            'samples', 'wait_classes', 'events', 'modules', 'programs',
            'avg_pga', 'max_pga', 'avg_temp', 'max_temp'
        ]

        # Convert PGA (memory) to GB
        for col in ['avg_pga', 'max_pga']:
            sql_metrics[f'{col}_gb'] = sql_metrics[col] / 1024 / 1024 / 1024

        # Convert temp space (disk) to GB
        for col in ['avg_temp', 'max_temp']:
            sql_metrics[f'{col}_gb'] = sql_metrics[col] / 1024 / 1024 / 1024

        # Calculate execution time and frequency
        sql_metrics['execution_period'] = (
            pd.to_datetime(sql_metrics['last_seen']) - 
            pd.to_datetime(sql_metrics['first_seen'])
        ).dt.total_seconds() / 3600  # Convert to hours

        # Calculate AAS (Average Active Sessions)
        # AAS = total_sessions * (10 seconds per sample) / total_seconds_in_period
        sql_metrics['average_active_sessions'] = (sql_metrics['total_sessions'] * 10) / time_range

        return sql_metrics

    def analyze_sql_wait_patterns(self):
        """Analyze wait patterns for each SQL"""
        # Group by SQL_ID and wait_class for session counts
        sql_waits = self.df.groupby(['sql_id', 'wait_class'])['session_count'].sum().reset_index()
        
        # Create pivot tables for wait patterns
        wait_patterns = sql_waits.pivot(
            index='sql_id',
            columns='wait_class',
            values='session_count'
        ).fillna(0)
        
        # Calculate percentages
        row_sums = wait_patterns.sum(axis=1)
        wait_patterns_pct = wait_patterns.div(row_sums, axis=0) * 100
        
        # Get memory metrics separately
        memory_metrics = self.df.groupby(['sql_id', 'wait_class']).agg({
            'pga_allocated': ['mean', 'max']  # Memory only
        }).reset_index()
        
        # Get disk metrics separately
        disk_metrics = self.df.groupby(['sql_id', 'wait_class']).agg({
            'temp_space_allocated': ['mean', 'max']  # Disk only
        }).reset_index()
        
        # Convert memory metrics to GB
        memory_metrics['pga_gb'] = memory_metrics['pga_allocated']['mean'] / 1024 / 1024 / 1024
        memory_metrics['pga_max_gb'] = memory_metrics['pga_allocated']['max'] / 1024 / 1024 / 1024
        
        # Convert disk metrics to GB
        disk_metrics['temp_space_gb'] = disk_metrics['temp_space_allocated']['mean'] / 1024 / 1024 / 1024
        disk_metrics['temp_space_max_gb'] = disk_metrics['temp_space_allocated']['max'] / 1024 / 1024 / 1024
        
        return wait_patterns, wait_patterns_pct, memory_metrics, disk_metrics

    def analyze_temporal_patterns(self):
        """Analyze SQL execution patterns over time"""
        # Calculate time range for each hour for AAS calculation
        self.df['hour'] = self.df['sample_time'].dt.hour
        time_by_hour = self.df.groupby(['sql_id', 'hour'])['sample_time'].agg(['min', 'max'])
        time_by_hour['seconds'] = (time_by_hour['max'] - time_by_hour['min']).dt.total_seconds()
        
        # Create hourly aggregations for session count
        session_patterns = self.df.groupby(['sql_id', 'hour'])['session_count'].sum().reset_index()
        
        # Calculate AAS by hour
        session_patterns = session_patterns.merge(
            time_by_hour.reset_index()[['sql_id', 'hour', 'seconds']],
            on=['sql_id', 'hour']
        )
        session_patterns['aas'] = (session_patterns['session_count'] * 10) / session_patterns['seconds']
        
        # Create pivot for AAS
        temporal_matrix = session_patterns.pivot(
            index='sql_id',
            columns='hour',
            values='aas'
        ).fillna(0)
        
        # Get memory metrics separately
        memory_patterns = self.df.groupby(['sql_id', 'hour']).agg({
            'pga_allocated': ['mean', 'max']  # Memory only
        }).reset_index()
        
        # Get disk metrics separately
        disk_patterns = self.df.groupby(['sql_id', 'hour']).agg({
            'temp_space_allocated': ['mean', 'max']  # Disk only
        }).reset_index()
        
        # Convert memory metrics to GB
        memory_patterns['pga_gb'] = memory_patterns['pga_allocated']['mean'] / 1024 / 1024 / 1024
        memory_patterns['pga_max_gb'] = memory_patterns['pga_allocated']['max'] / 1024 / 1024 / 1024
        
        # Convert disk metrics to GB
        disk_patterns['temp_space_gb'] = disk_patterns['temp_space_allocated']['mean'] / 1024 / 1024 / 1024
        disk_patterns['temp_space_max_gb'] = disk_patterns['temp_space_allocated']['max'] / 1024 / 1024 / 1024
        
        return temporal_matrix, memory_patterns, disk_patterns

    def generate_report(self):
        """Generate comprehensive SQL analysis report"""
        # Get SQL patterns
        sql_metrics = self.analyze_sql_patterns()
        wait_patterns, wait_patterns_pct, memory_metrics, disk_metrics = self.analyze_sql_wait_patterns()
        temporal_matrix, memory_patterns, disk_patterns = self.analyze_temporal_patterns()
        
        # Get top SQL IDs first
        top_sql = sql_metrics.nlargest(10, 'average_active_sessions')
        
        # Ensure we only use SQL IDs that exist in both datasets
        common_sql_ids = set(top_sql['sql_id']).intersection(set(wait_patterns_pct.index))
        top_sql = top_sql[top_sql['sql_id'].isin(common_sql_ids)]

        # Create visualizations
        if not top_sql.empty:
            # 1. Top SQL by Active Sessions
            plt.figure(figsize=(15, 7))
            sns.barplot(data=top_sql, x='sql_id', y='average_active_sessions')
            plt.title('Top 10 SQL by Average Active Sessions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.save_plot('top_sql_active_sessions')

            # 2. Memory and Disk Usage by Top SQL
            plt.figure(figsize=(15, 7))
            memory_data = pd.melt(
                top_sql,
                id_vars=['sql_id'],
                value_vars=['avg_pga_gb', 'avg_temp_gb'],
                var_name='Resource Type',
                value_name='GB'
            )
            sns.barplot(data=memory_data, x='sql_id', y='GB', hue='Resource Type')
            plt.title('Resource Usage by Top SQL')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.save_plot('top_sql_resource_usage')

            # 3. Wait Class Distribution Heatmap
            if not wait_patterns_pct.empty:
                plt.figure(figsize=(15, 8))
                top_sql_waits = wait_patterns_pct.loc[top_sql['sql_id']]
                sns.heatmap(top_sql_waits, cmap='YlOrRd', annot=True, fmt='.1f')
                plt.title('Wait Class Distribution for Top SQL')
                plt.tight_layout()
                self.save_plot('sql_wait_distribution')

            # 4. SQL Execution Timeline
            if not temporal_matrix.empty:
                plt.figure(figsize=(15, 7))
                temporal_data = temporal_matrix.loc[temporal_matrix.index.isin(top_sql['sql_id'])]
                sns.heatmap(temporal_data, cmap='YlOrRd', annot=True, fmt='.1f')
                plt.title('SQL Average Active Sessions by Hour')
                plt.tight_layout()
                self.save_plot('sql_temporal_patterns')

        # Prepare report data
        report = {
            'analysis_period': {
                'start': self.df['sample_time'].min().isoformat(),
                'end': self.df['sample_time'].max().isoformat(),
                'total_days': len(self.df['sample_time'].dt.date.unique())
            },
            'sql_patterns': {
                'total_unique_sql': len(sql_metrics),
                'top_sql': [
                    {
                        'sql_id': row['sql_id'],
                        'average_active_sessions': float(row['average_active_sessions']),
                        'execution_period_hours': float(row['execution_period']),
                        'total_samples': int(row['samples']),
                        'resource_usage': {
                            'memory': {
                                'avg_pga_gb': float(row['avg_pga_gb']),
                                'max_pga_gb': float(row['max_pga_gb'])
                            },
                            'disk': {
                                'avg_temp_gb': float(row['avg_temp_gb']),
                                'max_temp_gb': float(row['max_temp_gb'])
                            }
                        },
                        'primary_wait_class': max(row['wait_classes'].items(), key=lambda x: x[1])[0] if row['wait_classes'] else 'None',
                        'primary_event': max(row['events'].items(), key=lambda x: x[1])[0] if row['events'] else 'None',
                        'primary_module': max(row['modules'].items(), key=lambda x: x[1])[0] if row['modules'] else 'None'
                    }
                    for _, row in top_sql.iterrows()
                ]
            }
        }

        # Add wait patterns only if we have cluster data
        if not wait_patterns_pct.empty:
            report['wait_patterns'] = {
                'sql_wait_patterns': [
                    {
                        'sql_id': sql_id,
                        'wait_distribution': pattern.to_dict(),
                        'resource_usage': {
                            'memory': {
                                'pga_gb': float(memory_metrics[
                                    (memory_metrics['sql_id'] == sql_id)
                                ]['pga_gb'].mean()),
                                'pga_max_gb': float(memory_metrics[
                                    (memory_metrics['sql_id'] == sql_id)
                                ]['pga_max_gb'].mean())
                            },
                            'disk': {
                                'temp_gb': float(disk_metrics[
                                    (disk_metrics['sql_id'] == sql_id)
                                ]['temp_space_gb'].mean()),
                                'temp_max_gb': float(disk_metrics[
                                    (disk_metrics['sql_id'] == sql_id)
                                ]['temp_space_max_gb'].mean())
                            }
                        }
                    }
                    for sql_id, pattern in wait_patterns_pct.iterrows()
                    if sql_id in top_sql['sql_id'].values
                ]
            }

        # Add temporal patterns if we have data
        if not temporal_matrix.empty:
            report['temporal_patterns'] = {
                'peak_hours': [
                    {
                        'hour': hour,
                        'average_active_sessions': float(temporal_matrix[hour].mean()),
                        'resource_usage': {
                            'memory': {
                                'avg_pga_gb': float(memory_patterns[
                                    memory_patterns['hour'] == hour
                                ]['pga_gb'].mean()),
                                'max_pga_gb': float(memory_patterns[
                                    memory_patterns['hour'] == hour
                                ]['pga_max_gb'].mean())
                            },
                            'disk': {
                                'avg_temp_gb': float(disk_patterns[
                                    disk_patterns['hour'] == hour
                                ]['temp_space_gb'].mean()),
                                'max_temp_gb': float(disk_patterns[
                                    disk_patterns['hour'] == hour
                                ]['temp_space_max_gb'].mean())
                            }
                        }
                    }
                    for hour in temporal_matrix.sum().nlargest(3).index
                ]
            }

        # Save detailed report
        output_file = os.path.join(self.analysis_dir, 'sql_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    try:
        print("Starting SQL pattern analysis...")
        
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        analyzer = SQLPatternAnalyzer(paths['historical_data_file'])
        report = analyzer.generate_report()
        
        # Print business-focused summary
        print("\nSQL Workload Analysis Summary")
        print("=" * 50)
        
        print(f"\nAnalysis Period: {report['analysis_period']['total_days']} days")
        print(f"From: {report['analysis_period']['start']}")
        print(f"To: {report['analysis_period']['end']}")
        
        print(f"\nTotal Unique SQL: {report['sql_patterns']['total_unique_sql']}")
        
        if report['sql_patterns']['top_sql']:
            print("\nTop 5 SQL by Average Active Sessions:")
            for sql in report['sql_patterns']['top_sql'][:5]:
                print(f"\nSQL_ID: {sql['sql_id']}")
                print(f"- Average Active Sessions: {sql['average_active_sessions']:.2f}")
                print(f"- Primary Wait Class: {sql['primary_wait_class']}")
                print(f"- Primary Event: {sql['primary_event']}")
                print(f"- Primary Module: {sql['primary_module']}")
                print(f"- Execution Period: {sql['execution_period_hours']:.1f} hours")
                print("- Resource Usage:")
                mem = sql['resource_usage']['memory']
                disk = sql['resource_usage']['disk']
                print(f"  * Memory (PGA): Avg {mem['avg_pga_gb']:.2f} GB, Peak {mem['max_pga_gb']:.2f} GB")
                print(f"  * Disk (Temp): Avg {disk['avg_temp_gb']:.2f} GB, Peak {disk['max_temp_gb']:.2f} GB")
        
        if 'temporal_patterns' in report:
            print("\nPeak Execution Hours:")
            for peak in report['temporal_patterns']['peak_hours']:
                print(f"- Hour {peak['hour']:02d}:00")
                print(f"  * Average Active Sessions: {peak['average_active_sessions']:.2f}")
                mem = peak['resource_usage']['memory']
                disk = peak['resource_usage']['disk']
                print(f"  * Memory (PGA): Avg {mem['avg_pga_gb']:.2f} GB, Peak {mem['max_pga_gb']:.2f} GB")
                print(f"  * Disk (Temp): Avg {disk['avg_temp_gb']:.2f} GB, Peak {disk['max_temp_gb']:.2f} GB")
        
        print("\nDetailed Analysis Files:")
        print(f"- Full report: {os.path.join(analyzer.analysis_dir, 'sql_analysis.json')}")
        print(f"- Visualizations: {analyzer.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()