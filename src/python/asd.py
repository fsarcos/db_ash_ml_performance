import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from config import get_pdb_directories, ensure_directories_exist

class HourlyPatternAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with ASH data"""
        self.df = pd.read_csv(f"{data_file}.gz", compression='gzip')
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        
        # Setup directories
        paths = get_pdb_directories()
        self.analysis_dir = os.path.join(paths['awr_data_dir'], 'hourly_analysis')
        self.viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def save_plot(self, name, dpi=300):
        """Save the current plot with high resolution"""
        plt.savefig(os.path.join(self.viz_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def analyze_hourly_patterns(self):
        """Analyze hourly workload patterns"""
        # Extract hour and date from sample_time
        self.df['hour'] = self.df['sample_time'].dt.hour
        self.df['date'] = self.df['sample_time'].dt.date
        
        # Count samples per hour
        sample_counts = self.df.groupby(['date', 'hour'])['sample_time'].nunique().reset_index()
        sample_counts.columns = ['date', 'hour', 'sample_count']
        
        # Calculate metrics for each hour - using separate aggregations to avoid MultiIndex
        hourly_sessions = self.df.groupby(['date', 'hour'])['session_count'].sum().reset_index()
        hourly_instances = self.df.groupby(['date', 'hour'])['instance_number'].nunique().reset_index()
        hourly_pga = self.df.groupby(['date', 'hour'])['pga_allocated'].agg(['mean', 'max']).reset_index()
        hourly_temp = self.df.groupby(['date', 'hour'])['temp_space_allocated'].agg(['mean', 'max']).reset_index()
        
        # Merge all metrics
        hourly_metrics = hourly_sessions.merge(hourly_instances, on=['date', 'hour'])
        hourly_metrics = hourly_metrics.merge(hourly_pga, on=['date', 'hour'])
        hourly_metrics = hourly_metrics.merge(hourly_temp, on=['date', 'hour'])
        hourly_metrics = hourly_metrics.merge(sample_counts, on=['date', 'hour'])
        
        # Calculate AAS (Average Active Sessions)
        # AAS = total_sessions / number_of_samples
        hourly_metrics['aas'] = hourly_metrics['session_count'] / hourly_metrics['sample_count']
        
        # Convert memory metrics to GB
        hourly_metrics['pga_gb'] = hourly_metrics['mean_x'] / 1024 / 1024 / 1024
        hourly_metrics['pga_max_gb'] = hourly_metrics['max_x'] / 1024 / 1024 / 1024
        hourly_metrics['temp_space_gb'] = hourly_metrics['mean_y'] / 1024 / 1024 / 1024
        hourly_metrics['temp_space_max_gb'] = hourly_metrics['max_y'] / 1024 / 1024 / 1024
        
        return hourly_metrics

    def analyze_wait_class_patterns(self):
        """Analyze wait class patterns by hour"""
        self.df['hour'] = self.df['sample_time'].dt.hour
        self.df['date'] = self.df['sample_time'].dt.date
        
        # Count samples per hour and wait class
        sample_counts = self.df.groupby(['date', 'hour', 'wait_class'])['sample_time'].nunique().reset_index()
        sample_counts.columns = ['date', 'hour', 'wait_class', 'sample_count']
        
        # Group by hour and wait_class - using separate aggregations
        wait_sessions = self.df.groupby(['date', 'hour', 'wait_class'])['session_count'].sum().reset_index()
        wait_pga = self.df.groupby(['date', 'hour', 'wait_class'])['pga_allocated'].agg(['mean', 'max']).reset_index()
        wait_temp = self.df.groupby(['date', 'hour', 'wait_class'])['temp_space_allocated'].agg(['mean', 'max']).reset_index()
        
        # Merge metrics
        wait_metrics = wait_sessions.merge(wait_pga, on=['date', 'hour', 'wait_class'])
        wait_metrics = wait_metrics.merge(wait_temp, on=['date', 'hour', 'wait_class'])
        wait_metrics = wait_metrics.merge(sample_counts, on=['date', 'hour', 'wait_class'])
        
        # Calculate AAS per wait class
        wait_metrics['aas'] = wait_metrics['session_count'] / wait_metrics['sample_count']
        
        # Calculate average AAS by hour and wait class across all days
        avg_wait_by_hour = wait_metrics.groupby(['hour', 'wait_class'])['aas'].mean().reset_index()
        
        # Pivot for heatmap
        wait_heatmap = avg_wait_by_hour.pivot(
            index='hour',
            columns='wait_class',
            values='aas'
        ).fillna(0)
        
        return wait_metrics, wait_heatmap

    def generate_report(self):
        """Generate comprehensive hourly analysis report"""
        # Get hourly patterns
        hourly_metrics = self.analyze_hourly_patterns()
        wait_metrics, wait_heatmap = self.analyze_wait_class_patterns()
        
        # Calculate average metrics by hour across all days
        avg_by_hour = hourly_metrics.groupby('hour').agg({
            'aas': ['mean', 'std', 'min', 'max'],
            'instance_number': 'mean',
            'pga_gb': ['mean', 'max'],
            'temp_space_gb': ['mean', 'max'],
            'sample_count': 'mean'  # Add sample count to the report
        })

        # Create visualizations
        # 1. Average Active Sessions by Hour
        plt.figure(figsize=(15, 7))
        sns.boxplot(data=hourly_metrics, x='hour', y='aas')
        plt.title('Average Active Sessions Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Active Sessions')
        plt.grid(True, alpha=0.3)
        self.save_plot('hourly_aas_distribution')

        # 2. Wait Class Heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(wait_heatmap, cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title('Average Active Sessions by Wait Class and Hour')
        plt.xlabel('Wait Class')
        plt.ylabel('Hour of Day')
        self.save_plot('wait_class_heatmap')

        # 3. Resource Usage by Hour
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Memory (PGA) Usage
        sns.boxplot(data=hourly_metrics, x='hour', y='pga_gb', ax=ax1)
        ax1.set_title('PGA Memory Usage by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('PGA Usage (GB)')
        ax1.grid(True, alpha=0.3)
        
        # Temp Space Usage
        sns.boxplot(data=hourly_metrics, x='hour', y='temp_space_gb', ax=ax2)
        ax2.set_title('Temporary Tablespace Usage by Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Temp Space Usage (GB)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot('resource_usage_by_hour')

        # Prepare report data
        report = {
            'analysis_period': {
                'start': self.df['sample_time'].min().isoformat(),
                'end': self.df['sample_time'].max().isoformat(),
                'total_days': len(self.df['date'].unique())
            },
            'hourly_patterns': {
                'by_hour': [
                    {
                        'hour': hour,
                        'active_sessions': {
                            'mean': float(stats['aas']['mean']),
                            'std': float(stats['aas']['std']),
                            'min': float(stats['aas']['min']),
                            'max': float(stats['aas']['max'])
                        },
                        'resource_usage': {
                            'memory': {
                                'avg_pga_gb': float(stats['pga_gb']['mean']),
                                'max_pga_gb': float(stats['pga_gb']['max'])
                            },
                            'disk': {
                                'avg_temp_gb': float(stats['temp_space_gb']['mean']),
                                'max_temp_gb': float(stats['temp_space_gb']['max'])
                            }
                        },
                        'instances': float(stats['instance_number']),
                        'avg_samples': float(stats['sample_count']['mean'])
                    }
                    for hour, stats in avg_by_hour.iterrows()
                ]
            },
            'peak_hours': sorted(
                [
                    {
                        'hour': hour,
                        'average_active_sessions': float(stats['aas']['mean']),
                        'peak_active_sessions': float(stats['aas']['max']),
                        'avg_samples': float(stats['sample_count']['mean'])
                    }
                    for hour, stats in avg_by_hour.iterrows()
                ],
                key=lambda x: x['average_active_sessions'],
                reverse=True
            )[:5],
            'quiet_hours': sorted(
                [
                    {
                        'hour': hour,
                        'average_active_sessions': float(stats['aas']['mean']),
                        'peak_active_sessions': float(stats['aas']['max']),
                        'avg_samples': float(stats['sample_count']['mean'])
                    }
                    for hour, stats in avg_by_hour.iterrows()
                ],
                key=lambda x: x['average_active_sessions']
            )[:5]
        }

        # Save detailed report
        output_file = os.path.join(self.analysis_dir, 'hourly_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    try:
        print("Starting hourly pattern analysis...")
        
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        analyzer = HourlyPatternAnalyzer(paths['historical_data_file'])
        report = analyzer.generate_report()
        
        # Print business-focused summary
        print("\nHourly Workload Analysis Summary")
        print("=" * 50)
        
        print(f"\nAnalysis Period: {report['analysis_period']['total_days']} days")
        print(f"From: {report['analysis_period']['start']}")
        print(f"To: {report['analysis_period']['end']}")
        
        print("\nPeak Hours (Top 5):")
        for peak in report['peak_hours']:
            print(f"- Hour {peak['hour']:02d}:00")
            print(f"  * Average Active Sessions: {peak['average_active_sessions']:.2f}")
            print(f"  * Peak Active Sessions: {peak['peak_active_sessions']:.2f}")
            print(f"  * Average Samples: {peak['avg_samples']:.1f}")
        
        print("\nQuiet Hours (Top 5):")
        for quiet in report['quiet_hours']:
            print(f"- Hour {quiet['hour']:02d}:00")
            print(f"  * Average Active Sessions: {quiet['average_active_sessions']:.2f}")
            print(f"  * Peak Active Sessions: {quiet['peak_active_sessions']:.2f}")
            print(f"  * Average Samples: {quiet['avg_samples']:.1f}")
        
        print("\nDetailed Analysis Files:")
        print(f"- Full report: {os.path.join(analyzer.analysis_dir, 'hourly_analysis.json')}")
        print(f"- Visualizations: {analyzer.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()