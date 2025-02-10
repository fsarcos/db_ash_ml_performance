import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from config import get_pdb_directories, ensure_directories_exist

class ASHPatternAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with ASH data"""
        self.df = pd.read_csv(data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        
        # Setup directories
        paths = get_pdb_directories()
        self.analysis_dir = os.path.join(paths['awr_data_dir'], 'analysis')
        self.viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def save_plot(self, name, dpi=300):
        """Save the current plot with high resolution"""
        plt.savefig(os.path.join(self.viz_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def analyze_workload_patterns(self):
        """Analyze database workload patterns"""
        # First, create date and hour columns
        self.df['date'] = self.df['sample_time'].dt.date
        self.df['hour'] = self.df['sample_time'].dt.hour
        
        # Group by date, hour, and sample_time to get the total sessions at each sample point
        sample_metrics = self.df.groupby(['date', 'hour', 'sample_time']).agg({
            'session_count': 'sum',  # Total sessions at this sample point
            'instance_number': 'nunique',  # Number of instances
            'wait_class': lambda x: x.value_counts().to_dict()  # Wait class distribution
        }).reset_index()
        
        # Now group by date and hour to get hourly statistics
        hourly_metrics = sample_metrics.groupby(['date', 'hour']).agg({
            'session_count': 'mean',  # Average sessions across all sample points in the hour
            'instance_number': 'first',  # Number of instances remains the same
        }).reset_index()
        
        # The session_count is now our AAS
        hourly_metrics['aas'] = hourly_metrics['session_count']
        
        # Add wait class breakdowns
        wait_classes = self.df['wait_class'].unique()
        for wait_class in wait_classes:
            if pd.notna(wait_class):
                mask = self.df['wait_class'] == wait_class
                wait_class_samples = self.df[mask].groupby(['date', 'hour', 'sample_time'])['session_count'].sum()
                wait_class_hourly = wait_class_samples.groupby(['date', 'hour']).mean()
                hourly_metrics[f'wait_{wait_class}'] = hourly_metrics.apply(
                    lambda row: wait_class_hourly.get((row['date'], row['hour']), 0),
                    axis=1
                )

        return hourly_metrics

    def analyze_wait_classes(self):
        """Analyze wait class distribution and patterns"""
        # First group by sample_time and wait_class to get point-in-time totals
        sample_wait_totals = self.df.groupby(['sample_time', 'wait_class'])['session_count'].sum().reset_index()
        
        # Then calculate averages over all samples
        wait_class_avg = sample_wait_totals.groupby('wait_class').agg({
            'session_count': 'mean'  # Average sessions for this wait class across all samples
        }).reset_index()
        
        # Calculate percentages
        total_sessions = wait_class_avg['session_count'].sum()
        wait_class_avg['percentage'] = (wait_class_avg['session_count'] / total_sessions * 100)
        wait_class_avg = wait_class_avg.rename(columns={'session_count': 'average_sessions'})
        wait_class_avg = wait_class_avg.sort_values('average_sessions', ascending=False)
        
        # Calculate hourly averages for heatmap
        hourly_wait_class = self.df.groupby(['hour', 'wait_class', 'sample_time'])['session_count'].sum().reset_index()
        wait_class_hourly = hourly_wait_class.groupby(['hour', 'wait_class'])['session_count'].mean().unstack()
        
        return wait_class_avg, wait_class_hourly

    def generate_report(self):
        """Generate comprehensive workload analysis report"""
        # Get workload patterns
        hourly_metrics = self.analyze_workload_patterns()
        wait_class_avg, wait_class_hourly = self.analyze_wait_classes()

        # Create AAS visualization
        plt.figure(figsize=(15, 7))
        sns.boxplot(data=hourly_metrics, x='hour', y='aas')
        plt.title('Average Active Sessions Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Active Sessions')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        self.save_plot('aas_distribution')

        # Create wait class heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(wait_class_hourly, cmap='YlOrRd', annot=True, fmt='.1f')
        plt.title('Average Concurrent Sessions by Wait Class and Hour')
        plt.xlabel('Wait Class')
        plt.ylabel('Hour of Day')
        self.save_plot('wait_class_heatmap')

        # Calculate key statistics
        aas_stats = hourly_metrics.groupby('hour')['aas'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)

        # Find peak hours (top 3)
        peak_hours = aas_stats.nlargest(3, 'mean')

        # Find maintenance windows (lowest 4 consecutive hours)
        rolling_mean = aas_stats['mean'].rolling(window=4, center=True).mean()
        best_start_hour = rolling_mean.idxmin()

        report = {
            'analysis_period': {
                'start': self.df['sample_time'].min().isoformat(),
                'end': self.df['sample_time'].max().isoformat(),
                'total_days': len(self.df['date'].unique()),
                'total_samples': len(self.df),
                'instances': self.df['instance_number'].nunique()
            },
            'workload_patterns': {
                'average_active_sessions': {
                    'overall_mean': float(hourly_metrics['aas'].mean()),
                    'overall_peak': float(hourly_metrics['aas'].max()),
                    'by_hour': {
                        str(hour): {
                            'mean': float(stats['mean']),
                            'std': float(stats['std']),
                            'min': float(stats['min']),
                            'max': float(stats['max'])
                        }
                        for hour, stats in aas_stats.iterrows()
                    }
                },
                'peak_hours': [
                    {
                        'hour': str(hour),
                        'average_sessions': float(stats['mean']),
                        'max_sessions': float(stats['max'])
                    }
                    for hour, stats in peak_hours.iterrows()
                ],
                'maintenance_window': {
                    'start_hour': str(best_start_hour),
                    'duration': '4 hours',
                    'average_sessions': float(rolling_mean[best_start_hour])
                }
            },
            'wait_analysis': {
                'top_wait_classes': [
                    {
                        'wait_class': row['wait_class'],
                        'average_sessions': float(row['average_sessions']),
                        'percentage': float(row['percentage'])
                    }
                    for _, row in wait_class_avg.head().iterrows()
                ]
            }
        }

        # Save detailed report
        output_file = os.path.join(self.analysis_dir, 'workload_analysis.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    try:
        print("Starting ASH pattern analysis...")
        
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        analyzer = ASHPatternAnalyzer(paths['historical_data_file'])
        report = analyzer.generate_report()
        
        # Print business-focused summary
        print("\nWorkload Analysis Summary")
        print("=" * 50)
        
        print(f"\nAnalysis Period: {report['analysis_period']['total_days']} days")
        print(f"From: {report['analysis_period']['start']}")
        print(f"To: {report['analysis_period']['end']}")
        print(f"Database Instances: {report['analysis_period']['instances']}")
        
        print("\nKey Metrics:")
        print(f"- Average Active Sessions (AAS): {report['workload_patterns']['average_active_sessions']['overall_mean']:.2f}")
        print(f"- Peak AAS: {report['workload_patterns']['average_active_sessions']['overall_peak']:.2f}")
        
        print("\nPeak Hours (Top 3):")
        for peak in report['workload_patterns']['peak_hours']:
            print(f"- {peak['hour']}:00 - Avg: {peak['average_sessions']:.2f}, Max: {peak['max_sessions']:.2f}")
        
        print("\nRecommended Maintenance Window:")
        mw = report['workload_patterns']['maintenance_window']
        print(f"- Start: {mw['start_hour']}:00 ({mw['duration']})")
        print(f"- Average Sessions: {mw['average_sessions']:.2f}")
        
        print("\nTop Wait Classes:")
        for wait in report['wait_analysis']['top_wait_classes']:
            print(f"- {wait['wait_class']}: {wait['average_sessions']:.2f} sessions ({wait['percentage']:.1f}%)")
        
        print("\nDetailed Analysis Files:")
        print(f"- Full report: {os.path.join(analyzer.analysis_dir, 'workload_analysis.json')}")
        print(f"- Visualizations: {analyzer.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    