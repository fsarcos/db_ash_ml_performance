import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from config import get_pdb_directories, ensure_directories_exist

class ContentionAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with ASH data"""
        self.df = pd.read_csv(data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        
        # Setup directories
        paths = get_pdb_directories()
        self.analysis_dir = os.path.join(paths['awr_data_dir'], 'contention_analysis')
        self.viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def save_plot(self, name, dpi=300):
        """Save the current plot with high resolution"""
        plt.savefig(os.path.join(self.viz_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def analyze_blocking_chains(self):
        """Analyze blocking session chains and their impact"""
        # Filter for sessions with blocking sessions
        blocking_df = self.df[self.df['blocking_session'].notna()].copy()
        
        if blocking_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create blocking chain metrics
        blocking_metrics = blocking_df.groupby(['sample_time', 'blocking_session']).agg({
            'session_count': ['sum', 'count'],  # Total sessions affected and count of blocked sessions
            'wait_class': lambda x: x.value_counts().index[0],  # Most common wait class
            'event': lambda x: x.value_counts().index[0]  # Most common event
        }).reset_index()
        
        # Flatten column names
        blocking_metrics.columns = ['sample_time', 'blocking_session', 'total_sessions', 
                                  'blocked_count', 'wait_class', 'event']
        
        # Calculate chain statistics
        chain_stats = blocking_metrics.groupby('blocking_session').agg({
            'blocked_count': ['mean', 'max'],  # Average and max blocked sessions
            'total_sessions': 'sum',  # Total impact
            'wait_class': lambda x: x.value_counts().index[0],
            'event': lambda x: x.value_counts().index[0]
        })
        
        return blocking_metrics, chain_stats

    def analyze_resource_contention(self):
        """Analyze resource contention patterns"""
        # Group by sample time and resource-related wait classes
        resource_waits = ['User I/O', 'System I/O', 'Concurrency', 'Application']
        resource_df = self.df[self.df['wait_class'].isin(resource_waits)].copy()
        
        if resource_df.empty:
            return pd.DataFrame()
        
        # Calculate resource contention metrics
        resource_metrics = resource_df.groupby(['sample_time', 'wait_class', 'event']).agg({
            'session_count': 'sum',
            'instance_number': 'nunique'
        }).reset_index()
        
        # Add hour for temporal analysis
        resource_metrics['hour'] = resource_metrics['sample_time'].dt.hour
        
        return resource_metrics

    def analyze_instance_skew(self):
        """Analyze workload skew across RAC instances"""
        # Calculate per-instance metrics
        instance_metrics = self.df.groupby(['sample_time', 'instance_number']).agg({
            'session_count': 'sum',
            'wait_class': lambda x: x.value_counts().to_dict(),
            'blocking_session': lambda x: x.notna().sum()
        }).reset_index()
        
        # Calculate skew metrics
        total_sessions = instance_metrics.groupby('sample_time')['session_count'].sum()
        instance_metrics['total_sessions'] = instance_metrics['sample_time'].map(total_sessions)
        instance_metrics['workload_percentage'] = (instance_metrics['session_count'] / 
                                                 instance_metrics['total_sessions'] * 100)
        
        return instance_metrics

    def generate_report(self):
        """Generate comprehensive contention analysis report"""
        # Get contention metrics
        blocking_metrics, chain_stats = self.analyze_blocking_chains()
        resource_metrics = self.analyze_resource_contention()
        instance_metrics = self.analyze_instance_skew()

        # Create visualizations
        if not blocking_metrics.empty:
            # Blocking Chain Impact
            plt.figure(figsize=(15, 7))
            blocking_by_time = blocking_metrics.groupby('sample_time')['total_sessions'].sum()
            plt.plot(blocking_by_time.index, blocking_by_time.values)
            plt.title('Blocking Session Impact Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Blocked Sessions')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            self.save_plot('blocking_impact')

        if not resource_metrics.empty:
            # Resource Contention Heatmap
            plt.figure(figsize=(15, 8))
            pivot_data = resource_metrics.pivot_table(
                index='hour',
                columns='wait_class',
                values='session_count',
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.1f')
            plt.title('Resource Contention by Hour')
            plt.tight_layout()
            self.save_plot('resource_contention_heatmap')

        if not instance_metrics.empty:
            # Instance Workload Distribution
            plt.figure(figsize=(15, 7))
            sns.boxplot(data=instance_metrics, x='instance_number', y='workload_percentage')
            plt.title('Workload Distribution Across RAC Instances')
            plt.xlabel('Instance Number')
            plt.ylabel('Workload Percentage')
            plt.grid(True, alpha=0.3)
            self.save_plot('instance_workload_distribution')

        # Prepare report data
        report = {
            'analysis_period': {
                'start': self.df['sample_time'].min().isoformat(),
                'end': self.df['sample_time'].max().isoformat(),
                'total_days': len(self.df['sample_time'].dt.date.unique())
            },
            'blocking_analysis': {
                'total_blocking_incidents': len(blocking_metrics) if not blocking_metrics.empty else 0,
                'top_blockers': [
                    {
                        'blocking_session': int(session),
                        'avg_blocked_sessions': float(stats['blocked_count']['mean']),
                        'max_blocked_sessions': float(stats['blocked_count']['max']),
                        'total_impact': float(stats['total_sessions']),
                        'primary_wait_class': str(stats['wait_class']),  # Convert to string
                        'primary_event': str(stats['event'])  # Convert to string
                    }
                    for session, stats in chain_stats.iterrows()
                ] if not chain_stats.empty else []
            },
            'resource_contention': {
                'top_contentions': [
                    {
                        'wait_class': str(wait_class),  # Convert to string
                        'event': str(event),  # Convert to string
                        'avg_sessions': float(metrics['session_count'].mean()),
                        'peak_sessions': float(metrics['session_count'].max()),
                        'instances_affected': int(metrics['instance_number'].max())
                    }
                    for (wait_class, event), metrics in resource_metrics.groupby(['wait_class', 'event'])
                ] if not resource_metrics.empty else []
            },
            'instance_analysis': {
                'instance_metrics': [
                    {
                        'instance_number': int(instance),
                        'avg_workload_percentage': float(metrics['workload_percentage'].mean()),
                        'peak_workload_percentage': float(metrics['workload_percentage'].max()),
                        'blocking_incidents': int(metrics['blocking_session'].sum()),
                        'wait_classes': {
                            str(k): float(v) for k, v in metrics['wait_class'].iloc[0].items()
                        } if not metrics['wait_class'].empty else {}
                    }
                    for instance, metrics in instance_metrics.groupby('instance_number')
                ] if not instance_metrics.empty else []
            }
        }

        # Save detailed report
        output_file = os.path.join(self.analysis_dir, 'contention_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    try:
        print("Starting contention analysis...")
        
        paths = get_pdb_directories()
        ensure_directories_exist(paths)
        
        analyzer = ContentionAnalyzer(paths['historical_data_file'])
        report = analyzer.generate_report()
        
        # Print business-focused summary
        print("\nResource Contention Analysis Summary")
        print("=" * 50)
        
        print(f"\nAnalysis Period: {report['analysis_period']['total_days']} days")
        print(f"From: {report['analysis_period']['start']}")
        print(f"To: {report['analysis_period']['end']}")
        
        if report['blocking_analysis']['top_blockers']:
            print("\nBlocking Session Analysis:")
            print(f"Total Blocking Incidents: {report['blocking_analysis']['total_blocking_incidents']}")
            print("\nTop Blocking Sessions:")
            for blocker in report['blocking_analysis']['top_blockers'][:3]:
                print(f"\nSession {blocker['blocking_session']}:")
                print(f"- Average Blocked Sessions: {blocker['avg_blocked_sessions']:.2f}")
                print(f"- Maximum Blocked Sessions: {blocker['max_blocked_sessions']}")
                print(f"- Primary Wait Class: {blocker['primary_wait_class']}")
                print(f"- Primary Event: {blocker['primary_event']}")
        
        if report['resource_contention']['top_contentions']:
            print("\nTop Resource Contentions:")
            for contention in sorted(report['resource_contention']['top_contentions'], 
                                   key=lambda x: x['avg_sessions'], reverse=True)[:3]:
                print(f"\n{contention['wait_class']} - {contention['event']}:")
                print(f"- Average Sessions: {contention['avg_sessions']:.2f}")
                print(f"- Peak Sessions: {contention['peak_sessions']}")
                print(f"- Instances Affected: {contention['instances_affected']}")
        
        if report['instance_analysis']['instance_metrics']:
            print("\nInstance Workload Distribution:")
            for instance in report['instance_analysis']['instance_metrics']:
                print(f"\nInstance {instance['instance_number']}:")
                print(f"- Average Workload: {instance['avg_workload_percentage']:.1f}%")
                print(f"- Peak Workload: {instance['peak_workload_percentage']:.1f}%")
                print(f"- Blocking Incidents: {instance['blocking_incidents']}")
                if instance['wait_classes']:
                    print("- Top Wait Classes:")
                    for wait_class, count in sorted(instance['wait_classes'].items(), 
                                                  key=lambda x: x[1], reverse=True)[:3]:
                        print(f"  * {wait_class}: {count:.1f}")
        
        print("\nDetailed Analysis Files:")
        print(f"- Full report: {os.path.join(analyzer.analysis_dir, 'contention_analysis.json')}")
        print(f"- Visualizations: {analyzer.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()