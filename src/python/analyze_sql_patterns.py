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
        self.df = pd.read_csv(data_file)
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
        # Group by SQL_ID and calculate metrics
        sql_metrics = self.df.groupby('sql_id').agg({
            'session_count': 'sum',
            'sample_time': ['min', 'max', 'count'],
            'wait_class': lambda x: x.value_counts().to_dict(),
            'event': lambda x: x.value_counts().to_dict(),
            'module': lambda x: x.value_counts().to_dict(),
            'program': lambda x: x.value_counts().to_dict()
        }).reset_index()

        # Rename columns for clarity
        sql_metrics.columns = [
            'sql_id', 'total_sessions', 'first_seen', 'last_seen',
            'samples', 'wait_classes', 'events', 'modules', 'programs'
        ]

        # Calculate execution time and frequency
        sql_metrics['execution_period'] = (
            pd.to_datetime(sql_metrics['last_seen']) - 
            pd.to_datetime(sql_metrics['first_seen'])
        ).dt.total_seconds() / 3600  # Convert to hours

        # Calculate average active sessions
        sql_metrics['average_active_sessions'] = sql_metrics['total_sessions'] / sql_metrics['samples']

        return sql_metrics

    def analyze_sql_wait_patterns(self):
        """Analyze wait patterns for each SQL"""
        # Group by SQL_ID and wait_class
        sql_waits = self.df.groupby(['sql_id', 'wait_class'])['session_count'].sum().reset_index()
        
        # Pivot to get wait classes as columns
        wait_patterns = sql_waits.pivot(
            index='sql_id',
            columns='wait_class',
            values='session_count'
        ).fillna(0)
        
        # Calculate percentages
        row_sums = wait_patterns.sum(axis=1)
        wait_patterns_pct = wait_patterns.div(row_sums, axis=0) * 100
        
        return wait_patterns, wait_patterns_pct

    def identify_sql_clusters(self, wait_patterns_pct, n_clusters=5):
        """Identify SQL clusters based on wait patterns"""
        from sklearn.cluster import KMeans
        
        # Handle empty dataframe
        if wait_patterns_pct.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(wait_patterns_pct)), random_state=42)
        clusters = kmeans.fit_predict(wait_patterns_pct.fillna(0))
        
        # Create cluster profiles
        cluster_profiles = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=wait_patterns_pct.columns
        )
        
        # Add cluster assignments to wait patterns
        wait_patterns_pct['cluster'] = clusters
        
        return wait_patterns_pct, cluster_profiles

    def analyze_temporal_patterns(self):
        """Analyze SQL execution patterns over time"""
        # Create hourly aggregations
        self.df['hour'] = self.df['sample_time'].dt.hour
        temporal_patterns = self.df.groupby(['sql_id', 'hour'])['session_count'].sum().reset_index()
        
        # Pivot to get hours as columns
        temporal_matrix = temporal_patterns.pivot(
            index='sql_id',
            columns='hour',
            values='session_count'
        ).fillna(0)
        
        return temporal_matrix

    def generate_report(self):
        """Generate comprehensive SQL analysis report"""
        # Get SQL patterns
        sql_metrics = self.analyze_sql_patterns()
        wait_patterns, wait_patterns_pct = self.analyze_sql_wait_patterns()
        
        # Get top SQL IDs first
        top_sql = sql_metrics.nlargest(10, 'average_active_sessions')
        
        # Ensure we only use SQL IDs that exist in both datasets
        common_sql_ids = set(top_sql['sql_id']).intersection(set(wait_patterns_pct.index))
        top_sql = top_sql[top_sql['sql_id'].isin(common_sql_ids)]
        
        # Only proceed with clustering if we have data
        if not wait_patterns_pct.empty:
            wait_patterns_pct, cluster_profiles = self.identify_sql_clusters(wait_patterns_pct)
        else:
            cluster_profiles = pd.DataFrame()
        
        temporal_matrix = self.analyze_temporal_patterns()

        # Create visualizations only if we have data
        if not top_sql.empty:
            # 1. Top SQL by Active Sessions
            plt.figure(figsize=(15, 7))
            sns.barplot(data=top_sql, x='sql_id', y='average_active_sessions')
            plt.title('Top 10 SQL by Average Active Sessions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.save_plot('top_sql_active_sessions')

            # 2. Wait Class Distribution Heatmap
            if not wait_patterns_pct.empty:
                plt.figure(figsize=(15, 8))
                top_sql_waits = wait_patterns_pct.loc[top_sql['sql_id']]
                sns.heatmap(top_sql_waits, cmap='YlOrRd', annot=True, fmt='.1f')
                plt.title('Wait Class Distribution for Top SQL')
                plt.tight_layout()
                self.save_plot('sql_wait_distribution')

            # 3. SQL Execution Timeline
            if not temporal_matrix.empty:
                plt.figure(figsize=(15, 7))
                temporal_data = temporal_matrix.loc[temporal_matrix.index.isin(top_sql['sql_id'])]
                sns.heatmap(temporal_data, cmap='YlOrRd', annot=True, fmt='.1f')
                plt.title('SQL Execution Patterns by Hour')
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
                        'primary_wait_class': max(row['wait_classes'].items(), key=lambda x: x[1])[0] if row['wait_classes'] else 'None',
                        'primary_module': max(row['modules'].items(), key=lambda x: x[1])[0] if row['modules'] else 'None'
                    }
                    for _, row in top_sql.iterrows()
                ]
            }
        }

        # Add wait patterns only if we have cluster data
        if not cluster_profiles.empty:
            report['wait_patterns'] = {
                'cluster_profiles': [
                    {
                        'cluster_id': i,
                        'dominant_wait': profile.idxmax(),
                        'wait_distribution': profile.to_dict()
                    }
                    for i, profile in cluster_profiles.iterrows()
                ]
            }

        # Add temporal patterns if we have data
        if not temporal_matrix.empty:
            report['temporal_patterns'] = {
                'peak_hours': [
                    {
                        'hour': hour,
                        'total_sessions': int(temporal_matrix[hour].sum())
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
                print(f"- Primary Module: {sql['primary_module']}")
                print(f"- Execution Period: {sql['execution_period_hours']:.1f} hours")
        
        if 'wait_patterns' in report:
            print("\nSQL Clustering Analysis:")
            for profile in report['wait_patterns']['cluster_profiles']:
                print(f"\nCluster {profile['cluster_id']}:")
                print(f"- Dominant Wait Class: {profile['dominant_wait']}")
                waits = sorted(profile['wait_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
                print("- Top Wait Classes:")
                for wait, pct in waits:
                    print(f"  * {wait}: {pct:.1f}%")
        
        if 'temporal_patterns' in report:
            print("\nPeak Execution Hours:")
            for peak in report['temporal_patterns']['peak_hours']:
                print(f"- Hour {peak['hour']:02d}:00 - {peak['total_sessions']} total sessions")
        
        print("\nDetailed Analysis Files:")
        print(f"- Full report: {os.path.join(analyzer.analysis_dir, 'sql_analysis.json')}")
        print(f"- Visualizations: {analyzer.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()