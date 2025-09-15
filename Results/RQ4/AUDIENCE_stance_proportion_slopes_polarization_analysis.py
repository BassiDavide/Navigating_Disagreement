import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_dataset(main_folder_path):
    """Load dataset from JSONL files in nested folders"""
    
    print("Loading dataset...")
    all_comments = []
    
    for channel_folder in Path(main_folder_path).iterdir():
        if not channel_folder.is_dir():
            continue
            
        channel_name = channel_folder.name
        print(f"Processing: {channel_name}")
        
        for jsonl_file in channel_folder.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            comment = json.loads(line)
                            comment['Channel'] = channel_name
                            all_comments.append(comment)
            except Exception as e:
                print(f"  Error with {jsonl_file}: {e}")
                continue
    
    print(f"Loaded {len(all_comments):,} comments from {len(set(c['Channel'] for c in all_comments))} channels")
    return pd.DataFrame(all_comments)

def analyze_stance_stability(df):
    """Analyze stance distribution and stability by channel and year"""
    
    # Clean data
    df = df.dropna(subset=['predicted_stance', 'Timestamp'])
    df['year'] = pd.to_datetime(df['Timestamp']).dt.year
    df = df[df['year'] < 2025]
    
    print(f"Analysis dataset: {len(df):,} comments, {df['Channel'].nunique()} channels, years {df['year'].min()}-{df['year'].max()}")
    
    # Calculate yearly proportions by channel
    yearly_results = []
    
    for channel in df['Channel'].unique():
        channel_data = df[df['Channel'] == channel]
        leaning = channel_data['ChannelLeaning'].iloc[0]
        source = channel_data['Source'].iloc[0] if 'Source' in channel_data.columns else 'Unknown'
        
        for year in sorted(df['year'].unique()):
            year_data = channel_data[channel_data['year'] == year]
            if len(year_data) == 0:
                continue
                
            total = len(year_data)
            contra = (year_data['predicted_stance'] == 0).sum()
            neutral = (year_data['predicted_stance'] == 1).sum()
            pro = (year_data['predicted_stance'] == 2).sum()
            
            yearly_results.append({
                'Channel': channel,
                'Political_Leaning': leaning,
                'Source_Type': source,
                'Year': year,
                'Total_Comments': total,
                'Contra_Prop': contra / total,
                'Neutral_Prop': neutral / total,
                'Pro_Prop': pro / total
            })
    
    yearly_df = pd.DataFrame(yearly_results)
    
    # Calculate stability statistics
    stability_results = []
    
    for channel in yearly_df['Channel'].unique():
        channel_data = yearly_df[yearly_df['Channel'] == channel]
        
        if len(channel_data) < 5:  # Need at least 2 years
            continue
            
        leaning = channel_data['Political_Leaning'].iloc[0]
        source = channel_data['Source_Type'].iloc[0]
        
        # Variance for each stance
        contra_var = np.var(channel_data['Contra_Prop'])
        neutral_var = np.var(channel_data['Neutral_Prop'])
        pro_var = np.var(channel_data['Pro_Prop'])
        overall_var = (contra_var + neutral_var + pro_var) / 3
        
        # Coefficient of variation
        contra_cv = np.std(channel_data['Contra_Prop']) / np.mean(channel_data['Contra_Prop']) if np.mean(channel_data['Contra_Prop']) > 0 else 0
        neutral_cv = np.std(channel_data['Neutral_Prop']) / np.mean(channel_data['Neutral_Prop']) if np.mean(channel_data['Neutral_Prop']) > 0 else 0
        pro_cv = np.std(channel_data['Pro_Prop']) / np.mean(channel_data['Pro_Prop']) if np.mean(channel_data['Pro_Prop']) > 0 else 0
        
        # Trends (slopes) and significance
        years = channel_data['Year'].values
        contra_reg = stats.linregress(years, channel_data['Contra_Prop'])
        neutral_reg = stats.linregress(years, channel_data['Neutral_Prop'])
        pro_reg = stats.linregress(years, channel_data['Pro_Prop'])
        
        stability_results.append({
            'Channel': channel,
            'Political_Leaning': leaning,
            'Source_Type': source,
            'Years_Active': len(channel_data),
            'Total_Comments': channel_data['Total_Comments'].sum(),
            'Mean_Contra': np.mean(channel_data['Contra_Prop']),
            'Mean_Neutral': np.mean(channel_data['Neutral_Prop']),
            'Mean_Pro': np.mean(channel_data['Pro_Prop']),
            'Contra_Variance': contra_var,
            'Neutral_Variance': neutral_var,
            'Pro_Variance': pro_var,
            'Stability_Index': overall_var,
            'Contra_CV': contra_cv,
            'Neutral_CV': neutral_cv,
            'Pro_CV': pro_cv,
            'Contra_Slope': contra_reg.slope,
            'Contra_Slope_Pval': contra_reg.pvalue,
            'Neutral_Slope': neutral_reg.slope,
            'Neutral_Slope_Pval': neutral_reg.pvalue,
            'Pro_Slope': pro_reg.slope,
            'Pro_Slope_Pval': pro_reg.pvalue
        })
    
    stability_df = pd.DataFrame(stability_results)
    
    return yearly_df, stability_df

def analyze_cluster_groups(df):
    """Analyze stance distribution by cluster groups (Political_Leaning × Source_Type)"""
    
    # Clean data
    df = df.dropna(subset=['predicted_stance', 'Timestamp'])
    df['year'] = pd.to_datetime(df['Timestamp']).dt.year
    df = df[df['year'] < 2025]
    
    # Create cluster group identifier
    df['Cluster_Group'] = df['ChannelLeaning'] + '_' + df['Source']
    
    print(f"\nCluster Analysis - Groups found: {df['Cluster_Group'].unique()}")
    
    # Calculate yearly proportions by cluster group
    cluster_yearly_results = []
    
    for cluster in df['Cluster_Group'].unique():
        cluster_data = df[df['Cluster_Group'] == cluster]
        leaning = cluster_data['ChannelLeaning'].iloc[0]
        source = cluster_data['Source'].iloc[0]
        
        for year in sorted(df['year'].unique()):
            year_data = cluster_data[cluster_data['year'] == year]
            if len(year_data) == 0:
                continue
                
            total = len(year_data)
            contra = (year_data['predicted_stance'] == 0).sum()
            neutral = (year_data['predicted_stance'] == 1).sum()
            pro = (year_data['predicted_stance'] == 2).sum()
            
            cluster_yearly_results.append({
                'Cluster_Group': cluster,
                'Political_Leaning': leaning,
                'Source_Type': source,
                'Year': year,
                'Total_Comments': total,
                'Contra_Prop': contra / total,
                'Neutral_Prop': neutral / total,
                'Pro_Prop': pro / total,
                'Channels_Count': cluster_data['Channel'].nunique()
            })
    
    cluster_yearly_df = pd.DataFrame(cluster_yearly_results)
    
    # Calculate cluster stability statistics
    cluster_stability_results = []
    
    for cluster in cluster_yearly_df['Cluster_Group'].unique():
        cluster_data = cluster_yearly_df[cluster_yearly_df['Cluster_Group'] == cluster]
        
        if len(cluster_data) < 2:  # Need at least 2 years
            continue
            
        leaning = cluster_data['Political_Leaning'].iloc[0]
        source = cluster_data['Source_Type'].iloc[0]
        
        # Variance for each stance
        contra_var = np.var(cluster_data['Contra_Prop'])
        neutral_var = np.var(cluster_data['Neutral_Prop'])
        pro_var = np.var(cluster_data['Pro_Prop'])
        overall_var = (contra_var + neutral_var + pro_var) / 3
        
        # Coefficient of variation
        contra_cv = np.std(cluster_data['Contra_Prop']) / np.mean(cluster_data['Contra_Prop']) if np.mean(cluster_data['Contra_Prop']) > 0 else 0
        neutral_cv = np.std(cluster_data['Neutral_Prop']) / np.mean(cluster_data['Neutral_Prop']) if np.mean(cluster_data['Neutral_Prop']) > 0 else 0
        pro_cv = np.std(cluster_data['Pro_Prop']) / np.mean(cluster_data['Pro_Prop']) if np.mean(cluster_data['Pro_Prop']) > 0 else 0
        
        # Trends (slopes) and significance
        years = cluster_data['Year'].values
        contra_reg = stats.linregress(years, cluster_data['Contra_Prop'])
        neutral_reg = stats.linregress(years, cluster_data['Neutral_Prop'])
        pro_reg = stats.linregress(years, cluster_data['Pro_Prop'])
        
        cluster_stability_results.append({
            'Cluster_Group': cluster,
            'Political_Leaning': leaning,
            'Source_Type': source,
            'Years_Active': len(cluster_data),
            'Channels_Count': cluster_data['Channels_Count'].iloc[0],
            'Total_Comments': cluster_data['Total_Comments'].sum(),
            'Mean_Contra': np.mean(cluster_data['Contra_Prop']),
            'Mean_Neutral': np.mean(cluster_data['Neutral_Prop']),
            'Mean_Pro': np.mean(cluster_data['Pro_Prop']),
            'Contra_Variance': contra_var,
            'Neutral_Variance': neutral_var,
            'Pro_Variance': pro_var,
            'Stability_Index': overall_var,
            'Contra_CV': contra_cv,
            'Neutral_CV': neutral_cv,
            'Pro_CV': pro_cv,
            'Contra_Slope': contra_reg.slope,
            'Contra_Slope_Pval': contra_reg.pvalue,
            'Neutral_Slope': neutral_reg.slope,
            'Neutral_Slope_Pval': neutral_reg.pvalue,
            'Pro_Slope': pro_reg.slope,
            'Pro_Slope_Pval': pro_reg.pvalue
        })
    
    cluster_stability_df = pd.DataFrame(cluster_stability_results)
    
    return cluster_yearly_df, cluster_stability_df

def create_tables(yearly_df, stability_df):
    """Create formatted tables for appendix"""
    
    print("\n" + "="*80)
    print("TABLE A1: YEARLY STANCE DISTRIBUTION BY CHANNEL")
    print("="*80)
    
    for leaning in ['Left', 'Right']:
        print(f"\n{leaning}-Leaning Channels:")
        print("-" * 60)
        
        leaning_data = yearly_df[yearly_df['Political_Leaning'] == leaning]
        
        for channel in sorted(leaning_data['Channel'].unique()):
            channel_data = leaning_data[leaning_data['Channel'] == channel]
            source = channel_data['Source_Type'].iloc[0]
            
            print(f"\n{channel} ({source}):")
            
            # Format as percentages
            display_data = channel_data[['Year', 'Total_Comments', 'Contra_Prop', 'Neutral_Prop', 'Pro_Prop']].copy()
            display_data['Contra_%'] = (display_data['Contra_Prop'] * 100).round(1)
            display_data['Neutral_%'] = (display_data['Neutral_Prop'] * 100).round(1)
            display_data['Pro_%'] = (display_data['Pro_Prop'] * 100).round(1)
            
            print(display_data[['Year', 'Total_Comments', 'Contra_%', 'Neutral_%', 'Pro_%']].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE A2: CHANNEL STABILITY STATISTICS")
    print("="*80)
    
    # Format stability table
    stability_display = stability_df.copy()
    
    # Convert to percentages and round
    for col in ['Mean_Contra', 'Mean_Neutral', 'Mean_Pro']:
        stability_display[col] = (stability_display[col] * 100).round(1)
    
    for col in ['Contra_Variance', 'Neutral_Variance', 'Pro_Variance', 'Stability_Index']:
        stability_display[col] = stability_display[col].round(4)
    
    for col in ['Contra_CV', 'Neutral_CV', 'Pro_CV']:
        stability_display[col] = stability_display[col].round(2)
    
    display_cols = ['Channel', 'Political_Leaning', 'Source_Type', 'Years_Active', 'Total_Comments',
                   'Mean_Contra', 'Mean_Neutral', 'Mean_Pro', 'Stability_Index']
    
    print(stability_display[display_cols].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE A3: TREND ANALYSIS (ANNUAL SLOPES)")
    print("="*80)
    
    trend_display = stability_df[['Channel', 'Political_Leaning', 'Source_Type', 
                                 'Contra_Slope', 'Neutral_Slope', 'Pro_Slope']].copy()
    
    for col in ['Contra_Slope', 'Neutral_Slope', 'Pro_Slope']:
        trend_display[col] = trend_display[col].round(4)
    
    print(trend_display.to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE A4: CHANNEL TREND ANALYSIS WITH SIGNIFICANCE TESTS")
    print("="*80)
    
    trend_display = stability_df[['Channel', 'Political_Leaning', 'Contra_Slope', 'Contra_Slope_Pval',
                                'Neutral_Slope', 'Neutral_Slope_Pval', 
                                'Pro_Slope', 'Pro_Slope_Pval']].copy()
    
    # Round slopes and p-values
    for col in ['Contra_Slope', 'Neutral_Slope', 'Pro_Slope']:
        trend_display[col] = trend_display[col].round(4)
    
    for col in ['Contra_Slope_Pval', 'Neutral_Slope_Pval', 'Pro_Slope_Pval']:
        trend_display[col] = trend_display[col].round(4)
    
    # Add significance indicators
    trend_display['Contra_Sig'] = trend_display['Contra_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    trend_display['Neutral_Sig'] = trend_display['Neutral_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    trend_display['Pro_Sig'] = trend_display['Pro_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    
    print(trend_display.to_string(index=False))
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

def create_cluster_tables(cluster_yearly_df, cluster_stability_df):
    """Create formatted tables for cluster analysis"""
    
    print("\n\n" + "="*80)
    print("TABLE B1: YEARLY STANCE DISTRIBUTION BY CLUSTER GROUPS")
    print("="*80)
    
    for cluster in sorted(cluster_yearly_df['Cluster_Group'].unique()):
        cluster_data = cluster_yearly_df[cluster_yearly_df['Cluster_Group'] == cluster]
        leaning = cluster_data['Political_Leaning'].iloc[0]
        source = cluster_data['Source_Type'].iloc[0]
        channels_count = cluster_data['Channels_Count'].iloc[0]
        
        print(f"\n{leaning} {source} (n={channels_count} channels):")
        print("-" * 60)
        
        # Format as percentages
        display_data = cluster_data[['Year', 'Total_Comments', 'Contra_Prop', 'Neutral_Prop', 'Pro_Prop']].copy()
        display_data['Contra_%'] = (display_data['Contra_Prop'] * 100).round(1)
        display_data['Neutral_%'] = (display_data['Neutral_Prop'] * 100).round(1)
        display_data['Pro_%'] = (display_data['Pro_Prop'] * 100).round(1)
        
        print(display_data[['Year', 'Total_Comments', 'Contra_%', 'Neutral_%', 'Pro_%']].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE B2: CLUSTER GROUP STABILITY STATISTICS")
    print("="*80)
    
    # Format cluster stability table
    cluster_display = cluster_stability_df.copy()
    
    # Convert to percentages and round
    for col in ['Mean_Contra', 'Mean_Neutral', 'Mean_Pro']:
        cluster_display[col] = (cluster_display[col] * 100).round(1)
    
    for col in ['Contra_Variance', 'Neutral_Variance', 'Pro_Variance', 'Stability_Index']:
        cluster_display[col] = cluster_display[col].round(4)
    
    display_cols = ['Cluster_Group', 'Channels_Count', 'Years_Active', 'Total_Comments',
                   'Mean_Contra', 'Mean_Neutral', 'Mean_Pro', 'Stability_Index']
    
    print(cluster_display[display_cols].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE B3: CLUSTER TREND ANALYSIS WITH SIGNIFICANCE TESTS")
    print("="*80)
    
    trend_display = cluster_stability_df[['Cluster_Group', 'Contra_Slope', 'Contra_Slope_Pval',
                                        'Neutral_Slope', 'Neutral_Slope_Pval', 
                                        'Pro_Slope', 'Pro_Slope_Pval']].copy()
    
    # Round slopes and p-values
    for col in ['Contra_Slope', 'Neutral_Slope', 'Pro_Slope']:
        trend_display[col] = trend_display[col].round(4)
    
    for col in ['Contra_Slope_Pval', 'Neutral_Slope_Pval', 'Pro_Slope_Pval']:
        trend_display[col] = trend_display[col].round(4)
    
    # Add significance indicators
    trend_display['Contra_Sig'] = trend_display['Contra_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    trend_display['Neutral_Sig'] = trend_display['Neutral_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    trend_display['Pro_Sig'] = trend_display['Pro_Slope_Pval'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    
    print(trend_display.to_string(index=False))
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

def run_statistical_tests(stability_df):
    """Run key statistical tests"""
    
    print("\n\n" + "="*80)
    print("STATISTICAL TESTS - CHANNEL LEVEL")
    print("="*80)
    
    # Test 1: Left vs Right stability
    left_stability = stability_df[stability_df['Political_Leaning'] == 'Left']['Stability_Index']
    right_stability = stability_df[stability_df['Political_Leaning'] == 'Right']['Stability_Index']
    
    if len(left_stability) > 0 and len(right_stability) > 0:
        t_stat, p_val = stats.ttest_ind(left_stability, right_stability)
        print(f"Left vs Right Stability:")
        print(f"  Left mean: {left_stability.mean():.4f} (n={len(left_stability)})")
        print(f"  Right mean: {right_stability.mean():.4f} (n={len(right_stability)})")
        print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Test 2: Legacy Media vs Content Creator
    if 'Source_Type' in stability_df.columns:
        legacy = stability_df[stability_df['Source_Type'] == 'Legacy Media']['Stability_Index']
        creator = stability_df[stability_df['Source_Type'] == 'Content Creator']['Stability_Index']
        
        if len(legacy) > 0 and len(creator) > 0:
            t_stat, p_val = stats.ttest_ind(legacy, creator)
            print(f"\nLegacy Media vs Content Creator Stability:")
            print(f"  Legacy Media mean: {legacy.mean():.4f} (n={len(legacy)})")
            print(f"  Content Creator mean: {creator.mean():.4f} (n={len(creator)})")
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Summary statistics
    print(f"\nOverall Summary:")
    print(f"  Most stable: {stability_df.loc[stability_df['Stability_Index'].idxmin(), 'Channel']}")
    print(f"  Least stable: {stability_df.loc[stability_df['Stability_Index'].idxmax(), 'Channel']}")
    print(f"  Mean stability index: {stability_df['Stability_Index'].mean():.4f}")
    print(f"  Median stability index: {stability_df['Stability_Index'].median():.4f}")

def run_cluster_statistical_tests(cluster_stability_df):
    """Run statistical tests for cluster groups"""
    
    print("\n\n" + "="*80)
    print("STATISTICAL TESTS - CLUSTER LEVEL")
    print("="*80)
    
    # Test 1: Compare all four groups on stability
    groups = {}
    for _, row in cluster_stability_df.iterrows():
        group_name = f"{row['Political_Leaning']} {row['Source_Type']}"
        groups[group_name] = row['Stability_Index']
    
    print("Cluster Group Stability Comparison:")
    for group_name, stability in groups.items():
        print(f"  {group_name}: {stability:.4f}")
    
    # Test 2: Pairwise comparisons of mean stance proportions
    print(f"\nCluster Group Mean Stance Proportions:")
    for _, row in cluster_stability_df.iterrows():
        group_name = f"{row['Political_Leaning']} {row['Source_Type']}"
        print(f"  {group_name}:")
        print(f"    Contra: {row['Mean_Contra']:.1%}")
        print(f"    Neutral: {row['Mean_Neutral']:.1%}")
        print(f"    Pro: {row['Mean_Pro']:.1%}")
    
    # Test 3: Significant trends summary
    print(f"\nSignificant Trends (p < 0.05):")
    for _, row in cluster_stability_df.iterrows():
        group_name = f"{row['Political_Leaning']} {row['Source_Type']}"
        trends = []
        
        if row['Contra_Slope_Pval'] < 0.05:
            direction = "↑" if row['Contra_Slope'] > 0 else "↓"
            trends.append(f"Contra {direction} (p={row['Contra_Slope_Pval']:.3f})")
        
        if row['Neutral_Slope_Pval'] < 0.05:
            direction = "↑" if row['Neutral_Slope'] > 0 else "↓"
            trends.append(f"Neutral {direction} (p={row['Neutral_Slope_Pval']:.3f})")
        
        if row['Pro_Slope_Pval'] < 0.05:
            direction = "↑" if row['Pro_Slope'] > 0 else "↓"
            trends.append(f"Pro {direction} (p={row['Pro_Slope_Pval']:.3f})")
        
        if trends:
            print(f"  {group_name}: {', '.join(trends)}")
        else:
            print(f"  {group_name}: No significant trends")

def main(folder_path):
    """Run complete analysis"""
    
    # Load data
    df = load_dataset(folder_path)
    
    if len(df) == 0:
        print("No data found! Check your folder path.")
        return None, None, None, None
    
    # Individual channel analysis
    yearly_df, stability_df = analyze_stance_stability(df)
    
    # Cluster group analysis
    cluster_yearly_df, cluster_stability_df = analyze_cluster_groups(df)
    
    # Create tables
    create_tables(yearly_df, stability_df)
    create_cluster_tables(cluster_yearly_df, cluster_stability_df)
    
    # Statistical tests
    run_statistical_tests(stability_df)
    run_cluster_statistical_tests(cluster_stability_df)
    
    # Export all results
    yearly_df.to_csv('yearly_stance_distribution.csv', index=False)
    stability_df.to_csv('channel_stability_stats.csv', index=False)
    cluster_yearly_df.to_csv('cluster_yearly_stance_distribution.csv', index=False)
    cluster_stability_df.to_csv('cluster_stability_stats.csv', index=False)
    
    print(f"\nExported CSV files:")
    print("- yearly_stance_distribution.csv")
    print("- channel_stability_stats.csv")
    print("- cluster_yearly_stance_distribution.csv")
    print("- cluster_stability_stats.csv")
    
    return yearly_df, stability_df, cluster_yearly_df, cluster_stability_df

if __name__ == "__main__":
    # SET YOUR FOLDER PATH HERE
    folder_path = "Data/Stance_Dataset"  # Update this path accordingly
    
    print("Channel Stance Stability Analysis with Cluster Groups")
    print("=" * 60)
    print(f"Dataset folder: {folder_path}")
    print()
    
    # Uncomment to run:
    yearly_df, stability_df, cluster_yearly_df, cluster_stability_df = main(folder_path)
    
    #print("Set the folder_path variable and uncomment the last line to run the analysis.")
