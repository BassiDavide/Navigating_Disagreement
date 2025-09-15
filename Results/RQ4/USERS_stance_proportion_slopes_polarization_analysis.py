import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_user_dataset(json_file_path):
    """Load dataset from user groups JSON file"""
    
    print("Loading user dataset...")
    all_comments = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    
    excluded_groups = ["Left-ex", "Control-Ex"]
    group_counts = {}
    
    for username, user_info in user_data.items():
        group = user_info.get('group_classification', 'Unknown')
        
        # Skip excluded groups
        if group in excluded_groups:
            continue
            
        group_counts[group] = group_counts.get(group, 0) + 1
        
        for comment in user_info.get('comments', []):
            # Add user and group info to comment
            comment['Username'] = username
            comment['Group_Classification'] = group
            all_comments.append(comment)
    
    print(f"Loaded {len(all_comments):,} comments from {len(group_counts)} groups:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} users")
    
    df = pd.DataFrame(all_comments)
    
    # Filter out 2025
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    initial_count = len(df)
    df = df[df['year'] != 2025]
    print(f"Filtered out {initial_count - len(df):,} comments from 2025")
    
    return df

def analyze_user_stance_stability(df):
    """Analyze stance distribution and stability by user groups (user-level aggregation)"""
    
    # Clean data
    df = df.dropna(subset=['stance', 'timestamp'])
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    
    print(f"Analysis dataset: {len(df):,} comments, {df['Username'].nunique()} users, {df['Group_Classification'].nunique()} groups, years {df['year'].min()}-{df['year'].max()}")
    
    # Step 1: Calculate yearly proportions by USER first
    user_yearly_results = []
    
    for username in df['Username'].unique():
        user_data = df[df['Username'] == username]
        group = user_data['Group_Classification'].iloc[0]
        
        for year in sorted(df['year'].unique()):
            year_data = user_data[user_data['year'] == year]
            if len(year_data) == 0:
                continue
                
            total = len(year_data)
            contra = (year_data['stance'] == 0).sum()
            neutral = (year_data['stance'] == 1).sum()
            pro = (year_data['stance'] == 2).sum()
            
            user_yearly_results.append({
                'Username': username,
                'Group_Classification': group,
                'Year': year,
                'Total_Comments': total,
                'Contra_Prop': contra / total,
                'Neutral_Prop': neutral / total,
                'Pro_Prop': pro / total
            })
    
    user_yearly_df = pd.DataFrame(user_yearly_results)
    
    # Step 2: Aggregate by GROUP (averaging user proportions)
    group_yearly_results = []
    
    for group in user_yearly_df['Group_Classification'].unique():
        group_data = user_yearly_df[user_yearly_df['Group_Classification'] == group]
        
        for year in sorted(user_yearly_df['Year'].unique()):
            year_group_data = group_data[group_data['Year'] == year]
            if len(year_group_data) == 0:
                continue
            
            # Average user proportions (not weighted by comment count)
            avg_contra = np.mean(year_group_data['Contra_Prop'])
            avg_neutral = np.mean(year_group_data['Neutral_Prop'])
            avg_pro = np.mean(year_group_data['Pro_Prop'])
            
            group_yearly_results.append({
                'Group': group,
                'Year': year,
                'Users_Count': len(year_group_data),
                'Total_Comments': year_group_data['Total_Comments'].sum(),
                'Contra_Prop': avg_contra,
                'Neutral_Prop': avg_neutral,
                'Pro_Prop': avg_pro
            })
    
    yearly_df = pd.DataFrame(group_yearly_results)
    
    # Step 3: Calculate stability statistics by group
    stability_results = []
    
    for group in yearly_df['Group'].unique():
        group_data = yearly_df[yearly_df['Group'] == group]
        
        if len(group_data) < 2:  # Need at least 2 years
            continue
        
        # Variance for each stance
        contra_var = np.var(group_data['Contra_Prop'])
        neutral_var = np.var(group_data['Neutral_Prop'])
        pro_var = np.var(group_data['Pro_Prop'])
        overall_var = (contra_var + neutral_var + pro_var) / 3
        
        # Coefficient of variation
        contra_cv = np.std(group_data['Contra_Prop']) / np.mean(group_data['Contra_Prop']) if np.mean(group_data['Contra_Prop']) > 0 else 0
        neutral_cv = np.std(group_data['Neutral_Prop']) / np.mean(group_data['Neutral_Prop']) if np.mean(group_data['Neutral_Prop']) > 0 else 0
        pro_cv = np.std(group_data['Pro_Prop']) / np.mean(group_data['Pro_Prop']) if np.mean(group_data['Pro_Prop']) > 0 else 0
        
        # Trends (slopes) and significance
        years = group_data['Year'].values
        contra_reg = stats.linregress(years, group_data['Contra_Prop'])
        neutral_reg = stats.linregress(years, group_data['Neutral_Prop'])
        pro_reg = stats.linregress(years, group_data['Pro_Prop'])
        
        # Total users across all years (unique)
        total_users = user_yearly_df[user_yearly_df['Group_Classification'] == group]['Username'].nunique()
        
        stability_results.append({
            'Group': group,
            'Years_Active': len(group_data),
            'Total_Users': total_users,
            'Total_Comments': group_data['Total_Comments'].sum(),
            'Mean_Contra': np.mean(group_data['Contra_Prop']),
            'Mean_Neutral': np.mean(group_data['Neutral_Prop']),
            'Mean_Pro': np.mean(group_data['Pro_Prop']),
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
    
    return yearly_df, stability_df, user_yearly_df

def create_user_group_tables(yearly_df, stability_df):
    """Create formatted tables for user group analysis"""
    
    print("\n" + "="*80)
    print("TABLE A1: YEARLY STANCE DISTRIBUTION BY USER GROUP")
    print("="*80)
    
    for group in sorted(yearly_df['Group'].unique()):
        group_data = yearly_df[yearly_df['Group'] == group]
        total_users = group_data['Users_Count'].sum()  # This might double count, but we'll use stability_df
        
        print(f"\n{group}:")
        print("-" * 60)
        
        # Format as percentages
        display_data = group_data[['Year', 'Users_Count', 'Total_Comments', 'Contra_Prop', 'Neutral_Prop', 'Pro_Prop']].copy()
        display_data['Contra_%'] = (display_data['Contra_Prop'] * 100).round(1)
        display_data['Neutral_%'] = (display_data['Neutral_Prop'] * 100).round(1)
        display_data['Pro_%'] = (display_data['Pro_Prop'] * 100).round(1)
        
        print(display_data[['Year', 'Users_Count', 'Total_Comments', 'Contra_%', 'Neutral_%', 'Pro_%']].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE A2: USER GROUP STABILITY STATISTICS")
    print("="*80)
    
    # Format stability table
    stability_display = stability_df.copy()
    
    # Convert to percentages and round
    for col in ['Mean_Contra', 'Mean_Neutral', 'Mean_Pro']:
        stability_display[col] = (stability_display[col] * 100).round(1)
    
    for col in ['Contra_Variance', 'Neutral_Variance', 'Pro_Variance', 'Stability_Index']:
        stability_display[col] = stability_display[col].round(4)
    
    display_cols = ['Group', 'Total_Users', 'Years_Active', 'Total_Comments',
                   'Mean_Contra', 'Mean_Neutral', 'Mean_Pro', 'Stability_Index']
    
    print(stability_display[display_cols].to_string(index=False))
    
    print("\n\n" + "="*80)
    print("TABLE A3: USER GROUP TREND ANALYSIS WITH SIGNIFICANCE TESTS")
    print("="*80)
    
    trend_display = stability_df[['Group', 'Contra_Slope', 'Contra_Slope_Pval',
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

def run_user_group_statistical_tests(stability_df):
    """Run statistical tests for user groups"""
    
    print("\n\n" + "="*80)
    print("STATISTICAL TESTS - USER GROUP LEVEL")
    print("="*80)
    
    # Test 1: Overall group stability comparison
    print("User Group Stability Comparison:")
    for _, row in stability_df.iterrows():
        print(f"  {row['Group']}: {row['Stability_Index']:.4f} (n={row['Total_Users']} users)")
    
    # Test 2: Group mean stance proportions
    print(f"\nUser Group Mean Stance Proportions:")
    for _, row in stability_df.iterrows():
        print(f"  {row['Group']}:")
        print(f"    Contra: {row['Mean_Contra']:.1%}")
        print(f"    Neutral: {row['Mean_Neutral']:.1%}")
        print(f"    Pro: {row['Mean_Pro']:.1%}")
    
    # Test 3: Significant trends summary
    print(f"\nSignificant Trends (p < 0.05):")
    for _, row in stability_df.iterrows():
        group_name = row['Group']
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
    
    # Test 4: Pairwise group comparisons (if applicable)
    if len(stability_df) > 1:
        print(f"\nGroup Stability Rankings (most to least stable):")
        ranked = stability_df.sort_values('Stability_Index')
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {row['Group']}: {row['Stability_Index']:.4f}")

def main(json_file_path):
    """Run complete user group analysis"""
    
    # Load data
    df = load_user_dataset(json_file_path)
    
    if len(df) == 0:
        print("No data found! Check your file path.")
        return None, None, None
    
    # User group analysis
    yearly_df, stability_df, user_yearly_df = analyze_user_stance_stability(df)
    
    # Create tables
    create_user_group_tables(yearly_df, stability_df)
    
    # Statistical tests
    run_user_group_statistical_tests(stability_df)
    
    # Export all results
    yearly_df.to_csv('user_group_yearly_stance_distribution.csv', index=False)
    stability_df.to_csv('user_group_stability_stats.csv', index=False)
    user_yearly_df.to_csv('individual_user_yearly_stance.csv', index=False)
    
    print(f"\nExported CSV files:")
    print("- user_group_yearly_stance_distribution.csv")
    print("- user_group_stability_stats.csv") 
    print("- individual_user_yearly_stance.csv")
    
    return yearly_df, stability_df, user_yearly_df

if __name__ == "__main__":
    json_file_path = "Data/NY_Legacy_group_mapping.json"  
    
    print("User Group Stance Stability Analysis")
    print("=" * 50)
    print(f"Dataset file: {json_file_path}")
    print()
    
    # Uncomment to run:
    yearly_df, stability_df, user_yearly_df = main(json_file_path)
    
  #  print("Set the json_file_path variable and uncomment the last line to run the analysis.")
