#### This script compares if echo chambers groups (Contra-Echo and Contra-Balance) have different reactions to real-life events, compared to the other groups. 
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def load_data(file_path):
    """Load the unified user dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def parse_period(period_str):
    """Convert T1-20 → (2020, Q1)."""
    if not period_str or not isinstance(period_str, str):
        return (0, 0)
    match = re.match(r'T(\d+)-(\d+)', period_str)
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2))
        year = 2000 + year if year < 50 else 1900 + year
        return (year, quarter)
    return (0, 0)

def get_user_temporal_data(dataset):
    """Extract user-level temporal activity data for statistical analysis."""
    user_data = []
    all_periods = set()
    
    # First pass: collect all periods
    for username, user_info in dataset.items():
        for comment in user_info.get('comments', []):
            period = comment.get('period')
            if period and period != 'T1-25':
                all_periods.add(period)
    
    sorted_periods = sorted(all_periods, key=parse_period)
    
    # Second pass: create user-level activity vectors
    for username, user_info in dataset.items():
        subgroup = user_info.get('group_classification')
        if not subgroup:
            continue
            
        user_period_counts = defaultdict(int)
        total_user_comments = 0
        
        for comment in user_info.get('comments', []):
            period = comment.get('period')
            if period and period != 'T1-25':
                user_period_counts[period] += 1
                total_user_comments += 1
        
        # Calculate percentage distribution for this user
        if total_user_comments > 0:
            activity_vector = []
            for period in sorted_periods:
                percentage = (user_period_counts[period] / total_user_comments) * 100
                activity_vector.append(percentage)
            
            user_data.append({
                'username': username,
                'subgroup': subgroup,
                'activity_vector': activity_vector,
                'total_comments': total_user_comments
            })
    
    return user_data, sorted_periods

def test_distinct_temporal_patterns(user_data, periods):
    """Test if C-Echo and C-Balance have distinct temporal patterns."""
    
    # Convert to DataFrame for easier manipulation
    df_data = []
    for user in user_data:
        row = {'username': user['username'], 'subgroup': user['subgroup']}
        for i, period in enumerate(periods):
            row[f'period_{period}'] = user['activity_vector'][i]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Define groups
    target_groups = ['Contra-Echo', 'Contra-Balance']
    other_groups = ['Pro', 'Contra-Cross', 'Control']
    
    results = {}
    
    # 1. CLUSTERING ANALYSIS
    print("=== CLUSTERING ANALYSIS ===")
    
    # Prepare data for clustering (activity vectors only)
    activity_data = []
    group_labels = []
    for user in user_data:
        activity_data.append(user['activity_vector'])
        group_labels.append(user['subgroup'])
    
    activity_array = np.array(activity_data)
    scaler = StandardScaler()
    activity_scaled = scaler.fit_transform(activity_array)
    
    # K-means clustering with k=2 (target vs others)
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(activity_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(activity_scaled, cluster_labels)
    print(f"Silhouette Score for 2-cluster solution: {silhouette_avg:.3f}")
    
    # Check if target groups cluster together
    target_indices = [i for i, user in enumerate(user_data) if user['subgroup'] in target_groups]
    other_indices = [i for i, user in enumerate(user_data) if user['subgroup'] in other_groups]
    
    target_clusters = cluster_labels[target_indices]
    other_clusters = cluster_labels[other_indices]
    
    # Calculate purity: what % of target group users are in the same cluster?
    target_cluster_mode = stats.mode(target_clusters)[0][0]
    target_purity = np.mean(target_clusters == target_cluster_mode)
    
    print(f"Target group clustering purity: {target_purity:.3f}")
    print(f"Target groups predominantly in cluster: {target_cluster_mode}")
    
    # Chi-square test for cluster association
    contingency_table = np.zeros((2, 2))
    for i, user in enumerate(user_data):
        is_target = int(user['subgroup'] in target_groups)
        cluster = cluster_labels[i]
        contingency_table[is_target, cluster] += 1
    
    chi2, p_chi2 = stats.chi2_contingency(contingency_table)[:2]
    print(f"Chi-square test for cluster-group association: χ² = {chi2:.3f}, p = {p_chi2:.3f}")
    
    results['clustering'] = {
        'silhouette_score': silhouette_avg,
        'target_purity': target_purity,
        'chi2': chi2,
        'p_chi2': p_chi2
    }
    
    # 2. VARIANCE ANALYSIS
    print("\n=== VARIANCE ANALYSIS ===")
    
    # Calculate variance in activity for each user
    target_variances = []
    other_variances = []
    
    for user in user_data:
        variance = np.var(user['activity_vector'])
        if user['subgroup'] in target_groups:
            target_variances.append(variance)
        elif user['subgroup'] in other_groups:
            other_variances.append(variance)
    
    # Mann-Whitney U test for variance differences
    u_stat, p_variance = mannwhitneyu(target_variances, other_variances, alternative='two-sided')
    
    print(f"Target groups variance (median): {np.median(target_variances):.2f}")
    print(f"Other groups variance (median): {np.median(other_variances):.2f}")
    print(f"Mann-Whitney U test for variance: U = {u_stat:.1f}, p = {p_variance:.3f}")
    
    results['variance'] = {
        'target_median_var': np.median(target_variances),
        'other_median_var': np.median(other_variances),
        'u_stat': u_stat,
        'p_variance': p_variance
    }
    
    # 3. PEAK TIMING ANALYSIS
    print("\n=== PEAK TIMING ANALYSIS ===")
    
    # Find peak period for each user
    target_peaks = []
    other_peaks = []
    
    for user in user_data:
        peak_period_idx = np.argmax(user['activity_vector'])
        if user['subgroup'] in target_groups:
            target_peaks.append(peak_period_idx)
        elif user['subgroup'] in other_groups:
            other_peaks.append(peak_period_idx)
    
    # Kolmogorov-Smirnov test for peak timing distribution
    ks_stat, p_ks = stats.ks_2samp(target_peaks, other_peaks)
    
    print(f"Target groups peak timing (median period index): {np.median(target_peaks):.1f}")
    print(f"Other groups peak timing (median period index): {np.median(other_peaks):.1f}")
    print(f"KS test for peak timing: D = {ks_stat:.3f}, p = {p_ks:.3f}")
    
    results['peak_timing'] = {
        'target_median_peak': np.median(target_peaks),
        'other_median_peak': np.median(other_peaks),
        'ks_stat': ks_stat,
        'p_ks': p_ks
    }
    
    # 4. MULTIVARIATE ANALYSIS
    print("\n=== MULTIVARIATE ANALYSIS ===")
    
    # Create binary target variable
    y = np.array([1 if user['subgroup'] in target_groups else 0 for user in user_data])
    X = activity_scaled
    
    # MANOVA-like approach using distances
    target_center = np.mean(X[y == 1], axis=0)
    other_center = np.mean(X[y == 0], axis=0)
    
    # Calculate within-group and between-group distances
    within_target = np.mean([np.linalg.norm(x - target_center) for x in X[y == 1]])
    within_other = np.mean([np.linalg.norm(x - other_center) for x in X[y == 0]])
    between_groups = np.linalg.norm(target_center - other_center)
    
    # F-ratio like statistic
    f_ratio = between_groups / ((within_target + within_other) / 2)
    
    print(f"Between-group distance: {between_groups:.3f}")
    print(f"Within-group distance (target): {within_target:.3f}")
    print(f"Within-group distance (other): {within_other:.3f}")
    print(f"F-ratio-like statistic: {f_ratio:.3f}")
    
    results['multivariate'] = {
        'between_distance': between_groups,
        'within_target': within_target,
        'within_other': within_other,
        'f_ratio': f_ratio
    }
    
    # 5. PERIOD-BY-PERIOD ANALYSIS
    print("\n=== PERIOD-BY-PERIOD ANALYSIS ===")
    
    significant_periods = []
    for i, period in enumerate(periods):
        target_values = [user['activity_vector'][i] for user in user_data if user['subgroup'] in target_groups]
        other_values = [user['activity_vector'][i] for user in user_data if user['subgroup'] in other_groups]
        
        if len(target_values) > 0 and len(other_values) > 0:
            u_stat, p_val = mannwhitneyu(target_values, other_values, alternative='two-sided')
            if p_val < 0.05:
                significant_periods.append((period, p_val))
                print(f"  {period}: p = {p_val:.3f} *")
    
    print(f"\nPeriods with significant differences: {len(significant_periods)}")
    
    results['period_analysis'] = {
        'significant_periods': significant_periods,
        'n_significant': len(significant_periods)
    }
    
    return results

def create_diagnostic_plots(user_data, periods, results):
    """Create diagnostic plots for the statistical analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    target_groups = ['Contra-Echo', 'Contra-Balance']
    other_groups = ['Pro', 'Contra-Cross', 'Control']
    
    # Plot 1: Activity variance distribution
    target_variances = [np.var(user['activity_vector']) for user in user_data if user['subgroup'] in target_groups]
    other_variances = [np.var(user['activity_vector']) for user in user_data if user['subgroup'] in other_groups]
    
    axes[0,0].hist(target_variances, alpha=0.7, label='Target Groups', bins=15, color='red')
    axes[0,0].hist(other_variances, alpha=0.7, label='Other Groups', bins=15, color='blue')
    axes[0,0].set_xlabel('Activity Variance')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Activity Variance')
    axes[0,0].legend()
    
    # Plot 2: Peak timing distribution
    target_peaks = [np.argmax(user['activity_vector']) for user in user_data if user['subgroup'] in target_groups]
    other_peaks = [np.argmax(user['activity_vector']) for user in user_data if user['subgroup'] in other_groups]
    
    axes[0,1].hist(target_peaks, alpha=0.7, label='Target Groups', bins=len(periods), color='red')
    axes[0,1].hist(other_peaks, alpha=0.7, label='Other Groups', bins=len(periods), color='blue')
    axes[0,1].set_xlabel('Peak Period Index')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Peak Activity Timing')
    axes[0,1].legend()
    
    # Plot 3: Average activity patterns
    target_avg = np.mean([user['activity_vector'] for user in user_data if user['subgroup'] in target_groups], axis=0)
    other_avg = np.mean([user['activity_vector'] for user in user_data if user['subgroup'] in other_groups], axis=0)
    
    x_pos = range(len(periods))
    axes[1,0].plot(x_pos, target_avg, 'ro-', label='Target Groups', linewidth=2, markersize=6)
    axes[1,0].plot(x_pos, other_avg, 'bo-', label='Other Groups', linewidth=2, markersize=6)
    axes[1,0].set_xticks(x_pos[::3])  # Show every 3rd period
    axes[1,0].set_xticklabels([periods[i] for i in range(0, len(periods), 3)], rotation=45)
    axes[1,0].set_ylabel('Avg % Comments per User')
    axes[1,0].set_title('Average Temporal Activity Patterns')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Period-by-period effect sizes
    effect_sizes = []
    period_labels = []
    for i, period in enumerate(periods):
        target_values = [user['activity_vector'][i] for user in user_data if user['subgroup'] in target_groups]
        other_values = [user['activity_vector'][i] for user in user_data if user['subgroup'] in other_groups]
        
        if len(target_values) > 0 and len(other_values) > 0:
            # Cohen's d
            pooled_std = np.sqrt(((len(target_values)-1)*np.var(target_values) + 
                                (len(other_values)-1)*np.var(other_values)) / 
                               (len(target_values) + len(other_values) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(target_values) - np.mean(other_values)) / pooled_std
                effect_sizes.append(abs(cohens_d))
                period_labels.append(period)
    
    if effect_sizes:
        axes[1,1].bar(range(len(effect_sizes)), effect_sizes, alpha=0.7)
        axes[1,1].set_xticks(range(0, len(period_labels), 3))
        axes[1,1].set_xticklabels([period_labels[i] for i in range(0, len(period_labels), 3)], rotation=45)
        axes[1,1].set_ylabel('|Cohen\'s d|')
        axes[1,1].set_title('Effect Sizes by Period')
        axes[1,1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Medium Effect')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('temporal_statistical_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistical_summary(results):
    """Print a comprehensive summary of statistical results."""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n1. CLUSTERING EVIDENCE:")
    print(f"   - Silhouette Score: {results['clustering']['silhouette_score']:.3f}")
    print(f"   - Target Group Purity: {results['clustering']['target_purity']:.3f}")
    print(f"   - Chi-square p-value: {results['clustering']['p_chi2']:.3f}")
    
    significance = "SIGNIFICANT" if results['clustering']['p_chi2'] < 0.05 else "NOT SIGNIFICANT"
    print(f"   → Clustering association: {significance}")
    
    print(f"\n2. VARIANCE EVIDENCE:")
    print(f"   - Target median variance: {results['variance']['target_median_var']:.2f}")
    print(f"   - Other median variance: {results['variance']['other_median_var']:.2f}")
    print(f"   - Mann-Whitney p-value: {results['variance']['p_variance']:.3f}")
    
    significance = "SIGNIFICANT" if results['variance']['p_variance'] < 0.05 else "NOT SIGNIFICANT"
    print(f"   → Variance difference: {significance}")
    
    print(f"\n3. PEAK TIMING EVIDENCE:")
    print(f"   - Target median peak: {results['peak_timing']['target_median_peak']:.1f}")
    print(f"   - Other median peak: {results['peak_timing']['other_median_peak']:.1f}")
    print(f"   - KS test p-value: {results['peak_timing']['p_ks']:.3f}")
    
    significance = "SIGNIFICANT" if results['peak_timing']['p_ks'] < 0.05 else "NOT SIGNIFICANT"
    print(f"   → Peak timing difference: {significance}")
    
    print(f"\n4. MULTIVARIATE EVIDENCE:")
    print(f"   - Between-group distance: {results['multivariate']['between_distance']:.3f}")
    print(f"   - F-ratio statistic: {results['multivariate']['f_ratio']:.3f}")
    
    print(f"\n5. PERIOD-SPECIFIC EVIDENCE:")
    print(f"   - Significant periods: {results['period_analysis']['n_significant']}")
    if results['period_analysis']['significant_periods']:
        for period, p_val in results['period_analysis']['significant_periods'][:5]:  # Show first 5
            print(f"     * {period}: p = {p_val:.3f}")
    
    # Overall conclusion
    significant_tests = sum([
        results['clustering']['p_chi2'] < 0.05,
        results['variance']['p_variance'] < 0.05,
        results['peak_timing']['p_ks'] < 0.05,
        results['period_analysis']['n_significant'] > 0
    ])
    
    print(f"\n" + "="*60)
    print(f"OVERALL CONCLUSION:")
    print(f"Evidence supporting distinct patterns: {significant_tests}/4 tests significant")
    
    if significant_tests >= 2:
        print("→ STRONG evidence for distinct temporal patterns in target groups")
    elif significant_tests == 1:
        print("→ MODERATE evidence for distinct temporal patterns in target groups")
    else:
        print("→ WEAK evidence for distinct temporal patterns in target groups")
    
    print("="*60)

def main():
    file_path = 'ADD PATH'  # Update with your file path
    dataset = load_data(file_path)

    # Filter out excluded groups
    dataset = {username: user_info for username, user_info in dataset.items() 
               if user_info.get('group_classification') not in ['Left-ex', 'Control-Ex']}

    print(f"Loaded dataset with {len(dataset)} users after filtering")

    # Extract user-level temporal data
    user_data, periods = get_user_temporal_data(dataset)
    print(f"Extracted temporal data for {len(user_data)} users across {len(periods)} periods")

    # Run statistical tests
    results = test_distinct_temporal_patterns(user_data, periods)
    
    # Create diagnostic plots
    create_diagnostic_plots(user_data, periods, results)
    
    # Print comprehensive summary
    print_statistical_summary(results)

if __name__ == "__main__":
    main()
