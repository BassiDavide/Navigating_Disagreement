###### TOXICITY USER-LEVEL ANALYSIS ######

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_user_comments_data():
    """Load the user comments dataset with group classifications"""
    
    with open('/mnt/beegfs/home/davide.bassi/User_Study/Data/NY_Legacy_group_mapping.json', 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    
    return user_data

def extract_linguistic_features():
    """Define linguistic feature categories"""
    
    #macro_categories = {
    #    'Toxicity': [
    #        'toxicity_experimental_score',
    #        'severe_toxicity_experimental_score', 
    #        'identity_attack_experimental_score',
    #        'insult_experimental_score',
    #        'profanity_experimental_score',
    #        'threat_experimental_score'
    #    ],
    #    'Emotional_Tone': [
    #        'affinity_experimental_score',
    #        'compassion_experimental_score',
    #        'respect_experimental_score'
    #    ],
    #    'Cognitive_Style': [
    #        'curiosity_experimental_score',
    #        'nuance_experimental_score',
    #        'reasoning_experimental_score'
    #    ],
    #    'Communication_Style': [
    #        'personal_story_experimental_score',
    #        'sexually_explicit_score',
    #        'flirtation_score'
    #    ]
    #}

    macro_categories = {
        'Toxicity': [
            'toxicity_score',
            'severe_toxicity_score',
            'identity_attack_score',
            'insult_score',
            'profanity_score',
            'threat_score',
        ],
        'Communication_Style': [
            'attack_on_commenter_score',
            'inflammatory_score',
            'obscene_score',
            'spam_score',
            'unsubstantial_score'
        ]
    }
    
    #all_features = [
    #    'toxicity_experimental_score',
    #    'severe_toxicity_experimental_score',
    #    'identity_attack_experimental_score',
    #    'insult_experimental_score',
    #    'profanity_experimental_score',
    #    'threat_experimental_score',
    #    'sexually_explicit_score',
    #    'flirtation_score',
    #    'affinity_experimental_score',
    #    'compassion_experimental_score',
    #    'curiosity_experimental_score',
    #    'nuance_experimental_score',
    #    'personal_story_experimental_score',
    #    'reasoning_experimental_score',
    #    'respect_experimental_score'
    #]

    all_features = [
        'toxicity_score',
        'severe_toxicity_score',
        'identity_attack_score',
        'insult_score',
        'profanity_score',
        'threat_score',
        'attack_on_commenter_score',
        'inflammatory_score',
        'obscene_score',
        'spam_score',
        'unsubstantial_score'
    ]
    
    return macro_categories, all_features

def calculate_word_count(comment_text):
    """Calculate word count for a comment"""
    return len(comment_text.split())

def create_comment_dataframe(user_data):
    """Convert nested user data to flat comment dataframe"""
    
    _, all_features = extract_linguistic_features()
    comment_data = []
    
    for username, user_info in user_data.items():
        group = user_info['group_classification']
        comments = user_info['comments']
        
        for comment in comments:
            word_count = calculate_word_count(comment['comment_text'])
            
            # Determine comment type
            level = comment.get('Level', 0)
            comment_type = 'VideoComment' if level == 0 else 'Reply'
            
            comment_row = {
                'username': username,
                'group': group,
                'commentID': comment.get('commentID', ''),
                'comment_text': comment['comment_text'],
                'word_count': word_count,
                'stance': comment.get('stance', np.nan),
                'period': comment.get('period', ''),
                'channel_leaning': comment.get('ChannelLeaning', ''),
                'level': level,
                'comment_type': comment_type
            }
            
            # Add linguistic features
            for feature in all_features:
                comment_row[feature] = comment.get(feature, np.nan)
            
            comment_data.append(comment_row)
    
    df = pd.DataFrame(comment_data)
    
    # Filter groups as specified: include only Pro, Contra-Balance, Contra-Cross, Contra-Echo, Control
    # Exclude Control-Ex, Left-Ex, and any others
    allowed_groups = ['Pro', 'Contra-Balance', 'Contra-Cross', 'Contra-Echo', 'Control']
    df = df[df['group'].isin(allowed_groups)]
    
    print(f"Dataset created (filtered groups):")
    print(f"  Total comments: {len(df):,}")
    print(f"  Users: {df['username'].nunique()}")
    print(f"  Comments by group:")
    for group, count in df['group'].value_counts().items():
        print(f"    {group}: {count:,} comments")
    
    print(f"\nComment type distribution:")
    print(df['comment_type'].value_counts())
    
    return df

def create_length_strata(df, n_strata=3):
    """Create length-based strata for analysis"""
    
    if n_strata == 3:
        labels = ['Short', 'Medium', 'Long']
        bins = [0, 0.33, 0.67, 1.0]
    elif n_strata == 4:
        labels = ['Very_Short', 'Short', 'Medium', 'Long']
        bins = [0, 0.25, 0.5, 0.75, 1.0]
    else:
        labels = [f'Stratum_{i+1}' for i in range(n_strata)]
        bins = np.linspace(0, 1, n_strata + 1)
    
    df['length_stratum'] = pd.qcut(df['word_count'], 
                                   q=bins, 
                                   labels=labels,
                                   duplicates='drop')
    
    print(f"\nLength Strata Created:")
    for stratum in df['length_stratum'].unique():
        if pd.isna(stratum):
            continue
        subset = df[df['length_stratum'] == stratum]
        print(f"  {stratum}:")
        print(f"    Word count range: {subset['word_count'].min()}-{subset['word_count'].max()}")
        print(f"    Mean word count: {subset['word_count'].mean():.1f}")
        print(f"    Total comments: {len(subset):,}")
        print(f"    Group distribution: {dict(subset['group'].value_counts())}")
    
    return df

def analyze_stratum(df_stratum, stratum_name, linguistic_features):
    """Analyze one length stratum using USER-LEVEL aggregation"""
    
    print(f"\n" + "="*60)
    print(f"ANALYZING {stratum_name.upper()} COMMENTS (USER-LEVEL ANALYSIS)")
    print(f"Word count range: {df_stratum['word_count'].min()}-{df_stratum['word_count'].max()}")
    print(f"Mean word count: {df_stratum['word_count'].mean():.1f}")
    print("="*60)
    
    results = {
        'stratum_info': {
            'name': stratum_name,
            'word_count_range': [str(df_stratum['word_count'].min()), str(df_stratum['word_count'].max())],
            'mean_word_count': df_stratum['word_count'].mean(),
            'total_comments': len(df_stratum),
            'group_distribution': {k: str(v) for k, v in dict(df_stratum['group'].value_counts()).items()}
        },
        'macro_results': {},
        'micro_results': {},
        'significant_features': []
    }
    
    group_counts = df_stratum['group'].value_counts()
    user_counts = df_stratum.groupby('group')['username'].nunique()
    min_group_size = group_counts.min()
    min_user_count = user_counts.min()
    
    if min_group_size < 10:
        print(f"WARNING: Smallest group has only {min_group_size} comments. Results may be unreliable.")
    
    if min_user_count < 5:
        print(f"WARNING: Smallest group has only {min_user_count} users. User-level results may be unreliable.")
    
    print(f"Group sizes (comments): {dict(group_counts)}")
    print(f"Group sizes (users): {dict(user_counts)}")
    
    macro_categories, _ = extract_linguistic_features()
    
    print(f"\nMACRO-LEVEL ANALYSIS (USER-LEVEL):")
    print("-" * 40)
    
    for category, features in macro_categories.items():
        available_features = [f for f in features if f in df_stratum.columns]
        if not available_features:
            continue
        
        # MODIFIED: Calculate user-level means first, then group-level means
        # Step 1: Calculate category scores for each user
        user_category_scores = df_stratum.groupby(['username', 'group'])[available_features].mean().mean(axis=1)
        
        # Step 2: Calculate group means from user means
        group_stats = user_category_scores.groupby('group').mean()
        
        print(f"\n{category}:")
        for group, score in group_stats.items():
            print(f"  {group}: {score:.4f}")
        
        # MODIFIED: Use user-level data for ANOVA
        groups = df_stratum['group'].unique()
        group_data = [user_category_scores[user_category_scores.index.get_level_values('group') == group].dropna()
                     for group in groups]
        
        if all(len(data) >= 3 for data in group_data):  # Need at least 3 users per group
            f_stat, p_value = stats.f_oneway(*group_data)

            # --- Compute eta squared using user-level data ---
            grand_mean = user_category_scores.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_data)
            ss_total = sum(((g - grand_mean) ** 2).sum() for g in group_data)
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
            # --------------------------

            print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.4f}, eta²={eta_sq:.4f}")
            
            if p_value < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE ***")
                highest_group = group_stats.idxmax()
                lowest_group = group_stats.idxmin()
                print(f"  Highest: {highest_group} ({group_stats[highest_group]:.4f})")
                print(f"  Lowest: {lowest_group} ({group_stats[lowest_group]:.4f})")
            
            results['macro_results'][category] = {
                'group_means': dict(group_stats),
                'anova': {'f_stat': f_stat, 'p_value': p_value, 'eta_sq': eta_sq}
            }
    
    print(f"\nMICRO-LEVEL ANALYSIS (USER-LEVEL):")
    print("-" * 40)
    
    significant_features = []
    
    for feature in linguistic_features:
        if feature not in df_stratum.columns or df_stratum[feature].isna().all():
            continue
        
        # MODIFIED: Calculate user-level means first, then group-level means
        # Step 1: Calculate user means for this feature
        user_feature_means = df_stratum.groupby(['username', 'group'])[feature].mean()
        
        # Step 2: Calculate group means from user means
        group_stats = user_feature_means.groupby('group').mean()
        
        # MODIFIED: Use user-level data for ANOVA
        groups = df_stratum['group'].unique()
        group_data = [user_feature_means[user_feature_means.index.get_level_values('group') == group].dropna()
                     for group in groups]
        
        if all(len(data) >= 3 for data in group_data):  # Need at least 3 users per group
            f_stat, p_value = stats.f_oneway(*group_data)
            
            if p_value < 0.05:
                significant_features.append((feature, p_value, f_stat, dict(group_stats)))
                
                print(f"\n{feature}:")
                for group, score in group_stats.items():
                    print(f"  {group}: {score:.4f}")
                print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.4f} ***")
            
            results['micro_results'][feature] = {
                'group_means': dict(group_stats),
                'anova': {'f_stat': f_stat, 'p_value': p_value}
            }
    
    results['significant_features'] = significant_features
    
    if significant_features:
        print(f"\nSignificant features in {stratum_name}: {len(significant_features)}")
        for feature, p_val, f_stat, group_means in significant_features:
            highest_group = max(group_means.items(), key=lambda x: x[1])
            print(f"  {feature}: {highest_group[0]} highest ({highest_group[1]:.4f}), p={p_val:.4f}")
    else:
        print(f"\nNo significant features in {stratum_name}")
    
    return results

def stratified_analysis(df, linguistic_features, n_strata=3):
    print("="*80)
    print("LENGTH-STRATIFIED LINGUISTIC ANALYSIS (USER-LEVEL)")
    print("="*80)
    
    df = create_length_strata(df, n_strata)
    all_results = {}
    
    for stratum in df['length_stratum'].unique():
        if pd.isna(stratum):
            continue
        df_stratum = df[df['length_stratum'] == stratum]
        stratum_results = analyze_stratum(df_stratum, stratum, linguistic_features)
        all_results[stratum] = stratum_results
    
    return all_results, df

def create_stratified_visualizations(all_results, df):
    macro_categories, _ = extract_linguistic_features()
    strata = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Length-Stratified Linguistic Analysis (User-Level)', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    group_by_stratum = df.groupby(['length_stratum', 'group']).size().unstack(fill_value=0)
    group_by_stratum.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Comment Distribution by Length Stratum and Group')
    ax1.set_xlabel('Length Stratum')
    ax1.set_ylabel('Number of Comments')
    ax1.legend(title='Group')
    
    ax2 = axes[0, 1]
    for i, stratum in enumerate(strata):
        df_stratum = df[df['length_stratum'] == stratum]
        for group in df['group'].unique():
            group_data = df_stratum[df_stratum['group'] == group]['word_count']
            if len(group_data) > 0:
                ax2.hist(group_data, alpha=0.6, label=f'{group}-{stratum}', bins=20)
    ax2.set_title('Word Count Distribution by Group and Stratum')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3 = axes[1, 0]
    all_significant = {}
    for stratum, results in all_results.items():
        for feature, p_val, f_stat, group_means in results['significant_features']:
            if feature not in all_significant:
                all_significant[feature] = {}
            all_significant[feature][stratum] = p_val
    
    if all_significant:
        sig_df = pd.DataFrame(all_significant).fillna(1.0)
        sig_df = -np.log10(sig_df)
        
        sns.heatmap(sig_df, annot=True, fmt='.1f', cmap='Reds', ax=ax3,
                   cbar_kws={'label': '-log10(p-value)'})
        ax3.set_title('Significance Heatmap Across Strata')
        ax3.set_xlabel('Linguistic Features')
        ax3.set_ylabel('Length Strata')
    else:
        ax3.text(0.5, 0.5, 'No significant features found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Significance Heatmap Across Strata')
    
    ax4 = axes[1, 1]
    toxicity_trends = {}
    for stratum, results in all_results.items():
        if 'Toxicity' in results['macro_results']:
            toxicity_trends[stratum] = results['macro_results']['Toxicity']['group_means']
    
    if toxicity_trends:
        toxicity_df = pd.DataFrame(toxicity_trends).T
        toxicity_df.plot(kind='bar', ax=ax4, rot=45)
        ax4.set_title('Toxicity Scores by Length Stratum and Group (User-Level)')
        ax4.set_xlabel('Length Stratum')
        ax4.set_ylabel('Toxicity Score')
        ax4.legend(title='Group')
    
    plt.tight_layout()
    plt.savefig('length_stratified_analysis_user_level.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_stratified_summary(all_results):
    """Generate comprehensive summary of stratified analysis"""

    print(f"\n" + "="*80)
    print("STRATIFIED ANALYSIS SUMMARY (USER-LEVEL)")
    print("="*80)

    # Overall pattern analysis
    print(f"\nOVERALL PATTERNS:")
    print("-" * 40)

    # Count significant features by stratum
    sig_counts = {}
    for stratum, results in all_results.items():
        sig_counts[stratum] = len(results['significant_features'])

    print(f"Significant features by stratum:")
    for stratum, count in sig_counts.items():
        print(f"  {stratum}: {count} features")

    # Analyze consistency across strata
    print(f"\nCONSISTENCY ANALYSIS:")
    print("-" * 40)

    # Find features significant in multiple strata
    feature_significance = {}
    for stratum, results in all_results.items():
        for feature, p_val, f_stat, group_means in results['significant_features']:
            if feature not in feature_significance:
                feature_significance[feature] = []
            feature_significance[feature].append((stratum, p_val, group_means))

    consistent_features = {f: strata for f, strata in feature_significance.items() if len(strata) > 1}

    if consistent_features:
        print(f"Features significant in multiple strata:")
        for feature, strata_info in consistent_features.items():
            print(f"  {feature}: {len(strata_info)} strata")
            for stratum, p_val, group_means in strata_info:
                highest_group = max(group_means.items(), key=lambda x: x[1])
                print(f"    {stratum}: {highest_group[0]} highest, p={p_val:.4f}")
    else:
        print("No features consistently significant across strata")

    # Toxicity analysis across strata
    print(f"\nTOXICITY PATTERNS:")
    print("-" * 40)

    toxicity_winners = {}
    for stratum, results in all_results.items():
        if 'Toxicity' in results['macro_results']:
            group_means = results['macro_results']['Toxicity']['group_means']
            highest_group = max(group_means.items(), key=lambda x: x[1])
            p_value = results['macro_results']['Toxicity']['anova']['p_value']
            eta_sq = results['macro_results']['Toxicity']['anova'].get('eta_sq', np.nan)
            toxicity_winners[stratum] = (highest_group[0], highest_group[1], p_value, eta_sq)

    if toxicity_winners:
        for stratum, (group, score, p_val, eta_sq) in toxicity_winners.items():
            significance = "***" if p_val < 0.05 else ""
            print(f"  {stratum}: {group} highest ({score:.4f}), p={p_val:.4f}, η²={eta_sq:.3f} {significance}")

    # Final conclusions
    print(f"\nKEY FINDINGS (USER-LEVEL):")
    print("-" * 40)

    if toxicity_winners:
        # Check if results change across strata
        if len(set(winner[0] for winner in toxicity_winners.values())) == 1:
            consistent_winner = list(toxicity_winners.values())[0][0]
            # Check if effect sizes are meaningful
            meaningful_effects = [eta_sq for _, _, p_val, eta_sq in toxicity_winners.values()
                                  if p_val < 0.05 and eta_sq >= 0.01]
            if meaningful_effects:
                print(f"• CONSISTENT & MEANINGFUL: {consistent_winner} shows highest toxicity across all length strata with meaningful effect sizes")
            else:
                print(f"• CONSISTENT BUT WEAK: {consistent_winner} shows highest toxicity across all length strata but with negligible effect sizes")
        else:
            print(f"• INCONSISTENT: Different groups show highest toxicity in different length strata")
            for stratum, (group, score, p_val, eta_sq) in toxicity_winners.items():
                print(f"  {stratum}: {group} (η²={eta_sq:.3f})")

    if consistent_features:
        meaningful_consistent = sum(1 for strata_info in consistent_features.values() 
                                    if any(p_val < 0.05 for _, p_val, _ in strata_info))
        print(f"• {len(consistent_features)} features show consistent patterns across length strata")
        print(f"• {meaningful_consistent} of these have meaningful effect sizes in at least one stratum")
    else:
        print(f"• No consistent patterns across length strata - suggests length confounding was severe")

    # Save results
    with open('stratified_analysis_results_user_level.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDetailed results saved to 'stratified_analysis_results_user_level.json'")


def main():
    """Run complete length-stratified analysis for VideoComments and Replies using USER-LEVEL aggregation."""
    print("Starting Length-Stratified Linguistic Analysis (USER-LEVEL)...")

    # Load data and build dataframe
    user_data = load_user_comments_data()
    df = create_comment_dataframe(user_data)

    # Get linguistic features
    _, linguistic_features = extract_linguistic_features()

    all_results = {}
    dfs_by_type = {}

    for comment_type in ['VideoComment', 'Reply']:
        print("\n" + "="*80)
        print(f"ANALYSIS FOR: {comment_type.upper()} (USER-LEVEL)")
        print("="*80)

        df_type = df[df['comment_type'] == comment_type].copy()
        if df_type.empty:
            print(f"No comments of type '{comment_type}' found — skipping.")
            all_results[comment_type] = {}
            dfs_by_type[comment_type] = pd.DataFrame()  # empty placeholder
            continue

        # Run stratified analysis
        results, df_with_strata = stratified_analysis(df_type, linguistic_features, n_strata=3)

        # Visualizations and summary per comment type
        create_stratified_visualizations(results, df_with_strata)
        generate_stratified_summary(results)

        # Store results
        all_results[comment_type] = results
        dfs_by_type[comment_type] = df_with_strata

    # Save combined results for convenience (one file for all types)
    with open('user_level_stratified_analysis_all_types_Leagacy.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "="*80)
    print("USER-LEVEL STRATIFIED ANALYSIS COMPLETE")
    print("="*80)
    print("Generated files (per-type and combined):")
    print("  • length_stratified_analysis_user_level.png  (generated per comment_type)")
    print("  • stratified_analysis_results_user_level.json (generated per comment_type)")
    print("  • user_level_stratified_analysis_all_types.json (combined results)")

    # Return everything: results and per-type dataframes
    return all_results, dfs_by_type


if __name__ == "__main__":
    all_results, df_with_strata = main()
