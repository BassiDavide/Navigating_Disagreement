##### REPLY PATTERN USER LEVEL ANALYSIS VERSION #####

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
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

def analyze_temporal_activity(dataset):
    """Activity of each subgroup across periods (as % of total comments per user, then averaged by subgroup)."""
    # First, calculate user-level temporal distributions
    user_temporal_distributions = {}
    all_periods = set()
    
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
                all_periods.add(period)
        
        # Calculate percentage distribution for this user
        if total_user_comments > 0:
            user_temporal_distributions[username] = {
                'subgroup': subgroup,
                'period_percentages': {p: (count / total_user_comments) * 100 
                                     for p, count in user_period_counts.items()}
            }

    sorted_periods = sorted(all_periods, key=parse_period)
    
    # Now average by subgroup
    subgroup_activity_pct = defaultdict(lambda: defaultdict(list))
    
    # Collect user percentages by subgroup
    for username, user_data in user_temporal_distributions.items():
        subgroup = user_data['subgroup']
        for period in sorted_periods:
            percentage = user_data['period_percentages'].get(period, 0)
            subgroup_activity_pct[subgroup][period].append(percentage)
    
    # Calculate averages
    final_subgroup_activity = {}
    for subgroup, period_data in subgroup_activity_pct.items():
        final_subgroup_activity[subgroup] = {}
        for period in sorted_periods:
            user_percentages = period_data.get(period, [0])
            final_subgroup_activity[subgroup][period] = np.mean(user_percentages)

    return final_subgroup_activity, sorted_periods

def analyze_reply_behavior(dataset):
    """Reply vs top-level comments by subgroup (user-level analysis averaged by group)."""
    # First, calculate user-level reply behavior
    user_reply_data = {}
    
    for username, user_info in dataset.items():
        subgroup = user_info.get('group_classification')
        if not subgroup:
            continue
            
        video_replies = 0  # Level = 0
        user_replies = 0   # Level > 0
        total_comments = 0
        
        for c in user_info.get('comments', []):
            total_comments += 1
            if c.get('Level', 0) == 0:
                video_replies += 1
            else:
                user_replies += 1
        
        # Calculate proportions for this user
        if total_comments > 0:
            user_reply_data[username] = {
                'subgroup': subgroup,
                'video_proportion': video_replies / total_comments,
                'user_proportion': user_replies / total_comments,
                'total': total_comments
            }
    
    # Now average by subgroup
    subgroup_reply_data = defaultdict(lambda: {'video_proportions': [], 'user_proportions': [], 'total_users': 0})
    
    for username, user_data in user_reply_data.items():
        subgroup = user_data['subgroup']
        subgroup_reply_data[subgroup]['video_proportions'].append(user_data['video_proportion'])
        subgroup_reply_data[subgroup]['user_proportions'].append(user_data['user_proportion'])
        subgroup_reply_data[subgroup]['total_users'] += 1
    
    # Calculate final averages and format for compatibility with existing plotting code
    reply_data = {}
    for subgroup, data in subgroup_reply_data.items():
        if data['total_users'] > 0:
            avg_video_prop = np.mean(data['video_proportions'])
            avg_user_prop = np.mean(data['user_proportions'])
            
            # Format to match original output structure
            reply_data[subgroup] = {
                'video_replies': avg_video_prop,
                'user_replies': avg_user_prop,
                'total': 1.0  # Since these are now proportions, total should be 1.0
            }
    
    return reply_data

def create_temporal_activity_plot(subgroup_activity, periods):
    """Create temporal activity plot optimized for paper."""
    plt.figure(figsize=(8, 5))
    
    # Colors for each group (same as preference plot)
    colors = {'Pro': '#1f77b4', 'Contra-Echo': '#ff7f0e', 'Contra-Balance': '#2ca02c', 
              'Contra-Cross': '#d62728', 'Control': '#9467bd'}
    
    # Shortened names for legend
    name_mapping = {
        'Pro': 'Pro',
        'Contra-Echo': 'C-Echo', 
        'Contra-Balance': 'C-Balance',
        'Contra-Cross': 'C-Cross',
        'Control': 'Control'
    }
    
    # Plot each subgroup
    for sg, data in subgroup_activity.items():
        values = [data.get(p, 0) for p in periods]
        display_name = name_mapping.get(sg, sg)
        plt.plot(periods, values, label=display_name, linewidth=3, marker='o', markersize=6, 
                color=colors.get(sg, 'gray'))
    
    # Add vertical lines with letters for key events
    # Based on typical immigration timeline events, adjust positions as needed
    event_periods = {
        'T1-21': 'A',  # Biden inauguration 
        'T3-22': 'B',  # Migrant bussing controversies
        'T1-23': 'C',  # Title 42 expiration
        'T2-24': 'D'   # Election buildup
    }
    
    for period, letter in event_periods.items():
        if period in periods:
            period_idx = periods.index(period)
            plt.axvline(x=period_idx, color='black', linestyle='--', alpha=0.7, linewidth=2)
            plt.text(period_idx, plt.ylim()[0] + (plt.ylim()[1]-plt.ylim()[0])*0.05, letter, 
                ha='right', va='bottom', fontsize=16,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.xlabel('Time Period', fontsize=16)
    plt.ylabel('Avg % of Comments per User', fontsize=16)  # Updated label to reflect user-level averaging

    # Larger tick labels and rotate x-axis
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=14)
    
    # Legend in upper right with reduced fontsize
    plt.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('temporal_activity_user_level.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_reply_behavior_plot(reply_data):
    """Create reply behavior plot optimized for paper."""
    plt.figure(figsize=(8, 5))
    
    # Extract proportions (these are now already averaged user-level proportions)
    video_props = {sg: d['video_replies'] for sg, d in reply_data.items()}
    user_props = {sg: d['user_replies'] for sg, d in reply_data.items()}
    
    # Name mapping for two-line labels
    name_mapping = {
        'Pro': 'Pro',
        'Contra-Echo': 'Contra\nEcho', 
        'Contra-Balance': 'Contra\nBalance',
        'Contra-Cross': 'Contra\nCross',
        'Control': 'Control'
    }
    
    # Create grouped bar chart
    x = np.arange(len(video_props))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, list(video_props.values()), width, 
                   label='Video', alpha=0.8, color='orange')
    bars2 = plt.bar(x + width/2, list(user_props.values()), width, 
                   label='User', alpha=0.8, color='olivedrab')
    
    plt.xlabel('', fontsize=1)
    plt.ylabel('Avg Proportion per User', fontsize=16)  # Updated label to reflect user-level averaging
    
    # Set x-axis labels with formatted names
    formatted_labels = [name_mapping.get(sg, sg) for sg in video_props.keys()]
    plt.xticks(x, formatted_labels, fontsize=14, rotation=0, ha='center')
    plt.yticks(fontsize=14)
    
    # Legend with reduced fontsize
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('reply_behavior_user_level.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def print_user_level_statistics(dataset, reply_data):
    """Print statistics about the user-level analysis."""
    print("\n=== USER-LEVEL ANALYSIS STATISTICS ===")
    
    # Count users per group
    group_counts = defaultdict(int)
    for username, user_info in dataset.items():
        subgroup = user_info.get('group_classification')
        if subgroup:
            group_counts[subgroup] += 1
    
    print(f"\nUsers per group:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} users")
    
    print(f"\nReply behavior (averaged across users):")
    for group, data in sorted(reply_data.items()):
        video_pct = data['video_replies'] * 100
        user_pct = data['user_replies'] * 100
        print(f"  {group}: {video_pct:.1f}% video-directed, {user_pct:.1f}% user-directed")

def main():
    file_path = 'ADD PATH'  # Update with your file path
    dataset = load_data(file_path)

    # Filter out excluded groups
    dataset = {username: user_info for username, user_info in dataset.items() 
               if user_info.get('group_classification') not in ['Left-ex', 'Control-Ex']}

    print(f"Loaded dataset with {len(dataset)} users after filtering")

    # Analyze data using user-level approach
    subgroup_activity, periods = analyze_temporal_activity(dataset)
    reply_data = analyze_reply_behavior(dataset)

    # Print statistics
    print_user_level_statistics(dataset, reply_data)

    # Create plots
    print("\nCreating temporal activity plot (user-level averaged)...")
    create_temporal_activity_plot(subgroup_activity, periods)
    
    print("Creating reply behavior plot (user-level averaged)...")
    create_reply_behavior_plot(reply_data)
    
    print("Plots saved as temporal_activity_user_level.pdf and reply_behavior_user_level.pdf")

if __name__ == "__main__":
    main()
