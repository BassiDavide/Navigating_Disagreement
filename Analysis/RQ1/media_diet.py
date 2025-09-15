import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(file_path):
    """Load the dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_channel_distribution(data):
    """Analyze channel distribution by group and channel type using user-level averages"""
    # Groups to include (excluding Left-ex and Control-Ex)
    included_groups = ["Pro", "Control", "Contra-Echo", "Contra-Cross", "Contra-Balance"]
    
    # Track user-level relative frequencies
    group_user_frequencies = defaultdict(list)  # group -> list of user frequency dicts
    
    for username, user_data in data.items():
        group = user_data["group_classification"]
        
        # Skip excluded groups
        if group not in included_groups:
            continue
        
        # Count this user's comments by channel leaning and type
        user_counts = defaultdict(lambda: defaultdict(int))
        total_user_comments = 0
        
        for comment in user_data["comments"]:
            channel_leaning = comment["ChannelLeaning"]
            channel_type = comment["Source"]  # Assuming this field exists
            user_counts[channel_leaning][channel_type] += 1
            total_user_comments += 1
        
        # Calculate relative frequencies for this user
        user_frequencies = defaultdict(lambda: defaultdict(float))
        if total_user_comments > 0:
            for channel_leaning in user_counts:
                for channel_type in user_counts[channel_leaning]:
                    user_frequencies[channel_leaning][channel_type] = user_counts[channel_leaning][channel_type] / total_user_comments
        
        group_user_frequencies[group].append(user_frequencies)
    
    # Calculate average relative frequencies within each group
    # Convert to the same format as the original function but with user-level averages
    group_channel_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for group in included_groups:
        if group_user_frequencies[group]:  # If there are users in this group
            num_users = len(group_user_frequencies[group])
            
            # Sum up relative frequencies across all users in the group
            for user_frequencies in group_user_frequencies[group]:
                for channel_leaning in user_frequencies:
                    for channel_type in user_frequencies[channel_leaning]:
                        group_channel_counts[group][channel_leaning][channel_type] += user_frequencies[channel_leaning][channel_type]
            
            # Average by dividing by number of users - keep as proportions, not percentages
            for channel_leaning in group_channel_counts[group]:
                for channel_type in group_channel_counts[group][channel_leaning]:
                    group_channel_counts[group][channel_leaning][channel_type] = (group_channel_counts[group][channel_leaning][channel_type] / num_users)
    
    return group_channel_counts

def create_scientific_plot(group_channel_counts, output_file="channel_distribution.pdf"):
    """Create publication-ready plot optimized for 2-column layout"""
    
    # Check if we have data to plot
    if not group_channel_counts:
        print("Error: No data found to plot!")
        return
    
    # Set matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    })
    
    groups = list(group_channel_counts.keys())
    print(f"Found {len(groups)} groups: {groups}")
    
    # Calculate counts
    left_content_counts = [group_channel_counts[group]["Left"]["Content Creator"] for group in groups]
    left_news_counts = [group_channel_counts[group]["Left"]["News Channel"] for group in groups]
    right_content_counts = [group_channel_counts[group]["Right"]["Content Creator"] for group in groups]
    right_news_counts = [group_channel_counts[group]["Right"]["News Channel"] for group in groups]
    
    # Debug: Print total counts
    total_comments = sum(left_content_counts + left_news_counts + right_content_counts + right_news_counts)
    print(f"Total comments to plot: {total_comments}")
    
    # Create figure optimized for single column (3.5 inches wide is typical)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    
    x = range(len(groups))
    width = 0.35
    
    # Create stacked bars with distinct patterns
    ax.bar([i - width/2 for i in x], left_content_counts, width, 
           label='Left Content', color='#1f77b4', alpha=0.8, hatch='//')
    ax.bar([i - width/2 for i in x], left_news_counts, width, 
           bottom=left_content_counts, label='Left News', 
           color='#1f77b4', alpha=0.5, hatch='')
    
    ax.bar([i + width/2 for i in x], right_content_counts, width, 
           label='Right Content', color='#d62728', alpha=0.8, hatch='//')
    ax.bar([i + width/2 for i in x], right_news_counts, width, 
           bottom=right_content_counts, label='Right News', 
           color='#d62728', alpha=0.5, hatch='')
    
    # Optimize labels and formatting
    ax.set_xlabel('')
    ax.set_ylabel('Number of Comments')
    ax.set_xticks(x)
    
    # Rotate and adjust group labels for better fit
    group_labels = [g.replace('-', '\n') if '-' in g else g for g in groups]
    ax.set_xticklabels(group_labels, ha='center')
    
    # Position legend inside plot to save space
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper right', 
              frameon=True, handlelength=1.5, handletextpad=0.5)
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout with minimal padding
    plt.tight_layout(pad=0.3)
    
    # Save as high-quality PDF
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Plot saved as {output_file}")
    plt.show()

def main(file_path, output_file="channel_distribution.pdf"):
    """Main analysis function"""
    data = load_data(file_path)
    group_channel_counts = analyze_channel_distribution(data)
    create_scientific_plot(group_channel_counts, output_file)

if __name__ == "__main__":
    # Replace with your JSON file path
    file_path = "Data/users_group_toxicity_NEWclustering.json"
    main(file_path)
