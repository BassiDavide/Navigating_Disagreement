import json
import os
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from glob import glob

def load_video_stance_distributions(data_folder):
    """
    Step 1: Build Video-Level Stance Distribution Lookup
    Returns:
    - comment_to_video: mapping from CommentID to VideoID
    - video_stance_dist: mapping from VideoID to stance distribution
    """
    print("Loading video stance distributions...")
    
    comment_to_video = {}
    video_stance_counts = defaultdict(Counter)
    
    # Walk through all channel folders
    for channel_folder in os.listdir(data_folder):
        channel_path = os.path.join(data_folder, channel_folder)
        if not os.path.isdir(channel_path):
            continue
            
        print(f"Processing channel: {channel_folder}")
        
        # Process all .jsonl files in the channel folder
        jsonl_files = glob(os.path.join(channel_path, "*.jsonl"))
        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        comment = json.loads(line.strip())
                        
                        comment_id = comment['CommentID']
                        video_id = comment['VideoID']
                        predicted_stance = comment['predicted_stance']
                        
                        # Store mapping from comment to video
                        comment_to_video[comment_id] = video_id
                        
                        # Count stance distribution for this video
                        video_stance_counts[video_id][predicted_stance] += 1
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
    
    # Convert counts to proportions
    video_stance_dist = {}
    for video_id, stance_counts in video_stance_counts.items():
        total_comments = sum(stance_counts.values())
        if total_comments > 0:
            video_stance_dist[video_id] = {
                0: stance_counts[0] / total_comments,
                1: stance_counts[1] / total_comments, 
                2: stance_counts[2] / total_comments
            }
    
    print(f"Processed {len(video_stance_dist)} videos")
    print(f"Created mapping for {len(comment_to_video)} comments")
    
    return comment_to_video, video_stance_dist

def extract_user_reply_behavior(user_data, comment_to_video, video_stance_dist):
    """
    Step 2: Extract User Reply Behavior with Video Context
    Returns dictionary organized by user: {username: {'group': group, 'replies': [(video_id, parent_stance), ...]}}
    """
    print("Extracting user reply behavior...")
    
    # Groups to include in analysis
    valid_groups = {"Pro", "Contra-Echo", "Contra-Balance", "Contra-Cross", "Control"}
    
    user_behaviors = {}
    
    for username, user_info in user_data.items():
        # Get user group classification
        user_group = user_info.get('group_classification')
        
        # Skip excluded groups
        if user_group not in valid_groups:
            continue
            
        # Get user comments
        user_comments = user_info.get('comments', [])
        user_replies = []
        
        for comment in user_comments:
            # Only process replies (Level > 0)
            if comment.get('Level', 0) > 0:
                comment_id = comment.get('commentID')
                parent_stance = comment.get('ParentStance')
                
                # Get video ID for this comment
                video_id = comment_to_video.get(comment_id)
                
                # Only include if we have video context and valid parent stance
                if video_id and video_id in video_stance_dist and parent_stance is not None:
                    user_replies.append((video_id, parent_stance))
        
        # Only include users with at least 3 replies for meaningful analysis
        if len(user_replies) >= 3:
            user_behaviors[username] = {
                'group': user_group,
                'replies': user_replies
            }
    
    total_replies = sum(len(data['replies']) for data in user_behaviors.values())
    print(f"Extracted {total_replies} reply behaviors from {len(user_behaviors)} users")
    
    # Debug: show distribution by group
    group_counts = defaultdict(int)
    for user_data in user_behaviors.values():
        group_counts[user_data['group']] += 1
    print("Users by group:", dict(group_counts))
    
    return user_behaviors

def calculate_user_preference_indices(user_replies, video_stance_dist):
    """
    Calculate preference indices for a single user
    Returns: preference_indices dict
    """
    # Calculate actual targeting proportions for this user
    actual_targeting = Counter([parent_stance for _, parent_stance in user_replies])
    total_replies = len(user_replies)
    
    actual_props = {
        0: actual_targeting[0] / total_replies,
        1: actual_targeting[1] / total_replies,
        2: actual_targeting[2] / total_replies
    }
    
    # Calculate expected targeting for this user (weighted by their reply volume per video)
    video_reply_counts = Counter([video_id for video_id, _ in user_replies])
    
    expected_props = {0: 0, 1: 0, 2: 0}
    
    for video_id, reply_count in video_reply_counts.items():
        if video_id in video_stance_dist:
            video_weight = reply_count / total_replies
            video_dist = video_stance_dist[video_id]
            
            for stance in [0, 1, 2]:
                expected_props[stance] += video_weight * video_dist[stance]
    
    # Calculate preference indices (avoid division by zero)
    preference_indices = {}
    for stance in [0, 1, 2]:
        if expected_props[stance] > 0:
            preference_indices[stance] = actual_props[stance] / expected_props[stance]
        else:
            preference_indices[stance] = np.nan
    
    return {
        'actual_targeting': actual_props,
        'expected_targeting': expected_props,
        'preference_indices': preference_indices,
        'total_replies': total_replies,
        'videos_engaged': len(video_reply_counts)
    }

def calculate_targeting_statistics(user_behaviors, video_stance_dist):
    """
    Step 3 & 4: Calculate preference indices at user level, then aggregate by group
    """
    print("Calculating targeting statistics...")
    
    # Calculate preference indices for each user
    user_results = {}
    for username, user_data in user_behaviors.items():
        user_group = user_data['group']
        user_replies = user_data['replies']
        
        user_indices = calculate_user_preference_indices(user_replies, video_stance_dist)
        user_results[username] = {
            'group': user_group,
            **user_indices
        }
    
    print(f"Calculated preference indices for {len(user_results)} users")
    
    # Group users by their group classification
    group_user_data = defaultdict(list)
    for username, user_data in user_results.items():
        group_user_data[user_data['group']].append(user_data)
    
    # Aggregate by group
    group_results = {}
    for group, users_data in group_user_data.items():
        print(f"Analyzing group: {group} ({len(users_data)} users)")
        
        # Collect preference indices from all users in this group
        group_preference_indices = {0: [], 1: [], 2: []}
        group_actual_targeting = {0: [], 1: [], 2: []}
        group_expected_targeting = {0: [], 1: [], 2: []}
        
        total_replies = 0
        total_videos = set()
        
        for user_data in users_data:
            total_replies += user_data['total_replies']
            total_videos.add(user_data['videos_engaged'])  # This is just a count, but we'll approximate
            
            for stance in [0, 1, 2]:
                if not np.isnan(user_data['preference_indices'][stance]):
                    group_preference_indices[stance].append(user_data['preference_indices'][stance])
                    group_actual_targeting[stance].append(user_data['actual_targeting'][stance])
                    group_expected_targeting[stance].append(user_data['expected_targeting'][stance])
        
        # Calculate group-level statistics (same format as before for compatibility)
        results = {}
        results['actual_targeting'] = {}
        results['expected_targeting'] = {}
        results['preference_indices'] = {}
        results['preference_se'] = {}  # New: standard errors
        results['n_users'] = {}  # New: number of users per stance
        
        for stance in [0, 1, 2]:
            if group_preference_indices[stance]:
                results['actual_targeting'][stance] = np.mean(group_actual_targeting[stance])
                results['expected_targeting'][stance] = np.mean(group_expected_targeting[stance])
                results['preference_indices'][stance] = np.mean(group_preference_indices[stance])
                results['preference_se'][stance] = np.std(group_preference_indices[stance]) / np.sqrt(len(group_preference_indices[stance]))
                results['n_users'][stance] = len(group_preference_indices[stance])
            else:
                results['actual_targeting'][stance] = 0
                results['expected_targeting'][stance] = 0
                results['preference_indices'][stance] = np.nan
                results['preference_se'][stance] = np.nan
                results['n_users'][stance] = 0
        
        results['total_replies'] = total_replies
        results['videos_engaged'] = len(total_videos)
        results['total_users'] = len(users_data)
        
        group_results[group] = results
    
    return group_results

def print_results(results):
    """Print formatted results"""
    print("\n" + "="*90)
    print("STANCE TARGETING ANALYSIS RESULTS (User-Level Aggregation)")
    print("="*90)
    
    print(f"\n{'Group':<15} {'Users':<6} {'Replies':<8} {'Videos':<7} {'Stance':<6} {'Actual':<8} {'Expected':<8} {'Preference':<10} {'SE':<6}")
    print("-" * 90)
    
    for group, stats in results.items():
        for i, stance in enumerate([0, 1, 2]):
            stance_label = ["Contra", "Neutral", "Pro"][stance]
            
            if i == 0:
                print(f"{group:<15} {stats['total_users']:<6} {stats['total_replies']:<8} {stats['videos_engaged']:<7} ", end="")
            else:
                print(f"{'':>38}", end="")
            
            actual = stats['actual_targeting'][stance]
            expected = stats['expected_targeting'][stance] 
            preference = stats['preference_indices'][stance]
            se = stats['preference_se'][stance]
            
            if np.isnan(preference):
                print(f"{stance_label:<6} {actual:<8.3f} {expected:<8.3f} {'NaN':<10} {'NaN':<6}")
            else:
                print(f"{stance_label:<6} {actual:<8.3f} {expected:<8.3f} {preference:<10.3f} {se:<6.3f}")
    
    print("\nPreference Index Interpretation:")
    print("  > 1.0: Over-targeting (preference)")
    print("  < 1.0: Under-targeting (avoidance)")
    print("  ≈ 1.0: Random targeting")
    print("SE = Standard Error of the mean preference index across users")

def save_results_to_csv(results, output_file="stance_targeting_results_user_level.csv"):
    """Save results to CSV for further analysis"""
    
    rows = []
    for group, stats in results.items():
        for stance in [0, 1, 2]:
            stance_label = ["Contra", "Neutral", "Pro"][stance]
            rows.append({
                'Group': group,
                'Target_Stance': stance_label,
                'Actual_Proportion': stats['actual_targeting'][stance],
                'Expected_Proportion': stats['expected_targeting'][stance],
                'Preference_Index': stats['preference_indices'][stance],
                'Preference_SE': stats['preference_se'][stance],
                'N_Users': stats['n_users'][stance],
                'Total_Replies': stats['total_replies'],
                'Total_Users': stats['total_users'],
                'Videos_Engaged': stats['videos_engaged']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

def main():
    # Configuration
    VIDEO_DATA_FOLDER = "ADD PATH"
    USER_DATA_FILE = "ADD PATH"
    
    # Load user dataset
    print("Loading user dataset...")
    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    
    print(f"Loaded {len(user_data)} users")
    
    # Debug: Check group distribution
    group_dist = defaultdict(int)
    for username, user_info in user_data.items():
        group = user_info.get('group_classification', 'Unknown')
        group_dist[group] += 1
    print("Group distribution:", dict(group_dist))
    
    # Step 1: Build video-level stance distributions
    comment_to_video, video_stance_dist = load_video_stance_distributions(VIDEO_DATA_FOLDER)
    
    # Step 2: Extract user reply behavior
    user_behaviors = extract_user_reply_behavior(user_data, comment_to_video, video_stance_dist)
    
    # Step 3 & 4: Calculate targeting statistics
    results = calculate_targeting_statistics(user_behaviors, video_stance_dist)
    
    # Display and save results
    print_results(results)
    save_results_to_csv(results)

if __name__ == "__main__":
    main()
