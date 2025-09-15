from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import time


class YouTubeChannelQueryAnalyzer:
    def __init__(self, api_key):
        """Initialize the YouTube API client."""
        self.youtube = build('youtube', 'v3', developerKey=api_key)


        self.queries = [
            # Policy and control focus
            '"immigration policy" OR "border security" OR "border wall" OR "ICE" OR "deportation"',

            # People-focused terms
            '"immigrants" OR "refugees" OR "asylum seekers" OR "migrant caravan" OR "DACA"',

            # Crisis/issue framing
            '"immigration crisis" OR "border crisis" OR "illegal immigration" OR "undocumented immigrants"'
        ]

        self.verification_terms = {
            'immigr', 'migr', 'refugee', 'asylum', 'foreign', 'border', 'deport', 'admit',
            'integration', 'alien', 'illegal', 'undocument',

            'visa', 'citizenship', 'natural', 'green card', 'DACA', 'dreamer', 'ICE', 'CBP',
            'detention', 'separat', 'family', 'title 42', 'remain in mexico', 'MPP',

            'mexican border', 'southern border', 'northern border', 'wall', 'fence',
            'caravan', 'cross', 'entry', 'port of entry',

            'migrant', 'asylum seeker', 'unaccompanied', 'child', 'children', 'famil',

            'process', 'status', 'pathway', 'legal', 'amnest', 'crackdown', 'raid',
            'enforce', 'patrol', 'detain', 'arrest', 'court', 'judge', 'hearing'
        }


        self.total_videos_found = 0
        self.videos_filtered_out = 0
        self.filter_stats_by_channel = {}
        self.filter_stats_by_query = {}

    def verify_content_relevance(self, title, description):
        """
        Verify if the content is actually related to immigration topics
        by checking key terms in title or description.
        """
        content = (title + ' ' + description).lower()

        return any(term in content for term in self.verification_terms)

    def search_channel_videos(self, channel_id, channel_name, query):
        """Search for videos matching query within a specific channel and track all results."""
        videos = []
        next_page_token = None

        if channel_id not in self.filter_stats_by_channel:
            self.filter_stats_by_channel[channel_id] = {
                'name': channel_name,
                'total': 0,
                'filtered_out': 0,
                'kept': 0
            }

        if query not in self.filter_stats_by_query:
            self.filter_stats_by_query[query] = {
                'total': 0,
                'filtered_out': 0,
                'kept': 0
            }

        while True:
            try:
                search_params = {
                    'channelId': channel_id,
                    'q': query,
                    'type': 'video',
                    'part': 'id,snippet',
                    'maxResults': 50,
                    'order': 'relevance',
                    'relevanceLanguage': 'en',  # Add language restriction
                    'publishedAfter': '2020-01-01T00:00:00Z',  # Add date range filter
                    'publishedBefore': '2024-12-31T00:00:00Z'  # End date (current)
                }

                if next_page_token:
                    search_params['pageToken'] = next_page_token

                search_response = self.youtube.search().list(**search_params).execute()

                # Track videos found in this batch
                videos_in_batch = len(search_response['items'])
                self.total_videos_found += videos_in_batch
                self.filter_stats_by_channel[channel_id]['total'] += videos_in_batch
                self.filter_stats_by_query[query]['total'] += videos_in_batch

                filtered_in_batch = 0

                for item in search_response['items']:
                    title = item['snippet']['title']
                    description = item['snippet']['description']

                    # Check relevance
                    is_relevant = self.verify_content_relevance(title, description)

                    # Add all videos to the list, with a relevance flag
                    video_info = {
                        'video_id': item['id']['videoId'],
                        'channel_id': channel_id,
                        'channel_name': item['snippet']['channelTitle'],
                        'title': title,
                        'description': description,
                        'published_at': item['snippet']['publishedAt'],
                        'query': query,
                        'passed_filter': is_relevant  # Add this flag
                    }
                    videos.append(video_info)

                    # Track filtering results
                    if is_relevant:
                        filtered_in_batch += 1

                # Update filtered stats
                filtered_out_in_batch = videos_in_batch - filtered_in_batch
                self.videos_filtered_out += filtered_out_in_batch
                self.filter_stats_by_channel[channel_id]['filtered_out'] += filtered_out_in_batch
                self.filter_stats_by_channel[channel_id]['kept'] += filtered_in_batch
                self.filter_stats_by_query[query]['filtered_out'] += filtered_out_in_batch
                self.filter_stats_by_query[query]['kept'] += filtered_in_batch

                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break

            except Exception as e:
                print(f"Error searching videos for query '{query}' in channel {channel_id}: {e}")
                time.sleep(10)
                break

        return videos

    def get_comment_counts(self, video_ids):
        """Get only comment counts for a list of video IDs to minimize API usage."""
        comment_data = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i + 50]
            try:
                # Request only statistics part to minimize API usage
                response = self.youtube.videos().list(
                    part='statistics',
                    id=','.join(chunk),
                    fields='items(id,statistics/commentCount)'  # Request only what we need
                ).execute()

                for item in response['items']:
                    comment_data.append({
                        'video_id': item['id'],
                        'comment_count': int(item['statistics'].get('commentCount', 0))
                    })
            except Exception as e:
                print(f"Error getting comment counts: {e}")
                time.sleep(10)  # Wait before retrying
                continue

        return comment_data

    def analyze_channels(self, channels_file):
        """Analyze content from multiple channels loaded from CSV."""
        # Load channels from CSV
        channels_df = pd.read_csv(channels_file)

        all_videos = []
        unique_video_ids = set()

        # Process each channel
        for _, channel in channels_df.iterrows():
            print(f"\nProcessing channel: {channel['channel_name']}")

            # Search for each query in the channel
            for query in self.queries:
                print(f"  Searching for query: '{query}'")
                videos = self.search_channel_videos(channel['channel_id'], channel['channel_name'], query)
                print(
                    f"    Found {len([v for v in videos if v['passed_filter']])} relevant videos out of {len(videos)} total")

                # Add only new videos to the collection
                for video in videos:
                    if video['video_id'] not in unique_video_ids:
                        video['channel_leaning'] = channel['Leaning']
                        # Add Source field from the original CSV if it exists
                        if 'Source' in channel:
                            video['Source'] = channel['Source']
                        all_videos.append(video)
                        unique_video_ids.add(video['video_id'])

        print(f"Total unique videos found: {len(all_videos)}")
        print(f"Videos that passed filter: {len([v for v in all_videos if v['passed_filter']])}")
        print(f"Videos that failed filter: {len([v for v in all_videos if not v['passed_filter']])}")

        self.print_filter_statistics()

        # Create video DataFrame with all videos
        videos_df = pd.DataFrame(all_videos)

        # Get comment counts only for videos that passed the filter
        if len([v for v in all_videos if v['passed_filter']]) > 0:
            print("Fetching comment counts for videos that passed filter...")
            filtered_video_ids = [v['video_id'] for v in all_videos if v['passed_filter']]
            comment_data = self.get_comment_counts(filtered_video_ids)
            comments_df = pd.DataFrame(comment_data)

            # Merge comment counts with videos that passed filter
            videos_df = videos_df.merge(comments_df, on='video_id', how='left')

        # Calculate simple channel metrics
        channel_data = []
        for channel_id in videos_df['channel_id'].unique():
            channel_videos = videos_df[videos_df['channel_id'] == channel_id]
            filtered_videos = channel_videos[channel_videos['passed_filter'] == True]
            channel_info = channels_df[channels_df['channel_id'] == channel_id].iloc[0]

            channel_data.append({
                'channel_id': channel_id,
                'channel_name': channel_info['channel_name'],
                'channel_leaning': channel_info['Leaning'],
                'total_videos': len(channel_videos),
                'filtered_videos': len(filtered_videos),
                'queries_appeared_in': len(filtered_videos['query'].unique()) if len(filtered_videos) > 0 else 0,
                'top_query': filtered_videos['query'].mode().iloc[0] if len(filtered_videos) > 0 else 'None'
            })

        channels_df = pd.DataFrame(channel_data)
        channels_df = channels_df.sort_values('filtered_videos', ascending=False)

        return channels_df, videos_df

    def print_filter_statistics(self):
        """Print statistics about filtered videos."""
        print("\n==== FILTER STATISTICS ====")
        print(f"Total videos found: {self.total_videos_found}")
        print(
            f"Videos filtered out: {self.videos_filtered_out} ({self.videos_filtered_out / self.total_videos_found * 100:.2f}%)")
        print(
            f"Videos kept: {self.total_videos_found - self.videos_filtered_out} ({(self.total_videos_found - self.videos_filtered_out) / self.total_videos_found * 100:.2f}%)")

        print("\n== Filter Stats by Query ==")
        for query, stats in self.filter_stats_by_query.items():
            if stats['total'] > 0:
                print(f"Query: '{query}'")
                print(f"  Total: {stats['total']}")
                print(f"  Filtered out: {stats['filtered_out']} ({stats['filtered_out'] / stats['total'] * 100:.2f}%)")
                print(f"  Kept: {stats['kept']} ({stats['kept'] / stats['total'] * 100:.2f}%)")

        print("\n== Top 5 Channels by Filter Rate ==")
        channel_list = []
        for channel_id, stats in self.filter_stats_by_channel.items():
            if stats['total'] > 0:
                filter_rate = stats['filtered_out'] / stats['total'] * 100
                channel_list.append({
                    'name': stats['name'],
                    'total': stats['total'],
                    'filtered_out': stats['filtered_out'],
                    'kept': stats['kept'],
                    'filter_rate': filter_rate
                })

        # Sort by filter rate descending
        channel_list.sort(key=lambda x: x['filter_rate'], reverse=True)

        # Print top 5
        for i, channel in enumerate(channel_list[:5]):
            print(f"{i + 1}. {channel['name']}")
            print(f"  Total: {channel['total']}")
            print(f"  Filtered out: {channel['filtered_out']} ({channel['filter_rate']:.2f}%)")
            print(f"  Kept: {channel['kept']} ({100 - channel['filter_rate']:.2f}%)")


def main():
    API_KEY = 'ADD API KEY'


    analyzer = YouTubeChannelQueryAnalyzer(API_KEY)

    print("Starting analysis...")
    channels_df, videos_df = analyzer.analyze_channels(
        'ADD PATH')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


    channel_name_from_file = channels_df['channel_name'].iloc[0]



    channel_name_for_file = channel_name_from_file.replace(' ', '_')

    channels_df.to_csv(f'{channel_name_for_file}_channel_analysis_{timestamp}.csv', index=False)

    # Filter to only relevant videos
    relevant_videos = videos_df[videos_df['passed_filter'] == True].copy()

    # Reorder columns to match desired output format
    # If 'Source' doesn't exist, create it with empty values
    if 'Source' not in relevant_videos.columns:
        relevant_videos['Source'] = ''

    # Select and reorder columns for output
    relevant_columns = [
        'video_id', 'channel_id', 'channel_name', 'title',
        'description', 'published_at', 'channel_leaning',
        'comment_count', 'Source'
    ]

    # Save only the selected columns
    relevant_videos[relevant_columns].to_csv(f'{channel_name_for_file}_relevant_videos_{timestamp}.csv', index=False)

    print("\nChannel Analysis Summary:")
    print(channels_df[['channel_name', 'total_videos', 'filtered_videos',
                       'channel_leaning', 'queries_appeared_in']].head(10))

    print(f"\nTotal unique channels analyzed: {len(channels_df)}")
    print(f"Total unique videos found: {len(videos_df)}")
    print(f"Videos that passed filter: {len(videos_df[videos_df['passed_filter'] == True])}")
    print(f"Videos that failed filter: {len(videos_df[videos_df['passed_filter'] == False])}")


if __name__ == "__main__":
    main()
