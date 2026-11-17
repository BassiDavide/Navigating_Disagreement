# Data (dehydrated release)

The datasets are released in **dehydrated** form: comment texts and author names are not distributed.
The files contain only YouTube IDs, metadata, video titles and our annotations and model outputs.
`rehydrate.py` downloads the missing content from the YouTube Data API v3 and rebuilds every file in the
format used by the analysis and model scripts.

| File | Contents |
|---|---|
| `Dataset_Total/<Channel>.zip` | All comments used in the paper (1,579,789 comments, 3,377 videos, 13 channels), one `.jsonl` per video, with predicted stance |
| `active_users.json.zip` | The most active users (517 users, 30,258 comments): group classification, stance, toxicity attributes |
| `Training_Data.jsonl.zip` | NLI training data for the stance model (76,772 rows = 38,386 comments × 2 input types) |
| `Test_Data.jsonl` | Manually annotated gold test set (2,004 comments) |
| `rehydrate.py` | Restores the removed fields |

## Quick start

**Requirements:** Python 3.7 or later and an internet connection. No extra packages are needed.
Rehydrating `Dataset_Total` needs about 2 GB of RAM and 1.5 GB of disk.

1. **Get a YouTube Data API v3 key** (free):
   1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a project (or select an existing one).
   2. Open *APIs & Services → Library*, search for **YouTube Data API v3** and click **Enable**.
   3. Open *APIs & Services → Credentials*, click **Create credentials → API key** and copy the key.
2. **Add the key** to the top of `Data/rehydrate.py`:
   ```python
   API_KEY = "your-key-here"
   ```
   You can also leave the file unchanged and set an environment variable instead:
   `export YOUTUBE_API_KEY="your-key-here"`.
3. **Run the script:**
   ```bash
   cd Data
   python rehydrate.py                              # everything
   python rehydrate.py --datasets test,training     # only the stance-model data
   ```
   Available datasets: `test`, `training`, `active_users`, `total` (= `Dataset_Total`).
   They are processed in this order, smallest first.

### API quota

Each request retrieves 50 comments and costs 1 quota unit. The default free quota is 10,000 units per day.

| Dataset | Comments to download | Requests |
|---|---|---|
| `test` | ~3,000 | ~61 |
| `training` | ~44,700 | ~894 |
| `active_users` | ~45,600 | ~912 |
| `total` | ~1,580,000 | ~31,600 (about 3–4 days) |

All downloads are cached in `rehydrated/comments_cache.jsonl`. When the daily quota runs out, the script
writes every dataset it has already completed and stops. Run it again after the quota resets (midnight
Pacific Time) and it continues from where it stopped. You can also request a higher quota from Google.

## Output

Everything is written to `Data/rehydrated/`, in the same format as the original data:

| Output | Contents |
|---|---|
| `Test_Data.jsonl` | `CommentID`, `ParentCommentText`, `CommentText`, `AuthorName`, `Stance_Label` |
| `Training_Data.jsonl` | `premise`, `hypothesis`, `label`, `original_stance`, `input_type` |
| `active_users.json` | One entry per user, keyed by their YouTube handle |
| `Dataset_Total/<Channel>/Precomments_<VideoID>.predictions.jsonl` | One file per video |
| `missing_comments.txt` | IDs of comments that could not be retrieved |
| `comments_cache.jsonl` | Downloaded comments (lets a re-run skip the API) |

## Dehydrated format

All IDs are real YouTube IDs (`https://www.youtube.com/watch?v=<VideoID>`, `https://www.youtube.com/watch?v=<VideoID>&lc=<CommentID>`).

**`Dataset_Total`:** each record has `CommentID`, `VideoID`, `NumberOfLikes`, `IsReply`, `Timestamp`,
`ChannelLeaning`, `Source`, `PeriodTimestamp`, `VideoTitle` and `predicted_stance`.
`CommentText` and `AuthorName` are restored.

**`active_users`:** users are keyed by a placeholder (`user_0001`, …). After rehydration the key is the user's YouTube handle.
In each comment:
- `parent` is the video ID for top-level comments (`Level` 0) and the ID of the comment being replied to otherwise;
- `context` is the video title for top-level comments, and is restored from `parent` for replies;
- `comment_text` is restored.

**`Test_Data`:** the record has `CommentID` and `Stance_Label`. `ResponseToID` is the comment being replied to,
and `ParentCommentText` is restored from it. For top-level comments `ResponseToID` is `null`, and
`ParentCommentText` already contains the video title.

**`Training_Data`:** every comment appears twice, as a `child_only` row and a `parent_child` row. Each row has
`CommentID`, `hypothesis`, `label`, `original_stance` and `input_type`. The `premise` is rebuilt as:
- `child_only`: `"Context: <comment>"`;
- `parent_child`: `"Context: <parent>\nComment: <comment>"`, where `<parent>` is the text of `ResponseToID`,
  or the given `Context` (the video title) for top-level comments.

## Limitations

- **Unavailable comments.** Comments deleted, hidden or made private after collection cannot be retrieved.
  Their fields are set to `null`, and in the training data the whole `premise` is `null`.
  Their IDs are listed in `missing_comments.txt`. All records are kept so that labels stay aligned; drop the `null` rows before training.
- **Edited comments.** The API returns the current text, which may differ from the text that was annotated or classified.
- **Author names.** `AuthorName` is the current YouTube handle of the author, which may have changed since collection.
- `NumberOfLikes` and `Timestamp` keep their values at collection time.

The rehydrated data contains third-party content. Use it in line with the
[YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)
and do not redistribute it. The repository's `.gitignore` keeps `Data/rehydrated/` out of git.
