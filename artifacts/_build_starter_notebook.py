"""
Builds starter.ipynb — the kickoff notebook fellows run first.

Generator approach (rather than hand-crafting JSON) keeps cells readable in
source control and lets us re-emit cleanly when content changes.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


CELLS = [
    md("""
# Discussion Intelligence Toolkit — Starter Notebook

This notebook is your starting point. It uses **Reddit as the reference connector path** because the data is threaded, public, and easy to analyze at useful scale. The rest of the toolkit should stay source-agnostic.

By the end of it you'll have:

1. A small Reddit reference dataset built from filtered Pushshift exports
2. A working pipeline producing canonical `Conversation` objects
3. First EDA on thread structure
4. The ConvoKit Coarse Discourse training/dev/test splits
5. A working baseline through the evaluation harness — your floor for October

Run cells top to bottom. The Pushshift streaming cells take time (5–30 minutes depending on which subreddits you pick); everything after that runs in seconds.
"""),

    md("""
## 0. Setup

In Colab, uncomment the install + clone lines. Locally, assume the toolkit files (`schema.py`, `connector.py`, `reddit_dump.py`, `corpora.py`, `evaluation.py`) are on the Python path alongside this notebook.
"""),

    code("""
# Colab setup — uncomment these lines on first run:
# !pip install -q pandas pyarrow pydantic scikit-learn scipy datasets convokit matplotlib tqdm
# !git clone https://github.com/YOUR_TEAM_REPO.git toolkit
# %cd toolkit

# Optional: mount Drive so filtered parquet survives across Colab sessions
# from google.colab import drive
# drive.mount('/content/drive')
"""),

    code("""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# Toolkit modules
from schema import Conversation, Speaker, Utterance
from reddit_dump import RedditDumpConfig, RedditDumpConnector
from corpora import CoarseDiscourseLoader
from evaluation import evaluate_classifier
"""),

    md("""
## 1. Build the Reddit reference dataset

This notebook starts with Reddit because it gives you a concrete threaded-discussion source for the first connector. The rest of the toolkit should remain portable to other threaded sources such as GitHub discussions, forum dumps, or Discord-style exports.

The full Pushshift dump is 89 GB across two parquet files. We stream it from Hugging Face and keep only rows from the subreddits we care about. This is a **one-time prep step** — after this cell completes, the filtered parquets live on disk and the connector reads them directly.

**Time expectations:** streaming time depends entirely on how frequent your target subreddits are in the global stream. AskReddit hits in minutes; niche technical subreddits take longer. The `MAX_ITERATIONS` safety bound below caps the wait.
"""),

    code("""
# --- Configure your subreddits ---
# Team A (technical communities):
TARGET_SUBREDDITS = {"MachineLearning", "programming", "datascience"}

# Team B (advice / discussion communities) — uncomment and use instead:
# TARGET_SUBREDDITS = {"changemyview", "explainlikeimfive", "AskReddit"}

# Caps. Bump these later once you've validated the flow end-to-end.
MAX_SUBMISSIONS = 1_000
MAX_COMMENTS = 50_000
MAX_ITERATIONS = 5_000_000  # safety bound on how far to scan into the stream

OUTPUT_DIR = Path("./pushshift_filtered")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Targets: {TARGET_SUBREDDITS}")
print(f"Caps: {MAX_SUBMISSIONS} submissions, {MAX_COMMENTS} comments")
print(f"Output: {OUTPUT_DIR.resolve()}")
"""),

    code("""
# Stream submissions; keep rows matching TARGET_SUBREDDITS
from datasets import load_dataset

ds_subs = load_dataset("fddemarco/pushshift-reddit", split="train", streaming=True)
subs_filtered = []

for i, item in enumerate(tqdm(ds_subs, desc="Scanning submissions")):
    if i >= MAX_ITERATIONS:
        print(f"Hit iteration cap at {i:,}; collected {len(subs_filtered)}")
        break
    if item.get("subreddit") in TARGET_SUBREDDITS:
        subs_filtered.append(item)
    if len(subs_filtered) >= MAX_SUBMISSIONS:
        break

subs_df = pd.DataFrame(subs_filtered)
subs_path = OUTPUT_DIR / "submissions.parquet"
subs_df.to_parquet(subs_path)
print(f"\\nSaved {len(subs_df):,} submissions to {subs_path}")
print(f"Per-subreddit: {subs_df['subreddit'].value_counts().to_dict()}")
"""),

    code("""
# Stream comments; keep rows matching TARGET_SUBREDDITS
ds_comments = load_dataset("fddemarco/pushshift-reddit-comments", split="train", streaming=True)
comments_filtered = []

for i, item in enumerate(tqdm(ds_comments, desc="Scanning comments")):
    if i >= MAX_ITERATIONS:
        print(f"Hit iteration cap at {i:,}; collected {len(comments_filtered)}")
        break
    if item.get("subreddit") in TARGET_SUBREDDITS:
        comments_filtered.append(item)
    if len(comments_filtered) >= MAX_COMMENTS:
        break

comments_df = pd.DataFrame(comments_filtered)
comments_path = OUTPUT_DIR / "comments.parquet"
comments_df.to_parquet(comments_path)
print(f"\\nSaved {len(comments_df):,} comments to {comments_path}")
"""),

    md("""
## 2. Assemble Conversations via the Connector

The `RedditDumpConnector` reads the filtered parquet files and produces canonical `Conversation` objects. From here on, the toolkit downstream only sees `Conversation` — it never knows the data came from Pushshift.
"""),

    code("""
config = RedditDumpConfig(
    submissions_path=subs_path,
    comments_path=comments_path,
    min_comments=3,  # skip threads with fewer than 3 replies
)
connector = RedditDumpConnector(config)

# Materialize a few hundred for EDA; for full bulk processing use lazy iteration.
conversations = list(connector.list_conversations(limit=200))
print(f"Assembled {len(conversations)} conversations after min_comments filter")
"""),

    code("""
# Peek at one conversation end-to-end
sample = conversations[0]
print(f"ID:         {sample.id}")
print(f"Title:      {sample.title}")
print(f"Subreddit:  {sample.metadata.get('subreddit')}")
print(f"URL:        {sample.url}")
print(f"Utterances: {len(sample)}    Speakers: {len(sample.speakers)}")
print()
print("--- First 5 utterances in DFS order ---")
for u in list(sample.traverse())[:5]:
    speaker = sample.speaker_of(u)
    indent = "  " * u.depth
    snippet = u.text[:120] + ("..." if len(u.text) > 120 else "")
    print(f"{indent}@{speaker.handle} (score={u.score}): {snippet}")
"""),

    md("""
## 3. First EDA on thread structure

Before modeling, look at what we actually have. The distributions below directly inform decisions in September — choice of max sequence length, expected batch shapes, whether to truncate or filter long threads.
"""),

    code("""
# Thread size distribution
thread_sizes = [len(c) for c in conversations]
plt.figure(figsize=(10, 4))
plt.hist(thread_sizes, bins=30, edgecolor='black')
plt.xlabel("Utterances per conversation")
plt.ylabel("Frequency")
plt.title(f"Thread size distribution (n={len(conversations)})")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

ts = pd.Series(thread_sizes)
print(f"Median: {ts.median():.0f}    p95: {ts.quantile(0.95):.0f}    max: {ts.max()}")
"""),

    code("""
# Thread depth distribution
depths = [max(u.depth for u in c.utterances) for c in conversations]
plt.figure(figsize=(10, 4))
plt.hist(depths, bins=range(0, max(depths) + 2), edgecolor='black')
plt.xlabel("Max thread depth")
plt.ylabel("Conversations")
plt.title("Thread depth distribution")
plt.grid(True, alpha=0.3)
plt.show()
print(f"Mean depth: {pd.Series(depths).mean():.1f}    max: {max(depths)}")
"""),

    code("""
# Utterance score distribution (Reddit upvotes - downvotes)
all_scores = [u.score for c in conversations for u in c.utterances if u.score is not None]
plt.figure(figsize=(10, 4))
plt.hist(all_scores, bins=50, edgecolor='black')
plt.xlabel("Utterance score")
plt.ylabel("Frequency")
plt.title("Score distribution across all utterances")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
"""),
    md("""
## 4. Load the ConvoKit Coarse Discourse Corpus

The Reddit reference data has no discourse labels. To train the discourse-act classifier in October, we use Cornell's Coarse Discourse Corpus — about 115k Reddit utterances with human-annotated labels such as question, answer, agreement, disagreement, humor, announcement, appreciation, negative reaction, and elaboration.

The loader downloads the corpus on first call (about 50 MB) and produces leakage-free train/dev/test splits.
"""),

    code("""
loader = CoarseDiscourseLoader()
splits = loader.load()  # downloads if not cached locally
print(splits.summary())
print(f"Label inventory: {splits.labels}")
"""),

    code("""
# Label distribution in train
label_counts = Counter(ex.label for ex in splits.train)
labels, counts = zip(*sorted(label_counts.items(), key=lambda x: -x[1]))

plt.figure(figsize=(10, 4))
plt.bar(labels, counts, edgecolor='black')
plt.xlabel("Discourse act")
plt.ylabel("Train examples")
plt.title("Coarse Discourse label distribution (train split)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# A few examples per top label
for label in labels[:3]:
    examples = [ex for ex in splits.train if ex.label == label][:2]
    print(f"\\n--- {label} ---")
    for ex in examples:
        snippet = ex.text[:160].replace("\\n", " ")
        print(f"  {snippet}")
"""),

    md("""
## 5. Run a dummy baseline through the evaluation harness

This exercises the harness flow end-to-end. The majority-class baseline is your floor — anything you build in October needs to beat it comfortably. If it doesn't, something is wrong with your training loop, not your model.

Notice the model satisfies `TextClassifier` without inheriting from anything. Any object with a `.predict(texts) -> list[str]` method works. Use that flexibility.
"""),

    code("""
class MajorityBaseline:
    \"\"\"Always predicts the most common label seen in training.\"\"\"
    def __init__(self, train_labels):
        self.majority = Counter(train_labels).most_common(1)[0][0]

    def predict(self, texts):
        return [self.majority] * len(texts)


baseline = MajorityBaseline([ex.label for ex in splits.train])
print(f"Majority label: {baseline.majority}")

result = evaluate_classifier(
    baseline,
    texts=[ex.text for ex in splits.test],
    labels=[ex.label for ex in splits.test],
)
print()
print(f"Majority baseline on Coarse Discourse test split:")
print(f"  Accuracy:     {result.accuracy:.3f}")
print(f"  Macro F1:     {result.macro_f1:.3f}")
print(f"  Weighted F1:  {result.weighted_f1:.3f}")
print()
print("Per-class F1 (sorted):")
for lbl, f1 in sorted(result.per_class_f1.items(), key=lambda x: -x[1]):
    print(f"  {lbl:>22s}  {f1:.3f}")
"""),

    md("""
## What's next

You now have:

- A small Reddit reference dataset built from filtered Pushshift exports
- A working pipeline producing canonical `Conversation` objects
- First EDA on thread structure
- The ConvoKit Coarse Discourse training splits
- A baseline that establishes your floor

**September milestone:** EDA, schema normalization, baseline classifiers, and the first checked-in evaluation harness.

**October milestone:** Fine-tune a compact supervised model for discourse-act classification, add the quality scorer, and build the topic clusterer. Beat the baseline with measured gains, not anecdotes.

**November milestone:** Integrate components into a single `analyze(thread) -> InsightReport` pipeline. Add evidence retrieval and LLM synthesis. Package the toolkit so another user can run it without reading your notebook history.

Keep this notebook around. Return to the EDA cells whenever you change something about the data prep — the distributions are your reality check.
"""),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main() -> None:
    out = Path(__file__).parent / "starter.ipynb"
    nbf.write(build_notebook(), out)
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
