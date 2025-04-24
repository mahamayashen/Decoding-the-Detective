import json
import os
from bertopic import BERTopic
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "topic_modeling")

# Load your preprocessed data

novel_dir = os.path.join(PROJECT_ROOT, "data", "processed", "doyle_novels")
novel_file = os.path.join(novel_dir, "novel_chunks.jsonl")
with open(novel_file, 'r') as f:
    novel_chunks = [json.loads(line) for line in f]

script_dir = os.path.join(PROJECT_ROOT, "data", "processed", "cbs_elementary")
script_file = os.path.join(script_dir, "elementary_chunks.jsonl")
with open(script_file, 'r') as f:
    script_chunks = [json.loads(line) for line in f]

# Extract text from chunks
novel_texts = [chunk["text"] for chunk in novel_chunks]
script_texts = [chunk["text"] for chunk in script_chunks]

# Initialize and train BERTopic model
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    min_topic_size=20,
    verbose=True
)

# Train on both corpora to create a unified topic space
all_texts = novel_texts + script_texts
all_topics, all_probs = topic_model.fit_transform(all_texts)

# Save visualization of topic differences
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "topic_modeling")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# visualize topics
fig = topic_model.visualize_topics()
fig.update_layout(title_text="Topic Modeling Visualization", title_x=0.5)
fig.write_image(os.path.join(OUTPUT_DIR, "topic_model_visualization.png"))
# Save the model
# This file is very large, so be careful with it
# topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))
# Save the topics to a file
topics_file = os.path.join(OUTPUT_DIR, "topics.json")
with open(topics_file, 'w') as f:
    json.dump(topic_model.get_topic_info().to_dict(), f)

# Save the model to HTML for better visualization
model_html = topic_model.visualize_topics()


# Save the HTML visualization
model_html.write_html(os.path.join(OUTPUT_DIR, "topic_model_visualization.html"))