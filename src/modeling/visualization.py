import numpy as np
import os

def save_distribution_plot(fig, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"{name}.html"))

def visualize_most_confident_doc(topic_model, all_probs, output_dir):
    idx = np.argmax([probs.max() for probs in all_probs])
    # Check if the document probabilities are valid
    if np.all(all_probs[idx] == 0):
        print(f"Warning: No valid probabilities for document at index {idx}. Skipping visualization.")
        return None
    fig = topic_model.visualize_distribution(all_probs[idx], custom_labels=True)
    save_distribution_plot(fig, "most_confident_doc", output_dir)
    return idx

def visualize_most_ambiguous_doc(topic_model, all_probs, output_dir):
    idx = np.argmin([probs.max() for probs in all_probs])
    # Check if the document probabilities are valid
    if np.all(all_probs[idx] == 0):
        print(f"Warning: No valid probabilities for document at index {idx}. Skipping visualization.")
        return None
    fig = topic_model.visualize_distribution(all_probs[idx], custom_labels=True)
    save_distribution_plot(fig, "most_ambiguous_doc", output_dir)
    return idx


def visualize_random_novel_vs_script(topic_model, all_probs, doc_labels, output_dir):
    novel_idx = next(i for i, label in enumerate(doc_labels) if label == "novel")
    script_idx = next(i for i, label in enumerate(doc_labels) if label == "script")

    fig_novel = topic_model.visualize_distribution(all_probs[novel_idx], custom_labels=True)
    fig_script = topic_model.visualize_distribution(all_probs[script_idx], custom_labels=True)

    save_distribution_plot(fig_novel, "example_novel_doc", output_dir)
    save_distribution_plot(fig_script, "example_script_doc", output_dir)

    return novel_idx, script_idx

def visualize_topic_flagship_doc(topic_model, all_probs, topic_id, output_dir):
    idx = np.argmax([prob[topic_id] for prob in all_probs])
    fig = topic_model.visualize_distribution(all_probs[idx], custom_labels=True)
    save_distribution_plot(fig, f"topic_{topic_id}_flagship_doc", output_dir)
    return idx
