import os
import re
import json
import spacy
import pandas as pd
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chunk_and_clean import preprocess_file


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

# Load spaCy model for NER and linguistic analysis
nlp = spacy.load("en_core_web_sm")

# Define terms for cultural/temporal analysis
VICTORIAN_TERMS = {
    "telegram", "carriage", "hansom", "cab", "gaslight", "revolver", 
    "constable", "inspector", "pound", "sovereign", "shilling", "guinea",
    "magnifying", "microscope", "tobacco", "pipe", "opium", "cocaine", 
    "morphine", "hackney", "laboratory", "telegraph", "hat", "cane",
    "pocket-watch", "lodgings", "horseback", "railway", "steam"
}

MODERN_TERMS = {
    "smartphone", "text", "dna", "forensic", "database", "internet", 
    "computer", "email", "laptop", "digital", "gps", "nypd", "cell", 
    "rehab", "addiction", "recovery", "trauma", "therapy", "firewall",
    "wifi", "google", "social", "algorithm", "addiction", "scanner",
    "surveillance", "camera", "video", "tech", "app", "message"
}


def is_valid_episode(episode_name):
    # Returns False if episode_name starts with "~" (after stripping whitespace)
    return not episode_name.strip().startswith("~") and episode_name.strip() != ""

def extract_entities(text):
    """Extract named entities using spaCy."""
    doc = nlp(text[:1000000])  # Limit text length to avoid memory issues
    
    # Dictionary to hold entity counts by type
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    # Count entity occurrences
    for label in entities:
        entities[label] = dict(Counter(entities[label]).most_common(10))
    
    return entities

def count_cultural_terms(text):
    """Count Victorian and modern terms in text."""
    lower_text = text.lower()
    words = re.findall(r'\b\w+\b', lower_text)
    
    # Dictionary to track specific term occurrences
    term_mentions = {
        "victorian": {term: lower_text.count(term) for term in VICTORIAN_TERMS if term in lower_text},
        "modern": {term: lower_text.count(term) for term in MODERN_TERMS if term in lower_text}
    }
    
    # Remove terms with zero occurrences
    term_mentions["victorian"] = {k: v for k, v in term_mentions["victorian"].items() if v > 0}
    term_mentions["modern"] = {k: v for k, v in term_mentions["modern"].items() if v > 0}
    
    # Calculate totals
    victorian_count = sum(term_mentions["victorian"].values())
    modern_count = sum(term_mentions["modern"].values())
    
    return {
        "victorian_terms": victorian_count,
        "modern_terms": modern_count,
        "ratio": modern_count / (victorian_count + 1),  # Add 1 to avoid division by zero
        "term_mentions": term_mentions
    }

def analyze_linguistic_features(text):
    """Extract stylometric features using spaCy."""
    doc = nlp(text[:100000])  # Limit text size
    
    # Count sentence types
    sentence_types = {
        "declarative": 0,
        "interrogative": 0,
        "exclamatory": 0
    }
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text.endswith('?'):
            sentence_types["interrogative"] += 1
        elif sent_text.endswith('!'):
            sentence_types["exclamatory"] += 1
        else:
            sentence_types["declarative"] += 1
    
    # Calculate lexical diversity (type-token ratio)
    total_tokens = len(doc)
    unique_tokens = len(set(token.text.lower() for token in doc if not token.is_punct))
    type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Create part-of-speech distribution
    pos_counts = Counter([token.pos_ for token in doc])
    
    return {
        "sentence_types": sentence_types,
        "type_token_ratio": type_token_ratio,
        "avg_sentence_length": sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if doc.sents else 0,
        "pos_distribution": dict(pos_counts.most_common())
    }

def process_novels():
    """Process Doyle's Sherlock Holmes novels."""
    print("Processing Doyle novels...")
    novel_chunks = []
    chunk_id = 0
    
    # Process each novel
    novel_dir = os.path.join(PROJECT_ROOT, "data", "raw", "doyle_novels")
    novel_files = [f for f in os.listdir(novel_dir) if f.endswith(".txt")]
    for file in tqdm(novel_files):
        file_path = os.path.join(novel_dir, file)
        novel_name = file.replace(".txt", "")
        
        # Use preprocess_file to get basic processed chunks
        file_chunks = preprocess_file(
            file_path, 
            is_script=False,
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Process each chunk further for novel-specific metadata
        for i, chunk_data in enumerate(file_chunks):
            chunk_text = chunk_data["text"]
            
            # Extract additional metadata
            entities = extract_entities(chunk_text)
            term_counts = count_cultural_terms(chunk_text)
            
            # Create full chunk data with all required fields
            chunk_id += 1
            novel_chunks.append({
                "chunk_id": f"novel_{novel_name}_chunk_{chunk_id}",
                "source": "doyle",
                "novel": novel_name,
                "text": chunk_text,
                "entities": entities,
                "term_counts": term_counts,
                "linguistic_features": chunk_data["linguistic_features"]
            })
    
    # Save to JSONL file
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "doyle_novels")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "novel_chunks.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in novel_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✅ Processed {len(novel_chunks)} novel chunks from {len(novel_files)} novels")
    return novel_chunks


def process_elementary_scripts():
    """Process Elementary (CBS) scripts."""
    print("Processing Elementary scripts...")
    script_chunks = []
    chunk_id = 0
    
    # Process each script
    script_dir = os.path.join(PROJECT_ROOT, "data", "raw", "cbs_elementary")
    
    valid_episode_count = 0
    script_files = [f for f in os.listdir(script_dir) if f.endswith(".txt")]
    
    for file in tqdm(script_files):
        # Skip files that are not valid episodes
        if not is_valid_episode(file):
            print(f"Skipping invalid episode: {file}")
            continue
            
        valid_episode_count += 1
        file_path = os.path.join(script_dir, file)
        episode_info = file.replace(".txt", "")
        
        # Use preprocess_file to get basic processed chunks
        file_chunks = preprocess_file(
            file_path, 
            is_script=True,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Process each chunk further for script-specific metadata
        for i, chunk_data in enumerate(file_chunks):
            chunk_text = chunk_data["text"]
            
            # Extract additional metadata
            entities = extract_entities(chunk_text)
            term_counts = count_cultural_terms(chunk_text)
            
            # Create full chunk data with all required fields
            chunk_id += 1
            script_chunks.append({
                "chunk_id": f"elementary_{episode_info}_chunk_{chunk_id}",
                "source": "elementary",
                "episode": episode_info,
                "text": chunk_text,
                "entities": entities,
                "term_counts": term_counts,
                "linguistic_features": chunk_data["linguistic_features"]
            })
    
    # Save to JSONL file
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "cbs_elementary")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "elementary_chunks.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in script_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✅ Processed {len(script_chunks)} script chunks from {valid_episode_count} episodes")
    return script_chunks



# Generate summary statistics for the preprocessed data and save to JSON file
def generate_summary_stats(novel_chunks, script_chunks):
    """Generate summary statistics for preprocessed data."""
    print("Generating summary statistics...")
    
    # Calculate victorian and modern term frequencies
    novel_victorian = sum(c["term_counts"]["victorian_terms"] for c in novel_chunks)
    novel_modern = sum(c["term_counts"]["modern_terms"] for c in novel_chunks)
    script_victorian = sum(c["term_counts"]["victorian_terms"] for c in script_chunks)
    script_modern = sum(c["term_counts"]["modern_terms"] for c in script_chunks)
    
    novel_entities = {}
    for chunk in novel_chunks:
        for entity_type, entities in chunk["entities"].items():
            if entity_type not in novel_entities:
                novel_entities[entity_type] = Counter()
            for entity, count in entities.items():
                novel_entities[entity_type][entity] += count

    script_entities = {}
    for chunk in script_chunks:
        for entity_type, entities in chunk["entities"].items():
            if entity_type not in script_entities:
                script_entities[entity_type] = Counter()
            for entity, count in entities.items():
                script_entities[entity_type][entity] += count

    # Process for JSON output
    novel_top_entities = {}
    for entity_type, counter in novel_entities.items():
        novel_top_entities[entity_type] = dict(counter.most_common(10))

    script_top_entities = {}
    for entity_type, counter in script_entities.items():
        script_top_entities[entity_type] = dict(counter.most_common(10))
    
    # Create summary dictionary
    summary = {
        "corpus_stats": {
            "total_novel_chunks": len(novel_chunks),
            "total_script_chunks": len(script_chunks),
            "avg_novel_chunk_length": sum(len(c["text"]) for c in novel_chunks) / len(novel_chunks) if novel_chunks else 0,
            "avg_script_chunk_length": sum(len(c["text"]) for c in script_chunks) / len(script_chunks) if script_chunks else 0,
        },
        "cultural_terms": {
            "victorian_terms_in_novels": novel_victorian,
            "modern_terms_in_novels": novel_modern,
            "victorian_terms_in_scripts": script_victorian,
            "modern_terms_in_scripts": script_modern,
            "novel_ratio": novel_modern / (novel_victorian + 1),
            "script_ratio": script_modern / (script_victorian + 1),
        },
        "top_entities": {
            "novels": {entity_type: dict(counter.most_common(30)) 
                    for entity_type, counter in novel_entities.items()},
            "scripts": {entity_type: dict(counter.most_common(30)) 
                        for entity_type, counter in script_entities.items()},
    }
    }
    
    # Save summary
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist
    output_file = os.path.join(output_dir, "preprocessing_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Generated summary statistics")
    print(f"Novel chunks: {summary['corpus_stats']['total_novel_chunks']}, " +
          f"Script chunks: {summary['corpus_stats']['total_script_chunks']}")

def main():
    """Main preprocessing function."""
    print("Starting preprocessing pipeline...")
    # novel_chunks = process_novels()
    script_chunks = process_elementary_scripts()

    # use novel_chunks and script_chunks from the jsonl files
    novel_chunks = []
    # script_chunks = []
    # # Load novel chunks
    with open(os.path.join(PROJECT_ROOT, "data", "processed", "doyle_novels", "novel_chunks.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            novel_chunks.append(json.loads(line))
    
    # # Load script chunks
    # with open(os.path.join(PROJECT_ROOT, "data", "processed", "cbs_elementary", "elementary_chunks.jsonl"), 'r', encoding='utf-8') as f:
    #     for line in f:
    #         script_chunks.append(json.loads(line))

    generate_summary_stats(novel_chunks, script_chunks)
    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    main()
