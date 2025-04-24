import re
import spacy
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_stage_directions(text: str) -> str:
    """Remove stage directions from scripts."""
    # Remove text in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove text in brackets
    text = re.sub(r'\[[^\]]*\]', '', text)
    return text

def remove_musical_note(text: str) -> str:
    # Remove all occurrences of the musical note ♪ (\u266a)
    return text.replace('\u266a', '')

def clean_gutenberg_text(text):
    """
    Thoroughly clean Project Gutenberg texts by removing headers, footers, and license information.
    
    Args:
        text (str): The raw text content from a Project Gutenberg ebook
        
    Returns:
        str: The cleaned text with headers and footers removed
    """
    # First step: Find and remove everything after the END marker
    # This is the most reliable way to get rid of the license text
    end_patterns = [
        r'^\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$',
        r'^End of (the )?Project Gutenberg.*$',
        r'^END OF THE PROJECT GUTENBERG EBOOK.*$'
    ]
    
    # Find the earliest end marker
    end_index = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            end_index = min(end_index, match.start())
            break
    
    # If we found an end marker, truncate the text
    if end_index < len(text):
        text = text[:end_index].strip()
    
    # Second step: Find and remove the header
    start_patterns = [
        r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'The Project Gutenberg (EBook|eText) of',
        r'Project Gutenberg\'s.*?, by'
    ]
    
    # Find the latest start marker
    start_index = 0
    for pattern in start_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        if matches:
            # Get the last match (in case there are multiple references)
            match = matches[-1]
            potential_start = match.end()
            # Only use this if it's after our current start_index
            if potential_start > start_index:
                start_index = potential_start
    
    # If we found a start marker, trim the text
    if start_index > 0:
        # Skip empty lines after the header
        while start_index < len(text) and text[start_index:start_index+1].isspace():
            start_index += 1
        text = text[start_index:].strip()
    
    # Normalize Unicode characters
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart single quotes
    text = text.replace('\u2014', "--").replace('\u2013', "-") # Em/en dashes
    
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

def extract_linguistic_features(doc) -> Dict[str, Any]:
    """Extract linguistic features from a spaCy document."""
    # Count types of sentences
    sentence_types = {"statements": 0, "questions": 0, "exclamations": 0}
    for sent in doc.sents:
        text = sent.text.strip()
        if text.endswith('?'):
            sentence_types["questions"] += 1
        elif text.endswith('!'):
            sentence_types["exclamations"] += 1
        else:
            sentence_types["statements"] += 1
    
    # Calculate average sentence length
    sent_lengths = [len(sent) for sent in doc.sents]
    avg_sent_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    
    # Calculate lexical diversity
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / len(tokens) if tokens else 0
    
    return {
        "sentence_types": sentence_types,
        "avg_sentence_length": avg_sent_length,
        "type_token_ratio": type_token_ratio,
        "pos_counts": dict(sorted({pos: tokens.count(pos) for pos in set(token.pos_ for token in doc)}.items())),
    }

def preprocess_file(file_path: str, is_script: bool = False, 
                   chunk_size: int = 1000, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Process a single file completely."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Apply appropriate cleaning
    if is_script:
        text = remove_stage_directions(text)
        text = remove_musical_note(text)
        
    else:
        text = clean_gutenberg_text(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Process each chunk further
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Skip short chunks
        if len(chunk.strip()) < (50 if is_script else 100):
            continue
            
        # Process with spaCy for linguistic features
        doc = nlp(chunk[:100000])  # Limit size to avoid memory issues
        linguistic_features = extract_linguistic_features(doc)
        
        # Create chunk metadata
        chunk_data = {
            "chunk_id": f"chunk_{i}",
            "text": chunk,
            "linguistic_features": linguistic_features,
        }

        processed_chunks.append(chunk_data)
    
    return processed_chunks
