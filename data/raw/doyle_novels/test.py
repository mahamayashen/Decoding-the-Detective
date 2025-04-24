import re
import os
import requests

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


# let's test with the hound of the baskervilles

if __name__ == "__main__":    
    # Read the raw text file
    with open("the_hound_of_the_baskervilles.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Clean the text
    cleaned_text = clean_gutenberg_text(raw_text)
    
    # Print the first 500 characters of the cleaned text
    print(cleaned_text[:500])
    # Print the last 500 characters of the cleaned text
    print(cleaned_text[-500:])