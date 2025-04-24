import os
import requests

# Directory to save the novels
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "doyle_novels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

books = {
    "A Study in Scarlet": "https://www.gutenberg.org/files/244/244-0.txt",
    "The Sign of the Four": "https://www.gutenberg.org/files/2097/2097-0.txt",
    "The Adventures of Sherlock Holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "The Memoirs of Sherlock Holmes": "https://www.gutenberg.org/files/834/834-0.txt",
    "The Hound of the Baskervilles": "https://www.gutenberg.org/files/2852/2852-0.txt",
    "The Return of Sherlock Holmes": "https://www.gutenberg.org/files/108/108-0.txt",
    "His Last Bow": "https://www.gutenberg.org/files/2350/2350-0.txt",
    "The Valley of Fear": "https://www.gutenberg.org/files/3289/3289-0.txt",
    "The Case-Book of Sherlock Holmes": "https://www.gutenberg.org/files/69700/69700-0.txt"
}


for title, url in books.items():
    print(f"Downloading: {title}")
    response = requests.get(url)
    filename = title.replace(" ", "_").lower() + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)

print(f"All books downloaded into {OUTPUT_DIR}")
