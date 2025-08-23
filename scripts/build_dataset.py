import os
import re
import pandas as pd
import pdfplumber

# Set the base directory for the circulars dataset
base_dir = os.path.abspath("circulars_admin")
data = []         # To hold the original circulars data
augmented = []    # To hold chunked/augmented data

def extract_text(path):
    """
    Extracts all text from a PDF file using pdfplumber.
    Returns the text as a single string, or empty string on error.
    """
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def split_text_into_chunks(text, min_words=30, max_words=100):
    """
    Splits text into chunks for augmentation:
    - Chunks are at least min_words and at most max_words.
    - Paragraphs are combined, and long paragraphs are windowed/sliding.
    Returns a list of text chunks.
    """
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip().split()) > 0]
    chunks = []
    current = ""
    for p in paragraphs:
        # Merge paragraphs if chunk is not yet at max_words
        if len(current.split()) + len(p.split()) < max_words:
            current += " " + p
        else:
            if len(current.split()) > min_words:
                chunks.append(current.strip())
            current = p
    if len(current.split()) > min_words:
        chunks.append(current.strip())

    # Further split chunks that are too long using a sliding window
    windowed = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words // 2):
                window = words[i:i + max_words]
                if len(window) > min_words:
                    windowed.append(" ".join(window))
        else:
            windowed.append(chunk)
    return windowed

print("Scanning circular folders...")

# Walk through all files in base_dir recursively
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dir)
            parts = rel_path.split(os.sep)

            # Print for debugging/folder structure inspection
            print("Found:", rel_path)

            # Expect at least main category and subcategory folders
            if len(parts) < 3:
                print(f"Skipped {file} (unexpected path structure)")
                continue

            main = parts[0]
            sub = parts[1]

            # Extract circular number from filename using pattern (1234)
            match = re.search(r"\((\d{4})\)", file)
            number = match.group(1) if match else "0000"

            # Extract all text from the PDF
            text = extract_text(full_path)

            if text.strip():
                # Add to original dataset
                data.append({
                    "filename": file,
                    "main_category": main,
                    "subcategory": sub,
                    "circular_number": number,
                    "content": text
                })

                # Chunk the text and add augmented entries
                chunks = split_text_into_chunks(text)
                for idx, chunk in enumerate(chunks):
                    augmented.append({
                        "filename": f"{file}_part{idx + 1}",
                        "main_category": main,
                        "subcategory": sub,
                        "circular_number": number,
                        "content": chunk
                    })
            else:
                print(f"Empty content skipped: {file}")

# Save datasets to the dataset directory
os.makedirs("dataset", exist_ok=True)

# Save the original (non-augmented) dataset
df = pd.DataFrame(data)
print(f"Data collected: {len(df)} entries (original)")
print(df.head(2))  # Show a preview

try:
    df.to_csv("dataset/circulars_dataset.csv", index=False, quoting=1, escapechar='\\', encoding='utf-8')
    print("Dataset saved: dataset/circulars_dataset.csv")
except Exception as e:
    print(f"Failed to save dataset: {e}")

# Save the augmented dataset
aug_df = pd.DataFrame(augmented)
try:
    aug_df.to_csv("dataset/circulars_dataset_augmented.csv", index=False, quoting=1, escapechar='\\', encoding='utf-8')
    print("Augmented dataset saved: dataset/circulars_dataset_augmented.csv")
    print(f"Total augmented samples: {len(aug_df)}")
except Exception as e:
    print(f"Failed to save augmented dataset: {e}")
