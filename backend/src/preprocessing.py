import os
import re
import pandas as pd
from datasets import load_dataset

def clean_text(text):
    """
    Cleans input text by converting to lowercase, 
    removing punctuation, and stripping whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def main():
    print("Loading dataset 'IsmaelMousa/engsaf'...")
    dataset = load_dataset("IsmaelMousa/engsaf")
    print(dataset)
    
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    for split_name in dataset.keys():
        print(f"\nProcessing split: {split_name}")
        df = pd.DataFrame(dataset[split_name])
        
        df['student_answer_clean'] = df['student_answer'].apply(clean_text)
        df['reference_answer_clean'] = df['reference_answer'].apply(clean_text)
        
        df.dropna(subset=['student_answer_clean', 'reference_answer_clean', 'score'], inplace=True)
        df = df[(df['student_answer_clean'] != "") & (df['reference_answer_clean'] != "")]

        filename = f"engsaf_{split_name}.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved {filename} with {len(df)} records to {output_path}.")

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()