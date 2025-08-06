import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import MBart50TokenizerFast
import re
import torch
from tqdm import tqdm
import logging
import os
import sys
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "process.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
output_dir = os.path.join("processed")
os.makedirs(output_dir, exist_ok=True)

def setup_environment():
    """Configure environment and device settings."""
    os.makedirs("logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    return device

def load_dataset(file_path):
    """Load and validate the dataset."""
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate required columns for the simple dataset
        required_cols = ['tn', 'en']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
        logger.info(f"Dataset loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def clean_and_normalize_text(df):
    """Clean and normalize text in the dataset."""
    logger.info("Cleaning and normalizing text")
    
    # Check for missing values
    for col in ['tn', 'en']:
        missing = df[col].isnull().sum()
        logger.info(f"Missing values in '{col}' column: {missing}")
        
        # Fill missing values
        if missing > 0:
            df[col] = df[col].fillna("")
    
    # Remove rows with empty text
    df = df[(df['tn'].str.strip() != "") & 
            (df['en'].str.strip() != "")]
    
    return df

def normalize_tunisian_text(text):
    """Normalize Tunisian Arabic text."""
    if not isinstance(text, str):
        return ""
    
    # Normalization rules
    text = text.lower()
    
    # Normalize Arabic characters
    text = re.sub(r"[إأٱآا]", "ا", text)  # Normalize alef variants
    text = re.sub(r"[ة]", "ه", text)  # Normalize taa marbouta to haa
    text = re.sub(r"[ى]", "ي", text)  # Normalize alef maksura to yaa
    
    # Keep Arabic punctuation but normalize multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def normalize_english_text(text):
    """Normalize English text."""
    if not isinstance(text, str):
        return ""
    
    # Basic English normalization
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces
    
    return text.strip()

def setup_tokenizer():
    """Set up and return the tokenizer."""
    logger.info("Loading mBART-50 tokenizer")
    
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50",
            src_lang="ar_AR",
            tgt_lang="en_XX"
        )
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise

def tokenize_batch(texts, tokenizer, lang_code, max_length=128):
    """Tokenize a batch of texts."""
    if lang_code == "ar_AR":
        src_texts = texts
    elif lang_code == "en_XX":
        src_texts = texts
    else:
        raise ValueError(f"Unsupported language code: {lang_code}")
    
    # Process in batches to avoid memory issues
    batch_size = 64
    all_tokens = []
    
    for i in range(0, len(src_texts), batch_size):
        batch_texts = src_texts[i:i+batch_size]
        with torch.no_grad():
            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        all_tokens.extend(encodings.input_ids.tolist())
    
    return all_tokens

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and perform initial data processing."""
    try:
        df = pd.read_csv(file_path)
        
        # Calculate text lengths for analysis - updated column names
        df['tn_length'] = df['tn'].str.len()
        df['en_length'] = df['en'].str.len()
        
        # Split data into train/val/test sets
        total_len = len(df)
        train_df = df[:int(total_len * 0.8)]
        val_df = df[int(total_len * 0.8):int(total_len * 0.9)]
        test_df = df[int(total_len * 0.9):]
        
        return df, train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    try:
        # Setup environment
        device = setup_environment()
        logger.info(f"Using device: {device}")
        
        # Load dataset
        file_path = os.path.join("augmented", "augmented_dataset.csv")
        df = load_dataset(file_path)
        
        # Clean and normalize the dataset
        df = clean_and_normalize_text(df)
        
        # Apply text normalization
        logger.info("Applying text normalization")
        df['tn_normalized'] = df['tn'].apply(normalize_tunisian_text)
        df['en_normalized'] = df['en'].apply(normalize_english_text)
        
        # Set up tokenizer
        tokenizer = setup_tokenizer()
        
        # Tokenize in batches with progress bar
        logger.info("Tokenizing texts")
        tn_tokens = tokenize_batch(
            df['tn_normalized'].tolist(),
            tokenizer,
            lang_code="ar_AR"
        )
        
        en_tokens = tokenize_batch(
            df['en_normalized'].tolist(),
            tokenizer,
            lang_code="en_XX"
        )
        
        # Add tokenized data
        df['tn_tokens'] = [','.join(map(str, tokens)) for tokens in tn_tokens]
        df['en_tokens'] = [','.join(map(str, tokens)) for tokens in en_tokens]
        
        # Compute sequence lengths
        df['tn_length'] = df['tn_normalized'].apply(len)
        df['en_length'] = df['en_normalized'].apply(len)
        
        # Split dataset with stratification
        df['length_bin'] = pd.qcut(df['tn_length'], 10, labels=False, duplicates='drop')
        
        logger.info("Splitting dataset")
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42,
            stratify=df['length_bin']
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42,
            stratify=temp_df['length_bin']
        )
        
        # Remove stratification column
        for dataset in [train_df, val_df, test_df]:
            dataset.drop(columns=['length_bin'], inplace=True)
        
        # Save processed datasets
        logger.info("Saving processed datasets")
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val_dataset.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)
        
        # Save small test set
        train_small = train_df.sample(min(1000, len(train_df)), random_state=42)
        train_small.to_csv(os.path.join(output_dir, "train_small_dataset.csv"), index=False)
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        logger.info(f"Average Tunisian text length: {df['tn_length'].mean():.2f} chars")
        logger.info(f"Average English text length: {df['en_length'].mean():.2f} chars")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()