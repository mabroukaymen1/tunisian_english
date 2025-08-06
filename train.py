import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import json
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    MBart50Tokenizer,  # Changed from MBart50TokenizerFast
    MBartForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sacrebleu import corpus_bleu
import logging
import random
from pathlib import Path
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import psutil
from datetime import datetime, timedelta
import colorama
from colorama import Fore, Style
import signal
import sys
import io
import codecs

# Fix console output encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize colorama for colored terminal output
colorama.init()

try:
    # Import bitsandbytes for 8-bit quantization if available
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print(f"{Fore.YELLOW}bitsandbytes not installed. Install for 8-bit quantization with: pip install bitsandbytes{Style.RESET_ALL}")

# Enhanced logging setup with timestamp in filename
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'training_{current_time}.log'

# Define custom log formatter with colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.CYAN,
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{Style.RESET_ALL}"
        return log_message

# Create logger
logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler with colored output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = ColoredFormatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# Memory management functions
def cleanup_memory():
    """Perform memory cleanup for CPU and GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def log_memory_usage(tag=""):
    """Log memory usage with optional tag"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
    if tag:
        logger.info(f"{Fore.CYAN}Memory Usage [{tag}] - RAM: {ram_usage:.2f} MB, GPU: {gpu_memory:.2f} MB{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.CYAN}Memory Usage - RAM: {ram_usage:.2f} MB, GPU: {gpu_memory:.2f} MB{Style.RESET_ALL}")
    
    return ram_usage, gpu_memory

# Function to print a dynamic training status dashboard in terminal
def print_training_dashboard(epoch, total_epochs, current_step, total_steps, 
                            train_loss, val_loss, learning_rate, best_val_loss, 
                            shard_info=None, epoch_time=None, eta=None, gpu_memory=None, 
                            examples=None, translations=None):
    """Prints a clean, informative training dashboard"""
    # Use ANSI escape sequence to move cursor to home position and clear screen
    # This works better across different terminals
    print("\033[H\033[J", end="")
    
    # Calculate progress bar
    progress = int(30 * current_step / (total_steps or 1))
    progress_bar = '█' * progress + '░' * (30 - progress)
    
    # Header
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"   Neural Machine Translation Training - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    # Training progress
    if shard_info:
        shard_id = shard_info.get('shard_id', 0) + 1
        total_shards = shard_info.get('total_shards', 1)
        print(f"{Fore.YELLOW}Progress:{Style.RESET_ALL} Epoch {epoch}/{total_epochs} - Shard {shard_id}/{total_shards}")
    else:
        print(f"{Fore.YELLOW}Progress:{Style.RESET_ALL} Epoch {epoch}/{total_epochs}")
    
    print(f"{Fore.YELLOW}[{progress_bar}] {current_step/max(total_steps, 1)*100:.1f}% - Step {current_step}/{total_steps}{Style.RESET_ALL}")
    
    # Metrics
    print(f"\n{Fore.GREEN}Metrics:{Style.RESET_ALL}")
    print(f"  • Train Loss:    {train_loss:.4f}")
    print(f"  • Val Loss:      {val_loss:.4f} (Best: {best_val_loss:.4f})")
    print(f"  • Learning Rate: {learning_rate:.6f}")
    
    # Hardware / Time
    print(f"\n{Fore.MAGENTA}System:{Style.RESET_ALL}")
    if gpu_memory:
        print(f"  • GPU Memory:    {gpu_memory:.2f} MB")
    if epoch_time:
        print(f"  • Epoch Time:    {timedelta(seconds=epoch_time)}")
    if eta:
        print(f"  • ETA:           {timedelta(seconds=eta)}")
    
    # Example translations (if available)
    if examples and translations and len(examples) == len(translations):
        print(f"\n{Fore.BLUE}Sample Translations:{Style.RESET_ALL}")
        for i, (src, tgt) in enumerate(zip(examples, translations)):
            if i >= 2:  # Limit to 2 examples to avoid cluttering the display
                break
            print(f"  • {Fore.CYAN}Input:{Style.RESET_ALL}  {src}")
            print(f"    {Fore.GREEN}Output:{Style.RESET_ALL} {tgt}")
            print()
    
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

# Function to estimate training time
def estimate_training_time(num_examples, batch_size, num_epochs, gradient_accumulation_steps):
    """Estimates training time based on hardware and dataset size"""
    # RTX 3050 4GB estimated processing speed (examples per second)
    examples_per_second = 2.5  # Updated processing speed estimate for 8-bit quantization
    
    total_steps = (num_examples / (batch_size * gradient_accumulation_steps)) * num_epochs
    estimated_seconds = total_steps / examples_per_second
    
    return timedelta(seconds=estimated_seconds)

# Enhanced configuration for RTX 3050 4GB with sharding support
class TrainingConfig:
    def __init__(self):
        # Data paths
        self.train_path = "processed/train_dataset.csv"
        self.val_path = "processed/val_dataset.csv"
        self.test_path = "processed/test_dataset.csv"
        
        # Model parameters
        self.model_name = "facebook/mbart-large-50"
        self.src_lang = "ar_AR"  # Tunisian Arabic
        self.tgt_lang = "en_XX"
        
        # Training settings - optimized for RTX 4070
        self.batch_size = 4  # Increased for RTX 4070's 12GB VRAM
        self.eval_batch_size = 8
        self.max_length = 48
        self.eval_max_length = 64
        self.num_epochs = 100   # Increased to 100 epochs
        self.learning_rate = 2e-5  # Slightly adjusted for longer training
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 1000  # Increased for longer training
        self.gradient_accumulation_steps = 16  # Reduced due to larger batch size
        self.max_grad_norm = 1.0
        self.patience = 3  # Increased patience for longer training
        
        # Hardware/performance - optimized for RTX 4070
        self.num_workers = 2  # Increased for better data loading
        self.fp16 = True  # Enable mixed precision for RTX 4070
        self.use_8bit = False  # Not needed for 12GB VRAM
        self.cpu_offload = False  # Not needed for 12GB VRAM
        
        # Sharding configuration
        self.use_sharding = True
        self.num_shards = 4     # Reduced number of shards
        self.shard_cycles = 25  # Adjusted to achieve 100 epochs (4 * 25 = 100)
        
        # Monitoring
        self.logging_steps = 50
        self.dashboard_update_steps = 10
        self.save_steps = 500
        self.eval_steps = 200
        self.memory_log_steps = 100
        self.use_dashboard = True
        
        # Checkpointing
        self.checkpoint_interval = 5  # Save every 5 epochs
        self.keep_checkpoint_max = 3  # Keep last 3 checkpoints
        
        # Early stopping
        self.early_stopping = True
        self.early_stopping_min_delta = 0.0005  # More sensitive for longer training
        self.early_stopping_patience = 5  # More patience for longer training
        
        # Memory optimization
        self.gradient_checkpointing = False  # Not needed for RTX 4070
        
        # Directories
        self.output_dir = "models"
        self.cache_dir = "cache"
        self.tensorboard_dir = f"runs/{current_time}"
        self.log_dir = "logs"
        self.plot_dir = "plots"
        
        # Create directories
        for directory in [self.output_dir, self.cache_dir, self.tensorboard_dir, self.log_dir, self.plot_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save config for reproducibility
        with open(f"{self.output_dir}/config_{current_time}.json", 'w') as f:
            config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                          for k, v in self.__dict__.items()}
            json.dump(config_dict, f, indent=4)

# Function to handle graceful exit
def setup_graceful_exit():
    """Setup signal handling for graceful exit"""
    def signal_handler(sig, frame):
        logger.info(f"{Fore.YELLOW}Received interrupt signal, gracefully saving model before exit...{Style.RESET_ALL}")
        # Global flag to indicate exit request
        global exit_requested
        exit_requested = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global exit_requested
    exit_requested = False
    
    return exit_requested

# Function to create dataset shards
def create_dataset_shards(config):
        """
        Create shards from the full dataset to allow training on limited GPU memory.
        Returns information about the shards for cycling during training.
        """
        # Number of shards to create
        num_shards = config.num_shards
        
        # Calculate total number of samples in the full dataset
        logger.info(f"{Fore.BLUE}Counting samples in dataset...{Style.RESET_ALL}")
        
        # Count lines in file to determine dataset size
        try:
            total_train_samples = sum(1 for _ in open(config.train_path, 'r', encoding='utf-8')) - 1  # subtract header
            total_val_samples = sum(1 for _ in open(config.val_path, 'r', encoding='utf-8')) - 1  # subtract header
        except Exception as e:
            logger.error(f"Error counting samples in dataset: {e}")
            # Fallback to reasonable defaults
            total_train_samples = 20000
            total_val_samples = 2000
            logger.warning(f"Using default dataset size: {total_train_samples} train, {total_val_samples} val")
        
        logger.info(f"{Fore.GREEN}Full training dataset: {total_train_samples} samples{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Full validation dataset: {total_val_samples} samples{Style.RESET_ALL}")
        
        # Create shard information
        train_samples_per_shard = total_train_samples // num_shards
        val_samples_per_shard = total_val_samples // num_shards
        
        train_shards = []
        val_shards = []
        
        for i in range(num_shards):
            start_idx = i * train_samples_per_shard
            end_idx = (i + 1) * train_samples_per_shard if i < num_shards - 1 else total_train_samples
            
            train_shards.append({
                'shard_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': end_idx - start_idx
            })
            
            # Also create validation shards
            val_start_idx = i * val_samples_per_shard
            val_end_idx = (i + 1) * val_samples_per_shard if i < num_shards - 1 else total_val_samples
            
            val_shards.append({
                'shard_id': i,
                'start_idx': val_start_idx,
                'end_idx': val_end_idx,
                'size': val_end_idx - val_start_idx
            })
        
        shards_info = {
            'total_shards': num_shards,
            'train_shards': train_shards,
            'val_shards': val_shards,
            'total_train_samples': total_train_samples,
            'total_val_samples': total_val_samples
        }
        
        logger.info(f"{Fore.GREEN}Created {num_shards} dataset shards:{Style.RESET_ALL}")
        for i, shard in enumerate(train_shards):
            logger.info(f"  Shard {i+1}: {shard['size']} samples (indices {shard['start_idx']} to {shard['end_idx']})")
        
        return shards_info

# Function to load a specific dataset shard
def load_dataset_shard(path, shard_info, chunk_size=1000, use_full_validation=False):
    """
    Load a specific shard of the dataset.
    
    Args:
        path: Path to the dataset CSV
        shard_info: Information about the shard to load
        chunk_size: Size of chunks to load at once
        use_full_validation: For validation shards, whether to load the full validation set
                            instead of just the shard (recommended for consistent evaluation)
    
    Returns:
        src_texts, tgt_texts for the shard
    """
    shard_id = shard_info['shard_id']
    start_idx = shard_info['start_idx']
    end_idx = shard_info['end_idx']
    
    # For validation data, we usually want to use the full set for proper evaluation
    if "val" in path.lower() and use_full_validation:
        logger.info(f"{Fore.BLUE}Loading full validation dataset for consistent evaluation{Style.RESET_ALL}")
        start_idx = 0
        end_idx = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1  # All samples except header
    
    logger.info(f"{Fore.BLUE}Loading shard {shard_id+1} from {path} (samples {start_idx} to {end_idx}){Style.RESET_ALL}")
    
    chunks = []
    current_idx = 0
    rows_loaded = 0
    
    # Create progress bar
    total_rows = end_idx - start_idx
    progress_bar = tqdm(total=total_rows, desc=f"Loading shard {shard_id+1}", unit="rows")
    
    # Read the file in chunks to save memory
    for chunk in pd.read_csv(path, chunksize=chunk_size, encoding='utf-8'):
        # Get indices for this chunk
        chunk_start = current_idx
        chunk_end = current_idx + len(chunk)
        
        # Check if this chunk overlaps with our shard
        if chunk_end > start_idx and chunk_start < end_idx:
            # Calculate overlap
            overlap_start = max(chunk_start, start_idx) - chunk_start
            overlap_end = min(chunk_end, end_idx) - chunk_start
            
            # Get the overlapping rows
            shard_chunk = chunk.iloc[overlap_start:overlap_end]
            
            if len(shard_chunk) > 0:
                chunks.append(shard_chunk)
                rows_loaded += len(shard_chunk)
                progress_bar.update(len(shard_chunk))
        
        current_idx += len(chunk)
        
        # If we've passed the end of our shard, we can stop
        if chunk_start >= end_idx:
            break
    
    progress_bar.close()
    
    # Concatenate chunks
    if chunks:
        df = pd.concat(chunks)
        src_texts = df['tn_normalized'].tolist()
        tgt_texts = df['en_normalized'].tolist()
        
        # Clear DataFrame from memory
        del df
        cleanup_memory()
        
        logger.info(f"{Fore.GREEN}Loaded {rows_loaded} samples for shard {shard_id+1}{Style.RESET_ALL}")
        return src_texts, tgt_texts
    else:
        logger.warning(f"No data loaded for shard {shard_id+1} from {path}")
        return [], []

# Ultra memory-efficient dataset class
class EfficientTranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=32):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = "ar_AR"
        self.tgt_lang = "en_XX"
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        # Tokenize on-the-fly to save memory
        self.tokenizer.src_lang = self.src_lang
        src_encoding = self.tokenizer(
            self.src_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        self.tokenizer.tgt_lang = self.tgt_lang
        tgt_encoding = self.tokenizer(
            self.tgt_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": src_encoding.input_ids.squeeze(),
            "attention_mask": src_encoding.attention_mask.squeeze(),
            "labels": tgt_encoding.input_ids.squeeze()
        }

# Memory efficient evaluation with better visualization
def evaluate_model(model, tokenizer, val_loader, device, examples=None):
    """Evaluates model with minimal memory usage and returns translations for dashboard"""
    model.eval()
    total_val_loss = 0
    translations = []
    
    # Process in very small batches with nice progress bar
    eval_progress = tqdm(val_loader, desc=f"{Fore.BLUE}Evaluating{Style.RESET_ALL}", 
                         leave=False, ncols=80)
    
    for batch_idx, batch in enumerate(eval_progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_val_loss += loss
        
        # Release batch from GPU memory
        for k in batch:
            batch[k] = batch[k].cpu()
        del batch, outputs
        
        # Only clean memory occasionally during evaluation
        if batch_idx % 20 == 0:
            cleanup_memory()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Generate example translations if provided
    if examples and len(examples) > 0:
        for example in examples[:2]:
            tokenizer.src_lang = "ar_AR"
            inputs = tokenizer(example, return_tensors="pt", max_length=64, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=64,
                    num_beams=5,
                    early_stopping=True,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                    do_sample=False,
                    repetition_penalty=1.2,
                    length_penalty=1.1
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)
            
            # Release from GPU memory
            del inputs, outputs
    
    return avg_val_loss, translations

# Log GPU information with more details and color
def log_gpu_info():
    """Logs detailed GPU information"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"{Fore.GREEN}PyTorch version: {torch.__version__}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Found {device_count} CUDA device(s){Style.RESET_ALL}")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"{Fore.GREEN}CUDA Device {i}: {device_properties.name}{Style.RESET_ALL}")
            logger.info(f"  Total Memory: {device_properties.total_memory / (1024**3):.2f} GB")
            logger.info(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
            
            # Log more detailed GPU info
            try:
                gpu_utilization = torch.cuda.utilization(i)
                logger.info(f"  Current Utilization: {gpu_utilization}%")
            except:
                pass
    else:
        logger.warning(f"{Fore.RED}No CUDA device available, training will be on CPU (very slow){Style.RESET_ALL}")

# Create dynamic plots during training
def create_training_plots(train_losses, val_losses, lr_history, memory_history, output_dir, bleu_scores=None, val_improvements=None, translation_lengths=None):
    """Creates comprehensive training plots with detailed metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Convert lists to numpy arrays and ensure they have the same length
    max_len = min(len(train_losses), len(val_losses))
    train_losses = np.array(train_losses[:max_len])
    val_losses = np.array(val_losses[:max_len])
    lr_history = np.array(lr_history[:max_len])
    
    # Create figure for main metrics
    fig1 = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig1)
    
    # 1. Training and Validation Loss
    ax1 = fig1.add_subplot(gs[0, :2])
    ax1.plot(train_losses, 'b-', label='Training Loss', alpha=0.7)
    ax1.plot(val_losses, 'r-', label='Validation Loss', alpha=0.7)
    ax1.set_title('Loss Curves', fontsize=12, pad=10)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Learning Rate
    ax2 = fig1.add_subplot(gs[0, 2])
    ax2.plot(lr_history, 'g-', label='Learning Rate', alpha=0.7)
    ax2.set_title('Learning Rate Schedule', fontsize=12, pad=10)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Loss Distribution
    ax3 = fig1.add_subplot(gs[1, 0])
    ax3.hist(train_losses, bins=50, alpha=0.5, color='blue', label='Training')
    ax3.hist(val_losses, bins=50, alpha=0.5, color='red', label='Validation')
    ax3.set_title('Loss Distribution', fontsize=12, pad=10)
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 4. Memory Usage
    ax4 = fig1.add_subplot(gs[1, 1])
    if memory_history and len(memory_history) > 0:
        steps = np.array([x[0] for x in memory_history])
        ram = np.array([x[1] for x in memory_history])
        gpu = np.array([x[2] for x in memory_history])
        ax4.plot(steps, ram, 'm-', label='RAM', alpha=0.7)
        ax4.plot(steps, gpu, 'c-', label='GPU', alpha=0.7)
        ax4.set_title('Memory Usage Over Time', fontsize=12, pad=10)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Memory (MB)')
        ax4.legend()

    # 5. Loss Change Rate
    ax5 = fig1.add_subplot(gs[1, 2])
    train_changes = np.diff(train_losses)
    val_changes = np.diff(val_losses)
    ax5.plot(train_changes, 'b-', label='Train Loss Δ', alpha=0.5)
    ax5.plot(val_changes, 'r-', label='Val Loss Δ', alpha=0.5)
    ax5.set_title('Loss Change Rate', fontsize=12, pad=10)
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Loss Change')
    ax5.legend()

    # 6. Training Progress Matrix
    ax6 = fig1.add_subplot(gs[2, 0])
    # Reshape the data into a 2D matrix for heatmap
    matrix_size = int(np.sqrt(max_len))
    if matrix_size > 1:
        train_matrix = train_losses[:matrix_size**2].reshape(matrix_size, matrix_size)
        im = ax6.imshow(train_matrix, aspect='auto', cmap='viridis')
        ax6.set_title('Training Loss Matrix', fontsize=12, pad=10)
        plt.colorbar(im, ax=ax6)
    else:
        ax6.text(0.5, 0.5, 'Not enough data points', ha='center', va='center')
        ax6.set_title('Training Progress Matrix', fontsize=12, pad=10)

    # 7. Loss Ratio
    ax7 = fig1.add_subplot(gs[2, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        loss_ratio = np.where(val_losses != 0, train_losses/val_losses, 0)
    ax7.plot(loss_ratio, 'purple', alpha=0.7)
    ax7.set_title('Train/Val Loss Ratio', fontsize=12, pad=10)
    ax7.set_xlabel('Steps')
    ax7.set_ylabel('Ratio')
    ax7.axhline(y=1, color='r', linestyle='--', alpha=0.3)

    # 8. Moving Average Convergence
    ax8 = fig1.add_subplot(gs[2, 2])
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        if len(val_losses) >= window:
            ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            ax8.plot(ma, alpha=0.7, label=f'{window}-step MA')
    ax8.set_title('Validation Loss Moving Averages', fontsize=12, pad=10)
    ax8.set_xlabel('Steps')
    ax8.set_ylabel('Loss')
    ax8.legend()

    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
    plt.close()

def create_advanced_analysis_plots(train_losses, val_losses, lr_history, output_dir):
    """Creates additional plots for advanced training analysis."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Try to use seaborn style, fall back to default if not available
    try:
        import seaborn as sns
        sns.set_style("darkgrid")
    except ImportError:
        plt.style.use('default')
    
    fig2 = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig2)

    # 1. Loss Correlation
    ax1 = fig2.add_subplot(gs[0, 0])
    ax1.scatter(train_losses, val_losses, alpha=0.5, c='blue')
    ax1.set_title('Train vs Val Loss Correlation', fontsize=12, pad=10)
    ax1.set_xlabel('Training Loss')
    ax1.set_ylabel('Validation Loss')

    # 2. Loss Momentum
    ax2 = fig2.add_subplot(gs[0, 1])
    momentum_window = 5
    train_momentum = [sum(train_losses[i:i+momentum_window])/momentum_window for i in range(len(train_losses)-momentum_window)]
    val_momentum = [sum(val_losses[i:i+momentum_window])/momentum_window for i in range(len(val_losses)-momentum_window)]
    ax2.plot(train_momentum, 'b-', label='Train Momentum', alpha=0.7)
    ax2.plot(val_momentum, 'r-', label='Val Momentum', alpha=0.7)
    ax2.set_title('Loss Momentum', fontsize=12, pad=10)
    ax2.legend()

    # 3. Learning Rate Impact
    ax3 = fig2.add_subplot(gs[0, 2])
    lr_impact = pd.Series(train_losses).diff() / pd.Series(lr_history[:len(train_losses)]).diff()
    ax3.plot(lr_impact, 'g-', alpha=0.7)
    ax3.set_title('Learning Rate Impact on Loss', fontsize=12, pad=10)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss Change / LR Change')

    # 4. Convergence Speed
    ax4 = fig2.add_subplot(gs[1, 0])
    conv_window = 50
    conv_speed = [(val_losses[i] - val_losses[i-conv_window])/conv_window 
                 if i >= conv_window else 0 
                 for i in range(len(val_losses))]
    ax4.plot(conv_speed, 'purple', alpha=0.7)
    ax4.set_title('Convergence Speed', fontsize=12, pad=10)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Loss Change Rate')

    # 5. Training Stability
    ax5 = fig2.add_subplot(gs[1, 1])
    window = 20
    stability = pd.Series(val_losses).rolling(window).std()
    ax5.plot(stability, 'orange', alpha=0.7)
    ax5.set_title('Training Stability', fontsize=12, pad=10)
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Loss Standard Deviation')

    # 6. Optimization Landscape
    ax6 = fig2.add_subplot(gs[1, 2])
    x = np.linspace(0, len(train_losses)-1, len(train_losses))
    z = np.polyfit(x, train_losses, 3)
    p = np.poly1d(z)
    ax6.plot(x, train_losses, 'b.', alpha=0.3, label='Actual')
    ax6.plot(x, p(x), 'r-', alpha=0.7, label='Trend')
    ax6.set_title('Optimization Landscape', fontsize=12, pad=10)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Main training function with dataset sharding
def train_with_shards():
    # Setup graceful exit handling
    setup_graceful_exit()
    global exit_requested
    
    try:
        # Initialize config and log basic info
        start_time = time.time()
        config = TrainingConfig()
        
        # Print colorful banner
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.YELLOW}    NEURAL MACHINE TRANSLATION TRAINER WITH DATASET SHARDING")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Set environment variables for CUDA memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True")
        
        # Log system info
        log_gpu_info()
        ram, gpu = log_memory_usage("Initial")
        
        # Create shards information
        shards_info = create_dataset_shards(config)
        num_shards = shards_info['total_shards']
        
        # Calculate total epochs based on sharding
        total_epochs = num_shards * config.shard_cycles
        logger.info(f"{Fore.GREEN}Training plan: {config.shard_cycles} cycles through {num_shards} shards = {total_epochs} total epochs{Style.RESET_ALL}")
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=config.tensorboard_dir)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{Fore.GREEN}Using device: {device}{Style.RESET_ALL}")
        
        # Empty CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        tokenizer = MBart50Tokenizer.from_pretrained(  # Changed from MBart50TokenizerFast
            config.model_name,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            cache_dir=config.cache_dir
        )
        
        # Load model with 8-bit quantization if available
        logger.info(f"{Fore.BLUE}Loading model: {config.model_name}{Style.RESET_ALL}")
        
        if config.use_8bit and HAS_BNB:
            logger.info(f"{Fore.GREEN}Using 8-bit quantization with bitsandbytes{Style.RESET_ALL}")
            model = MBartForConditionalGeneration.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            # Standard loading with memory optimizations
            model = MBartForConditionalGeneration.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
            )
            # Move model to device
            model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info(f"{Fore.GREEN}Gradient checkpointing enabled{Style.RESET_ALL}")
        
        # Example sentences for monitoring - UPDATED: varied examples
        examples = [
            "هاو ار يو",          # How are you
            "شنوا اسمك",          # What's your name
            "وين الهاتف متاعي؟",  # Where is my phone?
            "شنيه الطقس اليوم؟",  # What's the weather today?
            "ذكرني باش نشرب دواء", # Remind me to take my medicine
            "باهي نتقابلو غدوة",   # Let's meet tomorrow
        ]
        
        # Prepare optimizer
        if config.use_8bit and HAS_BNB:
            # Use 8-bit optimizer if available
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
            logger.info(f"{Fore.GREEN}Using 8-bit AdamW optimizer{Style.RESET_ALL}")
        else:
            # Standard optimizer with parameter grouping
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
            
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                eps=config.adam_epsilon
            )
        
        # Calculate approximate total training steps across all shards
        avg_examples_per_shard = shards_info['total_train_samples'] / num_shards
        total_steps = int((avg_examples_per_shard / config.batch_size) * total_epochs / config.gradient_accumulation_steps)
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare mixed precision training if not using 8-bit quantization
        if config.fp16 and not config.use_8bit:
            scaler = torch.amp.GradScaler(enabled=config.fp16)
        else:
            scaler = None
        
        # Training statistics
        train_losses = []
        val_losses = []
        lr_history = []
        memory_history = []  # Store (step, ram, gpu) tuples
        bleu_scores = []     # Store BLEU scores for evaluation
        val_improvements = [] # Store validation loss improvements
        translation_lengths = [] # Store lengths of translations for analysis
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        global_step = 0
        
        # Add early stopping with a minimum number of steps
        min_training_steps = 1000  # Minimum steps before allowing early stopping
        
        # Begin training with shard cycling
        logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}STARTING TRAINING WITH DATASET SHARDING{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        logger.info(f"  Total Samples = {shards_info['total_train_samples']}")
        logger.info(f"  Shards = {num_shards}")
        logger.info(f"  Cycles = {config.shard_cycles}")
        logger.info(f"  Total Epochs = {total_epochs}")
        logger.info(f"  Approximate Samples per Shard = {int(avg_examples_per_shard)}")
        logger.info(f"  Batch size = {config.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {config.batch_size * config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")
        
        epoch_times = []
        current_translations = []
        
        # Load validation set for consistent evaluation across all shards
        # We use the full validation set to ensure fair comparisons between shards
        logger.info(f"{Fore.BLUE}Loading full validation set for consistent evaluation{Style.RESET_ALL}")
        val_src, val_tgt = load_dataset_shard(
            config.val_path,
            shards_info['val_shards'][0],
            use_full_validation=True
        )
        
        # Create validation dataset and loader
        val_dataset = EfficientTranslationDataset(
            val_src, val_tgt, tokenizer, 
            max_length=config.max_length
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False
        )
        
        # Run initial evaluation to get baseline performance
        logger.info(f"{Fore.BLUE}Running initial evaluation...{Style.RESET_ALL}")
        initial_val_loss, initial_translations = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            device=device,
            examples=random.sample(examples, 2) # Choose 2 random examples for variety
        )
        val_losses.append(initial_val_loss)
        current_translations = initial_translations
        best_val_loss = initial_val_loss
        
        logger.info(f"{Fore.GREEN}Initial validation loss: {initial_val_loss:.4f}{Style.RESET_ALL}")
        for ex, trans in zip(examples[:2], initial_translations):
            logger.info(f"  • {ex} → {trans}")
        
        # Main training loop with shard cycling
        for epoch in range(total_epochs):
            if exit_requested:
                logger.info("Exit requested, stopping training")
                break
                
            # Calculate which shard to use for this epoch
            shard_index = epoch % num_shards
            shard_cycle = epoch // num_shards + 1
            
            logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
            logger.info(f"{Fore.YELLOW}Epoch {epoch + 1}/{total_epochs} - Cycle {shard_cycle}/{config.shard_cycles} - Shard {shard_index + 1}/{num_shards}{Style.RESET_ALL}")
            logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
            
            epoch_start_time = time.time()
            
            # Load training shard for this epoch
            train_src, train_tgt = load_dataset_shard(
                config.train_path,
                shards_info['train_shards'][shard_index]
            )
            
            # Create training dataset and loader for this shard
            train_dataset = EfficientTranslationDataset(
                train_src, train_tgt, tokenizer, 
                max_length=config.max_length
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=False
            )
            
            logger.info(f"{Fore.GREEN}Training on shard {shard_index + 1} with {len(train_dataset)} samples{Style.RESET_ALL}")
            
            # Training - with enhanced progress bar
            model.train()
            total_train_loss = 0
            epoch_steps = 0
            
            # Information about current shard for display
            current_shard_info = {
                'shard_id': shard_index,
                'total_shards': num_shards,
                'cycle': shard_cycle,
                'total_cycles': config.shard_cycles
            }
            
            # Select new random examples for this epoch to show variety
            current_examples = random.sample(examples, 2)
            
            # Create progress bar with custom formatting
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"{Fore.GREEN}Epoch {epoch + 1}/{total_epochs} - Shard {shard_index + 1}{Style.RESET_ALL}",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100
            )
            
            # Update dashboard at the beginning of the epoch
            if config.use_dashboard:
                print_training_dashboard(
                    epoch=epoch+1, 
                    total_epochs=total_epochs,
                    current_step=0,
                    total_steps=len(train_loader),
                    train_loss=0.0 if len(train_losses) == 0 else train_losses[-1],
                    val_loss=val_losses[-1],
                    learning_rate=scheduler.get_last_lr()[0],
                    best_val_loss=best_val_loss,
                    shard_info=current_shard_info,
                    gpu_memory=gpu,
                    examples=current_examples,
                    translations=current_translations
                )
            
            # Training loop for this shard
            for step, batch in enumerate(train_loader):
                if exit_requested:
                    break
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass - handle with or without mixed precision
                if config.fp16 and scaler:
                    with torch.amp.autocast(device_type='cuda', enabled=config.fp16):
                        outputs = model(**batch)
                        loss = outputs.loss / config.gradient_accumulation_steps
                    
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                    loss.backward()
                
                # Track loss
                loss_value = loss.item() * config.gradient_accumulation_steps
                total_train_loss += loss_value
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Update parameters with or without scaler
                    if config.fp16 and scaler:
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        
                        # Update parameters with scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard gradient clipping and optimizer step
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Save learning rate for plotting
                    lr_history.append(scheduler.get_last_lr()[0])
                    
                    # Zero gradients - more memory efficient way
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Increment step counter
                    global_step += 1
                    
                    # Log metrics only at specified intervals
                    if global_step % config.logging_steps == 0:
                        # Log learning rate and loss
                        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        writer.add_scalar("loss", loss_value, global_step)
                    
                    # Log memory usage only occasionally
                    if global_step % config.memory_log_steps == 0:
                        ram, gpu = log_memory_usage(f"Step {global_step}")
                        memory_history.append((global_step, ram, gpu))
                        
                        # Log to TensorBoard
                        writer.add_scalar("ram_usage_mb", ram, global_step)
                        writer.add_scalar("gpu_usage_mb", gpu, global_step)
                    
                    # Update current loss for progress bar
                    avg_loss_so_far = total_train_loss / epoch_steps
                    train_losses.append(avg_loss_so_far)
                    
                    # Evaluate at specified intervals
                    if global_step % config.eval_steps == 0:
                        # Evaluate and log validation loss
                        logger.info(f"{Fore.BLUE}Evaluating at step {global_step}...{Style.RESET_ALL}")
                        # Get new examples for each evaluation for variety
                        eval_examples = random.sample(examples, 2)
                        val_loss, translations = evaluate_model(
                            model=model,
                            tokenizer=tokenizer,
                            val_loader=val_loader,
                            device=device,
                            examples=eval_examples
                        )
                        
                        val_losses.append(val_loss)
                        current_translations = translations
                        current_examples = eval_examples
                        
                        # Log to TensorBoard
                        writer.add_scalar("val_loss", val_loss, global_step)
                        
                        # Return to training mode
                        model.train()
                        
                        # Check for best model
                        if val_loss < best_val_loss - config.early_stopping_min_delta:
                            logger.info(f"{Fore.GREEN}New best validation loss: {val_loss:.4f} (was {best_val_loss:.4f}){Style.RESET_ALL}")
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                            
                            # Save best model
                            best_model_dir = os.path.join(config.output_dir, "best_model")
                            os.makedirs(best_model_dir, exist_ok=True)
                            
                            try:
                                # Save best model with appropriate method
                                if not config.use_8bit:
                                    model.save_pretrained(best_model_dir)
                                else:
                                    model.save_pretrained(best_model_dir, safe_serialization=False)
                                    
                                tokenizer.save_pretrained(best_model_dir)
                                logger.info(f"{Fore.GREEN}Best model saved to {best_model_dir}{Style.RESET_ALL}")
                            except Exception as e:
                                logger.error(f"Error saving best model: {e}")
                
                # Update progress bar with more info
                progress_bar.set_postfix({
                    "loss": f"{total_train_loss / epoch_steps:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    "mem": f"{torch.cuda.memory_allocated() / (1024 * 1024):.0f}MB" if torch.cuda.is_available() else "N/A"
                })
                
                # Update progress bar
                progress_bar.update(1)
                
                # Update dashboard occasionally
                if config.use_dashboard and step % config.dashboard_update_steps == 0:
                    print_training_dashboard(
                        epoch=epoch+1, 
                        total_epochs=total_epochs,
                        current_step=step+1,
                        total_steps=len(train_loader),
                        train_loss=total_train_loss / epoch_steps,
                        val_loss=val_losses[-1],
                        learning_rate=scheduler.get_last_lr()[0],
                        best_val_loss=best_val_loss,
                        shard_info=current_shard_info,
                        epoch_time=time.time() - epoch_start_time,
                        eta=(time.time() - epoch_start_time) * (len(train_loader) - step - 1) / (step + 1) if step > 0 else None,
                        gpu_memory=gpu,
                        examples=current_examples,
                        translations=current_translations
                    )
                
                # Clean up memory every few batches
                if step % 10 == 0:
                    # Release batch from GPU memory
                    for k in batch:
                        batch[k] = batch[k].cpu()
                    del batch, outputs, loss
                    cleanup_memory()
            
            # End of epoch/shard
            progress_bar.close()
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / max(epoch_steps, 1)
            
            # Evaluate at the end of each shard
            logger.info(f"{Fore.BLUE}Running evaluation after shard {shard_index + 1}...{Style.RESET_ALL}")
            
            # Select new examples for end-of-epoch evaluation
            eval_examples = random.sample(examples, 2)
            
            # We'll use the same validation set for all shards to ensure consistent evaluation
            val_loss, translations = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                val_loader=val_loader,
                device=device,
                examples=eval_examples
            )
            
            val_losses.append(val_loss)
            current_translations = translations
            current_examples = eval_examples
            
            # Log stats with color
            logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
            logger.info(f"{Fore.YELLOW}Epoch {epoch + 1}/{total_epochs} (Shard {shard_index + 1}) completed{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}  Train Loss: {avg_train_loss:.4f}{Style.RESET_ALL}")
            logger.info(f"{Fore.GREEN}  Val Loss:   {val_loss:.4f} (Best: {best_val_loss:.4f}){Style.RESET_ALL}")
            
            # Log example translations
            logger.info(f"{Fore.BLUE}Example translations:{Style.RESET_ALL}")
            for ex, trans in zip(eval_examples[:2], translations):
                logger.info(f"  {ex} → {trans}")
            
            # Record epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Format timings in a human-readable way
            logger.info(f"{Fore.MAGENTA}Epoch time: {timedelta(seconds=epoch_time)}{Style.RESET_ALL}")
            
            # Calculate estimated remaining time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = total_epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * avg_epoch_time
            
            logger.info(f"{Fore.MAGENTA}Estimated remaining time: {timedelta(seconds=estimated_remaining_time)}{Style.RESET_ALL}")
            
            # Create and save plots for monitoring progress
            create_training_plots(
                train_losses=train_losses, 
                val_losses=val_losses,
                lr_history=lr_history,
                memory_history=memory_history,
                bleu_scores=bleu_scores,
                val_improvements=val_improvements,
                translation_lengths=translation_lengths,
                output_dir=config.plot_dir
            )
            
            # Major cleanup between epochs/shards
            cleanup_memory()
            
            # Free up memory from the datasets we don't need anymore
            del train_dataset, train_loader, train_src, train_tgt
            cleanup_memory()  # Extra cleanup
            
            # Check for early stopping
            if epoch >= min_training_steps and epochs_no_improve >= config.early_stopping_patience:
                logger.info(f"{Fore.YELLOW}Early stopping after {epochs_no_improve} epochs with no improvement{Style.RESET_ALL}")
                break
            
            # Save checkpoint at the end of each shard
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-e{epoch+1}-s{shard_index+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            logger.info(f"{Fore.BLUE}Saving checkpoint to {checkpoint_dir}...{Style.RESET_ALL}")
            
            try:
                # Save model with appropriate method
                if not config.use_8bit:
                    model.save_pretrained(checkpoint_dir)
                else:
                    model.save_pretrained(checkpoint_dir, safe_serialization=False)
                    
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Save optimizer and scheduler states
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                
                # Save training info
                train_info = {
                    "epoch": epoch + 1,
                    "shard": shard_index + 1,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "current_val_loss": val_loss,
                    "train_loss": avg_train_loss,
                    "shard_cycle": shard_cycle,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
                    json.dump(train_info, f, indent=4)
                    
                logger.info(f"{Fore.GREEN}Checkpoint saved{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
        
        # End of training
        logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}TRAINING COMPLETE{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        
        # Log total training time
        total_time = time.time() - start_time
        logger.info(f"{Fore.GREEN}Total training time: {timedelta(seconds=total_time)}{Style.RESET_ALL}")
        
        # Final evaluation
        logger.info(f"{Fore.BLUE}Running final evaluation...{Style.RESET_ALL}")
        # Use more examples for final evaluation
        final_examples = random.sample(examples, min(4, len(examples)))
        final_val_loss, final_translations = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            val_loader=val_loader,
            device=device,
            examples=final_examples[:2]  # Just use 2 for display purposes
        )
        
        logger.info(f"{Fore.GREEN}Final validation loss: {final_val_loss:.4f}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Best validation loss:  {best_val_loss:.4f}{Style.RESET_ALL}")
        
        # Log final example translations
        logger.info(f"{Fore.BLUE}Final translations:{Style.RESET_ALL}")
        for ex, trans in zip(final_examples[:2], final_translations):
            logger.info(f"  {ex} → {trans}")
        
        # Save final model
        final_model_dir = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        logger.info(f"{Fore.BLUE}Saving final model to {final_model_dir}...{Style.RESET_ALL}")
        
        try:
            # Save model with appropriate method
            if not config.use_8bit:
                model.save_pretrained(final_model_dir)
            else:
                model.save_pretrained(final_model_dir, safe_serialization=False)
                
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"{Fore.GREEN}Final model saved{Style.RESET_ALL}")
            
            # Save complete training info
            final_info = {
                "total_epochs": total_epochs,
                "completed_epochs": epoch + 1,
                "total_steps": global_step,
                "best_val_loss": best_val_loss,
                "final_val_loss": final_val_loss,
                "training_time_seconds": total_time,
                "total_train_samples": shards_info['total_train_samples'],
                "total_val_samples": shards_info['total_val_samples'],
                "date_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(final_model_dir, "training_summary.json"), "w") as f:
                json.dump(final_info, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving final model: {e}")
        
        # Final plots with all data
        create_training_plots(
            train_losses=train_losses, 
            val_losses=val_losses,
            lr_history=lr_history,
            memory_history=memory_history,
            bleu_scores=bleu_scores,
            val_improvements=val_improvements,
            translation_lengths=translation_lengths,
            output_dir=config.plot_dir
        )
        
        # Close TensorBoard writer
        writer.close()
        
        # Print final banner
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.YELLOW}    TRAINING WITH DATASET SHARDING COMPLETED SUCCESSFULLY")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}Best validation loss: {best_val_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Total training time: {timedelta(seconds=total_time)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Models saved to: {config.output_dir}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Training logs: {log_file}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Training plots: {config.plot_dir}/training_plots.png{Style.RESET_ALL}")

        return final_val_loss, best_val_loss, model, tokenizer
        
    except Exception as e:
        logger.error(f"{Fore.RED}Error during training: {str(e)}{Style.RESET_ALL}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None

# Function to get and display evaluation metrics on test set
def evaluate_on_test_set(model, tokenizer, config, device):
    """Evaluate the model on test set with BLEU score calculation"""
    logger.info(f"{Fore.BLUE}Evaluating model on test set...{Style.RESET_ALL}")
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_src, test_tgt = load_dataset_shard(
        config.test_path,
        {'shard_id': 0, 'start_idx': 0, 'end_idx': 100000},  # Load all test data
        use_full_validation=True
    )
    
    # Create test dataset
    test_dataset = EfficientTranslationDataset(
        test_src, test_tgt, tokenizer, 
        max_length=config.max_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Evaluate loss
    model.eval()
    total_loss = 0
    
    # Process in batches
    eval_progress = tqdm(test_loader, desc=f"{Fore.BLUE}Evaluating test set{Style.RESET_ALL}", 
                        leave=True, ncols=100)
    
    all_translations = []
    references = []
    
    for batch_idx, batch in enumerate(eval_progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            # Get loss
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss
            
            # Generate translations with forced BOS token
            gen_outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.eval_max_length,
                num_beams=5,
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                repetition_penalty=1.2,
                length_penalty=1.1
            )
            
            # Decode generated outputs
            translations = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            all_translations.extend(translations)
            
            # Get references
            refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            references.extend(refs)
        
        # Release batch from GPU memory
        for k in batch:
            batch[k] = batch[k].cpu()
        del batch, outputs, gen_outputs
        
        # Only clean memory occasionally
        if batch_idx % 20 == 0:
            cleanup_memory()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    
    # Calculate BLEU score
    bleu_score = corpus_bleu(all_translations, [references]).score
    
    # Log results
    logger.info(f"{Fore.GREEN}Test set evaluation complete:{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}  Test Loss: {avg_loss:.4f}{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}  BLEU Score: {bleu_score:.2f}{Style.RESET_ALL}")
    
    # Show some example translations
    logger.info(f"{Fore.BLUE}Example test translations:{Style.RESET_ALL}")
    for i in range(min(5, len(all_translations))):
        logger.info(f"  Source: {test_src[i]}")
        logger.info(f"  Reference: {references[i]}")
        logger.info(f"  Translation: {all_translations[i]}")
        logger.info("")
    
    # Save results to file
    results_file = os.path.join(config.output_dir, "test_results.json")
    test_results = {
        "test_loss": avg_loss,
        "bleu_score": bleu_score,
        "num_test_examples": len(test_dataset),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=4)
    
    logger.info(f"{Fore.GREEN}Test results saved to {results_file}{Style.RESET_ALL}")
    
    return avg_loss, bleu_score, all_translations, references

# Translation service function for interactive testing
def interactive_translation(model, tokenizer, device, max_length=64):
    """Interactive console for testing translations"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.YELLOW}    INTERACTIVE TRANSLATION CONSOLE")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"\nType 'quit' or 'exit' to end the session.")
    
    while True:
        # Get input from user
        print(f"\n{Fore.GREEN}Enter Tunisian Arabic text to translate:{Style.RESET_ALL}")
        text = input("> ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        # Translate
        tokenizer.src_lang = "ar_AR"
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Fast translation
            outputs_greedy = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=1,  # Greedy search
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                repetition_penalty=1.2,
                length_penalty=1.1
            )
            
            # Better quality translation
            outputs_beam = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=5,  # Beam search
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                repetition_penalty=1.2,
                length_penalty=1.1
            )
        
        translation_time = time.time() - start_time
        
        # Decode outputs
        translation_greedy = tokenizer.decode(outputs_greedy[0], skip_special_tokens=True)
        translation_beam = tokenizer.decode(outputs_beam[0], skip_special_tokens=True)
        
        # Print results
        print(f"\n{Fore.BLUE}Translation results:{Style.RESET_ALL}")
        print(f"  • {Fore.YELLOW}Input:{Style.RESET_ALL} {text}")
        print(f"  • {Fore.GREEN}Greedy search:{Style.RESET_ALL} {translation_greedy}")
        print(f"  • {Fore.GREEN}Beam search:{Style.RESET_ALL} {translation_beam}")
        print(f"  • {Fore.CYAN}Translation time:{Style.RESET_ALL} {translation_time:.3f} seconds")
        
        # Release memory
        del inputs, outputs_greedy, outputs_beam
        cleanup_memory()
    
    print(f"\n{Fore.YELLOW}Translation session ended.{Style.RESET_ALL}")

# Main function to run the training and evaluation
def main():
    """Main function to run the entire pipeline"""
    # Set critical environment variables for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
    
    # Clear CUDA cache at the beginning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print welcome banner
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.YELLOW}    NEURAL MACHINE TRANSLATION TRAINING PIPELINE")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    # Configuration
    config = TrainingConfig()
    
    # Log execution information
    logger.info(f"{Fore.GREEN}Starting training pipeline with dataset sharding{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}Log file: {log_file}{Style.RESET_ALL}")
    
    # Show the user what will happen
    num_shards = config.num_shards
    shard_cycles = config.shard_cycles
    total_epochs = num_shards * shard_cycles
    
    print(f"\n{Fore.GREEN}Training Configuration:{Style.RESET_ALL}")
    print(f"  • Model: {config.model_name}")
    print(f"  • Number of shards: {num_shards}")
    print(f"  • Cycles through all shards: {shard_cycles}")
    print(f"  • Total epochs: {total_epochs}")
    print(f"  • 8-bit quantization: {'Enabled' if config.use_8bit else 'Disabled'}")
    print(f"  • Gradient checkpointing: {'Enabled' if config.gradient_checkpointing else 'Disabled'}")
    print(f"  • Output directory: {config.output_dir}")
    
    # Ask for confirmation
    print(f"\n{Fore.YELLOW}Ready to start training? (y/n){Style.RESET_ALL}")
    choice = input("> ").strip().lower()
    
    if choice != 'y':
        print(f"{Fore.RED}Training aborted by user.{Style.RESET_ALL}")
        return
    
    # Start training
    val_loss, best_val_loss, model, tokenizer = train_with_shards()
    
    if model is None:
        logger.error("Training failed, exiting.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate on test set
    test_loss, bleu_score, _, _ = evaluate_on_test_set(model, tokenizer, config, device)
    
    # Print summary of results
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.YELLOW}    TRAINING RESULTS SUMMARY")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"  • {Fore.GREEN}Best validation loss:{Style.RESET_ALL} {best_val_loss:.4f}")
    print(f"  • {Fore.GREEN}Final validation loss:{Style.RESET_ALL} {val_loss:.4f}")
    print(f"  • {Fore.GREEN}Test loss:{Style.RESET_ALL} {test_loss:.4f}")
    print(f"  • {Fore.GREEN}BLEU score:{Style.RESET_ALL} {bleu_score:.2f}")
    
    # Ask if user wants to test translations interactively
    print(f"\n{Fore.YELLOW}Would you like to try interactive translations? (y/n){Style.RESET_ALL}")
    choice = input("> ").strip().lower()
    
    if choice == 'y':
        interactive_translation(model, tokenizer, device, max_length=config.max_length*2)
    
    print(f"\n{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()