# Tunisian to English AI Translation

AI model to translate Tunisian dialect to English language.

## Description

This project implements an AI-based translation system for converting Tunisian Arabic dialect to English.

## Setup

[Project setup instructions will go here]
  - Gradient checkpointing (reduce memory by 30-40%)
  - Mixed-precision training (FP16/FP32 hybrid)
  - 8-bit quantization (optional for low-memory devices)
- **Advanced Monitoring**:
  - Real-time training dashboard
  - TensorBoard integration
  - GPU memory profiling
  - Translation quality sampling

### Technical Innovations
- **Dynamic Batch Processing**: Automaticallly adjusts batch size based on available memory
- **Curriculum Learning**: Progressive difficulty scheduling
- **Noise Injection**: Improves model robustness
- **Attention Visualization**: Interpret model decisions

## 🧠 Model Architecture

### mBART-50 Foundation
The system builds on mBART-50, a multilingual sequence-to-sequence transformer pretrained on 50 languages. Key specifications:

| Parameter          | Value               |
|--------------------|---------------------|
| Architecture       | Transformer (seq2seq) |
| Layers             | 12 encoder, 12 decoder |
| Attention Heads    | 16                  |
| Hidden Size        | 1024                |
| Parameters         | 610M                |
| Pretraining        | 50 languages        |

### Custom Modifications
1. **Dialect-Specific Tokenization**:
   - Extended vocabulary for Tunisian Arabic
   - Special tokens for code-switching markers
   - Subword regularization for dialectal variations

2. **Memory Optimization**:
   ```python
   # Example of gradient checkpointing implementation
   model.config.use_cache = False  # Disable cache for checkpointing
   model.gradient_checkpointing_enable()
   ```

3. **Attention Mechanisms**:
   - Multi-head attention with relative position bias
   - Learned attention temperature scaling

## 📊 Performance Metrics

| Metric            | Validation Score | Test Score |
|-------------------|------------------|------------|
| BLEU-4            | 32.7             | 31.2       |
| TER               | 45.3             | 46.8       |
| ChrF             | 58.1             | 56.9       |
| Inference Speed   | 42 tokens/sec (RTX 3090) |

![Training Metrics](docs/images/training_curve.png)

## 🛠️ Installation

### Prerequisites
- NVIDIA GPU (recommended) with CUDA 11.7+
- Python 3.8+

### Setup
```bash
git clone https://github.com/mabroukaymen1/tunisian_english.git
cd tunisian-translator

# Create and activate environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install with memory-efficient options
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# Optional 8-bit support
pip install bitsandbytes accelerate
```

## 🚀 Usage

### 1. Data Processing
```bash
python process.py \
    --input_dir data/rawww \
    --output_dir processed \
    --max_length 128 \
    --shard_size 50000
```

### 2. Training
```bash
python train.py \
    --model_name facebook/mbart-large-50 \
    --train_shards processed/train_* \
    --val_shards processed/val_* \
    --batch_size 32 \
    --gradient_accumulation 4 \
    --fp16 \
    --use_8bit \
    --max_epochs 10
```

### 3. Interactive Translation
```python
from translator import InteractiveTranslator

translator = InteractiveTranslator("models/best_checkpoint")
while True:
    text = input("Enter Tunisian Arabic: ")
    translation = translator.predict(text)
    print(f"English: {translation['text']}")
    print(f"Confidence: {translation['confidence']:.2%}")
    print(f"Attention: {translation['attention_heatmap']}")
```

## 📂 Project Structure

```
tunisian-translator/
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed shards
├── docs/
│   ├── architecture.md       # Detailed model docs
│   └── evaluation.md         # Metric explanations
├── models/                   # Saved checkpoints
├── src/
│   ├── dataloader.py         # Memory-efficient loading
│   ├── model.py             # Architecture modifications
│   ├── trainer.py           # Training logic
│   └── utils/               # Helper functions
├── process.py               # Data preprocessing
├── train.py                # Main training script
├── evaluate.py             # Model evaluation
└── requirements.txt        # Dependencies
```

## 📈 Training Optimization

### Memory Management Techniques
1. **Gradient Checkpointing**:
   - Only stores certain layer activations
   - Recomputes others during backward pass
   - Reduces memory by ~30% with 20% speed tradeoff

2. **8-bit Quantization**:
   ```python
   from bitsandbytes import quantize
   model = quantize(model, bits=8)
   ```

3. **Dynamic Batching**:
   ```python
   # Automatically adjusts batch size
   batch_size = max(1, free_memory // mem_per_sample)
   ```

## 🔍 Model Interpretability

### Attention Visualization
![Attention Heatmap](docs/images/attention.png)

```python
def visualize_attention(text, model):
    outputs = model.generate(..., return_attention=True)
    attention = outputs.attentions[-1]  # Final layer attention
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(attention.cpu().numpy()[0])
    plt.show()
```

## 📚 Theoretical Background

### Handling Dialectal Variations
The model addresses key challenges in Tunisian Arabic translation:

1. **Morphological Complexity**:
   - Uses subword tokenization with SentencePiece
   - Implements aggressive wordpiece merging for dialect words

2. **Code-Switching**:
   ```python
   # Special tokens for language switching
   LANG_TOKENS = {
       'ar': "[AR]",
       'fr': "[FR]", 
       'en': "[EN]"
   }
   ```

3. **Data Augmentation**:
   - Random word deletion (10% probability)
   - Synonym replacement using dialect dictionaries
   - Synthetic code-switching injection

## 🤖 Deployment Options

### Production Serving
```bash
# FastAPI endpoint
uvicorn api:app --host 0.0.0.0 --port 8000
```

### ONNX Export
```python
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    opset_version=13,
    dynamic_axes={'input_ids': [0], 'attention_mask': [0]}
)
```




## 🙏 Acknowledgments

- Facebook AI for mBART-50
- Hugging Face for Transformers library
- TUNIZI corpus contributors
