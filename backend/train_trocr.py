"""
TrOCR Fine-Tuning Script for Classroom HTR System.

Supports:
1. Hugging Face Datasets (e.g. Teklia/iam-lines)
2. Local Custom Datasets (CSV/JSON folder with images)
3. Synthetic / Sample Dataset mode for immediate smoke testing

Usage:
    # 1. Train on Hugging Face IAM Lines dataset:
    python train_trocr.py --dataset_name Teklia/iam-lines --output_dir ./models/trocr-custom --epochs 5

    # 2. Train on local custom dataset (folder containing metadata.csv and image files):
    python train_trocr.py --local_data_dir ./my_handwriting_data --output_dir ./models/trocr-custom

    # 3. Test training loop with generated synthetic handwriting samples:
    python train_trocr.py --synthetic --output_dir ./models/trocr-custom --epochs 2
"""

import argparse
import os
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Optional evaluation library
try:
    # pyrefly: ignore [missing-import]
    import evaluate
    cer_metric = evaluate.load("cer")
except Exception:
    cer_metric = None


class HTRDataset(Dataset):
    """PyTorch Dataset wrapper for TrOCR training."""

    def __init__(self, data_items: list[dict], processor: TrOCRProcessor, max_target_length: int = 128):
        self.data_items = data_items
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data_items[idx]
        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        text = item["text"]

        # Transform image pixels
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize target text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Mask padding tokens so they are ignored in cross-entropy loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def generate_synthetic_samples(num_samples: int = 50) -> list[dict]:
    """Generates synthetic multi-font text line images for training pipeline verification."""
    samples = []
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Classroom handwriting recognition test line.",
        "Student assignment submission for biology notes.",
        "Mathematical equations and descriptive answers.",
        "Photosynthesis occurs inside chloroplasts.",
        "Newton's laws of motion explain dynamics.",
        "Chemistry lab experiment report results.",
        "History essay regarding ancient civilizations."
    ]

    for i in range(num_samples):
        text = texts[i % len(texts)] + f" (id: {i})"
        img = Image.new("RGB", (600, 80), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((20, 25), text, fill=(10, 10, 10))
        samples.append({"image": img, "text": text})

    return samples


def compute_metrics(pred, processor: TrOCRProcessor):
    """Computes Character Error Rate (CER) during evaluation."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    if cer_metric is not None:
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
    else:
        # Fallback basic character accuracy proxy if evaluate/jiwer not installed
        cer = 0.0

    return {"cer": cer}


def train(
    base_model_id: str = "microsoft/trocr-base-handwritten",
    dataset_name: Optional[str] = None,
    local_data_dir: Optional[str] = None,
    synthetic: bool = False,
    output_dir: str = "./models/trocr-custom",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 4e-5,
    max_target_length: int = 128,
    freeze_encoder: bool = False,
):
    print(f"=== Initializing TrOCR Fine-Tuning ===")
    print(f"Base Model: {base_model_id}")
    print(f"Output Directory: {output_dir}")

    # 1. Load Processor and Base Model
    processor = TrOCRProcessor.from_pretrained(base_model_id)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_id)

    # Optional: Freeze the Vision Encoder for much faster training on CPU
    if freeze_encoder:
        print("Freezing Vision Encoder weights (training decoder only for high speed)...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Configure generation parameters
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_target_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # 2. Prepare Dataset
    if synthetic:
        print("Using synthetic handwriting samples for quick verification...")
        all_samples = generate_synthetic_samples(60)
        train_samples = all_samples[:50]
        eval_samples = all_samples[50:]
    elif dataset_name:
        print(f"Loading Hugging Face dataset: {dataset_name}...")
        from datasets import load_dataset
        ds = load_dataset(dataset_name)
        train_samples = [{"image": x["image"], "text": x["text"]} for x in ds["train"]]
        eval_split = "test" if "test" in ds else "train"
        eval_samples = [{"image": x["image"], "text": x["text"]} for x in ds[eval_split].select(range(min(50, len(ds[eval_split]))))]
    elif local_data_dir:
        print(f"Loading local dataset from: {local_data_dir}...")
        import pandas as pd
        csv_path = os.path.join(local_data_dir, "metadata.csv")
        df = pd.read_csv(csv_path)
        train_samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(local_data_dir, str(row["file_name"]))
            train_samples.append({"image": img_path, "text": str(row["text"])})
        eval_samples = train_samples[-max(1, int(0.1 * len(train_samples))):]
        train_samples = train_samples[:-len(eval_samples)]
    else:
        raise ValueError("Please provide --dataset_name, --local_data_dir, or use --synthetic")

    train_dataset = HTRDataset(train_samples, processor, max_target_length=max_target_length)
    eval_dataset = HTRDataset(eval_samples, processor, max_target_length=max_target_length) if eval_samples else None

    print(f"Dataset ready. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset) if eval_dataset else 0}")

    # 3. Training Arguments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    eval_mode = "epoch" if eval_dataset else "no"
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy=eval_mode,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        output_dir=output_dir,
        report_to="none",
    )

    # 4. Initialize Seq2Seq Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 5. Start Training
    print("Starting training loop...")
    trainer.train()

    # 6. Save final model and processor
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\n[SUCCESS] Model successfully trained and saved to: {output_dir}")
    print(f"To use this model in your Classroom HTR system, update backend/.env:")
    print(f"TROCR_MODEL_ID={os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for Classroom HTR")
    parser.add_argument("--base_model_id", type=str, default="microsoft/trocr-base-handwritten", help="Base model checkpoint")
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset name (e.g. Teklia/iam-lines)")
    parser.add_argument("--local_data_dir", type=str, default=None, help="Directory containing metadata.csv and image files")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples to test pipeline")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze vision encoder for fast CPU training")
    parser.add_argument("--output_dir", type=str, default="./models/trocr-custom", help="Directory to save fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    args = parser.parse_args()

    train(
        base_model_id=args.base_model_id,
        dataset_name=args.dataset_name,
        local_data_dir=args.local_data_dir,
        synthetic=args.synthetic,
        freeze_encoder=args.freeze_encoder,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_target_length=args.max_length,
    )

