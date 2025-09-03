import os
import random
import re
import argparse
import numpy as np
import torch
import yaml
from captum.attr import LayerIntegratedGradients
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm

# Global seed and determinism configuration
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load YAML configuration for paths
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['paths']

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description="Interpretability with Captum script")
    parser.add_argument('--model', type=str, required=True, help="Model ID")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--checkpoint', type=str, required=True, help="Checkpoint name")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config YAML file")
    return parser.parse_args()

# Dataset loading
def load_dataset(dataset_name, paths):
    path = os.path.join(paths['datasets'], dataset_name)
    return load_from_disk(path)

# Model/tokenizer loading
def load_model_and_tokenizer(model_id, checkpoint, dataset, paths):
    model_path = os.path.join(paths['models'], model_id)
    checkpoint_filename = f"{model_id}_best_{dataset}_{checkpoint}.pth"
    checkpoint_path = os.path.join(paths['checkpoints'], checkpoint_filename)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tokenizer, device

def process_text(text, tokenizer):
    pattern = r"[^\w\s]"
    cleaned_text = re.sub(pattern, '', text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    # Remove [CLS] and [SEP] for attributions
    cleaned_input_ids = [tid for tid in input_ids.tolist() if tid != cls_token_id and tid != sep_token_id]
    cleaned_input_ids_tensor = torch.tensor(cleaned_input_ids).unsqueeze(0)
    tokens = tokenizer.convert_ids_to_tokens(cleaned_input_ids)
    return cleaned_input_ids_tensor, attention_mask.unsqueeze(0), tokens, cleaned_text

def forward_func(inputs, attention_mask, model):
    outputs = model(inputs, attention_mask=attention_mask)
    logits = outputs.logits
    return torch.softmax(logits, dim=1)

def main():
    args = get_args()
    paths = load_config(args.config)
    print(f"Model: {args.model}\nDataset: {args.dataset}\nCheckpoint: {args.checkpoint}")

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.checkpoint, args.dataset, paths)
    dataset = load_dataset(args.dataset, paths)

    split = 'validation' if 'sst2' in args.dataset else 'test'
    dataset_test = dataset[split]
    output_dir = os.path.join(paths['output'], args.model, args.dataset, "captum")
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(tqdm(dataset_test, desc="Processing records")):
        text_raw = data.get('text', '')
        if not text_raw or len(text_raw.split()) < 2:
            print(f"Text too short at index {i}, skipping... {text_raw}")
            continue

        cleaned_input_ids_tensor, attention_mask_tensor, tokens, cleaned_text = process_text(text_raw, tokenizer)

        cleaned_input_ids_tensor = cleaned_input_ids_tensor.to(device)
        attention_mask_tensor = attention_mask_tensor.to(device)

        # Captum forward wrapper
        def forward(inputs):
            return forward_func(inputs, attention_mask_tensor, model)

        # Choose embedding layer depending on model architecture
        if 'roberta' in args.model.lower():
            ig = LayerIntegratedGradients(forward, model.roberta.embeddings.word_embeddings)
        else:
            ig = LayerIntegratedGradients(forward, model.bert.embeddings.word_embeddings)

        attributions, delta = ig.attribute(
            cleaned_input_ids_tensor,
            target=1,
            return_convergence_delta=True,
            n_steps=50
        )

        attributions_sum = attributions.sum(dim=2).squeeze(0)
        attributions_norm = attributions_sum / torch.norm(attributions_sum)

        # Pair tokens with attribution weights
        top_words = [
            {
                "word": token,
                "class": '1' if weight > 0 else '0',
                "weight": float(weight)
            }
            for token, weight in zip(tokens, attributions_norm)
        ]

        # Save the results
        df = pd.DataFrame(top_words)
        df.to_csv(os.path.join(output_dir, f"captum_top_predictions_{i}.csv"), index=False)

if __name__ == "__main__":
    main()
