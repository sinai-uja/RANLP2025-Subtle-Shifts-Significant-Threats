import os
import random
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import shap
import scipy as sp
import yaml

# Global: reproducibility seed and determinism
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load YAML configuration for all paths
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['paths']

# Script arguments
def get_args():
    parser = argparse.ArgumentParser(description="SHAP explanation script")
    parser.add_argument('--model', type=str, required=True, help="Model identifier.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Model checkpoint name.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML config.")
    return parser.parse_args()

# Dataset loading
def load_dataset(dataset_name, paths):
    path = os.path.join(paths['datasets'], dataset_name)
    return load_from_disk(path)

# Model and tokenizer loading
def load_model_and_tokenizer(model_id, checkpoint, dataset_name, paths):
    model_path = os.path.join(paths['models'], model_id)
    checkpoint_filename = f"{model_id}_best_{dataset_name}_{checkpoint}.pth"
    checkpoint_path = os.path.join(paths['checkpoints'], checkpoint_filename)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tokenizer, device

# Predictor for SHAP and softmax predictions
def get_predictor(model, tokenizer, device):
    def predictor(texts):
        with torch.no_grad():
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**batch)
            probas = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probas
    return predictor

def custom_shap_fn(texts, tokenizer, model, device):
    # Vectorized for SHAP
    token_ids = [tokenizer.encode(t, padding="max_length", max_length=512, truncation=True) for t in texts]
    tv = torch.tensor(token_ids).to(device)
    with torch.no_grad():
        logits = model(tv).logits.cpu().numpy()
        scores = (np.exp(logits).T / np.exp(logits).sum(-1)).T  # softmax
        val = sp.special.logit(scores[:, 1])  # logit for class "1"
    return val

def process_text(text, tokenizer):
    pattern = r"[^\w\s]"
    cleaned_text = re.sub(pattern, '', text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"][0].tolist()
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    cleaned_input_ids = [tid for tid in input_ids if tid != cls_token_id and tid != sep_token_id]
    tokens = tokenizer.convert_ids_to_tokens(cleaned_input_ids)
    return cleaned_text, tokens

def main():
    args = get_args()
    paths = load_config(args.config)
    print(f"Model: {args.model}\nDataset: {args.dataset}\nCheckpoint: {args.checkpoint}")

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.checkpoint, args.dataset, paths)
    dataset = load_dataset(args.dataset, paths)

    predictor = get_predictor(model, tokenizer, device)
    split = 'validation' if 'sst2' in args.dataset else 'test'
    dataset_test = dataset[split]
    output_dir = os.path.join(paths['output'], args.model, args.dataset, "shap")
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(tqdm(dataset_test, desc="Processing records")):
        text_raw = data.get('text', '')
        if not text_raw or len(text_raw.split()) < 2:
            print(f"Text too short at index {i}, skipping... {text_raw}")
            continue

        cleaned_text, tokens = process_text(text_raw, tokenizer)
        probas = predictor([text_raw])
        prediction = probas.argmax(axis=1)[0]
        predicted_prob = probas[0, prediction]

        # SHAP explainer
        shap_fn = lambda x: custom_shap_fn(x, tokenizer, model, device)
        explainer = shap.Explainer(shap_fn, tokenizer)
        shap_values = explainer([cleaned_text])  # one text only

        # Save results in sorted DataFrame
        top_words = []
        for shap_ in shap_values.values:
            for token, shap_score in zip(tokens, shap_):
                associated_class = '1' if shap_score > 0 else '0'
                top_words.append({
                    "word": token,
                    "class": associated_class,
                    "weight": shap_score,
                    "probability": predicted_prob
                })

        df = pd.DataFrame(top_words).sort_values(by="weight", ascending=False)
        df.to_csv(os.path.join(output_dir, f"shap_top_predictions_{i}.csv"), index=False)

if __name__ == "__main__":
    main()
