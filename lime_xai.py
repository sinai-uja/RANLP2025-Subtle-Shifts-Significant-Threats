import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import yaml

# Global seed configuration for reproducibility
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

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description="LIME explanation script")
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
def load_model_and_tokenizer(model_id, checkpoint, dataset, paths):
    model_path = os.path.join(paths['models'], model_id)
    checkpoint_filename = f"{model_id}_best_{dataset}_{checkpoint}.pth"
    checkpoint_path = os.path.join(paths['checkpoints'], checkpoint_filename)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

# Predictor for LIME
def get_predictor(model, tokenizer, device):
    def predictor(texts):
        with torch.no_grad():
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**batch)
            probas = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probas
    return predictor

# Main LIME processing
def run_lime(args, paths):
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.checkpoint, args.dataset, paths)
    dataset = load_dataset(args.dataset, paths)
    class_names = ['0', '1']

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
    split = 'validation' if 'sst2' in args.dataset else 'test'
    dataset_test = dataset[split]

    output_dir = os.path.join(paths['output'], args.model, args.dataset, "lime")
    os.makedirs(output_dir, exist_ok=True)

    predictor = get_predictor(model, tokenizer, device)

    for i, data in enumerate(tqdm(dataset_test, desc="Processing records")):
        text = data.get('text', '')
        if not text or len(text.split()) < 2:
            print(f"Text too short at index {i}, skipping... {text}")
            continue

        exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=250)
        top_words = []
        for word, weight in exp.as_list():
            associated_class = '1' if weight > 0 else '0'
            top_words.append({
                "word": word,
                "class": associated_class,
                "weight": weight
            })

        df = pd.DataFrame(top_words)
        df.to_csv(os.path.join(output_dir, f"lime_top_predictions_{i}.csv"), index=False)

if __name__ == "__main__":
    args = get_args()
    paths = load_config(args.config)
    print(f"Model: {args.model}\nDataset: {args.dataset}\nCheckpoint: {args.checkpoint}")
    run_lime(args, paths)
