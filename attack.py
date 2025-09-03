import argparse
import random
import os
import re
import sys
import time
import gc
import pandas as pd
from datetime import datetime, time as dt_time
from datasets import load_from_disk
from tqdm import tqdm
import yaml

from openai import OpenAI, InternalServerError


# Global configuration
def set_seed(seed=42):
    random.seed(seed)

set_seed()

# Load YAML configuration for all API and paths
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['api'], config['paths']


# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description="Contextual word replacement attack script using OpenAI")
    parser.add_argument('--model_decoder', type=str, required=True, help="Decoder model to use")
    parser.add_argument('--model_encoder', type=str, required=True, help="Encoder model to use")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use")
    parser.add_argument('--attack_type', type=str, required=True, help="Type of attack or explainability method")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML config file")
    return parser.parse_args()


# Initialize OpenAI client from configuration
def init_openai_client(api_config):
    return OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )


def attack_word(word, phrase, model_decoder, client, max_retries=10):
    """
    Request a contextual synonym replacement from the OpenAI API for a word/phrase.
    Retries up to max_retries if the response is unsuccessful or unchanged.
    """
    initial_prompt = (
        "You are an assistant that replaces a single given word in a provided phrase "
        "with a contextually appropriate synonym.\n\n"
        "Instructions:\n"
        "1. Carefully analyze the meaning and usage of the word within the context of the entire phrase provided.\n"
        "2. Respond with a single replacement word that best fits the intended meaning in that specific context.\n"
        "3. Your response must contain only the replacement word, with no extra text, punctuation, or explanation.\n"
        "4. If no suitable contextual synonym exists, return a word that is as close as possible in meaning, even if not perfect.\n"
        "5. Consider the phrase's tone, register, and intended meaning to ensure the replacement is natural and appropriate."
    )
    text_prompt = f"Replace the word: {word} in this phrase: {phrase}"

    # Compose message according to the decoder model's type
    if any(x in model_decoder for x in ['instruct', 'Chat', 'Instruct', 'it']):
        if any(x in model_decoder for x in ['Mistral', 'gemma']):
            message = [{"role": "user", "content": f"{initial_prompt}\n{text_prompt}"}]
        elif any(x in model_decoder for x in ['Llama', 'Qwen']):
            message = [
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": text_prompt}
            ]
    else:
        message = [{"role": "user", "content": text_prompt}]

    retry = 0
    while retry < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_decoder,
                messages=message,
                max_tokens=16,
            )
            new_word = response.choices[0].message.content.strip()
            if new_word != word:
                return new_word
            retry += 1
            time.sleep(1)
        except InternalServerError as e:
            print(f"InternalServerError on attempt {retry + 1}: {e}, waiting 60 seconds before retry...")
            time.sleep(60)
            retry += 1
        except Exception as e:
            print(f"Unexpected error on attempt {retry + 1}: {e}, waiting 60 seconds before retry...")
            time.sleep(60)
            retry += 1

    # Return original word if all retries failed
    return word


def change_words(text, words_dict):
    """
    Replace words in the text according to the given dictionary.
    """
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    replacement_dict = {str(k): str(v) for k, v in words_dict.items() if str(k) != str(v)}
    changes_count = 0

    for i, token in enumerate(tokens):
        new_word = replacement_dict.get(token)
        if new_word:
            tokens[i] = new_word
            changes_count += 1

    new_text = " ".join(tokens)
    return new_text, changes_count


def attack_dataset(dataset, model_decoder, model_encoder, dataset_name, attack_type, paths, client):
    print("Starting dataset attack...")
    all_results = []

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
        original_text = data['text']
        label = data['label']

        predictions_path = os.path.join(
            paths['csv_predictions'],
            model_encoder,
            dataset_name,
            attack_type,
            f"{attack_type}_top_predictions_{i}.csv"
        )
        if not os.path.isfile(predictions_path):
            print(f"Prediction file for index {i} not found, skipping.")
            continue

        try:
            top_predictions_df = pd.read_csv(predictions_path)
        except Exception as e:
            print(f"Error reading {predictions_path}: {e}")
            continue

        words_to_replace = {}

        for word, cls in zip(top_predictions_df['word'], top_predictions_df['class']):
            word = str(word)
            add_word = False

            # Filtering logic for special tokens and conditions
            if ('##' not in word and 'Ċ' not in word):
                if ("shap" in attack_type or "captum" in attack_type) and model_encoder != 'bert-base-cased':
                    if 'Ġ' in word:
                        word = word.replace("Ġ", "")
                        add_word = True
                    if '▁' in word and "shap" in attack_type:
                        word = word.replace("▁", "")
                        add_word = True
                if model_encoder == 'bert-base-cased' or attack_type == 'lime':
                    add_word = True
            if word.strip() == '' or word.isdigit():
                add_word = False

            if add_word and label == int(cls):
                replacement_word = attack_word(word, original_text, model_decoder, client)
                words_to_replace[word] = replacement_word

        attacked_text, total_changes = change_words(str(original_text), words_to_replace)
        attacked_text = re.sub(r'\n{2,}', '\n', attacked_text).rstrip('\n')

        all_results.append({
            'index': i,
            'original_text': original_text,
            'attacked_text': attacked_text,
            'changes_count': total_changes
        })

        gc.collect()

    save_dir = os.path.join(
        paths['attacked_data'],
        model_encoder,
        model_decoder,
        dataset_name,
        attack_type
    )
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{dataset_name}_{attack_type}_attacked.csv")

    pd.DataFrame(all_results).to_csv(output_file, index=False)
    print(f"Attacked dataset saved to: {output_file}")


def main():
    args = get_args()
    api_config, paths = load_config(args.config)

    client = init_openai_client(api_config)

    already_attacked_path = os.path.join(
        paths['attacked_data'],
        args.model_encoder,
        args.model_decoder,
        args.dataset,
        args.attack_type,
        f"{args.dataset}_{args.attack_type}_attacked.csv"
    )
    if os.path.exists(already_attacked_path):
        print(f"Dataset already attacked at: {already_attacked_path}")
        sys.exit(0)

    dataset_dir = os.path.join(paths['datasets'], args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory does not exist at expected path: {dataset_dir}")
        sys.exit(1)

    dataset = load_from_disk(dataset_dir)
    dataset_split = dataset['validation'] if 'sst2' in args.dataset else dataset['test']

    attack_dataset(dataset_split, args.model_decoder, args.model_encoder, args.dataset, args.attack_type, paths, client)


if __name__ == "__main__":
    main()
