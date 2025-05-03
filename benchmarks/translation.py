import os
import time
import torch
import numpy as np
from sacrebleu import corpus_bleu
from datasets import load_dataset

def load_translation_dataset(dataset_name, source_lang, target_lang, split="test"):
    """
    Load a translation dataset using Hugging Face's datasets library.
    """
    dataset = load_dataset(dataset_name, f"{source_lang}-{target_lang}", split=split)
    return dataset

def construct_few_shot_prompt(source_text, few_shot_examples):
    """
    Construct a few-shot prompt for translation.
    
    Args:
        source_text (str): The sentence to translate.
        few_shot_examples (list of tuples): List of (source, target) translation pairs.
    
    Returns:
        str: The formatted prompt for GPT-2.
    """
    prompt = ""
    for src, tgt in few_shot_examples:
        prompt += f"French: {src}\nEnglish: {tgt}\n\n"
    prompt += f"French: {source_text}\nEnglish:"
    return prompt

def get_few_shot_examples(dataset_name, source_lang, target_lang, num_examples=10):
    """
    Fetch real translation examples from the dataset.
    """
    dataset = load_translation_dataset(dataset_name, source_lang, target_lang, split="validation")
    examples = [(ex['translation'][source_lang], ex['translation'][target_lang]) for ex in dataset][:num_examples]
    return examples

def translate_batch(model, dataset_name, encode, decode, source_texts, source_lang, target_lang, device, max_length=40):
    """
    Translate a batch of source texts using GPT-2 few-shot prompting.
    """
    model.eval()
    few_shot_examples = get_few_shot_examples(dataset_name, source_lang, target_lang, num_examples=10)

    translations = []
    for text in source_texts:
        prompt = construct_few_shot_prompt(text, few_shot_examples)
        input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.generate_with_stop(input_ids, max_length, "French", decode)

        raw_output = output[1]

        # Extract translation by removing prompt
        translated_text = raw_output.replace(prompt, "").strip().split("\n")[0]
        translations.append(translated_text)

    return translations

def evaluate_translation(model, dataset_name, encode, decode, dataset, source_lang, target_lang, device):
    """
    Evaluate a translation model using BLEU score.
    """
    source_texts = [example['translation'][source_lang] for example in dataset][:100]
    reference_texts = [[example['translation'][target_lang]] for example in dataset][:100]

    translations = translate_batch(model, dataset_name, encode, decode, source_texts, source_lang, target_lang, device)

    f = [i for s in reference_texts for i in s]
    for t, ref in zip(translations, f):
        print("translated text: ", t)
        print("reference text: ", ref)
        print("------------------------------")

    bleu = corpus_bleu(translations, reference_texts)
    return bleu.score

def benchmark_translation(model, encode, decode, dataset_name, source_lang, target_lang, device):
    """
    Benchmark translation performance on a specific dataset.
    """
    print(f"Loading dataset {dataset_name} ({source_lang} -> {target_lang})...")
    dataset = load_translation_dataset(dataset_name, source_lang, target_lang)

    print("Evaluating translations...")
    start_time = time.time()
    bleu_score = evaluate_translation(model, dataset_name, encode, decode, dataset, source_lang, target_lang, device)
    end_time = time.time()

    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")

    return bleu_score