#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import argparse
import gc
import json
import numpy as np
import subprocess
import sys
from collections import defaultdict
import os.path

from utils_new_11_3_updated import (
    init_aya_expanse_gpu,
    UNIVERSAL_POS_TAGS,
    UNIVERSAL_DEPREL_LABELS
)

RESULTS_PATH = os.path.join(os.getenv("SLURM_SUBMIT_DIR", "."), "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

def load_conllu_input_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_words = []
        current_tokens = []
        current_multi_tokens = []
        
        for line in f:
            line = line.strip()
            
            if not line:
                if current_words:
                    sentences.append((current_words, current_tokens, current_multi_tokens))
                    current_words = []
                    current_tokens = []
                    current_multi_tokens = []
                continue
            
            if line.startswith('#'):
                continue
            
            parts = line.split('\t')
            
            if '-' in parts[0]:
                id_range = parts[0]
                word = parts[1]
                lemma = parts[2] if len(parts) > 2 else word
                
                start_id, end_id = map(int, id_range.split('-'))
                current_multi_tokens.append((id_range, start_id, end_id, word, lemma))
                continue
            
            token_id = int(parts[0])
            word = parts[1]
            lemma = parts[2] if len(parts) > 2 else word
            
            current_words.append(word)
            current_tokens.append((token_id, word, lemma))
        
        if current_words:
            sentences.append((current_words, current_tokens, current_multi_tokens))
    
    return sentences

def load_conllu_gold_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_words = []
        current_pos = []
        current_heads = []
        current_deps = []
        current_lemmas = []
        current_multi_tokens = []
        
        for line in f:
            line = line.strip()
            
            if not line:
                if current_words:
                    sentences.append((current_words, (current_pos, current_heads, current_deps, current_lemmas), current_multi_tokens))
                    current_words = []
                    current_pos = []
                    current_heads = []
                    current_deps = []
                    current_lemmas = []
                    current_multi_tokens = []
                continue
            
            if line.startswith('#'):
                continue
            
            parts = line.split('\t')
            
            if '-' in parts[0]:
                id_range = parts[0]
                word = parts[1]
                lemma = parts[2] if len(parts) > 2 else word
                
                start_id, end_id = map(int, id_range.split('-'))
                current_multi_tokens.append((id_range, start_id, end_id, word, lemma))
                continue
            
            token_id = int(parts[0])
            word = parts[1]
            lemma = parts[2] if len(parts) > 2 else word
            pos = parts[3]
            head = int(parts[6])
            dep = parts[7]
            
            current_words.append(word)
            current_lemmas.append(lemma)
            current_pos.append(pos)
            current_heads.append(head)
            current_deps.append(dep)
        
        if current_words:
            sentences.append((current_words, (current_pos, current_heads, current_deps, current_lemmas), current_multi_tokens))
    
    return sentences

def load_conllu_examples_file(file_path):
    return load_conllu_gold_file(file_path)

def create_conllu_prompt(current_sentence, example_sentences, num_examples):
    words, tokens, multi_tokens = current_sentence

    prompt = """Please annotate the following sentence using the Universal Dependencies standard, ensuring the output adheres to the CoNLL-U format strictly (10 fields, tab-separated, with ID starting each line). The input provides ID, FORM, and LEMMA. Your output should complete all 10 columns of the CoNLL-U format.

"""

    if num_examples > 0 and example_sentences:
        prompt += "# Reference examples below (for guidance only)\n\n"
        
        selected_examples = example_sentences[:num_examples]
        
        for i, example in enumerate(selected_examples):
            example_words, example_annot, example_multi_tokens = example
            example_pos, example_heads, example_deps, example_lemmas = example_annot
            
            prompt += "I:\n"
            prompt += f"# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n"
            prompt += f"# sent_id = example{i+1}\n"
            prompt += f"# text = {' '.join(example_words)}\n"
            
            multi_token_map = {}
            for id_range, start_id, end_id, mw_word, mw_lemma in example_multi_tokens:
                multi_token_map[start_id] = (id_range, mw_word, mw_lemma)
            
            for j, word in enumerate(example_words):
                token_id = j + 1
                
                if token_id in multi_token_map:
                    id_range, mw_word, mw_lemma = multi_token_map[token_id]
                    prompt += f"{id_range} {mw_word} {mw_lemma}\n"
                
                lemma = example_lemmas[j] if isinstance(example_lemmas, list) else word
                prompt += f"{token_id} {word} {lemma}\n"
            
            prompt += "\nO:\n"
            prompt += f"# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n"
            prompt += f"# sent_id = example{i+1}\n"
            prompt += f"# text = {' '.join(example_words)}\n"
            
            for j, word in enumerate(example_words):
                token_id = j + 1
                
                if token_id in multi_token_map:
                    id_range, mw_word, mw_lemma = multi_token_map[token_id]
                    prompt += f"{id_range}\t{mw_word}\t{mw_lemma}\t_\t_\t_\t_\t_\t_\t_\n"
                
                pos = example_pos[j]
                head = example_heads[j]
                dep = example_deps[j]
                lemma = example_lemmas[j] if isinstance(example_lemmas, list) else word
                prompt += f"{token_id}\t{word}\t{lemma}\t{pos}\t_\t_\t{head}\t{dep}\t_\t_\n"
            
            prompt += "\n"
    
    multi_token_map = {}
    for id_range, start_id, end_id, mw_word, mw_lemma in multi_tokens:
        multi_token_map[start_id] = (id_range, mw_word, mw_lemma)
    
    for token_id, word, lemma in tokens:
        if token_id in multi_token_map:
            id_range, mw_word, mw_lemma = multi_token_map[token_id]
            prompt += f"{id_range}\t{mw_word}\t{mw_lemma}\n"
        
        prompt += f"{token_id}\t{word}\t{lemma}\n"
    
    prompt += f"\n# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n"
    prompt += f"# sent_id = current\n"
    prompt += f"# text = {' '.join(words)}\n"
    
    return prompt

def get_constrained_logits(model, tokenizer, input_ids, constraints=None, temperature=0.3):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1].clone()

    if temperature != 1.0:
        logits = logits / temperature

    if constraints:
        mask = torch.ones_like(logits) * float('-inf')
        
        for constraint in constraints:
            if isinstance(constraint, str):
                constraint_tokens = tokenizer.encode(constraint, add_special_tokens=False)
                if constraint_tokens:
                    token_id = constraint_tokens[0]
                    mask[token_id] = 0
            else:
                mask[constraint] = 0
        
        logits = logits + mask

    return logits

def get_token_probabilities(model, tokenizer, prompt, constraints=None, temperature=0.3):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(model.device)

    logits = get_constrained_logits(
        model, tokenizer, input_ids, 
        constraints=constraints,
        temperature=temperature
    )

    probs = F.softmax(logits, dim=-1)

    token_probs = {}
    for constraint in constraints:
        if isinstance(constraint, str):
            constraint_tokens = tokenizer.encode(constraint, add_special_tokens=False)
            if constraint_tokens:
                token_id = constraint_tokens[0]
                token_probs[constraint] = probs[token_id].item()

    return token_probs

def get_best_token(model, tokenizer, prompt, constraints=None, temperature=0.3):
    token_probs = get_token_probabilities(model, tokenizer, prompt, constraints, temperature)
    best_token = max(token_probs.items(), key=lambda x: x[1])[0] if token_probs else constraints[0]
    return best_token

def get_refined_head_candidates(i, sentence_length, pred_pos, words, temperature=0.3):
    valid_heads = list(range(sentence_length + 1))
    valid_heads.remove(i + 1)
    
    current_pos = pred_pos[i]
    
    if current_pos == "PUNCT":
        nearby_content_indices = [j for j in range(max(0, i-5), min(sentence_length, i+6)) 
                                if j != i and pred_pos[j] not in ["PUNCT", "DET", "PART"]]
        if nearby_content_indices:
            if i == sentence_length - 1:
                valid_heads = [str(j+1) for j in nearby_content_indices] + ["0"]
            else:
                valid_heads = [str(j+1) for j in nearby_content_indices]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) if j != i and pred_pos[j] != "PUNCT"] + ["0"]
    
    elif current_pos == "DET":
        noun_indices = [j for j in range(i+1, min(sentence_length, i+4)) 
                       if pred_pos[j] in ["NOUN", "PROPN"]]
        if not noun_indices:
            noun_indices = [j for j in range(max(0, i-3), i) 
                           if pred_pos[j] in ["NOUN", "PROPN"]]
        if noun_indices:
            valid_heads = [str(j+1) for j in noun_indices]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) 
                          if j != i and pred_pos[j] in ["NOUN", "PROPN", "ADJ", "VERB"]]
    
    elif current_pos == "ADP":
        noun_indices = [j for j in range(i+1, min(sentence_length, i+5)) 
                       if pred_pos[j] in ["NOUN", "PROPN", "PRON"]]
        verb_indices = [j for j in range(sentence_length) 
                       if j != i and pred_pos[j] in ["VERB", "AUX"]]
        
        if noun_indices or verb_indices:
            valid_heads = [str(j+1) for j in set(noun_indices + verb_indices)]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) 
                          if j != i and pred_pos[j] in ["NOUN", "PROPN", "VERB", "ADJ"]]
    
    elif current_pos == "ADJ":
        noun_indices = [j for j in range(max(0, i-3), min(sentence_length, i+4)) 
                       if pred_pos[j] in ["NOUN", "PROPN"]]
        if noun_indices:
            valid_heads = [str(j+1) for j in noun_indices]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) 
                          if j != i and pred_pos[j] in ["NOUN", "PROPN", "VERB", "ADJ"]] + ["0"]
    
    elif current_pos in ["NOUN", "PROPN", "PRON"]:
        verb_indices = [j for j in range(sentence_length) if pred_pos[j] in ["VERB", "AUX"]]
        noun_indices = [j for j in range(sentence_length) 
                       if j != i and pred_pos[j] in ["NOUN", "PROPN"]]
        
        if verb_indices or noun_indices:
            valid_heads = [str(j+1) for j in range(sentence_length) if j != i] + ["0"]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) if j != i] + ["0"]
    
    elif current_pos == "ADV":
        modifier_indices = [j for j in range(sentence_length) 
                           if j != i and pred_pos[j] in ["VERB", "ADJ", "ADV", "AUX"]]
        if modifier_indices:
            valid_heads = [str(j+1) for j in modifier_indices]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) 
                          if j != i and pred_pos[j] not in ["PUNCT", "DET", "PART"]] + ["0"]
    
    elif current_pos == "AUX":
        verb_indices = [j for j in range(sentence_length) if j != i and pred_pos[j] == "VERB"]
        if verb_indices:
            valid_heads = [str(j+1) for j in verb_indices]
        else:
            valid_heads = [str(j+1) for j in range(sentence_length) 
                          if j != i and pred_pos[j] in ["NOUN", "PROPN", "ADJ"]] + ["0"]
    
    if not valid_heads:
        valid_heads = [str(j+1) for j in range(sentence_length) if j != i] + ["0"]
    
    valid_heads = [str(h) for h in valid_heads]
    
    return valid_heads

def get_enhanced_deps_for_pos(pos_tag, head_pos=None, head_idx=None, token_idx=None, sentence_length=None):
    if head_idx == 0:
        if pos_tag == "VERB":
            return ["root"]
        elif pos_tag in ["NOUN", "PROPN"]:
            return ["root"]
        elif pos_tag == "ADJ":
            return ["root"]
        else:
            return ["root"]
    
    if head_pos and token_idx is not None and head_idx is not None:
        is_before_head = token_idx < head_idx
        
        if pos_tag == "PUNCT":
            return ["punct"]
        
        if pos_tag == "DET" and head_pos in ["NOUN", "PROPN"]:
            return ["det"]
        
        if pos_tag == "ADP":
            if head_pos in ["NOUN", "PROPN", "PRON"]:
                return ["case"]
            elif head_pos in ["VERB", "AUX"]:
                return ["mark", "case"]
            else:
                return ["case", "mark"]
        
        if pos_tag in ["NOUN", "PROPN", "PRON"]:
            if head_pos in ["VERB", "AUX"]:
                if is_before_head:
                    return ["nsubj", "nsubj:pass", "csubj"]
                else:
                    return ["obj", "iobj", "obl"]
            elif head_pos in ["NOUN", "PROPN"]:
                if pos_tag == "PRON" and head_pos == "NOUN":
                    return ["nmod:poss"]
                else:
                    return ["nmod", "appos", "compound"]
            elif head_pos == "ADJ":
                return ["nsubj", "obl"]
            else:
                return ["nmod", "obl", "nsubj"]
        
        if pos_tag == "ADJ":
            if head_pos in ["NOUN", "PROPN"]:
                return ["amod"]
            elif head_pos in ["VERB", "AUX"]:
                if is_before_head:
                    return ["amod", "advcl"]
                else:
                    return ["xcomp", "amod"]
            else:
                return ["amod", "conj"]
        
        if pos_tag == "ADV":
            if head_pos in ["VERB", "AUX"]:
                return ["advmod"]
            elif head_pos == "ADJ":
                return ["advmod"]
            elif head_pos == "ADV":
                return ["advmod"]
            else:
                return ["advmod"]
        
        if pos_tag in ["VERB", "AUX"]:
            if head_pos in ["VERB", "AUX"]:
                if is_before_head:
                    return ["advcl", "csubj", "conj"]
                else:
                    return ["xcomp", "ccomp", "conj", "advcl"]
            elif head_pos in ["NOUN", "PROPN"]:
                return ["acl", "acl:relcl"]
            else:
                return ["advcl", "ccomp", "xcomp"]
        
        if pos_tag == "CCONJ" and head_idx < sentence_length:
            return ["cc"]
        
        if pos_tag == "SCONJ":
            return ["mark"]
        
        if pos_tag == "PART":
            if head_pos in ["VERB", "AUX"]:
                return ["mark", "advmod"]
            else:
                return ["mark"]
        
        if pos_tag == "NUM" and head_pos in ["NOUN", "PROPN"]:
            return ["nummod"]
        
        if pos_tag == "INTJ":
            return ["discourse"]
    
    basic_deps = {
        "NOUN": ["nsubj", "obj", "iobj", "obl", "nmod", "appos", "conj", "flat", "compound", "parataxis", "dep"],
        "PROPN": ["nsubj", "obj", "iobj", "obl", "nmod", "appos", "conj", "flat", "compound", "parataxis", "dep"],
        "PRON": ["nsubj", "obj", "iobj", "obl", "nmod", "conj", "dep"],
        "VERB": ["ccomp", "xcomp", "advcl", "acl", "conj", "parataxis", "dep"],
        "ADJ": ["amod", "conj", "advcl", "acl", "parataxis", "dep"],
        "ADV": ["advmod", "conj", "dep"],
        "ADP": ["case", "mark", "fixed", "dep"],
        "DET": ["det", "dep"],
        "NUM": ["nummod", "conj", "dep"],
        "CCONJ": ["cc", "dep"],
        "SCONJ": ["mark", "dep"],
        "PART": ["mark", "dep"],
        "INTJ": ["discourse", "dep"],
        "PUNCT": ["punct"],
        "SYM": ["dep"],
        "AUX": ["aux", "cop", "dep"],
        "X": ["dep"]
    }
    
    return basic_deps.get(pos_tag, ["dep"])

def find_best_root(pos_tags):
    for i, pos in enumerate(pos_tags):
        if pos == "VERB":
            return i

    for i, pos in enumerate(pos_tags):
        if pos == "AUX":
            return i

    for i, pos in enumerate(pos_tags):
        if pos in ["NOUN", "PROPN"]:
            return i

    for i, pos in enumerate(pos_tags):
        if pos in ["ADJ", "ADV", "PRON", "NUM"]:
            return i

    for i, pos in enumerate(pos_tags):
        if pos != "PUNCT":
            return i

    return 0

def fix_dependency_tree(heads, deps, pos_tags, root_idx, sentence_length):
    if heads[root_idx] != 0:
        heads[root_idx] = 0
        deps[root_idx] = "root"
    
    for i in range(sentence_length):
        if i == root_idx:
            continue
            
        visited = set()
        current = i
        path = []
        
        while current != root_idx and current not in visited:
            visited.add(current)
            path.append(current)
            head_idx = heads[current] - 1
            
            if head_idx < 0 or head_idx >= sentence_length:
                heads[current] = root_idx + 1
                deps[current] = "dep"
                break
            
            current = head_idx
        
        if current in visited:
            cycle_start = path.index(current)
            cycle = path[cycle_start:]
            
            heads[cycle[0]] = root_idx + 1
            
            if pos_tags[cycle[0]] == "VERB":
                deps[cycle[0]] = "parataxis"
            elif pos_tags[cycle[0]] in ["NOUN", "PROPN"]:
                deps[cycle[0]] = "obl"
            elif pos_tags[cycle[0]] == "ADJ":
                deps[cycle[0]] = "amod"
            else:
                deps[cycle[0]] = "dep"
    
    for i in range(sentence_length):
        if heads[i] == i + 1:
            heads[i] = root_idx + 1
            deps[i] = "dep"

def get_joint_head_dep_scores(model, tokenizer, token_prompt, i, pred_pos, sentence_length, temperature=0.3):
    valid_heads = get_refined_head_candidates(i, sentence_length, pred_pos, None, temperature)
    
    head_probs = get_token_probabilities(model, tokenizer, token_prompt, valid_heads, temperature)
    
    joint_scores = {}
    
    for head_str, head_prob in head_probs.items():
        head = int(head_str)
        head_idx = head - 1 if head > 0 else -1
        head_pos = pred_pos[head_idx] if 0 <= head_idx < len(pred_pos) else None
        
        valid_deps = get_enhanced_deps_for_pos(
            pred_pos[i], head_pos, head, i+1, sentence_length
        )
        
        dep_prompt = token_prompt + f"{head}\t"
        dep_probs = get_token_probabilities(model, tokenizer, dep_prompt, valid_deps, temperature)
        
        for dep, dep_prob in dep_probs.items():
            joint_scores[(head, dep)] = head_prob * dep_prob
    
    return joint_scores

def parse_sentence_joint(current_sentence, model, tokenizer, example_sentences=None, num_examples=0, temperature=0.3):
    words, tokens, multi_tokens = current_sentence
    sentence_length = len(tokens)
    
    base_prompt = create_conllu_prompt(current_sentence, example_sentences, num_examples)
    
    pred_pos = []
    
    for i in range(sentence_length):
        (token_id, word, lemma) = tokens[i]
        
        token_prompt = base_prompt
        
        for id_range, start_id, end_id, mw_word, mw_lemma in multi_tokens:
            if start_id == token_id:
                token_prompt += f"{id_range}\t{mw_word}\t{mw_lemma}\t_\t_\t_\t_\t_\t_\t_\n"
        
        for j in range(i):
            prev_token_id, prev_word, prev_lemma = tokens[j]
            prev_pos = pred_pos[j]
            token_prompt += f"{prev_token_id}\t{prev_word}\t{prev_lemma}\t{prev_pos}\t_\t_\t_\t_\t_\t_\n"
        
        token_prompt += f"{token_id}\t{word}\t{lemma}\t"
        
        try:
            best_pos = get_best_token(
                model, tokenizer, token_prompt, 
                constraints=UNIVERSAL_POS_TAGS,
                temperature=temperature
            )
            
            pred_pos.append(best_pos)
            
        except Exception as e:
            print(f"Error processing POS tag for token {token_id} ({word}): {str(e)}")
            if word[0].isupper() and word.lower() != word.upper():
                pred_pos.append("PROPN")
            elif word in ",.;:!?":
                pred_pos.append("PUNCT")
            else:
                pred_pos.append("NOUN")
    
    root_idx = find_best_root(pred_pos)
    
    pred_heads = []
    pred_deps = []
    
    for i in range(sentence_length):
        if i == root_idx:
            pred_heads.append(0)
            pred_deps.append("root")
        else:
            pred_heads.append(root_idx + 1)
            pred_deps.append("dep")
    
    for i in range(sentence_length):
        if i == root_idx:
            continue
            
        (token_id, word, lemma) = tokens[i]
        
        token_prompt = base_prompt
        
        multi_token_map = {}
        for id_range, start_id, end_id, mw_word, mw_lemma in multi_tokens:
            multi_token_map[start_id] = (id_range, mw_word, mw_lemma)
        
        for j in range(sentence_length):
            prev_token_id, prev_word, prev_lemma = tokens[j]
            prev_pos = pred_pos[j]
            
            if prev_token_id in multi_token_map:
                id_range, mw_word, mw_lemma = multi_token_map[prev_token_id]
                token_prompt += f"{id_range}\t{mw_word}\t{mw_lemma}\t_\t_\t_\t_\t_\t_\t_\n"
            
            if j < i and j != root_idx:
                prev_head = pred_heads[j]
                prev_dep = pred_deps[j]
                token_prompt += f"{prev_token_id}\t{prev_word}\t{prev_lemma}\t{prev_pos}\t_\t_\t{prev_head}\t{prev_dep}\t_\t_\n"
            elif j == root_idx:
                token_prompt += f"{prev_token_id}\t{prev_word}\t{prev_lemma}\t{prev_pos}\t_\t_\t0\troot\t_\t_\n"
            elif j != i:
                token_prompt += f"{prev_token_id}\t{prev_word}\t{prev_lemma}\t{prev_pos}\t_\t_\t_\t_\t_\t_\n"
        
        if token_id in multi_token_map:
            id_range, mw_word, mw_lemma = multi_token_map[token_id]
            token_prompt += f"{id_range}\t{mw_word}\t{mw_lemma}\t_\t_\t_\t_\t_\t_\t_\n"
        
        token_prompt += f"{token_id}\t{word}\t{lemma}\t{pred_pos[i]}\t_\t_\t"
        
        try:
            joint_scores = get_joint_head_dep_scores(
                model, tokenizer, token_prompt, i, pred_pos, sentence_length, temperature
            )
            
            if joint_scores:
                best_head, best_dep = max(joint_scores.items(), key=lambda x: x[1])[0]
                
                if best_head == i + 1:
                    best_head = root_idx + 1
                    
                    head_pos = pred_pos[root_idx]
                    valid_deps = get_enhanced_deps_for_pos(
                        pred_pos[i], head_pos, best_head, i+1, sentence_length
                    )
                    
                    dep_prompt = token_prompt + f"{best_head}\t"
                    best_dep = get_best_token(
                        model, tokenizer, dep_prompt, 
                        constraints=valid_deps,
                        temperature=temperature
                    )
                
                pred_heads[i] = best_head
                pred_deps[i] = best_dep
            else:
                valid_heads = get_refined_head_candidates(
                    i, sentence_length, pred_pos, words, temperature
                )
                
                best_head = int(get_best_token(
                    model, tokenizer, token_prompt, 
                    constraints=valid_heads,
                    temperature=temperature
                ))
                
                if best_head == i + 1:
                    best_head = root_idx + 1
                
                head_idx = best_head - 1 if best_head > 0 else -1
                head_pos = pred_pos[head_idx] if 0 <= head_idx < len(pred_pos) else None
                
                dep_prompt = token_prompt + f"{best_head}\t"
                
                valid_deps = get_enhanced_deps_for_pos(
                    pred_pos[i], head_pos, best_head, i+1, sentence_length
                )
                
                best_dep = get_best_token(
                    model, tokenizer, dep_prompt, 
                    constraints=valid_deps,
                    temperature=temperature
                )
                
                pred_heads[i] = best_head
                pred_deps[i] = best_dep
            
        except Exception as e:
            print(f"Error processing dependencies for token {token_id} ({word}): {str(e)}")
    
    fix_dependency_tree(pred_heads, pred_deps, pred_pos, root_idx, sentence_length)
    
    return pred_pos, pred_heads, pred_deps, multi_tokens

def parse_with_cpu_fallback(current_sentence, model, tokenizer, example_sentences=None, num_examples=0, temperature=0.3):
    try:
        return parse_sentence_joint(current_sentence, model, tokenizer, example_sentences, num_examples, temperature)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA out of memory, falling back to CPU for this sentence: {' '.join([word for _, word, _ in current_sentence[1]])}")
            device = next(model.parameters()).device
            model = model.cpu()
            result = parse_sentence_joint(current_sentence, model, tokenizer, example_sentences, num_examples, temperature)
            if torch.cuda.is_available():
                model = model.to(device)
            return result
        else:
            raise e

def run_eval_script(gold_file, system_file, eval_script_path="eval.py"):
    try:
        eval_script_path = os.path.abspath(os.path.expanduser(eval_script_path))
        
        if not os.path.exists(eval_script_path):
            print(f"Warning: Evaluation script not found at {eval_script_path}")
            return {"error": f"Evaluation script not found at {eval_script_path}"}
        
        cmd = [sys.executable, eval_script_path, "--gold", gold_file, "--system", system_file]
        print(f"Running evaluation command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        metrics = {}
        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '%' in value:
                    try:
                        value = float(value.replace('%', ''))
                    except ValueError:
                        pass
                
                metrics[key] = value
        
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running eval.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return {"error": str(e)}
    except FileNotFoundError:
        print(f"Error: eval.py script not found at {eval_script_path}")
        return {"error": f"eval.py script not found at {eval_script_path}"}
    except Exception as e:
        print(f"Unexpected error running eval.py: {str(e)}")
        return {"error": str(e)}

def process_sentences(input_sentences, gold_sentences, example_sentences, model_path, num_examples, output_file, temperature=0.3, checkpoint_file=None, eval_script_path="eval.py"):
    start_idx = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data.get('next_sentence_idx', 0)
                print(f"Resuming from checkpoint at sentence {start_idx}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    actual_num_examples = min(num_examples, len(example_sentences)) if example_sentences else 0
    if actual_num_examples != num_examples:
        print(f"Warning: Requested {num_examples} examples, but only {actual_num_examples} are available.")
    
    print("Initializing model (one-time)...")
    model = init_aya_expanse_gpu(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-32b" if model_path is None else model_path)
    
    if torch.cuda.is_available():
        if not hasattr(model, 'hf_device_map') or not any(device == 'cpu' for device in model.hf_device_map.values()):
            model = model.half()
            print("Using half precision (fp16)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n")
    
    for idx in range(start_idx, len(input_sentences)):
        current_sentence = input_sentences[idx]
        words, tokens, multi_tokens = current_sentence
        
        print(f"\nProcessing sentence {idx+1}/{len(input_sentences)}: {' '.join(words)}")
        
        try:
            pred_pos, pred_heads, pred_deps, multi_tokens = parse_with_cpu_fallback(
                current_sentence, model, tokenizer, example_sentences, actual_num_examples,
                temperature=temperature
            )
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"# sent_id = {idx+1}\n")
                f.write(f"# text = {' '.join(words)}\n")
                
                multi_token_map = {}
                for id_range, start_id, end_id, mw_word, mw_lemma in multi_tokens:
                    multi_token_map[start_id] = (id_range, mw_word, mw_lemma)
                
                for (token_id, word, lemma), pos, head, dep in zip(tokens, pred_pos, pred_heads, pred_deps):
                    if token_id in multi_token_map:
                        id_range, mw_word, mw_lemma = multi_token_map[token_id]
                        f.write(f"{id_range}\t{mw_word}\t{mw_lemma}\t_\t_\t_\t_\t_\t_\t_\n")
                    
                    f.write(f"{token_id}\t{word}\t{lemma}\t{pos}\t_\t_\t{head}\t{dep}\t_\t_\n")
                
                f.write("\n")
            
            if checkpoint_file:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'next_sentence_idx': idx + 1,
                        'total_sentences': len(input_sentences),
                        'last_processed': idx
                    }, f)
            
            print(f"  Completed sentence {idx+1}/{len(input_sentences)}")
            
        except Exception as e:
            print(f"Error processing sentence: {' '.join(words)}")
            print(f"Error details: {str(e)}")
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"# sent_id = {idx+1}\n")
                f.write(f"# text = {' '.join(words)}\n")
                f.write(f"# ERROR: {str(e)}\n\n")
            
            continue
    
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\nProcessing complete. Parsed {len(input_sentences)} sentences.")
    print(f"Output saved to {output_file}")
    
    print("\nRunning evaluation with eval.py...")
    gold_file = os.path.abspath(args.gold)
    system_file = os.path.abspath(output_file)
    print(f"Gold file: {gold_file}")
    print(f"System file (output file): {system_file}")
    metrics = run_eval_script(gold_file, system_file, eval_script_path)
    
    print("\n=== Evaluation Results ===")
    if "error" in metrics:
        print(f"Error during evaluation: {metrics['error']}")
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        metrics_file = f"{os.path.splitext(output_file)[0]}_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"Examples: {actual_num_examples}\n")
            f.write(f"Processed: {len(input_sentences)} sentences\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nMetrics saved to {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description='Joint Dependency Parser')
    parser.add_argument('--input', type=str, required=True, help='Input CoNLL-U file')
    parser.add_argument('--gold', type=str, required=True, help='Gold CoNLL-U file')
    parser.add_argument('--examples', type=str, help='Examples CoNLL-U file for in-context learning')
    parser.add_argument('--output', type=str, default='output.conllu', help='Output file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--num_examples', type=int, default=0, help='Number of examples to use')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for softmax (default: 0.3)')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='File to save/load checkpoint information')
    parser.add_argument('--eval_script', type=str, default='eval.py', help='Path to the evaluation script (default: eval.py)')
    global args
    args = parser.parse_args()
    
    print("Loading data...")
    input_sentences = load_conllu_input_file(args.input)
    gold_sentences = load_conllu_gold_file(args.gold)
    
    example_sentences = []
    if args.examples:
        print("Loading examples...")
        example_sentences = load_conllu_examples_file(args.examples)
        print(f"Loaded {len(example_sentences)} examples")
    
    output_file = f"{os.path.splitext(args.output)[0]}_joint_{args.num_examples}_examples.conllu"
    
    print(f"Processing with {args.num_examples} examples using joint prediction approach...")
    
    process_sentences(
        input_sentences, gold_sentences, example_sentences,
        args.model_path, args.num_examples, output_file,
        temperature=args.temperature,
        checkpoint_file=args.checkpoint_file,
        eval_script_path=args.eval_script
    )

if __name__ == "__main__":
    main()