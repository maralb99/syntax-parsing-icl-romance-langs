from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoTokenizer
from accelerate import dispatch_model
import os
import torch

UNIVERSAL_POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

UNIVERSAL_DEPREL_LABELS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp",
    "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp"
]

def init_aya_expanse_gpu(model_path=None):
    if model_path is None:
        model_path = "CohereForAI/aya-expanse-32b"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model

def load_conllu_input_file(file_path):
    sentences = []
    current_sentence = []
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                if current_sentence:
                    sentences.append((current_text, current_sentence))
                    current_sentence = []
                    current_text = []
                continue
                
            parts = line.split('\t')
            if '-' in parts[0]:
                continue
                
            if len(parts) >= 2:
                token_id = parts[0]
                word = parts[1]
                lemma = parts[2] if len(parts) > 2 else '_'
                
                current_text.append(word)
                current_sentence.append((token_id, word, lemma))
    
    if current_sentence:
        sentences.append((current_text, current_sentence))
        
    return sentences

def load_conllu_gold_file(file_path):
    sentences = []
    current_text = []
    current_pos = []
    current_heads = []
    current_deps = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                if current_text:
                    sentences.append((
                        current_text, 
                        [current_pos, current_heads, current_deps]
                    ))
                    current_text = []
                    current_pos = []
                    current_heads = []
                    current_deps = []
                continue
                
            parts = line.split('\t')
            if '-' in parts[0]:
                continue
                
            word = parts[1]
            pos = parts[3]
            head = int(parts[6]) if parts[6] != '_' else -1
            dep = parts[7].split(':')[0]
            
            current_text.append(word)
            current_pos.append(pos)
            current_heads.append(head)
            current_deps.append(dep)
    
    if current_text:
        sentences.append((
            current_text, 
            [current_pos, current_heads, current_deps]
        ))
        
    return sentences

def load_conllu_examples_file(file_path):
    return load_conllu_gold_file(file_path)

def ud_pos_accuracy(gold, pred):
    correct = 0
    total = 0
    
    for g_sent, p_sent in zip(gold, pred):
        for g_tag, p_tag in zip(g_sent, p_sent):
            if g_tag != '_':
                total += 1
                if g_tag == p_tag:
                    correct += 1
    
    return (correct / total * 100) if total > 0 else 0

def ud_dep_metrics(gold_heads, gold_deps, pred_heads, pred_deps):
    correct_head = 0
    correct_head_and_rel = 0
    total = 0
    
    for g_heads, g_deps, p_heads, p_deps in zip(gold_heads, gold_deps, pred_heads, pred_deps):
        for g_head, g_dep, p_head, p_dep in zip(g_heads, g_deps, p_heads, p_deps):
            if g_head <= 0:
                continue
            
            total += 1
            if g_head == p_head:
                correct_head += 1
                if g_dep == p_dep:
                    correct_head_and_rel += 1
    
    uas = (correct_head / total * 100) if total > 0 else 0
    las = (correct_head_and_rel / total * 100) if total > 0 else 0
    
    return uas, las
