#!/usr/bin/env python3
import argparse
import os
import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from difflib import SequenceMatcher
from collections import Counter

output_file = None

def print_and_save(message):
    print(message)
    if output_file:
        output_file.write(message + '\n')
        output_file.flush()

def parse_conllu(file_path):
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append({'tokens': current_sentence})
                    current_sentence = []
            elif not line.startswith('#'):
                fields = line.split('\t')
                if len(fields) >= 8 and '-' not in fields[0]:
                    token = {
                        'form': fields[1],
                        'upos': fields[3],
                        'head': fields[6],
                        'deprel': fields[7]
                    }
                    current_sentence.append(token)
    
    if current_sentence:
        sentences.append({'tokens': current_sentence})
    
    return sentences

def create_signature(sentence):
    tokens = sentence['tokens']
    pos_tags = [t['upos'] for t in tokens if t['upos'] != '_']
    dep_labels = [t['deprel'] for t in tokens if t['deprel'] != '_']
    
    patterns = []
    for token in tokens:
        if token['head'] not in ['0', '_']:
            try:
                head_idx = int(token['head']) - 1
                if 0 <= head_idx < len(tokens):
                    head_pos = tokens[head_idx]['upos']
                    patterns.append(f"{token['deprel']}({head_pos}->{token['upos']})")
            except:
                pass
    
    if not patterns and not pos_tags and not dep_labels:
        return f"FALLBACK_LEN_{len(tokens)}"
    
    return ' '.join(pos_tags + dep_labels + patterns + [f"LEN_{len(tokens)//5*5}"])

def sequence_similarity(sig1, sig2):
    if not sig1.strip() or not sig2.strip():
        return 0.0
    return SequenceMatcher(None, sig1.split(), sig2.split()).ratio()

def ngram_overlap_similarity(sig1, sig2, n=2):
    if not sig1.strip() or not sig2.strip():
        return 0.0
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    tokens1 = sig1.split()
    tokens2 = sig2.split()
    
    if len(tokens1) < n or len(tokens2) < n:
        return 0.0
    
    ngrams1 = set(get_ngrams(tokens1, n))
    ngrams2 = set(get_ngrams(tokens2, n))
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0

def hybrid_similarity(sig1, sig2, tfidf_sim):
    seq_sim = sequence_similarity(sig1, sig2)
    ngram_sim = ngram_overlap_similarity(sig1, sig2)
    return 0.4 * tfidf_sim + 0.4 * seq_sim + 0.2 * ngram_sim

def calculate_similarity(example_sentences, gold_sentences):
    if not example_sentences:
        return np.random.rand(len(gold_sentences)) * 0.1
    
    first_example = example_sentences[0]
    example_sig = create_signature(first_example)
    gold_sigs = [create_signature(s) for s in gold_sentences]
    
    example_sig = example_sig if example_sig.strip() else "EMPTY_EX_0"
    gold_sigs = [sig if sig.strip() else f"EMPTY_GOLD_{i}" for i, sig in enumerate(gold_sigs)]
    
    try:
        tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),      
            min_df=1,                 
            max_df=0.8,                
            sublinear_tf=True,         
            norm='l2'                   
        )
        
        all_sigs = [example_sig] + gold_sigs
        tfidf_matrix = tfidf.fit_transform(all_sigs)
        tfidf_similarities = cosine_similarity(tfidf_matrix[:1], tfidf_matrix[1:])
        
        similarities = []
        for j in range(len(gold_sigs)):
            sim = hybrid_similarity(example_sig, gold_sigs[j], tfidf_similarities[0, j])
            similarities.append(sim)
        
        return np.array(similarities)
    except:
        return np.random.rand(len(gold_sentences)) * 0.1

def calculate_performance(gold_sentences, system_sentences):
    upos_scores = []
    las_scores = []
    uas_scores = []
    
    min_len = min(len(gold_sentences), len(system_sentences))
    
    for i in range(min_len):
        gold_tokens = gold_sentences[i]['tokens']
        system_tokens = system_sentences[i]['tokens']
        
        if len(gold_tokens) != len(system_tokens):
            continue
        
        upos_correct = sum(1 for g, s in zip(gold_tokens, system_tokens) if g['upos'] == s['upos'])
        las_correct = sum(1 for g, s in zip(gold_tokens, system_tokens) 
                         if g['head'] == s['head'] and g['deprel'] == s['deprel'])
        uas_correct = sum(1 for g, s in zip(gold_tokens, system_tokens) if g['head'] == s['head'])
        
        token_count = len(gold_tokens)
        upos_scores.append(upos_correct / token_count)
        las_scores.append(las_correct / token_count)
        uas_scores.append(uas_correct / token_count)
    
    return upos_scores, las_scores, uas_scores

def is_oneshot(filename):
    basename = os.path.basename(filename)
    return ('_1_' in basename or basename.endswith('_1.conllu'))

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (mean2 - mean1) / pooled_std

def compare_random_vs_similar(gold_file, example_file, system_file):
    print_and_save(f"Analyzing: {os.path.basename(system_file)}")
    
    if not is_oneshot(system_file):
        print_and_save("  Skipped: Not a 1-shot file")
        return None
    
    gold_sentences = parse_conllu(gold_file)
    example_sentences = parse_conllu(example_file)
    system_sentences = parse_conllu(system_file)
    
    if not all([gold_sentences, example_sentences, system_sentences]):
        print_and_save("  Error: Empty files")
        return None
    
    similarities = calculate_similarity(example_sentences, gold_sentences)
    upos_scores, las_scores, uas_scores = calculate_performance(gold_sentences, system_sentences)
    
    min_len = min(len(similarities), len(upos_scores))
    similarities = similarities[:min_len]
    upos_scores = upos_scores[:min_len]
    las_scores = las_scores[:min_len]
    uas_scores = uas_scores[:min_len]
    
    if len(similarities) < 4:
        print_and_save("  Error: Insufficient data for comparison")
        return None
    
    try:
        median_sim = np.median(similarities)
        
        random_indices = [i for i, sim in enumerate(similarities) if sim <= median_sim]
        similar_indices = [i for i, sim in enumerate(similarities) if sim > median_sim]
        
        if len(random_indices) < 2 or len(similar_indices) < 2:
            print_and_save("  Error: Insufficient data in groups")
            return None
        
        print_and_save(f"  Sentences analyzed: {min_len}")
        print_and_save(f"  Random group: {len(random_indices)}, Similar group: {len(similar_indices)}")
        
        random_upos = [upos_scores[i] for i in random_indices]
        similar_upos = [upos_scores[i] for i in similar_indices]
        random_las = [las_scores[i] for i in random_indices]
        similar_las = [las_scores[i] for i in similar_indices]
        random_uas = [uas_scores[i] for i in random_indices]
        similar_uas = [uas_scores[i] for i in similar_indices]
        
        t_upos, p_upos = stats.ttest_ind(similar_upos, random_upos)
        t_las, p_las = stats.ttest_ind(similar_las, random_las)
        t_uas, p_uas = stats.ttest_ind(similar_uas, random_uas)
        
        cohens_d_upos = calculate_cohens_d(random_upos, similar_upos)
        cohens_d_las = calculate_cohens_d(random_las, similar_las)
        cohens_d_uas = calculate_cohens_d(random_uas, similar_uas)
        
        mean_random_upos = np.mean(random_upos)
        mean_similar_upos = np.mean(similar_upos)
        mean_random_las = np.mean(random_las)
        mean_similar_las = np.mean(similar_las)
        mean_random_uas = np.mean(random_uas)
        mean_similar_uas = np.mean(similar_uas)
        
        upos_improvement = mean_similar_upos - mean_random_upos
        las_improvement = mean_similar_las - mean_random_las
        uas_improvement = mean_similar_uas - mean_random_uas
        
        def is_supported(improvement, p_val, cohens_d):
            return improvement > 0 and p_val <= 0.05 and abs(cohens_d) >= 0.2
        
        result = {
            'file': os.path.basename(system_file),
            'upos': {
                'random_mean': mean_random_upos,
                'similar_mean': mean_similar_upos,
                'improvement': upos_improvement,
                'p': p_upos,
                'cohens_d': cohens_d_upos,
                'supported': is_supported(upos_improvement, p_upos, cohens_d_upos)
            },
            'las': {
                'random_mean': mean_random_las,
                'similar_mean': mean_similar_las,
                'improvement': las_improvement,
                'p': p_las,
                'cohens_d': cohens_d_las,
                'supported': is_supported(las_improvement, p_las, cohens_d_las)
            },
            'uas': {
                'random_mean': mean_random_uas,
                'similar_mean': mean_similar_uas,
                'improvement': uas_improvement,
                'p': p_uas,
                'cohens_d': cohens_d_uas,
                'supported': is_supported(uas_improvement, p_uas, cohens_d_uas)
            }
        }
        
        print_and_save(f"  UPOS: Random={mean_random_upos:.3f}, Similar={mean_similar_upos:.3f}, Δ={upos_improvement:.3f}, p={p_upos:.3f}, d={cohens_d_upos:.3f} -> {'SUPPORTED' if result['upos']['supported'] else 'NOT SUPPORTED'}")
        print_and_save(f"  LAS:  Random={mean_random_las:.3f}, Similar={mean_similar_las:.3f}, Δ={las_improvement:.3f}, p={p_las:.3f}, d={cohens_d_las:.3f} -> {'SUPPORTED' if result['las']['supported'] else 'NOT SUPPORTED'}")
        print_and_save(f"  UAS:  Random={mean_random_uas:.3f}, Similar={mean_similar_uas:.3f}, Δ={uas_improvement:.3f}, p={p_uas:.3f}, d={cohens_d_uas:.3f} -> {'SUPPORTED' if result['uas']['supported'] else 'NOT SUPPORTED'}")
        
        return result
        
    except Exception as e:
        print_and_save(f"  Error: {e}")
        return None

def main():
    global output_file
    
    parser = argparse.ArgumentParser(description='1-Shot Random vs Similar Performance Comparison')
    parser.add_argument('--output-folder', help='Output folder path')
    parser.add_argument('--gold-folder', help='Gold files folder path')
    parser.add_argument('--annotations-folder', help='Annotations folder path')
    parser.add_argument('--output', help='Save results to text file')
    
    args = parser.parse_args()
    
    if not all([args.output_folder, args.gold_folder, args.annotations_folder]):
        print("Error: All three folder paths required")
        return
    
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
        print(f"Saving results to: {args.output}")
    
    try:
        languages = [d for d in os.listdir(args.output_folder) 
                    if os.path.isdir(os.path.join(args.output_folder, d)) and len(d) == 2]
        
        for lang in sorted(languages):
            print_and_save(f"\n=== PROCESSING {lang.upper()} ===")
            
            gold_file = os.path.join(args.gold_folder, f'gold_file_{lang}.conllu')
            if not os.path.exists(gold_file):
                print_and_save(f"Gold file not found: {gold_file}")
                continue
            
            system_files = glob.glob(os.path.join(args.output_folder, lang, f'{lang}_*.conllu'))
            oneshot_files = [f for f in system_files if is_oneshot(f)]
            
            if not oneshot_files:
                print_and_save(f"No 1-shot files found for {lang}")
                continue
            
            results = []
            for system_file in oneshot_files:
                basename = os.path.basename(system_file)
                example_file = None
                
                if '_1_long' in basename:
                    source_lang = basename.split('_')[1]
                    example_file = os.path.join(args.annotations_folder, f'example_annotation_{source_lang}_1S_long.conllu')
                elif '_1_short' in basename:
                    source_lang = basename.split('_')[1]
                    example_file = os.path.join(args.annotations_folder, f'example_annotation_{source_lang}_1S_short.conllu')
                else:
                    example_file = os.path.join(args.annotations_folder, f'{lang}_annotations.conllu')
                
                if not os.path.exists(example_file):
                    print_and_save(f"Example file not found: {example_file}")
                    continue
                
                result = compare_random_vs_similar(gold_file, example_file, system_file)
                if result:
                    results.append(result)
            
            if results:
                upos_supported = sum(1 for r in results if r['upos']['supported'])
                las_supported = sum(1 for r in results if r['las']['supported'])
                uas_supported = sum(1 for r in results if r['uas']['supported'])
                
                print_and_save(f"\nSUMMARY for {lang}:")
                print_and_save(f"Files tested: {len(results)}")
                print_and_save(f"UPOS supported: {upos_supported}/{len(results)} ({100*upos_supported/len(results):.1f}%)")
                print_and_save(f"LAS supported: {las_supported}/{len(results)} ({100*las_supported/len(results):.1f}%)")
                print_and_save(f"UAS supported: {uas_supported}/{len(results)} ({100*uas_supported/len(results):.1f}%)")
    
    finally:
        if output_file:
            output_file.close()
            print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
