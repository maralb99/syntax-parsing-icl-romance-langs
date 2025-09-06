#!/usr/bin/env python3
import sys
import argparse
import os
import re
import glob
from collections import Counter, defaultdict
import conllu

def load_conllu(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return conllu.parse(f.read())
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def map_line_numbers(file_path):
    line_map = {}
    current_sent_id = None
    token_index = 0
    line_number = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                
                if not line:
                    token_index = 0
                    continue
                
                if line.startswith('# sent_id = '):
                    current_sent_id = line.split('=', 1)[1].strip()
                    line_map[current_sent_id] = {}
                    token_index = 0
                    continue
                
                if not line.startswith('#') and current_sent_id is not None:
                    line_map[current_sent_id][token_index] = line_number
                    token_index += 1
    
    except Exception as e:
        print(f"Error mapping line numbers in {file_path}: {e}")
    
    return line_map

class EnhancedErrorAnalyzer:
    def __init__(self, gold_file, system_file):
        self.gold_file = gold_file
        self.system_file = system_file
        
        self.gold_data = load_conllu(gold_file)
        self.system_data = load_conllu(system_file)
        
        self.gold_sentences = {sent.metadata.get('sent_id', f'sent_{i}'): sent 
                              for i, sent in enumerate(self.gold_data)}
        self.system_sentences = {sent.metadata.get('sent_id', f'sent_{i}'): sent 
                                for i, sent in enumerate(self.system_data)}
        
        self.gold_line_map = map_line_numbers(gold_file)
        self.system_line_map = map_line_numbers(system_file)
        
        self.error_stats = {
            'upos': Counter(),       
            'head': Counter(),       
            'deprel': Counter(),    
            'upos_by_gold': defaultdict(Counter), 
            'deprel_by_gold': defaultdict(Counter),
            'head_distance': Counter(),
            'error_examples': defaultdict(list),
            'error_lines': defaultdict(list),
            'error_contexts': defaultdict(list),
            'token_pos_errors': Counter(),
            'token_form_errors': Counter(),
            'context_errors': defaultdict(Counter),
            'error_words_by_type': {
                'upos': defaultdict(Counter),
                'deprel': defaultdict(Counter),
                'head': defaultdict(Counter)
            },
            'total_tokens': 0,
            'total_errors': {
                'upos': 0,
                'head': 0,
                'deprel': 0
            }
        }
    
    def analyze_errors(self):
        matched_sentences = []
        
        for sent_id, gold_sent in self.gold_sentences.items():
            if sent_id not in self.system_sentences:
                continue
            
            system_sent = self.system_sentences[sent_id]
            
            if len(gold_sent) != len(system_sent):
                continue
            
            token_match = True
            for g_token, s_token in zip(gold_sent, system_sent):
                if g_token['form'] != s_token['form']:
                    token_match = False
                    break
            
            if token_match:
                matched_sentences.append((sent_id, gold_sent, system_sent))
        
        for sent_id, gold_sent, system_sent in matched_sentences:
            sentence_text = ' '.join(token['form'] for token in gold_sent)
            
            for i, (g_token, s_token) in enumerate(zip(gold_sent, system_sent)):
                token_line = self.gold_line_map.get(sent_id, {}).get(i, 0)
                if token_line == 0:
                    token_line = f"unknown (sent_id={sent_id}, token_idx={i})"
                
                self.error_stats['total_tokens'] += 1
                
                position = "Beginning" if i < len(gold_sent) // 3 else "End" if i > 2 * len(gold_sent) // 3 else "Middle"
                token_form = g_token['form'].lower()
                
                g_upos = g_token.get('upos', '_')
                s_upos = s_token.get('upos', '_')
                
                if g_upos != s_upos:
                    self.error_stats['total_errors']['upos'] += 1
                    error_key = f"{g_upos} → {s_upos}"
                    self.error_stats['upos'][error_key] += 1
                    self.error_stats['upos_by_gold'][g_upos][s_upos] += 1
                    self.error_stats['token_pos_errors'][position] += 1
                    self.error_stats['token_form_errors'][token_form] += 1
                    
                    self.error_stats['error_words_by_type']['upos'][error_key][token_form] += 1
                    
                    self.error_stats['error_lines'][error_key].append(token_line)
                    
                    prev_token = gold_sent[i-1]['form'] if i > 0 else "START"
                    next_token = gold_sent[i+1]['form'] if i < len(gold_sent) - 1 else "END"
                    self.error_stats['context_errors']['prev'][prev_token] += 1
                    self.error_stats['context_errors']['next'][next_token] += 1
                    
                    context_info = {
                        'sent_id': sent_id,
                        'token_index': i,
                        'token_form': g_token['form'],
                        'sentence': sentence_text,
                        'context': f"{prev_token} {g_token['form']} {next_token}".strip(),
                        'line_number': token_line,
                        'gold_upos': g_upos,
                        'system_upos': s_upos
                    }
                    
                    self.error_stats['error_examples'][error_key].append(context_info)
                    self.error_stats['error_contexts'][error_key].append(context_info)
                
                g_head = g_token.get('head', 0)
                s_head = s_token.get('head', 0)
                
                try:
                    g_head = int(g_head) if g_head != '_' else 0
                except (ValueError, TypeError):
                    g_head = 0
                
                try:
                    s_head = int(s_head) if s_head != '_' else 0
                except (ValueError, TypeError):
                    s_head = 0
                
                if g_head != s_head:
                    self.error_stats['total_errors']['head'] += 1
                    
                    if g_head == 0 or s_head == 0:
                        distance = "root_error"
                    else:
                        distance = abs(g_head - s_head)
                    
                    self.error_stats['head_distance'][distance] += 1
                    
                    g_head_form = "ROOT" if g_head == 0 else gold_sent[g_head-1]['form']
                    s_head_form = "ROOT" if s_head == 0 else system_sent[s_head-1]['form']
                    
                    error_key = f"{g_head} ({g_head_form}) → {s_head} ({s_head_form})"
                    self.error_stats['head'][error_key] += 1
                    self.error_stats['error_lines'][error_key].append(token_line)
                    
                    self.error_stats['error_words_by_type']['head'][error_key][token_form] += 1
                    
                    context_info = {
                        'sent_id': sent_id,
                        'token_index': i,
                        'token_form': g_token['form'],
                        'sentence': sentence_text,
                        'gold_head_index': g_head,
                        'system_head_index': s_head,
                        'line_number': token_line
                    }
                    
                    self.error_stats['error_examples'][error_key].append(context_info)
                    self.error_stats['error_contexts'][error_key].append(context_info)
                
                g_deprel = g_token.get('deprel', '_')
                s_deprel = s_token.get('deprel', '_')
                
                if g_deprel != s_deprel:
                    self.error_stats['total_errors']['deprel'] += 1
                    error_key = f"{g_deprel} → {s_deprel}"
                    self.error_stats['deprel'][error_key] += 1
                    self.error_stats['deprel_by_gold'][g_deprel][s_deprel] += 1
                    self.error_stats['error_lines'][error_key].append(token_line)
                    
                    self.error_stats['error_words_by_type']['deprel'][error_key][token_form] += 1
                    
                    context_info = {
                        'sent_id': sent_id,
                        'token_index': i,
                        'token_form': g_token['form'],
                        'sentence': sentence_text,
                        'upos': g_upos,
                        'line_number': token_line,
                        'gold_deprel': g_deprel,
                        'system_deprel': s_deprel
                    }
                    
                    self.error_stats['error_examples'][error_key].append(context_info)
                    self.error_stats['error_contexts'][error_key].append(context_info)
        
        return self.error_stats

    def generate_summary_report(self, output_file=None):
        if self.error_stats['total_tokens'] == 0:
            self.analyze_errors()
        
        report_lines = []
        report_lines.append(f"SUMMARY ERROR ANALYSIS: {os.path.basename(self.system_file)} vs. {os.path.basename(self.gold_file)}\n")
        report_lines.append("=" * 80 + "\n\n")
        
        report_lines.append("GENERAL STATISTICS:\n")
        report_lines.append(f"Analyzed tokens: {self.error_stats['total_tokens']}\n")
        
        for error_type in ['upos', 'head', 'deprel']:
            error_count = self.error_stats['total_errors'][error_type]
            error_pct = error_count / self.error_stats['total_tokens'] * 100 if self.error_stats['total_tokens'] > 0 else 0
            report_lines.append(f"{error_type.upper()} errors: {error_count} ({error_pct:.2f}%)\n")
        
        report_lines.append("\n" + "=" * 80 + "\n\n")
        
        report_lines.append("MOST COMMON ERROR-CAUSING WORDS (TOP 10):\n")
        report_lines.append("-" * 50 + "\n")
        for word, count in self.error_stats['token_form_errors'].most_common(10):
            pct = count / sum(self.error_stats['total_errors'].values()) * 100 if sum(self.error_stats['total_errors'].values()) > 0 else 0
            report_lines.append(f"'{word}': {count} errors ({pct:.2f}%)\n")
        report_lines.append("\n")
        
        report_lines.append("TOP 10 UPOS ERRORS WITH EXAMPLES AND MOST COMMON ERROR-CAUSING WORDS:\n")
        for error, count in self.error_stats['upos'].most_common(10):
            pct = count / self.error_stats['total_errors']['upos'] * 100 if self.error_stats['total_errors']['upos'] > 0 else 0
            report_lines.append(f"{error}: {count} occurrences ({pct:.2f}%)\n")
            
            if error in self.error_stats['error_words_by_type']['upos']:
                top_words = self.error_stats['error_words_by_type']['upos'][error].most_common(5)
                if top_words:
                    report_lines.append("Most common error-causing words:\n")
                    for word, word_count in top_words:
                        word_pct = word_count / count * 100
                        report_lines.append(f"  '{word}': {word_count} times ({word_pct:.1f}% of this error)\n")
                    report_lines.append("\n")
            
            if error in self.error_stats['error_examples']:
                report_lines.append("Examples:\n")
                for i, example in enumerate(self.error_stats['error_examples'][error][:5], 1):
                    report_lines.append(f"  {i}. Line {example['line_number']}: '{example['token_form']}' in sentence {example['sent_id']}\n")
                    report_lines.append(f"     Context: {example['context']}\n")
                    report_lines.append(f"     Sentence: {example['sentence']}\n")
                if len(self.error_stats['error_examples'][error]) > 5:
                    report_lines.append(f"     ... and {len(self.error_stats['error_examples'][error]) - 5} more occurrences\n")
                report_lines.append("\n")
        
        report_lines.append("\n" + "-" * 80 + "\n\n")
        
        report_lines.append("TOP 10 DEPREL ERRORS WITH EXAMPLES AND MOST COMMON ERROR-CAUSING WORDS:\n")
        for error, count in self.error_stats['deprel'].most_common(10):
            pct = count / self.error_stats['total_errors']['deprel'] * 100 if self.error_stats['total_errors']['deprel'] > 0 else 0
            report_lines.append(f"{error}: {count} occurrences ({pct:.2f}%)\n")
            
            if error in self.error_stats['error_words_by_type']['deprel']:
                top_words = self.error_stats['error_words_by_type']['deprel'][error].most_common(5)
                if top_words:
                    report_lines.append("Most common error-causing words:\n")
                    for word, word_count in top_words:
                        word_pct = word_count / count * 100
                        report_lines.append(f"  '{word}': {word_count} times ({word_pct:.1f}% of this error)\n")
                    report_lines.append("\n")
            
            if error in self.error_stats['error_examples']:
                report_lines.append("Examples:\n")
                for i, example in enumerate(self.error_stats['error_examples'][error][:5], 1):
                    report_lines.append(f"  {i}. Line {example['line_number']}: '{example['token_form']}' (UPOS: {example['upos']}) in sentence {example['sent_id']}\n")
                    report_lines.append(f"     Sentence: {example['sentence']}\n")
                if len(self.error_stats['error_examples'][error]) > 5:
                    report_lines.append(f"     ... and {len(self.error_stats['error_examples'][error]) - 5} more occurrences\n")
                report_lines.append("\n")
        
        report_text = ''.join(report_lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Summary error analysis saved to {output_file}.")
        else:
            print(report_text)
        
        return report_text

    def generate_complete_report(self, output_file=None):
        if self.error_stats['total_tokens'] == 0:
            self.analyze_errors()
        
        report_lines = []
        report_lines.append(f"COMPLETE ERROR ANALYSIS: {os.path.basename(self.system_file)} vs. {os.path.basename(self.gold_file)}\n")
        report_lines.append("=" * 80 + "\n\n")
        
        report_lines.append("GENERAL STATISTICS:\n")
        report_lines.append(f"Analyzed tokens: {self.error_stats['total_tokens']}\n")
        
        for error_type in ['upos', 'head', 'deprel']:
            error_count = self.error_stats['total_errors'][error_type]
            error_pct = error_count / self.error_stats['total_tokens'] * 100 if self.error_stats['total_tokens'] > 0 else 0
            report_lines.append(f"{error_type.upper()} errors: {error_count} ({error_pct:.2f}%)\n")
        
        report_lines.append("\n" + "=" * 80 + "\n\n")
        
        report_lines.append("COMPLETE WORD ERROR STATISTICS:\n")
        report_lines.append("=" * 50 + "\n\n")
        report_lines.append("ALL ERROR-CAUSING WORDS (sorted by frequency):\n")
        for word, count in self.error_stats['token_form_errors'].most_common():
            pct = count / sum(self.error_stats['total_errors'].values()) * 100 if sum(self.error_stats['total_errors'].values()) > 0 else 0
            report_lines.append(f"'{word}': {count} total errors ({pct:.2f}%)\n")
        report_lines.append("\n" + "-" * 50 + "\n\n")
        
        report_lines.append("ALL UPOS ERRORS WITH ALL OCCURRENCES AND WORD STATISTICS:\n")
        report_lines.append("=" * 50 + "\n\n")
        
        for error, count in self.error_stats['upos'].most_common():
            pct = count / self.error_stats['total_errors']['upos'] * 100 if self.error_stats['total_errors']['upos'] > 0 else 0
            report_lines.append(f"ERROR: {error}\n")
            report_lines.append(f"Occurrences: {count} ({pct:.2f}%)\n")
            
            if error in self.error_stats['error_words_by_type']['upos']:
                all_words = self.error_stats['error_words_by_type']['upos'][error].most_common()
                if all_words:
                    report_lines.append("All error-causing words for this error:\n")
                    for word, word_count in all_words:
                        word_pct = word_count / count * 100
                        report_lines.append(f"  '{word}': {word_count} times ({word_pct:.1f}%)\n")
                    report_lines.append("\n")
            
            if error in self.error_stats['error_lines']:
                lines = self.error_stats['error_lines'][error]
                report_lines.append(f"All lines: {', '.join(map(str, lines))}\n")
            
            if error in self.error_stats['error_contexts']:
                report_lines.append("All occurrences:\n")
                for i, example in enumerate(self.error_stats['error_contexts'][error], 1):
                    report_lines.append(f"  {i}. Line {example['line_number']}: '{example['token_form']}' in sentence {example['sent_id']}\n")
                    report_lines.append(f"     Gold UPOS: {example['gold_upos']} → System UPOS: {example['system_upos']}\n")
                    report_lines.append(f"     Context: {example['context']}\n")
                    report_lines.append(f"     Full sentence: {example['sentence']}\n")
                    report_lines.append("\n")
            
            report_lines.append("-" * 50 + "\n\n")
        
        report_lines.append("ALL DEPREL ERRORS WITH ALL OCCURRENCES AND WORD STATISTICS:\n")
        report_lines.append("=" * 50 + "\n\n")
        
        for error, count in self.error_stats['deprel'].most_common():
            pct = count / self.error_stats['total_errors']['deprel'] * 100 if self.error_stats['total_errors']['deprel'] > 0 else 0
            report_lines.append(f"ERROR: {error}\n")
            report_lines.append(f"Occurrences: {count} ({pct:.2f}%)\n")
            
            if error in self.error_stats['error_words_by_type']['deprel']:
                all_words = self.error_stats['error_words_by_type']['deprel'][error].most_common()
                if all_words:
                    report_lines.append("All error-causing words for this error:\n")
                    for word, word_count in all_words:
                        word_pct = word_count / count * 100
                        report_lines.append(f"  '{word}': {word_count} times ({word_pct:.1f}%)\n")
                    report_lines.append("\n")
            
            if error in self.error_stats['error_lines']:
                lines = self.error_stats['error_lines'][error]
                report_lines.append(f"All lines: {', '.join(map(str, lines))}\n")
            
            if error in self.error_stats['error_contexts']:
                report_lines.append("All occurrences:\n")
                for i, example in enumerate(self.error_stats['error_contexts'][error], 1):
                    report_lines.append(f"  {i}. Line {example['line_number']}: '{example['token_form']}' (UPOS: {example['upos']}) in sentence {example['sent_id']}\n")
                    report_lines.append(f"     Gold DEPREL: {example['gold_deprel']} → System DEPREL: {example['system_deprel']}\n")
                    report_lines.append(f"     Full sentence: {example['sentence']}\n")
                    report_lines.append("\n")
            
            report_lines.append("-" * 50 + "\n\n")
        
        report_text = ''.join(report_lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Complete error analysis saved to {output_file}.")
        else:
            print(report_text)
        
        return report_text

def extract_shot_scenario(filename):
    basename = os.path.basename(filename)
    
    if re.search(r'_0\.conllu$', basename):
        return "0-Shot"
    
    mixed_match = re.search(r'_mixed_(\d+)\.conllu$', basename)
    if mixed_match:
        return f"Mixed-{mixed_match.group(1)}"
    
    if '_1_short' in basename:
        return "1-Shot (short)"
    elif '_1_long' in basename:
        return "1-Shot (long)"
    elif re.search(r'_1\.conllu$', basename):
        return "1-Shot"
    
    for shot in [3, 5, 10]:
        if re.search(rf'_{shot}\.conllu$', basename):
            return f"{shot}-Shot"
    
    return "Unknown"

def extract_language_pair(filename):
    basename = os.path.basename(filename)
    
    if '_mixed_' in basename:
        return basename.split('_mixed_')[0] + '_mixed'
    
    if re.search(r'_0\.conllu$', basename):
        return basename.split('_0')[0] + '_0'
    
    match = re.match(r'([a-z]{2})_([a-z]{2})_', basename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    return "unknown"

def group_files_by_scenario_and_language(files):
    grouped = defaultdict(lambda: defaultdict(list))
    
    for file in files:
        scenario = extract_shot_scenario(file)
        language_pair = extract_language_pair(file)
        
        grouped[scenario][language_pair].append(file)
    
    return grouped

def calculate_average_errors_by_language_pair(gold_file, system_files):
    grouped_files = group_files_by_scenario_and_language(system_files)
    
    language_pair_stats = defaultdict(lambda: {
        'total_tokens': 0,
        'total_upos_errors': 0,
        'total_deprel_errors': 0,
        'total_head_errors': 0,
        'error_types': {
            'upos': Counter(),
            'deprel': Counter(),
            'head': Counter()
        },
        'all_error_lines': {
            'upos': defaultdict(set),
            'deprel': defaultdict(set),
            'head': defaultdict(set)
        },
        'error_words': Counter(),
        'scenarios_analyzed': set()
    })
    
    for scenario, language_pairs in grouped_files.items():
        for language_pair, files in language_pairs.items():
            for system_file in files:
                analyzer = EnhancedErrorAnalyzer(gold_file, system_file)
                error_stats = analyzer.analyze_errors()
                
                language_pair_stats[language_pair]['total_tokens'] += error_stats['total_tokens']
                language_pair_stats[language_pair]['total_upos_errors'] += error_stats['total_errors']['upos']
                language_pair_stats[language_pair]['total_deprel_errors'] += error_stats['total_errors']['deprel']
                language_pair_stats[language_pair]['total_head_errors'] += error_stats['total_errors']['head']
                language_pair_stats[language_pair]['scenarios_analyzed'].add(scenario)
                
                for word, count in error_stats['token_form_errors'].items():
                    language_pair_stats[language_pair]['error_words'][word] += count
                
                for error_type in ['upos', 'deprel', 'head']:
                    for error, count in error_stats[error_type].items():
                        language_pair_stats[language_pair]['error_types'][error_type][error] += count
                    
                    for error, lines in error_stats['error_lines'].items():
                        if '→' in error and error in error_stats['upos']:
                            err_type = 'upos'
                        elif '→' in error and error in error_stats['deprel']:
                            err_type = 'deprel'
                        else:
                            err_type = 'head'
                            
                        for line in lines:
                            language_pair_stats[language_pair]['all_error_lines'][err_type][error].add(line)
    
    for language_pair, stats in language_pair_stats.items():
        total_tokens = stats['total_tokens']
        if total_tokens > 0:
            stats['avg_upos_error_rate'] = stats['total_upos_errors'] / total_tokens * 100
            stats['avg_deprel_error_rate'] = stats['total_deprel_errors'] / total_tokens * 100
            stats['avg_head_error_rate'] = stats['total_head_errors'] / total_tokens * 100
        else:
            stats['avg_upos_error_rate'] = 0
            stats['avg_deprel_error_rate'] = 0
            stats['avg_head_error_rate'] = 0
    
    sorted_pairs = sorted(language_pair_stats.keys())
    
    report_lines = []
    report_lines.append("AVERAGE ERROR RATES AND MOST COMMON ERRORS BY LANGUAGE PAIRING:\n")
    report_lines.append("=" * 80 + "\n\n")
    
    report_lines.append("AVERAGE ERROR RATES BY LANGUAGE PAIRING:\n\n")
    for language_pair in sorted_pairs:
        stats = language_pair_stats[language_pair]
        report_lines.append(f"LANGUAGE PAIR: {language_pair}\n")
        report_lines.append("-" * 40 + "\n")
        report_lines.append(f"Average UPOS error rate: {stats['avg_upos_error_rate']:.2f}%\n")
        report_lines.append(f"Average DEPREL error rate: {stats['avg_deprel_error_rate']:.2f}%\n")
        report_lines.append(f"Average HEAD error rate: {stats['avg_head_error_rate']:.2f}%\n")
        
        if stats['error_words']:
            top_words = stats['error_words'].most_common(5)
            report_lines.append("Most common error-causing words:\n")
            for word, count in top_words:
                report_lines.append(f"  '{word}': {count} errors\n")
        
        report_lines.append(f"Analyzed scenarios: {', '.join(sorted(stats['scenarios_analyzed']))}\n")
        report_lines.append("\n")
    
    report_lines.append("\n" + "=" * 80 + "\n\n")
    
    report_lines.append("TOP 2 MOST COMMON ERRORS BY LANGUAGE PAIRING (with ALL line numbers):\n\n")

    for language_pair in sorted_pairs:
        stats = language_pair_stats[language_pair]
        report_lines.append(f"LANGUAGE PAIR: {language_pair}\n")
        report_lines.append("-" * 40 + "\n")
        
        if stats['error_types']['upos']:
            top_upos_errors = stats['error_types']['upos'].most_common(2)
            report_lines.append("Top 2 UPOS errors:\n")
            for i, (error, count) in enumerate(top_upos_errors, 1):
                report_lines.append(f"  {i}. {error} ({count} occurrences)\n")
                
                if error in stats['all_error_lines']['upos']:
                    all_lines = stats['all_error_lines']['upos'][error]
                    report_lines.append(f"     All lines: {', '.join(map(str, sorted(all_lines)))}\n")
        
        if stats['error_types']['deprel']:
            top_deprel_errors = stats['error_types']['deprel'].most_common(2)
            report_lines.append("Top 2 DEPREL errors:\n")
            for i, (error, count) in enumerate(top_deprel_errors, 1):
                report_lines.append(f"  {i}. {error} ({count} occurrences)\n")
                
                if error in stats['all_error_lines']['deprel']:
                    all_lines = stats['all_error_lines']['deprel'][error]
                    report_lines.append(f"     All lines: {', '.join(map(str, sorted(all_lines)))}\n")
        
        if stats['error_types']['head']:
            top_head_errors = stats['error_types']['head'].most_common(2)
            report_lines.append("Top 2 HEAD errors:\n")
            for i, (error, count) in enumerate(top_head_errors, 1):
                report_lines.append(f"  {i}. {error} ({count} occurrences)\n")
                
                if error in stats['all_error_lines']['head']:
                    all_lines = stats['all_error_lines']['head'][error]
                    report_lines.append(f"     All lines: {', '.join(map(str, sorted(all_lines)))}\n")
        
        report_lines.append(f"Analyzed scenarios: {', '.join(sorted(stats['scenarios_analyzed']))}\n")
        report_lines.append("\n")
    
    return ''.join(report_lines)

def main():
    parser = argparse.ArgumentParser(description='Enhanced CoNLL-U Error Analysis with Accurate Line Numbers')
    parser.add_argument('gold_file', help='Path to gold standard CoNLL-U file')
    parser.add_argument('system_files', nargs='+', help='Paths to system output CoNLL-U files')
    parser.add_argument('--output-dir', '-o', help='Directory for output files (optional)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Generate only summary reports (most common errors with examples)')
    parser.add_argument('--complete-only', action='store_true',
                        help='Generate only complete reports (all errors with all occurrences)')
    parser.add_argument('--both', action='store_true',
                        help='Generate both summary and complete reports (default)')
    parser.add_argument('--lang', '-l', help='Language code for simplified command usage')
    parser.add_argument('--base-dir', '-b', default='/Users/marlene/Downloads',
                        help='Base directory for simplified command usage')
    
    args = parser.parse_args()
    
    if args.lang:
        lang = args.lang
        base_dir = args.base_dir
        
        gold_file = os.path.join(base_dir, 'gold_files', f'gold_file_{lang}.conllu')
        system_pattern = os.path.join(base_dir, 'output', lang, f'{lang}_*.conllu')
        output_dir = args.output_dir or os.path.join(base_dir, 'error_analysis_new', lang)
        
        system_files = glob.glob(system_pattern)
        
        print(f"Language: {lang}")
        print(f"Gold file: {gold_file}")
        print(f"System files pattern: {system_pattern}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(system_files)} system files")
        
        if not os.path.exists(gold_file):
            print(f"Error: Gold file not found: {gold_file}")
            return
        
        if not system_files:
            print(f"Error: No system files found matching: {system_pattern}")
            return
    else:
        gold_file = args.gold_file
        system_files = []
        output_dir = args.output_dir
        
        for pattern in args.system_files:
            expanded = glob.glob(pattern)
            if expanded:
                system_files.extend(expanded)
            else:
                system_files.append(pattern)
    
    if not (args.summary_only or args.complete_only):
        args.both = True
    
    all_system_files = []
    for pattern in system_files:
        expanded = glob.glob(pattern)
        if expanded:
            all_system_files.extend(expanded)
        else:
            all_system_files.append(pattern)
    
    valid_system_files = []
    for file_path in all_system_files:
        if os.path.isfile(file_path) and file_path.endswith('.conllu'):
            valid_system_files.append(file_path)
        else:
            print(f"Warning: Skipping '{file_path}' - not a valid CoNLL-U file.")
    
    if not valid_system_files:
        print("Error: No valid CoNLL-U system files found.")
        return
    
    print(f"Found {len(valid_system_files)} valid system files:")
    for f in valid_system_files:
        print(f"  - {os.path.basename(f)}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        if len(valid_system_files) > 1:
            print("\nPerforming language pair analysis...")
            language_pair_report = calculate_average_errors_by_language_pair(gold_file, valid_system_files)
            
            language_pair_output_file = os.path.join(output_dir, "average_errors_by_language_pair.txt")
            
            with open(language_pair_output_file, 'w', encoding='utf-8') as f:
                f.write(language_pair_report)
            print(f"Language pair analysis saved to {language_pair_output_file}.")
    
    for system_file in valid_system_files:
        print(f"\nAnalyzing {os.path.basename(system_file)}...")
        
        analyzer = EnhancedErrorAnalyzer(gold_file, system_file)
        
        if args.summary_only or args.both:
            summary_output_file = None
            if output_dir:
                base_name = os.path.basename(system_file).replace('.conllu', '')
                summary_output_file = os.path.join(output_dir, f"summary_{base_name}.txt")
            
            print(f"  Generating summary report...")
            analyzer.generate_summary_report(summary_output_file)
        
        if args.complete_only or args.both:
            complete_output_file = None
            if output_dir:
                base_name = os.path.basename(system_file).replace('.conllu', '')
                complete_output_file = os.path.join(output_dir, f"complete_{base_name}.txt")
            
            print(f"  Generating complete report...")
            analyzer.generate_complete_report(complete_output_file)
    
    print(f"\nAnalysis completed!")
    if output_dir:
        print(f"Reports saved to: {output_dir}")
        print("Summary reports: summary_*.txt (most common errors with examples)")
        print("Complete reports: complete_*.txt (all errors with all line occurrences)")

if __name__ == "__main__":
    main()
