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

class MinimalErrorAnalyzer:
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
        
        self.error_stats = {
            'upos': Counter(),
            'head': Counter(),
            'deprel': Counter(),
            'error_lines': defaultdict(list),
            'token_form_errors': Counter(),
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
            for i, (g_token, s_token) in enumerate(zip(gold_sent, system_sent)):
                token_line = self.gold_line_map.get(sent_id, {}).get(i, 0)
                if token_line == 0:
                    token_line = f"unknown (sent_id={sent_id}, token_idx={i})"
                
                self.error_stats['total_tokens'] += 1
                
                token_form = g_token['form'].lower()
                
                g_upos = g_token.get('upos', '_')
                s_upos = s_token.get('upos', '_')
                
                if g_upos != s_upos:
                    self.error_stats['total_errors']['upos'] += 1
                    error_key = f"{g_upos} → {s_upos}"
                    self.error_stats['upos'][error_key] += 1
                    self.error_stats['token_form_errors'][token_form] += 1
                    self.error_stats['error_lines'][error_key].append(token_line)
                
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
                    
                    g_head_form = "ROOT" if g_head == 0 else gold_sent[g_head-1]['form']
                    s_head_form = "ROOT" if s_head == 0 else system_sent[s_head-1]['form']
                    
                    error_key = f"{g_head} ({g_head_form}) → {s_head} ({s_head_form})"
                    self.error_stats['head'][error_key] += 1
                    self.error_stats['error_lines'][error_key].append(token_line)
                
                g_deprel = g_token.get('deprel', '_')
                s_deprel = s_token.get('deprel', '_')
                
                if g_deprel != s_deprel:
                    self.error_stats['total_errors']['deprel'] += 1
                    error_key = f"{g_deprel} → {s_deprel}"
                    self.error_stats['deprel'][error_key] += 1
                    self.error_stats['error_lines'][error_key].append(token_line)
        
        return self.error_stats

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
                analyzer = MinimalErrorAnalyzer(gold_file, system_file)
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
    parser = argparse.ArgumentParser(description='Minimal CoNLL-U Error Analysis - Language Pair Analysis Only')
    parser.add_argument('gold_file', help='Path to gold standard CoNLL-U file')
    parser.add_argument('system_files', nargs='+', help='Paths to system output CoNLL-U files')
    parser.add_argument('--output-dir', '-o', help='Directory for output files (optional)')
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
    
    if len(valid_system_files) > 1:
        print("\nPerforming language pair analysis...")
        language_pair_report = calculate_average_errors_by_language_pair(gold_file, valid_system_files)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            language_pair_output_file = os.path.join(output_dir, "average_errors_by_language_pair.txt")
            
            with open(language_pair_output_file, 'w', encoding='utf-8') as f:
                f.write(language_pair_report)
            print(f"Language pair analysis saved to {language_pair_output_file}.")
        else:
            print(language_pair_report)
    else:
        print("Error: Need multiple system files for language pair analysis.")
    
    print(f"\nAnalysis completed!")

if __name__ == "__main__":
    main()
