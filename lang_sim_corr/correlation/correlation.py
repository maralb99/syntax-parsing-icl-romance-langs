import pandas as pd
import sys
from scipy.stats import pearsonr, spearmanr
import numpy as np

def analyze_by_target_language_with_output(file_path, output_file=None):
    if output_file is None:
        output_file = "correlation.txt"
    
    def write_output(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + "\n")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            write_output("LANGUAGE SIMILARITY AND MODEL PERFORMANCE ANALYSIS", f)
            write_output("", f)
            
            df = pd.read_csv(file_path)
            if len(df.columns) == 1 and ';' in df.columns[0]:
                df = pd.read_csv(file_path, sep=';')
            
            write_output("ALL COLUMNS IN CSV FILE:", f)
            write_output("-" * 40, f)
            for i, col in enumerate(df.columns, 1):
                write_output(f"{i:2}. {col}", f)
            write_output("", f)
            
            write_output("FIRST 5 ROWS OF DATA:", f)
            write_output("-" * 40, f)
            write_output(df.head().to_string(), f)
            write_output("", f)
            
            source_lang_col = None
            target_lang_col = None
            similarity_col = None
            upos_col = None
            uas_col = None
            las_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'source' in col_lower and 'language' in col_lower:
                    source_lang_col = col
                elif 'target' in col_lower and 'language' in col_lower:
                    target_lang_col = col
                elif 'similarity' in col_lower or 'score' in col_lower:
                    similarity_col = col
                elif 'upos' in col_lower:
                    upos_col = col
                elif 'uas' in col_lower:
                    uas_col = col
                elif 'las' in col_lower:
                    las_col = col
            
            write_output("COLUMN IDENTIFICATION:", f)
            write_output("-" * 40, f)
            write_output(f"- Target Language column: {target_lang_col}", f)
            write_output(f"- Source Language column: {source_lang_col}", f)
            write_output(f"- Similarity column: {similarity_col}", f)
            write_output(f"- UPOS column: {upos_col}", f)
            write_output(f"- UAS column: {uas_col}", f)
            write_output(f"- LAS column: {las_col}", f)
            write_output("", f)
            
            if not source_lang_col or not target_lang_col or not similarity_col:
                write_output("ERROR: Could not find required columns", f)
                return
            
            write_output("VALUES IN LANGUAGE COLUMNS:", f)
            write_output("-" * 50, f)
            write_output(f"SOURCE languages: {sorted(df[source_lang_col].unique())}", f)
            write_output(f"TARGET languages: {sorted(df[target_lang_col].unique())}", f)
            write_output("", f)
            
            performance_cols = [col for col in [upos_col, uas_col, las_col] if col is not None]
            numeric_cols = [similarity_col] + performance_cols
            
            for col in numeric_cols:
                if col and col in df.columns:
                    try:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace(',', '.').str.replace('%', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        pass
            
            valid_cols = [source_lang_col, target_lang_col, similarity_col] + performance_cols
            df_clean = df[valid_cols].dropna()
            
            target_languages = sorted(df_clean[target_lang_col].unique())
            
            for target_lang in target_languages:
                write_output("="*80, f)
                write_output(f"ANALYSIS FOR TARGET LANGUAGE: {target_lang}", f)
                write_output(f"(best SOURCE language for {target_lang})", f)
                write_output("="*80, f)
                write_output("", f)
                
                target_data = df_clean[df_clean[target_lang_col] == target_lang]
                
                write_output("CORRELATION ANALYSIS:", f)
                write_output(f"Question: For {target_lang}, does similarity predict performance?", f)
                write_output("-" * 60, f)
                
                for perf_col in performance_cols:
                    if perf_col and perf_col in target_data.columns:
                        try:
                            if target_data[similarity_col].nunique() < 2 or target_data[perf_col].nunique() < 2:
                                continue
                                
                            pearson_corr, pearson_p = pearsonr(target_data[similarity_col], target_data[perf_col])
                            spearman_corr, spearman_p = spearmanr(target_data[similarity_col], target_data[perf_col])
                            
                            write_output(f"{perf_col}:", f)
                            write_output(f"  Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})", f)
                            write_output(f"  Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})", f)
                            write_output("", f)
                            
                        except Exception as e:
                            continue
                
                write_output("PERFORMANCE RANKING:", f)
                write_output(f"Which source language works best for {target_lang}?", f)
                write_output("-" * 50, f)
                
                source_langs_for_target = sorted(target_data[source_lang_col].unique())
                summary_data = []
                
                for source_lang in source_langs_for_target:
                    source_subset = target_data[target_data[source_lang_col] == source_lang]
                    
                    row_data = {
                        'Source': source_lang,
                        'N': len(source_subset),
                        'Similarity': source_subset[similarity_col].mean()
                    }
                    
                    for perf_col in performance_cols:
                        if perf_col in source_subset.columns:
                            row_data[perf_col] = source_subset[perf_col].mean()
                    
                    summary_data.append(row_data)
                
                summary_df = pd.DataFrame(summary_data)
                
                write_output("SUMMARY TABLE:", f)
                
                header_line = f"{'Source Lang':<12} {'N':<3} {'Similarity':<12}"
                for perf_col in performance_cols:
                    if perf_col:
                        header_line += f" {perf_col:<12}"
                
                write_output(header_line, f)
                write_output("-" * len(header_line), f)
                
                for _, row in summary_df.iterrows():
                    line = f"{row['Source']:<12} {row['N']:<3.0f} {row['Similarity']:<12.3f}"
                    for perf_col in performance_cols:
                        if perf_col and perf_col in row:
                            line += f" {row[perf_col]:<12.3f}"
                    write_output(line, f)
                
                write_output("", f)
                write_output(f"Best to worst source language for {target_lang}:", f)
                write_output("-" * 60, f)
                
                for perf_col in performance_cols:
                    if perf_col and perf_col in summary_df.columns:
                        write_output(f"By {perf_col} performance:", f)
                        perf_ranking = summary_df.sort_values(perf_col, ascending=False)
                        for i, (_, row) in enumerate(perf_ranking.iterrows(), 1):
                            sim_score = row['Similarity']
                            perf_score = row[perf_col]
                            write_output(f"  {i}. {row['Source']:<12} → {perf_col}: {perf_score:.3f} (similarity: {sim_score:.3f})", f)
                        write_output("", f)
                
                write_output("", f)
        
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_by_target_language_with_output(file_path, output_file)
    else:
        file_path = input("Path to CSV file: ")
        output_file = input("Enter output file name: ").strip()
        if not output_file:
            output_file = None
        analyze_by_target_language_with_output(file_path, output_file)
