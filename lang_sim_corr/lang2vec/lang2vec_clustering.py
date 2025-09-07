import os
import numpy as np
from iso639 import Lang
import lang2vec.lang2vec as l2v
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

LANG_OPTIONS = ['fr', 'es', 'pt', 'ca', 'oc', 'gl', 'eu', 'it', 'ro']

LANGUAGE_NAMES = {
    'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese', 
    'ca': 'Catalan', 'oc': 'Occitan', 'gl': 'Galician', 'eu': 'Basque',
    'it': 'Italian', 'ro': 'Romanian'
}

def _lang_features(langs, sim_metric):
    if sim_metric in ['lexical']:
        lang3_codes = langs
    else:
        lang3_codes = [Lang(lang).pt3 if lang != 'iw' else Lang('he').pt3 for lang in langs]
    
    if sim_metric == 'syntax':
        v_dict = l2v.get_features(lang3_codes, "syntax_average", minimal=True)
    elif sim_metric == 'fam':  
        v_dict = l2v.get_features(lang3_codes, "fam", minimal=True)
    elif sim_metric == 'geo':
        v_dict = l2v.get_features(lang3_codes, "geo", minimal=True)
    
    print(f"Loaded {sim_metric} features")
    return v_dict

def calculate_similarity_matrix(langs, sim_metric):
    v_dict = _lang_features(langs, sim_metric)
    
    if sim_metric in ['lexical']:
        lang3_codes = langs
    else:
        lang3_codes = [Lang(lang).pt3 if lang != 'iw' else Lang('he').pt3 for lang in langs]
    
    n = len(langs)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0  
            else:
                lang_i = lang3_codes[i]
                lang_j = lang3_codes[j]
                
                try:
                    vec_i = np.array([float(v) if v != '--' else np.nan for v in v_dict[lang_i]])
                    vec_j = np.array([float(v) if v != '--' else np.nan for v in v_dict[lang_j]])
                    
                    mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
                    if np.sum(mask) == 0:
                        similarity = 0.0
                    else:
                        vec_i_masked = vec_i[mask]
                        vec_j_masked = vec_j[mask]
                        
                        norm_i = np.sqrt(np.sum(vec_i_masked ** 2))
                        norm_j = np.sqrt(np.sum(vec_j_masked ** 2))
                        
                        if norm_i == 0 or norm_j == 0:
                            similarity = 0.0
                        else:
                            dot_product = np.sum(vec_i_masked * vec_j_masked)
                            similarity = dot_product / (norm_i * norm_j)
                except Exception as e:
                    similarity = 0.0
                
                similarity_matrix[i][j] = similarity
    
    return similarity_matrix

def print_similarity_matrix(matrix, langs, feature_type):
    print(f"\n--- Similarity Matrix ({feature_type}) ---")
    print("Language".ljust(12), end="")
    for lang in langs:
        print(LANGUAGE_NAMES[lang][:9].ljust(10), end="")
    print()
    
    for i, lang1 in enumerate(langs):
        print(LANGUAGE_NAMES[lang1][:10].ljust(12), end="")
        for j, lang2 in enumerate(langs):
            print(f"{matrix[i][j]:.3f}".ljust(10), end="")
        print()

def print_most_similar_languages(matrix, langs, feature_type):
    print(f"\n--- Most similar languages ({feature_type}) ---")
    for i, lang in enumerate(langs):
        similarities = [(langs[j], matrix[i][j]) 
                       for j in range(len(langs)) if i != j]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{LANGUAGE_NAMES[lang]}: " + ", ".join(
            [f"{LANGUAGE_NAMES[sim_lang]} ({sim:.3f})" for sim_lang, sim in similarities[:3]]))

def main():
    feature_types = ['syntax', 'geo', 'fam']
    similarity_matrices = {}
    
    print("LANGUAGE SIMILARITY ANALYSIS")
    print("="*50)
    
    for feature_type in feature_types:
        try:
            print(f"\n=== {feature_type.upper()} FEATURES ===")
            similarity_matrices[feature_type] = calculate_similarity_matrix(LANG_OPTIONS, feature_type)
            print_similarity_matrix(similarity_matrices[feature_type], LANG_OPTIONS, feature_type)
            print_most_similar_languages(similarity_matrices[feature_type], LANG_OPTIONS, feature_type)
            
        except Exception as e:
            print(f"Error processing {feature_type}: {e}")
    
    if similarity_matrices:
        print(f"\n{'='*60}")
        print("AVERAGE RESULTS ACROSS ALL FEATURES")
        print(f"{'='*60}")
        
        avg_matrix = sum(similarity_matrices.values()) / len(similarity_matrices)
        print_similarity_matrix(avg_matrix, LANG_OPTIONS, "Average")
        print_most_similar_languages(avg_matrix, LANG_OPTIONS, "Average")
        
        print(f"\nSimilarity matrices calculated for: {', '.join(feature_types)}")
        print("Matrices are available in the 'similarity_matrices' dictionary")
        
        return similarity_matrices

if __name__ == "__main__":
    matrices = main()
