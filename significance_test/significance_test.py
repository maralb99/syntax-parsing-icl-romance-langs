import numpy as np
import pandas as pd
import os
from itertools import combinations
from scipy.stats import norm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Focused Statistical Significance Testing for NLP Parsing')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                        help='Directory to save output files (default: ./output)')
    return parser.parse_args()

def load_complete_data():
    language_data = {
        "fr_fr": {
            "0-Shot": {"UPOS": 95.73, "UAS": 48.91, "LAS": 43.13},
            "1-Shot-short": {"UPOS": 94.22, "UAS": 47.07, "LAS": 41.46},
            "1-Shot": {"UPOS": 93.80, "UAS": 46.57, "LAS": 41.79},
            "1-Shot-long": {"UPOS": 95.90, "UAS": 48.32, "LAS": 43.80},
            "3-Shot": {"UPOS": 95.31, "UAS": 48.16, "LAS": 42.96},
            "5-Shot": {"UPOS": 95.56, "UAS": 48.58, "LAS": 42.88},
            "10-Shot": {"UPOS": 95.81, "UAS": 49.92, "LAS": 44.05}
        },
        "fr_es": {
            "0-Shot": {"UPOS": 95.73, "UAS": 48.91, "LAS": 43.13},
            "1-Shot-short": {"UPOS": 93.63, "UAS": 45.90, "LAS": 41.29},
            "1-Shot": {"UPOS": 93.47, "UAS": 47.24, "LAS": 42.96},
            "1-Shot-long": {"UPOS": 94.22, "UAS": 48.24, "LAS": 43.30},
            "3-Shot": {"UPOS": 95.31, "UAS": 48.66, "LAS": 43.38},
            "5-Shot": {"UPOS": 94.64, "UAS": 47.57, "LAS": 42.63},
            "10-Shot": {"UPOS": 94.39, "UAS": 47.82, "LAS": 42.80}
        },
        "fr_oc": {
            "0-Shot": {"UPOS": 95.73, "UAS": 48.91, "LAS": 43.13},
            "1-Shot-short": {"UPOS": 94.30, "UAS": 45.64, "LAS": 41.54},
            "1-Shot-standard": {"UPOS": 95.23, "UAS": 49.16, "LAS": 43.89},
            "1-Shot-long": {"UPOS": 93.63, "UAS": 47.40, "LAS": 42.71},
            "3-Shot": {"UPOS": 95.14, "UAS": 49.08, "LAS": 44.22},
            "5-Shot": {"UPOS": 95.14, "UAS": 48.16, "LAS": 43.72},
            "10-Shot": {"UPOS": 94.97, "UAS": 50.25, "LAS": 45.14}
        },
        "fr_ca": {
            "0-Shot": {"UPOS": 95.73, "UAS": 48.91, "LAS": 43.13},
            "1-Shot-short": {"UPOS": 95.48, "UAS": 47.65, "LAS": 41.12},
            "1-Shot": {"UPOS": 94.89, "UAS": 49.75, "LAS": 44.14},
            "1-Shot-long": {"UPOS": 95.39, "UAS": 49.58, "LAS": 44.81},
            "3-Shot": {"UPOS": 95.31, "UAS": 49.58, "LAS": 44.39},
            "5-Shot": {"UPOS": 94.97, "UAS": 49.33, "LAS": 44.30},
            "10-Shot": {"UPOS": 95.56, "UAS": 48.74, "LAS": 43.38}
        },
        "fr_mixed": {
            "0-Shot": {"UPOS": 95.73, "UAS": 48.91, "LAS": 43.13},
            "3-Shot": {"UPOS": 95.39, "UAS": 47.82, "LAS": 43.05},
            "5-Shot": {"UPOS": 95.14, "UAS": 48.83, "LAS": 43.47},
            "10-Shot": {"UPOS": 95.23, "UAS": 50.17, "LAS": 44.81}
        }, 
        "es_fr": {
            "0-Shot": {"UPOS": 92.56, "UAS": 43.73, "LAS": 36.29},
            "1-Shot-short": {"UPOS": 93.10, "UAS": 43.37, "LAS": 37.28},
            "1-Shot": {"UPOS": 93.73, "UAS": 46.42, "LAS": 39.96},
            "1-Shot-long": {"UPOS": 93.19, "UAS": 44.09, "LAS": 39.25},
            "3-Shot": {"UPOS": 93.28, "UAS": 44.98, "LAS": 38.62},
            "5-Shot": {"UPOS": 93.55, "UAS": 45.79, "LAS": 40.05},
            "10-Shot": {"UPOS": 93.19, "UAS": 44.89, "LAS": 38.71}
        },
        "es_es": {
            "0-Shot": {"UPOS": 92.56, "UAS": 43.73, "LAS": 36.29},
            "1-Shot-short": {"UPOS": 91.58, "UAS": 44.27, "LAS": 38.62},
            "1-Shot": {"UPOS": 93.55, "UAS": 44.53, "LAS": 38.53},
            "1-Shot-long": {"UPOS": 93.46, "UAS": 45.79, "LAS": 39.70},
            "3-Shot": {"UPOS": 93.91, "UAS": 46.51, "LAS": 39.52},
            "5-Shot": {"UPOS": 93.91, "UAS": 46.06, "LAS": 39.87},
            "10-Shot": {"UPOS": 93.55, "UAS": 45.25, "LAS": 39.25}
        },
        "es_ca": {
            "0-Shot": {"UPOS": 92.56, "UAS": 43.73, "LAS": 36.29},
            "1-Shot-short": {"UPOS": 93.82, "UAS": 43.82, "LAS": 37.01},
            "1-Shot": {"UPOS": 93.10, "UAS": 47.13, "LAS": 40.32},
            "1-Shot-long": {"UPOS": 94.27, "UAS": 45.70, "LAS": 40.50},
            "3-Shot": {"UPOS": 93.10, "UAS": 45.16, "LAS": 38.53},
            "5-Shot": {"UPOS": 93.01, "UAS": 44.89, "LAS": 38.44},
            "10-Shot": {"UPOS": 93.55, "UAS": 45.61, "LAS": 39.16}
        },
        "es_mixed": {
            "0-Shot": {"UPOS": 92.56, "UAS": 43.73, "LAS": 36.29},
            "3-Shot": {"UPOS": 94.35, "UAS": 46.15, "LAS": 39.78},
            "5-Shot": {"UPOS": 94.09, "UAS": 44.98, "LAS": 39.43},
            "10-Shot": {"UPOS": 93.55, "UAS": 46.06, "LAS": 39.43}
        },
        "pt_fr": {
            "0-Shot": {"UPOS": 91.52, "UAS": 49.03, "LAS": 41.44},
            "1-Shot-short": {"UPOS": 91.60, "UAS": 46.20, "LAS": 41.20},
            "1-Shot": {"UPOS": 92.08, "UAS": 47.66, "LAS": 41.60},
            "1-Shot-long": {"UPOS": 92.08, "UAS": 47.58, "LAS": 42.25},
            "3-Shot": {"UPOS": 92.25, "UAS": 49.03, "LAS": 43.46},
            "5-Shot": {"UPOS": 92.16, "UAS": 47.66, "LAS": 42.33},
            "10-Shot": {"UPOS": 91.60, "UAS": 46.20, "LAS": 40.95}
        },
        "pt_pt": {
            "0-Shot": {"UPOS": 91.52, "UAS": 49.03, "LAS": 41.44},
            "1-Shot-short": {"UPOS": 92.08, "UAS": 47.01, "LAS": 41.28},
            "1-Shot": {"UPOS": 92.00, "UAS": 48.47, "LAS": 43.54},
            "1-Shot-long": {"UPOS": 93.13, "UAS": 47.98, "LAS": 42.57},
            "3-Shot": {"UPOS": 92.89, "UAS": 48.87, "LAS": 43.94},
            "5-Shot": {"UPOS": 92.33, "UAS": 47.74, "LAS": 42.97},
            "10-Shot": {"UPOS": 93.78, "UAS": 49.11, "LAS": 44.02}
        },
        "pt_es": {
            "0-Shot": {"UPOS": 91.52, "UAS": 49.03, "LAS": 41.44},
            "1-Shot-short": {"UPOS": 91.28, "UAS": 44.67, "LAS": 40.23},
            "1-Shot": {"UPOS": 91.44, "UAS": 46.12, "LAS": 41.76},
            "1-Shot-long": {"UPOS": 92.08, "UAS": 46.93, "LAS": 42.65},
            "3-Shot": {"UPOS": 91.92, "UAS": 48.30, "LAS": 43.21},
            "5-Shot": {"UPOS": 91.11, "UAS": 46.04, "LAS": 42.00},
            "10-Shot": {"UPOS": 91.28, "UAS": 46.28, "LAS": 41.36}
        },
        "pt_gl": {
            "0-Shot": {"UPOS": 91.52, "UAS": 49.03, "LAS": 41.44},
            "1-Shot-short": {"UPOS": 92.08, "UAS": 47.74, "LAS": 42.00},
            "1-Shot": {"UPOS": 92.25, "UAS": 47.25, "LAS": 42.08},
            "1-Shot-long": {"UPOS": 92.00, "UAS": 49.27, "LAS": 43.54},
            "3-Shot": {"UPOS": 91.76, "UAS": 47.66, "LAS": 43.54},
            "5-Shot": {"UPOS": 92.33, "UAS": 47.33, "LAS": 42.73},
            "10-Shot": {"UPOS": 92.41, "UAS": 46.45, "LAS": 41.68}
        },
        "pt_mixed": {
            "0-Shot": {"UPOS": 91.52, "UAS": 49.03, "LAS": 41.44},
            "3-Shot": {"UPOS": 92.00, "UAS": 48.71, "LAS": 43.70},
            "5-Shot": {"UPOS": 91.92, "UAS": 47.98, "LAS": 43.38},
            "10-Shot": {"UPOS": 92.89, "UAS": 46.61, "LAS": 42.25}
        },
        "eu_fr": {
            "0-Shot": {"UPOS": 76.29, "UAS": 35.05, "LAS": 15.46},
            "1-Shot-short": {"UPOS": 75.60, "UAS": 35.57, "LAS": 17.35},
            "1-Shot": {"UPOS": 73.88, "UAS": 34.36, "LAS": 15.81},
            "1-Shot-long": {"UPOS": 76.63, "UAS": 38.83, "LAS": 17.70},
            "3-Shot": {"UPOS": 75.77, "UAS": 37.46, "LAS": 16.84},
            "5-Shot": {"UPOS": 74.23, "UAS": 37.11, "LAS": 17.35},
            "10-Shot": {"UPOS": 77.49, "UAS": 39.00, "LAS": 18.21}
        },
        "eu_eu": {
            "0-Shot": {"UPOS": 76.29, "UAS": 35.05, "LAS": 15.46},
            "1-Shot-short": {"UPOS": 76.98, "UAS": 39.00, "LAS": 17.70},
            "1-Shot": {"UPOS": 78.35, "UAS": 36.94, "LAS": 18.73},
            "1-Shot-long": {"UPOS": 84.36, "UAS": 43.99, "LAS": 21.13},
            "3-Shot": {"UPOS": 82.99, "UAS": 41.58, "LAS": 19.76},
            "5-Shot": {"UPOS": 84.36, "UAS": 41.24, "LAS": 21.31},
            "10-Shot": {"UPOS": 85.57, "UAS": 41.75, "LAS": 21.31}
        },
        "eu_es": {
            "0-Shot": {"UPOS": 76.29, "UAS": 35.05, "LAS": 15.46},
            "1-Shot-short": {"UPOS": 72.16, "UAS": 34.36, "LAS": 15.81},
            "1-Shot": {"UPOS": 74.57, "UAS": 39.00, "LAS": 18.38},
            "1-Shot-long": {"UPOS": 76.29, "UAS": 39.00, "LAS": 19.07},
            "3-Shot": {"UPOS": 78.35, "UAS": 37.80, "LAS": 18.21},
            "5-Shot": {"UPOS": 78.52, "UAS": 39.35, "LAS": 18.73},
            "10-Shot": {"UPOS": 80.07, "UAS": 37.80, "LAS": 17.70}
        },
        "eu_oc": {
            "0-Shot": {"UPOS": 76.29, "UAS": 35.05, "LAS": 15.46},
            "1-Shot (short)": {"UPOS": 76.29, "UAS": 35.57, "LAS": 17.01},
            "1-Shot": {"UPOS": 77.15, "UAS": 40.38, "LAS": 19.24},
            "1-Shot (long)": {"UPOS": 75.95, "UAS": 35.91, "LAS": 16.49},
            "3-Shot": {"UPOS": 77.15, "UAS": 38.49, "LAS": 20.96},
            "5-Shot": {"UPOS": 75.95, "UAS": 37.80, "LAS": 19.24},
            "10-Shot": {"UPOS": 73.37, "UAS": 35.74, "LAS": 17.53}
    },
        "eu_mixed": {
            "0-Shot": {"UPOS": 76.29, "UAS": 35.05, "LAS": 15.46},
            "3-Shot": {"UPOS": 76.63, "UAS": 37.46, "LAS": 18.38},
            "5-Shot": {"UPOS": 78.52, "UAS": 37.63, "LAS": 17.70},
            "10-Shot" : {"UPOS": 77.84, "UAS": 38.14, "LAS": 18.73}
        },
        "ca_ca": {
            "0-Shot": {"UPOS": 92.36, "UAS": 44.34, "LAS": 36.31},
            "1-Shot-short": {"UPOS": 92.64, "UAS": 44.23, "LAS": 36.14},
            "1-Shot": {"UPOS": 93.70, "UAS": 45.01, "LAS": 37.87},
            "1-Shot-long": {"UPOS": 93.70, "UAS": 46.12, "LAS": 39.43},
            "3-Shot": {"UPOS": 94.70, "UAS": 45.34, "LAS": 38.82},
            "5-Shot": {"UPOS": 94.53, "UAS": 45.34, "LAS": 38.93},
            "10-Shot": {"UPOS": 95.59, "UAS": 45.51, "LAS": 39.04}
        },
        "ca_es": {
            "0-Shot": {"UPOS": 92.36, "UAS": 44.34, "LAS": 36.31},
            "1-Shot-short": {"UPOS": 92.08, "UAS": 43.73, "LAS": 37.03},
            "1-Shot": {"UPOS": 91.19, "UAS": 43.39, "LAS": 37.98},
            "1-Shot-long": {"UPOS": 92.69, "UAS": 44.73, "LAS": 38.43},
            "3-Shot": {"UPOS": 92.47, "UAS": 44.73, "LAS": 38.65},
            "5-Shot": {"UPOS": 92.69, "UAS": 45.06, "LAS": 39.04},
            "10-Shot": {"UPOS": 92.58, "UAS": 43.39, "LAS": 37.48}
        },
        "ca_oc": {
            "0-Shot": {"UPOS": 92.36, "UAS": 44.34, "LAS": 36.31},
            "1-Shot-short": {"UPOS": 92.19, "UAS": 44.56, "LAS": 37.09},
            "1-Shot": {"UPOS": 93.03, "UAS": 46.63, "LAS": 38.82},
            "1-Shot-long": {"UPOS": 92.53, "UAS": 44.56, "LAS": 38.04},
            "3-Shot": {"UPOS": 92.02, "UAS": 44.95, "LAS": 38.43},
            "5-Shot": {"UPOS": 91.52, "UAS": 44.79, "LAS": 38.43},
            "10-Shot": {"UPOS": 92.25, "UAS": 45.45, "LAS": 39.10}
        },
        "ca_mixed": {
            "0-Shot": {"UPOS": 92.36, "UAS": 44.34, "LAS": 36.31},
            "3-Shot": {"UPOS": 91.91, "UAS": 45.06, "LAS": 38.32},
            "5-Shot": {"UPOS": 92.69, "UAS": 44.95, "LAS": 38.43},
            "10-Shot": {"UPOS": 92.81, "UAS": 46.18, "LAS": 39.60}
        },
        "oc_fr": {
            "0-Shot": {"UPOS": 85.47, "UAS": 44.48, "LAS": 33.67},
            "1-Shot-short": {"UPOS": 87.61, "UAS": 46.28, "LAS": 35.25},
            "1-Shot": {"UPOS": 87.61, "UAS": 46.28, "LAS": 35.25},
            "1-Shot-long": {"UPOS": 89.64, "UAS": 46.40, "LAS": 37.05},
            "3-Shot": {"UPOS": 87.61, "UAS": 46.62, "LAS": 36.60},
            "5-Shot": {"UPOS": 88.06, "UAS": 46.85, "LAS": 37.73},
            "10-Shot": {"UPOS": 88.74, "UAS": 46.96, "LAS": 36.82}
        },
        "oc_oc": {
            "0-Shot": {"UPOS": 85.47, "UAS": 44.48, "LAS": 33.67},
            "1-Shot-short": {"UPOS": 88.29, "UAS": 45.38, "LAS": 36.49},
            "1-Shot": {"UPOS": 89.08, "UAS": 46.51, "LAS": 36.71},
            "1-Shot-long": {"UPOS": 89.64, "UAS": 46.40, "LAS": 37.05},
            "3-Shot": {"UPOS": 88.63, "UAS": 44.93, "LAS": 35.36},
            "5-Shot": {"UPOS": 88.85, "UAS": 45.38, "LAS": 36.04},
            "10-Shot": {"UPOS": 89.08, "UAS": 47.07, "LAS": 38.18}
        },
        "oc_ca": {
            "0-Shot": {"UPOS": 85.47, "UAS": 44.48, "LAS": 33.67},
            "1-Shot-short": {"UPOS": 87.61, "UAS": 44.93, "LAS": 34.23},
            "1-Shot": {"UPOS": 88.40, "UAS": 46.28, "LAS": 36.37},
            "1-Shot-long": {"UPOS": 88.18, "UAS": 46.17, "LAS": 36.15},
            "3-Shot": {"UPOS": 87.84, "UAS": 45.72, "LAS": 35.81},
            "5-Shot": {"UPOS": 87.84, "UAS": 45.83, "LAS": 35.47},
            "10-Shot": {"UPOS": 87.50, "UAS": 44.26, "LAS": 34.12}
        },
        "oc_mixed": {
            "0-Shot": {"UPOS": 85.47, "UAS": 44.48, "LAS": 33.67},
            "3-Shot": {"UPOS": 86.26, "UAS": 45.72, "LAS": 35.47},
            "5-Shot": {"UPOS": 88.06, "UAS": 45.83, "LAS": 36.37},
            "10-Shot": {"UPOS": 88.18, "UAS": 47.75, "LAS": 37.50}
        },
        "gl_gl": {
            "0-Shot": {"UPOS": 91.22, "UAS": 44.79, "LAS": 38.55},
            "1-Shot-short": {"UPOS": 91.16, "UAS": 43.95, "LAS": 37.82},
            "1-Shot": {"UPOS": 91.95, "UAS": 44.57, "LAS": 39.67},
            "1-Shot-long": {"UPOS": 92.07, "UAS": 46.31, "LAS": 41.31},
            "3-Shot": {"UPOS": 92.40, "UAS": 45.92, "LAS": 41.70},
            "5-Shot": {"UPOS": 92.35, "UAS": 45.69, "LAS": 41.31},
            "10-Shot": {"UPOS": 93.70, "UAS": 44.46, "LAS": 40.29}
        },
        "gl_es": {
            "0-Shot": {"UPOS": 91.22, "UAS": 44.79, "LAS": 38.55},
            "1-Shot-short": {"UPOS": 90.77, "UAS": 44.18, "LAS": 38.83},
            "1-Shot": {"UPOS": 90.77, "UAS": 44.74, "LAS": 39.05},
            "1-Shot-long": {"UPOS": 91.28, "UAS": 45.69, "LAS": 40.12},
            "3-Shot": {"UPOS": 90.71, "UAS": 45.41, "LAS": 40.41},
            "5-Shot": {"UPOS": 90.83, "UAS": 44.85, "LAS": 39.62},
            "10-Shot": {"UPOS": 90.26, "UAS": 43.84, "LAS": 39.00}
        },
        "gl_pt": {
            "0-Shot": {"UPOS": 91.22, "UAS": 44.79, "LAS": 38.55},
            "1-Shot-short": {"UPOS": 91.05, "UAS": 45.58, "LAS": 38.89},
            "1-Shot": {"UPOS": 91.90, "UAS": 46.15, "LAS": 40.52},
            "1-Shot-long": {"UPOS": 92.01, "UAS": 45.86, "LAS": 39.90},
            "3-Shot": {"UPOS": 92.12, "UAS": 46.48, "LAS": 41.19},
            "5-Shot": {"UPOS": 91.95, "UAS": 46.43, "LAS": 41.59},
            "10-Shot": {"UPOS": 92.40, "UAS": 46.15, "LAS": 40.97}
        },
        "gl_mixed": {
            "0-Shot": {"UPOS": 91.22, "UAS": 44.79, "LAS": 38.55},
            "3-Shot": {"UPOS": 91.90, "UAS": 45.81, "LAS": 40.29},
            "5-Shot": {"UPOS": 92.07, "UAS": 46.54, "LAS": 41.19},
            "10-Shot": {"UPOS": 92.12, "UAS": 45.86, "LAS": 41.02}
        }
    }
    return language_data

def get_token_count(language_pair):
    target_language = language_pair.split('_')[0]
    token_counts = {
        'pt': 1238, 'fr': 1194, 'es': 1116, 'eu': 582,
        'ca': 1793, 'oc': 824, 'gl': 1777
    }
    return token_counts.get(target_language, 1000)

def z_test_proportions(p1, p2, n1, n2):
    p1 = p1 / 100 if p1 > 1 else p1
    p2 = p2 / 100 if p2 > 1 else p2
    pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    if se == 0:
        return 0, 1.0
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value

def calculate_cohens_h(p1, p2):
    p1 = p1 / 100 if p1 > 1 else p1
    p2 = p2 / 100 if p2 > 1 else p2
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return h

def interpret_cohens_h(h):
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negligible"
    elif h_abs < 0.5:
        return "small"
    elif h_abs < 0.8:
        return "medium"
    else:
        return "large"

def analyze_shot_settings(language_data):
    metrics = ["UPOS", "UAS", "LAS"]
    results = []
    
    for language_pair in language_data:
        n_tokens = get_token_count(language_pair)
        target_lang = language_pair.split('_')[0]
        source_lang = language_pair.split('_')[1]
        
        for metric in metrics:
            shot_settings = list(language_data[language_pair].keys())
            
            for setting1, setting2 in combinations(shot_settings, 2):
                score1 = language_data[language_pair][setting1][metric]
                score2 = language_data[language_pair][setting2][metric]
                
                z_stat, p_value = z_test_proportions(score1/100, score2/100, n_tokens, n_tokens)
                cohens_h = calculate_cohens_h(score1, score2)
                
                results.append({
                    'comparison_type': 'shot_settings',
                    'language_pair': language_pair,
                    'target_language': target_lang,
                    'source_language': source_lang,
                    'metric': metric,
                    'setting1': setting1,
                    'setting2': setting2,
                    'score1': score1,
                    'score2': score2,
                    'diff': score2 - score1,
                    'n_tokens': n_tokens,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'cohens_h': cohens_h,
                    'effect_size': interpret_cohens_h(cohens_h),
                    'significant': p_value < 0.05
                })
    
    return results

def analyze_source_language_effects(language_data):
    metrics = ["UPOS", "UAS", "LAS"]
    results = []
    
    target_languages = set([pair.split('_')[0] for pair in language_data.keys()])
    
    for target_lang in target_languages:
        pairs_for_target = [pair for pair in language_data.keys() if pair.startswith(target_lang + '_')]
        
        if len(pairs_for_target) > 1:
            common_settings = set(language_data[pairs_for_target[0]].keys())
            for pair in pairs_for_target[1:]:
                common_settings = common_settings.intersection(set(language_data[pair].keys()))
            
            for shot_setting in common_settings:
                for metric in metrics:
                    for pair1, pair2 in combinations(pairs_for_target, 2):
                        score1 = language_data[pair1][shot_setting][metric]
                        score2 = language_data[pair2][shot_setting][metric]
                        
                        n_tokens = get_token_count(pair1)
                        
                        z_stat, p_value = z_test_proportions(score1/100, score2/100, n_tokens, n_tokens)
                        cohens_h = calculate_cohens_h(score1, score2)
                        
                        results.append({
                            'comparison_type': 'source_language',
                            'target_language': target_lang,
                            'source_lang1': pair1.split('_')[1],
                            'source_lang2': pair2.split('_')[1],
                            'shot_setting': shot_setting,
                            'metric': metric,
                            'score1': score1,
                            'score2': score2,
                            'diff': score2 - score1,
                            'n_tokens': n_tokens,
                            'z_statistic': z_stat,
                            'p_value': p_value,
                            'cohens_h': cohens_h,
                            'effect_size': interpret_cohens_h(cohens_h),
                            'significant': p_value < 0.05
                        })
    
    return results

def main():
    args = parse_arguments()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output files will be saved to: {os.path.abspath(output_dir)}")
    print("Loading data...")
    language_data = load_complete_data()
    
    print("Analyzing shot settings...")
    shot_results = analyze_shot_settings(language_data)
    
    print("Analyzing source language effects...")
    source_results = analyze_source_language_effects(language_data)
    
    shot_df = pd.DataFrame(shot_results)
    source_df = pd.DataFrame(source_results)
    
    shot_df = shot_df.sort_values(by='p_value')
    source_df = source_df.sort_values(by='p_value')
    
    sig_shot_df = shot_df[shot_df['significant']]
    sig_source_df = source_df[source_df['significant']]
    
    shot_df.to_csv(os.path.join(output_dir, 'shot_settings_significance.csv'), index=False)
    source_df.to_csv(os.path.join(output_dir, 'source_language_significance.csv'), index=False)
    
    sig_shot_df.to_csv(os.path.join(output_dir, 'significant_shot_settings.csv'), index=False)
    sig_source_df.to_csv(os.path.join(output_dir, 'significant_source_language.csv'), index=False)
    
    print("\nSummary of Results:")
    print(f"- Shot Settings: {len(sig_shot_df)} significant differences out of {len(shot_df)} comparisons")
    print(f"- Source Language Effects: {len(sig_source_df)} significant differences out of {len(source_df)} comparisons")
    
    print("\nTop 5 Most Significant Shot Setting Comparisons:")
    if not sig_shot_df.empty:
        top_shot = sig_shot_df.head(5)
        for _, row in top_shot.iterrows():
            print(f"  {row['language_pair']} - {row['metric']}: {row['setting1']} vs {row['setting2']}: "
                  f"diff={row['diff']:.2f}, p={row['p_value']:.6f}, effect={row['effect_size']}")
    
    print("\nTop 5 Most Significant Source Language Effects:")
    if not sig_source_df.empty:
        top_source = sig_source_df.head(5)
        for _, row in top_source.iterrows():
            print(f"  {row['target_language']} - {row['metric']} ({row['shot_setting']}): "
                  f"{row['source_lang1']} vs {row['source_lang2']}: "
                  f"diff={row['diff']:.2f}, p={row['p_value']:.6f}, effect={row['effect_size']}")
    
    print("\nResults saved to CSV files in the output directory:")
    print("- All shot setting comparisons: shot_settings_significance.csv")
    print("- All source language effects: source_language_significance.csv")
    print("- Significant shot setting comparisons: significant_shot_settings.csv")
    print("- Significant source language effects: significant_source_language.csv")

if __name__ == "__main__":
    main()
