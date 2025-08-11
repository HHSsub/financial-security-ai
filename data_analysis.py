"""
Data analysis module for Financial Security AI Model
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

def load_data(file_path):
    """Load the data from CSV file"""
    return pd.read_csv(file_path)

def analyze_question_types(df):
    """Analyze question types (objective vs subjective)"""
    # Function to determine if a question is multiple choice
    def is_multiple_choice(question):
        # Check for patterns like "1. option" or "1) option" or "1 option"
        return bool(re.search(r'\n\d[\.\)\s]', question))
    
    df['is_multiple_choice'] = df['Question'].apply(is_multiple_choice)
    
    return {
        'multiple_choice': df['is_multiple_choice'].sum(),
        'subjective': (~df['is_multiple_choice']).sum(),
        'total': len(df)
    }

def extract_num_options(question):
    """Extract the number of options in a multiple choice question"""
    if not question:
        return 0
        
    # Find all option patterns like "1. ", "2) ", "3 " etc.
    options = re.findall(r'\n\d[\.\)\s]', question)
    return len(options)

def analyze_question_length(df):
    """Analyze question length distribution"""
    df['question_length'] = df['Question'].apply(lambda x: len(x))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['question_length'], bins=20)
    plt.title('Question Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Count')
    plt.savefig('question_length_dist.png')
    
    return {
        'min_length': df['question_length'].min(),
        'max_length': df['question_length'].max(),
        'mean_length': df['question_length'].mean(),
        'median_length': df['question_length'].median()
    }

def analyze_multiple_choice_options(df):
    """Analyze the distribution of number of options in multiple choice questions"""
    # Filter multiple choice questions
    mc_df = df[df['is_multiple_choice']]
    
    # Extract number of options for each question
    mc_df['num_options'] = mc_df['Question'].apply(extract_num_options)
    
    # Count the frequency of each option count
    option_counts = Counter(mc_df['num_options'])
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(option_counts.keys()), y=list(option_counts.values()))
    plt.title('Number of Options in Multiple Choice Questions')
    plt.xlabel('Number of Options')
    plt.ylabel('Count')
    plt.savefig('mc_options_dist.png')
    
    return dict(option_counts)

def analyze_data(data_path, output_dir="analysis_results"):
    """Main function to perform all analyses"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Basic info
    print(f"Total number of questions: {len(df)}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Question type analysis
    question_types = analyze_question_types(df)
    print(f"\nQuestion type distribution:")
    print(f"- Multiple choice: {question_types['multiple_choice']}")
    print(f"- Subjective: {question_types['subjective']}")
    
    # Question length analysis
    length_stats = analyze_question_length(df)
    print(f"\nQuestion length statistics:")
    print(f"- Min length: {length_stats['min_length']:.2f}")
    print(f"- Max length: {length_stats['max_length']:.2f}")
    print(f"- Mean length: {length_stats['mean_length']:.2f}")
    print(f"- Median length: {length_stats['median_length']:.2f}")
    
    # Multiple choice options analysis
    if question_types['multiple_choice'] > 0:
        option_dist = analyze_multiple_choice_options(df)
        print(f"\nMultiple choice options distribution:")
        for num_options, count in sorted(option_dist.items()):
            print(f"- {num_options} options: {count} questions")
    
    return {
        "question_types": question_types,
        "length_stats": length_stats,
        "option_dist": option_dist if question_types['multiple_choice'] > 0 else {}
    }

if __name__ == "__main__":
    test_path = "/workspace/uploads/test.csv"
    analyze_data(test_path)