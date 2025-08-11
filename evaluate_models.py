"""
Model evaluation module for Financial Security AI Model
"""

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import re

class ModelEvaluator:
    def __init__(self, models_to_evaluate=None):
        """
        Initialize the model evaluator
        
        Args:
            models_to_evaluate: List of model paths to evaluate
        """
        self.models_to_evaluate = models_to_evaluate or [
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/gemma-7b-it",
            "01-ai/Yi-6B-Chat"
        ]
        
        self.loaded_models = {}
        self.results = {}
    
    def load_model(self, model_name):
        """
        Load a model for evaluation
        
        Args:
            model_name: HuggingFace path to the model
            
        Returns:
            model, tokenizer
        """
        print(f"Loading model: {model_name}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return model, tokenizer
    
    def _is_multiple_choice(self, question):
        """Check if the question is multiple choice"""
        # Look for patterns like "1. ", "1) ", etc.
        return bool(re.search(r'\n\d[\.\)\s]', question))
    
    def _format_prompt(self, question):
        """Format prompt for question"""
        if self._is_multiple_choice(question):
            return f"""### 지시사항: 
금융보안 전문가로서, 아래 주어진 객관식 문제의 정답을 선택해주세요. 
1부터 5까지의 숫자로만 답해주세요. 
설명은 필요하지 않습니다.

### 입력:
{question}

### 출력:
"""
        else:
            return f"""### 지시사항: 
금융보안 전문가로서, 다음 금융보안 관련 주관식 문제에 답변해주세요.
간결하고 명확하게 답변해주세요.

### 입력:
{question}

### 출력:
"""

    def _extract_answer(self, response, prompt, question):
        """Extract the answer from the response"""
        # Remove the prompt from the response
        answer_text = response.replace(prompt, "").strip()
        
        # For multiple choice questions, extract just the number
        if self._is_multiple_choice(question):
            # Look for a single digit at the beginning of the answer or after standard phrases
            number_match = re.search(r'^(\d)[^\d]|답[은|:]?\s*(\d)|정답[은|:]?\s*(\d)', answer_text)
            if number_match:
                # Return the first captured group that is not None
                for group in number_match.groups():
                    if group is not None:
                        return group
            
            # If no match found through regex, look for the first digit
            for char in answer_text:
                if char.isdigit():
                    return char
            
            # Default to 0 if no number found
            return "0"
        
        # For subjective questions, return the full answer
        return answer_text
    
    def evaluate_model(self, model_name, test_df, num_samples=None):
        """
        Evaluate a model on test data
        
        Args:
            model_name: Name of the model to evaluate
            test_df: DataFrame containing test data
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            DataFrame with model predictions
        """
        if num_samples:
            test_df = test_df.sample(num_samples)
        
        # Load model if not already loaded
        if model_name not in self.loaded_models:
            model, tokenizer = self.load_model(model_name)
            self.loaded_models[model_name] = (model, tokenizer)
        else:
            model, tokenizer = self.loaded_models[model_name]
        
        # Initialize results
        results = []
        
        # Process each question
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
            question_id = row['ID']
            question = row['Question']
            
            # Format prompt
            prompt = self._format_prompt(question)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True
                )
            inference_time = time.time() - start_time
            
            # Decode output
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract answer
            answer = self._extract_answer(response, prompt, question)
            
            # Store result
            results.append({
                'ID': question_id,
                'Question': question,
                'Answer': answer,
                'InferenceTime': inference_time
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        self.results[model_name] = results_df
        
        return results_df
    
    def evaluate_all_models(self, test_file_path, num_samples=None, output_dir="evaluation_results"):
        """
        Evaluate all models on the test data
        
        Args:
            test_file_path: Path to the test CSV file
            num_samples: Number of samples to evaluate (None for all)
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary mapping model names to result DataFrames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_df = pd.read_csv(test_file_path)
        
        # Evaluate each model
        for model_name in self.models_to_evaluate:
            print(f"\nEvaluating model: {model_name}")
            results_df = self.evaluate_model(model_name, test_df, num_samples)
            
            # Save model results
            results_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_results.csv")
            results_df.to_csv(results_path, index=False)
            
            print(f"Saved results to {results_path}")
            
            # Calculate statistics
            total_time = results_df['InferenceTime'].sum()
            avg_time = results_df['InferenceTime'].mean()
            max_time = results_df['InferenceTime'].max()
            
            print(f"Inference statistics:")
            print(f"- Total time: {total_time:.2f}s")
            print(f"- Average time per question: {avg_time:.2f}s")
            print(f"- Maximum time for a question: {max_time:.2f}s")
        
        return self.results
    
    def compare_inference_times(self, output_dir="evaluation_results"):
        """
        Compare inference times across all evaluated models
        
        Args:
            output_dir: Directory to save comparison results
            
        Returns:
            DataFrame with inference time statistics
        """
        if not self.results:
            print("No evaluation results available")
            return None
        
        # Initialize statistics
        stats = []
        
        # Calculate statistics for each model
        for model_name, results_df in self.results.items():
            model_stats = {
                'model_name': model_name,
                'total_time': results_df['InferenceTime'].sum(),
                'avg_time': results_df['InferenceTime'].mean(),
                'max_time': results_df['InferenceTime'].max(),
                'min_time': results_df['InferenceTime'].min(),
                'median_time': results_df['InferenceTime'].median(),
                'num_questions': len(results_df)
            }
            
            # Calculate estimated time for full dataset (9000 questions)
            model_stats['estimated_full_dataset_time'] = model_stats['avg_time'] * 9000 / 60  # in minutes
            
            stats.append(model_stats)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Average inference time
        plt.subplot(2, 1, 1)
        plt.bar(stats_df['model_name'], stats_df['avg_time'])
        plt.title('Average Inference Time per Question')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        
        # Estimated time for full dataset
        plt.subplot(2, 1, 2)
        plt.bar(stats_df['model_name'], stats_df['estimated_full_dataset_time'])
        plt.title('Estimated Time for Full Dataset (9000 questions)')
        plt.ylabel('Time (minutes)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=270, color='r', linestyle='--', label='Time Limit (270 min)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
        
        # Save statistics
        stats_df.to_csv(os.path.join(output_dir, 'inference_time_statistics.csv'), index=False)
        
        return stats_df
    
    def select_best_model(self):
        """
        Select the best model based on inference speed and estimated performance
        
        Returns:
            Name of the best model
        """
        if not self.results:
            print("No evaluation results available")
            return None
        
        # Calculate statistics for each model
        model_stats = []
        
        for model_name, results_df in self.results.items():
            avg_time = results_df['InferenceTime'].mean()
            estimated_full_time = avg_time * 9000  # 9000 questions
            
            # Check if model meets time constraint
            meets_time_constraint = estimated_full_time <= (270 * 60)  # 270 minutes in seconds
            
            model_stats.append({
                'model_name': model_name,
                'avg_time': avg_time,
                'estimated_full_time': estimated_full_time / 60,  # in minutes
                'meets_time_constraint': meets_time_constraint
            })
        
        # Filter models that meet time constraint
        valid_models = [m for m in model_stats if m['meets_time_constraint']]
        
        if not valid_models:
            print("WARNING: No model meets the time constraint!")
            # Return the fastest model
            return min(model_stats, key=lambda x: x['avg_time'])['model_name']
        
        # Among valid models, select based on model size (preferring larger models within constraint)
        model_sizes = {
            "upstage/SOLAR-10.7B-Instruct-v1.0": 10.7,
            "mistralai/Mistral-7B-Instruct-v0.2": 7.0,
            "google/gemma-7b-it": 7.0,
            "01-ai/Yi-6B-Chat": 6.0
        }
        
        valid_models_with_size = [(m['model_name'], model_sizes.get(m['model_name'], 0)) 
                                 for m in valid_models]
        
        # Select the largest model that meets time constraint
        best_model = max(valid_models_with_size, key=lambda x: x[1])[0]
        
        return best_model

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models on a small subset for testing
    test_file = "/workspace/uploads/test.csv"
    results = evaluator.evaluate_all_models(test_file, num_samples=5)
    
    # Compare inference times
    stats = evaluator.compare_inference_times()
    
    # Select the best model
    best_model = evaluator.select_best_model()
    print(f"\nBest model for financial security tasks: {best_model}")