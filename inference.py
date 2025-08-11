"""
Inference module for Financial Security AI Model
This module handles the inference process for the FSKU evaluation benchmark
"""

import os
import re
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class FinancialSecurityModelInference:
    def __init__(
        self,
        model_path,
        peft_model_path=None,
        device="cuda",
        load_in_4bit=True,
        load_in_8bit=False
    ):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the base model
            peft_model_path: Path to PEFT adapter (if using LoRA)
            device: Device to run inference on (cuda/cpu)
            load_in_4bit: Whether to load in 4-bit quantization
            load_in_8bit: Whether to load in 8-bit quantization
        """
        self.model_path = model_path
        self.peft_model_path = peft_model_path
        self.device = device
        
        print(f"Loading model from {model_path}")
        start_time = time.time()
        
        # Configure quantization
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT adapter if specified
        if peft_model_path:
            print(f"Loading PEFT adapter from {peft_model_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_model_path
            )
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def generate_response(
        self,
        question,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1
    ):
        """
        Generate a response for a given question
        
        Args:
            question: The question to answer
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated answer
        """
        # Format prompt based on whether it's multiple choice or subjective
        if self._is_multiple_choice(question):
            prompt = self._format_mc_prompt(question)
        else:
            prompt = self._format_subjective_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.1
            )
        
        # Decode output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract answer from response
        answer = self._extract_answer(response, prompt, question)
        
        return answer
    
    def _is_multiple_choice(self, question):
        """Check if the question is multiple choice"""
        # Look for patterns like "1. ", "1) ", etc.
        return bool(re.search(r'\n\d[\.\)\s]', question))
    
    def _format_mc_prompt(self, question):
        """Format prompt for multiple choice questions"""
        return f"""### 지시사항: 
금융보안 전문가로서, 아래 주어진 객관식 문제의 정답을 선택해주세요. 
1부터 5까지의 숫자로만 답해주세요. 
설명은 필요하지 않습니다.

### 입력:
{question}

### 출력:
"""
    
    def _format_subjective_prompt(self, question):
        """Format prompt for subjective questions"""
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
    
    def process_test_file(self, test_file, output_file):
        """
        Process a test file and generate answers
        
        Args:
            test_file: Path to the test CSV file
            output_file: Path to save the output CSV
            
        Returns:
            DataFrame with the generated answers
        """
        print(f"Processing test file: {test_file}")
        
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Initialize results list
        results = []
        
        # Process each question
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating answers"):
            question_id = row['ID']
            question = row['Question']
            
            # Generate answer
            start_time = time.time()
            answer = self.generate_response(question)
            inference_time = time.time() - start_time
            
            # Store result
            results.append({
                'ID': question_id,
                'Answer': answer,
                'InferenceTime': inference_time
            })
            
            print(f"Question ID: {question_id} | Inference Time: {inference_time:.2f}s")
            print(f"Question: {question[:100]}...")
            print(f"Answer: {answer}")
            print("-" * 50)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to output file
        results_df[['ID', 'Answer']].to_csv(output_file, index=False)
        
        # Calculate statistics
        total_time = results_df['InferenceTime'].sum()
        avg_time = results_df['InferenceTime'].mean()
        max_time = results_df['InferenceTime'].max()
        
        print(f"\nInference completed:")
        print(f"Total questions: {len(results_df)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per question: {avg_time:.2f}s")
        print(f"Maximum time for a question: {max_time:.2f}s")
        
        return results_df

if __name__ == "__main__":
    # Path to model weights and test data
    model_path = "upstage/SOLAR-10.7B-Instruct-v1.0"
    test_file = "/workspace/uploads/test.csv"
    output_file = "/workspace/uploads/results.csv"
    
    # Initialize inference model
    inference = FinancialSecurityModelInference(
        model_path=model_path,
        load_in_4bit=True
    )
    
    # Process test file
    results = inference.process_test_file(test_file, output_file)