"""
Model selection module for Financial Security AI Model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# List of potential models that meet the requirements:
# 1. Released before Aug 1, 2025
# 2. Open source with appropriate license
# 3. Can be run locally on RTX 4090 (24GB VRAM)
# 4. Suitable for financial/security domain knowledge

MODEL_CANDIDATES = [
    {
        "name": "Llama-2-13B-chat",
        "hf_path": "meta-llama/Llama-2-13b-chat-hf",
        "description": "Meta's Llama 2 model, 13B parameters, finetuned for chat",
        "quantization": "4bit",
        "context_length": 4096,
        "strengths": "Good general knowledge, instruction following",
        "weaknesses": "May require domain-specific finetuning"
    },
    {
        "name": "Mistral-7B-v0.1",
        "hf_path": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral AI's 7B parameter model",
        "quantization": "4bit",
        "context_length": 8192,
        "strengths": "Strong performance despite smaller size, long context",
        "weaknesses": "May need instruction finetuning"
    },
    {
        "name": "Mistral-7B-Instruct-v0.2",
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.2", 
        "description": "Instruction-tuned version of Mistral-7B",
        "quantization": "4bit",
        "context_length": 8192,
        "strengths": "Good instruction following, strong reasoning",
        "weaknesses": "Less domain-specific knowledge than larger models"
    },
    {
        "name": "SOLAR-10.7B-Instruct-v1.0",
        "hf_path": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "description": "Upstage's 10.7B parameter instruction-tuned model",
        "quantization": "4bit",
        "context_length": 4096,
        "strengths": "Good reasoning, knowledge, and instruction following",
        "weaknesses": "Relatively new, less tested in financial domain"
    },
    {
        "name": "Gemma-7B-it",
        "hf_path": "google/gemma-7b-it",
        "description": "Google's 7B instruction-tuned model",
        "quantization": "4bit",
        "context_length": 8192,
        "strengths": "Strong general knowledge, compact size",
        "weaknesses": "May require financial domain adaptation"
    },
    {
        "name": "Yi-6B-Chat", 
        "hf_path": "01-ai/Yi-6B-Chat",
        "description": "01.AI's 6B parameter chat model",
        "quantization": "4bit",
        "context_length": 4096,
        "strengths": "Lightweight, good multilingual support",
        "weaknesses": "Smaller size may impact domain knowledge"
    }
]

def load_model_for_eval(model_name, quantization="4bit"):
    """
    Load a model for evaluation with appropriate quantization
    """
    # Find the model info
    model_info = next((m for m in MODEL_CANDIDATES if m["name"] == model_name), None)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in candidates")
    
    model_path = model_info["hf_path"]
    
    # Configure quantization
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        bnb_config = None
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def compare_model_stats():
    """
    Print comparison of candidate model statistics
    """
    print("Model Comparison:")
    print("-" * 100)
    print(f"{'Name':<25} {'Size':<10} {'Context Len':<15} {'Quantization':<15} {'Main Strengths'}")
    print("-" * 100)
    
    for model in MODEL_CANDIDATES:
        name = model["name"]
        size = name.split("-")[1] if "-" in name else "Unknown"
        context = str(model["context_length"])
        quant = model["quantization"]
        strengths = model["strengths"]
        
        print(f"{name:<25} {size:<10} {context:<15} {quant:<15} {strengths}")

if __name__ == "__main__":
    compare_model_stats()