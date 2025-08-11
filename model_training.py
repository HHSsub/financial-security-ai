"""
Model training module for Financial Security AI Model
"""

import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

class FinancialSecurityModelTrainer:
    def __init__(
        self,
        base_model_name,
        output_dir="./trained_model",
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        use_4bit=True
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer()
    
    def _initialize_model_and_tokenizer(self):
        """Initialize the model and tokenizer with appropriate quantization"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare quantization config if needed
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # Load base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare model for training if using quantization
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self._get_target_modules()
        )
        
        self.model = get_peft_model(self.model, peft_config)
    
    def _get_target_modules(self):
        """Get target modules for LoRA based on model architecture"""
        # Default targets that work for many models
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _prepare_training_data(self, data_path, validation_split=0.1, max_length=512):
        """Prepare training and validation datasets"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Function to format data into instruction-response format
        def format_instruction(row):
            question = row['Question']
            answer = str(row.get('Answer', ''))  # If Answer column exists
            
            # Format for instruction-based training
            return f"""### 지시사항: 
다음 금융보안 관련 문제에 정확하게 답변하세요.

### 입력:
{question}

### 출력:
{answer}"""
        
        # Apply formatting
        df['formatted_text'] = df.apply(format_instruction, axis=1)
        
        # Convert to Dataset format
        dataset = Dataset.from_pandas(df[['formatted_text']])
        
        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["formatted_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["formatted_text"]
        )
        
        # Split into training and validation sets
        tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=validation_split
        )
        
        return tokenized_dataset
    
    def train(
        self, 
        training_data_path, 
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        validation_split=0.1,
        max_length=512
    ):
        """Train the model on financial security data"""
        # Prepare datasets
        datasets = self._prepare_training_data(
            training_data_path, 
            validation_split=validation_split,
            max_length=max_length
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
            logging_steps=10,
            optim="paged_adamw_8bit"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")
        
        return f"{self.output_dir}/final"