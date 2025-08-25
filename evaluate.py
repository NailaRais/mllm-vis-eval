#!/usr/bin/env python3
"""
Universal MLLM Visual Evaluator (MLLM-VisEval)
An open-source tool to evaluate any Multimodal LLM on the Do-You-See-Me benchmark.
"""

import argparse
import re
from datasets import load_dataset
from transformers import (
    AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer,
    AutoImageProcessor, AutoConfig, pipeline
)
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
from enum import Enum


class ModelType(Enum):
    """Enumeration of different model types we can handle"""
    VISION2SEQ = "vision2seq"  # Standard vision-to-sequence models
    IMAGE_TEXT = "image_text"  # Models that take both image and text
    VISION_ONLY = "vision_only"  # Models that only process images
    PIPELINE = "pipeline"  # Use HuggingFace pipeline API
    CUSTOM = "custom"  # Custom handling required


class UniversalMLLMEvaluator:
    """Universal class for evaluating any MLLM on the Do-You-See-Me benchmark."""
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype=torch.float16):
        """
        Initialize the evaluator with a model.
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to load the model on
            torch_dtype: Torch data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_type = None
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
        print(f"Loading model {model_name}...")
        
        # Try to detect model type and load appropriately
        self._load_model()
        
        # Load the dataset
        print("Loading Do-You-See-Me dataset...")
        self.dataset = load_dataset("microsoft/Do-You-See-Me", split="train")
        
        # Category mapping for better organization
        self.category_names = {
            "visual_figure_ground": "Visual Figure-Ground",
            "visual_spatial": "Visual Spatial",
            "visual_form_constancy": "Visual Form Constancy",
            "shape_discrimination": "Shape Discrimination",
            "shape_color_discrimination": "Shape-Color Discrimination",
            "letter_disambiguation": "Letter Disambiguation",
            "visual_closure": "Visual Closure"
        }
    
    def _load_model(self):
        """Try different approaches to load the model"""
        try:
            # First try standard vision2seq loading
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model_type = ModelType.VISION2SEQ
            print("Loaded as Vision2Seq model")
            return
        except Exception as e:
            print(f"Vision2Seq loading failed: {e}")
        
        try:
            # Try loading as a generic model with separate components
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model_type = ModelType.IMAGE_TEXT
            print("Loaded as generic image-text model")
            return
        except Exception as e:
            print(f"Generic model loading failed: {e}")
        
        try:
            # Try using pipeline API
            self.pipe = pipeline(
                "image-to-text",
                model=self.model_name,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            self.model_type = ModelType.PIPELINE
            print("Loaded using pipeline API")
            return
        except Exception as e:
            print(f"Pipeline loading failed: {e}")
        
        # If all else fails, try to get config and determine model type
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            arch = config.architectures[0] if config.architectures else "Unknown"
            print(f"Model architecture: {arch}")
            
            # Try to load based on architecture name
            if "Vision" in arch or "Image" in arch:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self.model_type = ModelType.CUSTOM
                print("Loaded as custom vision model")
            else:
                raise ValueError(f"Unsupported architecture: {arch}")
        except Exception as e:
            print(f"All loading methods failed: {e}")
            raise ValueError(f"Could not load model {self.model_name} with any method")
    
    def _prepare_inputs(self, image: Image.Image, question: str) -> Any:
        """Prepare inputs based on model type"""
        if self.model_type == ModelType.VISION2SEQ:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            return self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        
        elif self.model_type == ModelType.IMAGE_TEXT:
            # For generic models, we need to handle image and text separately
            image_inputs = self.image_processor(images=image, return_tensors="pt").to(self.model.device)
            text_inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
            return {**image_inputs, **text_inputs}
        
        elif self.model_type == ModelType.PIPELINE:
            # Pipeline handles everything internally
            return image, question
        
        elif self.model_type == ModelType.CUSTOM:
            # Custom handling - try different approaches
            try:
                # First try vision processor if available
                if hasattr(self, 'processor'):
                    return self.processor(image, question, return_tensors="pt").to(self.model.device)
                else:
                    # Fallback to separate processing
                    vision_inputs = self.image_processor(image, return_tensors="pt").to(self.model.device)
                    text_inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
                    return {**vision_inputs, **text_inputs}
            except Exception as e:
                print(f"Custom input preparation failed: {e}")
                # Last resort: just return the image
                return image
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _generate_response(self, inputs: Any) -> str:
        """Generate response based on model type"""
        try:
            if self.model_type == ModelType.VISION2SEQ:
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False
                    )
                return self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            elif self.model_type == ModelType.IMAGE_TEXT:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Try to extract text from outputs - this is model-specific
                if hasattr(outputs, 'logits'):
                    # If we have logits, we need to decode them
                    predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
                    return self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
                else:
                    # Try to find text in outputs
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 2:
                            predicted_token_ids = torch.argmax(value, dim=-1)
                            return self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
                    return "Could not decode model output"
            
            elif self.model_type == ModelType.PIPELINE:
                result = self.pipe(inputs[0], prompt=inputs[1])
                return result[0]['generated_text']
            
            elif self.model_type == ModelType.CUSTOM:
                # Custom models might need special handling
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Try various ways to extract text
                if hasattr(outputs, 'last_hidden_state'):
                    # This is a common output format
                    return "Custom model output - needs specific handling"
                else:
                    return str(outputs)
            
            else:
                return "Unknown model type"
                
        except Exception as e:
            print(f"Generation failed: {e}")
            return f"Error during generation: {str(e)}"
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the model's answer from its response text.
        Improved parsing with multiple strategies.
        """
        if not text or text.strip() == "":
            return "N/A"
        
        # Clean the text
        text = text.strip()
        
        # Strategy 1: Look for "Option X" pattern
        option_match = re.search(r'Option\s*([1-4])', text, re.IGNORECASE)
        if option_match:
            return option_match.group(1)
        
        # Strategy 2: Look for just the number in certain contexts
        number_match = re.search(r'(?:answer|choose|select|option|response|output).*?([1-4])', text, re.IGNORECASE)
        if number_match:
            return number_match.group(1)
        
        # Strategy 3: Look for the number at the very end if it's a short response
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line in ['1', '2', '3', '4']:
                return last_line
        
        # Strategy 4: Look for numbers in the entire text
        all_numbers = re.findall(r'\b[1-4]\b', text)
        if all_numbers:
            return all_numbers[-1]  # Return the last number found
        
        # Strategy 5: For models that output full sentences, try to find the answer
        if "option 1" in text.lower():
            return "1"
        elif "option 2" in text.lower():
            return "2"
        elif "option 3" in text.lower():
            return "3"
        elif "option 4" in text.lower():
            return "4"
        
        return "N/A"
    
    def evaluate(self, num_samples: int = -1, categories: List[str] = None) -> Dict:
        """
        Run evaluation on the dataset.
        
        Args:
            num_samples (int): Number of samples to evaluate (-1 for all)
            categories (List[str]): Specific categories to evaluate
            
        Returns:
            Dict: Evaluation results
        """
        # Filter dataset if needed
        eval_dataset = self.dataset
        if num_samples > 0:
            eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
        
        if categories:
            eval_dataset = eval_dataset.filter(lambda x: x['sweep'] in categories)
        
        results = []
        correct = 0
        
        print(f"Evaluating on {len(eval_dataset)} samples...")
        
        # Run evaluation
        for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            try:
                # Get sample data
                image = sample['image']
                question = sample['question']
                true_answer = str(sample['answer'])
                category = sample['sweep']
                
                # Prepare inputs based on model type
                inputs = self._prepare_inputs(image, question)
                
                # Generate response
                generated_text = self._generate_response(inputs)
                
                # Extract the predicted answer
                predicted_answer = self.extract_answer(generated_text)
                
                # Check if correct
                is_correct = (predicted_answer == true_answer)
                if is_correct:
                    correct += 1
                
                # Store result
                results.append({
                    'id': i,
                    'category': category,
                    'question': question,
                    'true_answer': true_answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'model_output': generated_text
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                results.append({
                    'id': i,
                    'category': sample.get('sweep', 'unknown'),
                    'question': sample.get('question', ''),
                    'true_answer': sample.get('answer', ''),
                    'predicted_answer': 'ERROR',
                    'is_correct': False,
                    'model_output': f"Error: {str(e)}"
                })
        
        # Calculate overall accuracy
        accuracy = (correct / len(results)) * 100 if results else 0
        
        # Calculate accuracy by category
        category_stats = {}
        if results:
            results_df = pd.DataFrame(results)
            for category in results_df['category'].unique():
                cat_results = results_df[results_df['category'] == category]
                cat_accuracy = (cat_results['is_correct'].sum() / len(cat_results)) * 100
                category_name = self.category_names.get(category, category)
                category_stats[category_name] = {
                    'accuracy': float(cat_accuracy),
                    'total': int(len(cat_results)),
                    'correct': int(cat_results['is_correct'].sum())
                }
        
        return {
            'model_name': self.model_name,
            'model_type': str(self.model_type),
            'overall_accuracy': float(accuracy),
            'total_samples': int(len(results)),
            'correct_predictions': int(correct),
            'category_stats': category_stats,
            'detailed_results': results
        }
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save evaluation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sanitize model name for filename
        model_safe_name = self.model_name.replace("/", "-").replace("\\", "-")
        
        # Save detailed results as CSV
        detailed_df = pd.DataFrame(results['detailed_results'])
        detailed_path = os.path.join(output_dir, f"detailed_{model_safe_name}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save summary as JSON with proper serialization
        summary = {
            'model': results['model_name'],
            'model_type': results['model_type'],
            'overall_accuracy': results['overall_accuracy'],
            'total_samples': results['total_samples'],
            'correct_predictions': results['correct_predictions'],
            'category_stats': results['category_stats']
        }
        
        summary_path = os.path.join(output_dir, f"summary_{model_safe_name}.json")
        
        # Custom JSON encoder to handle NumPy data types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"Results saved to {output_dir}/")
        return detailed_path, summary_path


def main():
    """Command line interface for the evaluator."""
    parser = argparse.ArgumentParser(description='Evaluate an MLLM on the Do-You-See-Me benchmark.')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Hugging Face model ID')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of samples to test (use -1 for all)')
    parser.add_argument('--categories', type=str, nargs='+', 
                       help='Specific categories to evaluate (e.g. visual_figure_ground visual_spatial)')
    parser.add_argument('--output_dir', type=str, default="results",
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default="auto",
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--precision', type=str, default="float16", choices=["float16", "float32", "bfloat16"],
                       help='Precision for model weights')
    
    args = parser.parse_args()
    
    # Set torch dtype
    if args.precision == "float16":
        torch_dtype = torch.float16
    elif args.precision == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    try:
        # Initialize evaluator
        evaluator = UniversalMLLMEvaluator(
            model_name=args.model_name,
            device=args.device,
            torch_dtype=torch_dtype
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            num_samples=args.num_samples,
            categories=args.categories
        )
        
        # Print summary
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {args.model_name}")
        print("="*60)
        print(f"Model Type: {results['model_type']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Correct: {results['correct_predictions']}/{results['total_samples']}")
        print("\nCategory-wise Performance:")
        
        for category, stats in results['category_stats'].items():
            print(f"  {category}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
    except Exception as e:
        print(f"Failed to evaluate model: {str(e)}")
        print("This model might not be compatible with the evaluator.")
        print("Consider trying a different model or checking the model's documentation.")


if __name__ == "__main__":
    main()
