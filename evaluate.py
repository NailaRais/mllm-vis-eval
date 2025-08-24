#!/usr/bin/env python3
"""
MLLM Visual Evaluator (MLLM-VisEval)
An open-source tool to evaluate Multimodal LLMs on the Do-You-See-Me benchmark.
"""

import argparse
import re
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
import json


class MLLMEvaluator:
    """Main class for evaluating MLLMs on the Do-You-See-Me benchmark."""
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype=torch.float16):
        """
        Initialize the evaluator with a model.
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to load the model on
            torch_dtype: Torch data type for model weights
        """
        self.model_name = model_name
        print(f"Loading model {model_name}...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True  # Some models need this
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
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
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the model's answer from its response text.
        Improved parsing with multiple strategies.
        """
        # Strategy 1: Look for "Option X" pattern
        option_match = re.search(r'Option\s*([1-4])', text, re.IGNORECASE)
        if option_match:
            return option_match.group(1)
        
        # Strategy 2: Look for just the number in certain contexts
        number_match = re.search(r'(?:answer|choose|select|option).*?([1-4])', text, re.IGNORECASE)
        if number_match:
            return number_match.group(1)
        
        # Strategy 3: Look for the number at the very end if it's a short response
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line in ['1', '2', '3', '4']:
                return last_line
        
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
                
                # Prepare prompt - this format works for most instruction-tuned models
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                
                # Prepare inputs
                inputs = self.processor(
                    text=prompt, 
                    images=image, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Generate response
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False  # More deterministic for evaluation
                    )
                
                # Decode response
                generated_text = self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
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
                category_stats[category] = {
                    'accuracy': cat_accuracy,
                    'total': len(cat_results),
                    'correct': cat_results['is_correct'].sum()
                }
        
        return {
            'model_name': self.model_name,
            'overall_accuracy': accuracy,
            'total_samples': len(results),
            'correct_predictions': correct,
            'category_stats': category_stats,
            'detailed_results': results
        }
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save evaluation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sanitize model name for filename
        model_safe_name = self.model_name.replace("/", "-")
        
        # Save detailed results as CSV
        detailed_df = pd.DataFrame(results['detailed_results'])
        detailed_path = os.path.join(output_dir, f"detailed_{model_safe_name}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save summary as JSON
        summary = {
            'model': results['model_name'],
            'overall_accuracy': results['overall_accuracy'],
            'total_samples': results['total_samples'],
            'correct_predictions': results['correct_predictions'],
            'category_stats': results['category_stats']
        }
        
        summary_path = os.path.join(output_dir, f"summary_{model_safe_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
        return detailed_path, summary_path


def main():
    """Command line interface for the evaluator."""
    parser = argparse.ArgumentParser(description='Evaluate an MLLM on the Do-You-See-Me benchmark.')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Hugging Face model ID (e.g. "llava-hf/llava-1.5-7b-hf")')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of samples to test (use -1 for all)')
    parser.add_argument('--categories', type=str, nargs='+', 
                       help='Specific categories to evaluate (e.g. visual_figure_ground visual_spatial)')
    parser.add_argument('--output_dir', type=str, default="results",
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default="auto",
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--precision', type=str, default="float16", choices=["float16", "float32"],
                       help='Precision for model weights')
    
    args = parser.parse_args()
    
    # Set torch dtype
    torch_dtype = torch.float16 if args.precision == "float16" else torch.float32
    
    # Initialize evaluator
    evaluator = MLLMEvaluator(
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
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {args.model_name}")
    print("="*50)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct: {results['correct_predictions']}/{results['total_samples']}")
    print("\nCategory-wise Performance:")
    
    for category, stats in results['category_stats'].items():
        category_name = evaluator.category_names.get(category, category)
        print(f"  {category_name}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save results
    evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
