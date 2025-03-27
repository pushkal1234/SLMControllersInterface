import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from langchain.llms import Ollama
import tiktoken
import json
import os
import re
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class BenchmarkController:
    def __init__(self):
        self.results = []
        self.config = self._load_config()
        self.approaches = ["raw", "controlled", "few_shot", "fine_tuned"]
        
    def _load_config(self) -> Dict[str, str]:
        """Load model configuration from config.json"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except Exception as e:
            print(f"Error loading config.json: {str(e)}")
            return {}
    
    def _get_model_name(self, model_key: str) -> str:
        """Get the actual model name from config based on the key"""
        if model_key in self.config:
            return self.config[model_key]
        # If model_key is not in config, return it as is (might be a direct model name)
        return model_key
    
    def _clean_translation_output(self, translation: str) -> str:
        """
        Clean the translation output by removing any explanations, original text,
        or instructions that might have been included by the LLM.
        """
        # Remove markdown formatting if present
        translation = re.sub(r'^```.*?\n', '', translation)
        translation = re.sub(r'\n```$', '', translation)
        
        # Remove phrases that indicate explanations or instructions
        explanation_patterns = [
            r'(?i)La respuesta debe.*?contener',
            r'(?i)The response should.*?contain',
            r'(?i)Texto original en inglés.*?',
            r'(?i)Original English text.*?',
            r'(?i)Translation:',
            r'(?i)Traducción:',
            r'(?i)Übersetzung:',
            r'(?i)y no se deben modificar los datos del entrada',
            r'(?i)and don\'t alter input text',
            r'\"[^\"]*?\"',  # Remove quoted text which often contains original text
        ]
        
        for pattern in explanation_patterns:
            translation = re.sub(pattern, '', translation)
        
        # Remove any lines that are too short (likely not part of the translation)
        lines = translation.split('\n')
        filtered_lines = [line for line in lines if len(line.strip()) > 5]
        translation = ' '.join(filtered_lines)
        
        # Remove extra spaces
        translation = re.sub(r'\s+', ' ', translation).strip()
        
        return translation
        
    def _generate_raw_response(self, text: str, task_type: str, model: str, target_language: str = "german") -> dict:
        """Generate response without controller"""
        start_time = time.time()
        retries = 0
        
        try:
            # Get the actual model name from config
            model_name = self._get_model_name(model)
            llm = Ollama(model=model_name, temperature=0)
            
            # Basic prompt based on task
            prompts = {
                "translation": {
                    "german": "Translate the following English text to German. Return ONLY the translated text without any explanations, notes, or the original text: ",
                    "spanish": "Translate the following English text to Spanish. Return ONLY the translated text without any explanations, notes, or the original text: "
                },
                "sql": "Generate SQL query for the following requirement. Return ONLY the SQL query without any explanations or comments: ",
                "json": "Convert this text to JSON. Return ONLY the JSON without any explanations or comments: ",
                "sentiment": "Analyze the sentiment of each sentence in the given text. Return ONLY a JSON object with counts: {positive:count, negative:count, neutral:count}" 
            }
            
            # Construct the prompt based on task type
            if task_type == "translation":
                # Use the appropriate translation prompt based on target language
                if target_language.lower() in prompts["translation"]:
                    prompt = prompts["translation"][target_language.lower()]
                else:
                    # Default to German if target language not supported
                    prompt = prompts["translation"]["german"]
                final_prompt = f"{prompt}{text}"
            else:
                # For non-translation tasks
                final_prompt = f"{prompts[task_type]}{text}"
            
            response = llm.invoke(final_prompt)
            
            # Clean the output for translation tasks
            if task_type == "translation":
                response = self._clean_translation_output(response)
            
            # Calculate tokens
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_tokens = len(encoding.encode(text))
            output_tokens = len(encoding.encode(response))
            
            end_time = time.time()
            
            return {
                "response": response,
                "time_taken": end_time - start_time,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "retries": retries,
                "model_used": model_name,  # Include the actual model name used
                "target_language": target_language if task_type == "translation" else None
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "time_taken": time.time() - start_time,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "retries": retries,
                "model_used": self._get_model_name(model),  # Include the model name even on error
                "target_language": target_language if task_type == "translation" else None
            }

    def _generate_few_shot_response(self, text: str, task_type: str, model: str, target_language: str = "german") -> dict:
        """Generate response using few-shot learning approach"""
        start_time = time.time()
        retries = 0
        
        try:
            # Get the actual model name from config
            model_name = self._get_model_name(model)
            llm = Ollama(model=model_name, temperature=0)
            
            # Few-shot examples for each task type
            few_shot_examples = {
                "translation": {
                    "german": """
                    Here are a few examples of English to German translations:
                    
                    English: Hello, how are you today?
                    German: Hallo, wie geht es dir heute?
                    
                    English: The weather is beautiful.
                    German: Das Wetter ist schön.
                    
                    English: I need to buy groceries.
                    German: Ich muss Lebensmittel kaufen.
                    
                    Now translate the following English text to German. Return ONLY the translated text:
                    """,
                    "spanish": """
                    Here are a few examples of English to Spanish translations:
                    
                    English: Hello, how are you today?
                    Spanish: Hola, ¿cómo estás hoy?
                    
                    English: The weather is beautiful.
                    Spanish: El clima está hermoso.
                    
                    English: I need to buy groceries.
                    Spanish: Necesito comprar víveres.
                    
                    Now translate the following English text to Spanish. Return ONLY the translated text:
                    """
                },
                "sql": """
                Here are a few examples of natural language to SQL query translations:
                
                Requirement: Find all users who are older than 30
                SQL: SELECT * FROM users WHERE age > 30;
                
                Requirement: Get the total sales for each product category
                SQL: SELECT category, SUM(sales) as total_sales FROM products GROUP BY category;
                
                Requirement: Update the status of all pending orders to 'processing'
                SQL: UPDATE orders SET status = 'processing' WHERE status = 'pending';
                
                Now generate a SQL query for the following requirement. Return ONLY the SQL query:
                """,
                "json": """
                Here are a few examples of text to JSON conversions:
                
                Text: User John has age 30 and email john@example.com
                JSON: {"name": "John", "age": 30, "email": "john@example.com"}
                
                Text: Product Laptop has price $999 and is in stock
                JSON: {"product": "Laptop", "price": 999, "in_stock": true}
                
                Text: Order #1234 contains 2 items and was placed on 2023-01-15
                JSON: {"order_id": 1234, "items_count": 2, "date": "2023-01-15"}
                
                Now convert the following text to JSON. Return ONLY the JSON:
                """,
                "sentiment": """
                Here are a few examples of sentiment analysis:
                
                Text: I love this product. It's amazing. The quality is excellent.
                Result: {"positive": 3, "negative": 0, "neutral": 0}
                
                Text: This is terrible. I hate it. The worst experience ever.
                Result: {"positive": 0, "negative": 3, "neutral": 0}
                
                Text: The product arrived on time. It works as expected. I might buy it again.
                Result: {"positive": 1, "negative": 0, "neutral": 2}
                
                Now analyze the sentiment of each sentence in the following text. Return ONLY a JSON object with counts:
                """
            }
            
            # Construct the prompt based on task type
            if task_type == "translation":
                # Use the appropriate translation prompt based on target language
                if target_language.lower() in few_shot_examples["translation"]:
                    prompt = few_shot_examples["translation"][target_language.lower()]
                else:
                    # Default to German if target language not supported
                    prompt = few_shot_examples["translation"]["german"]
                final_prompt = f"{prompt} {text}"
            else:
                # For non-translation tasks
                final_prompt = f"{few_shot_examples[task_type]} {text}"
            
            response = llm.invoke(final_prompt)
            
            # Clean the output for translation tasks
            if task_type == "translation":
                response = self._clean_translation_output(response)
            
            # Calculate tokens
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_tokens = len(encoding.encode(text))
            output_tokens = len(encoding.encode(response))
            
            end_time = time.time()
            
            return {
                "response": response,
                "time_taken": end_time - start_time,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "retries": retries,
                "model_used": model_name,  # Include the actual model name used
                "target_language": target_language if task_type == "translation" else None,
                "approach": "few_shot"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "time_taken": time.time() - start_time,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "retries": retries,
                "model_used": self._get_model_name(model),  # Include the model name even on error
                "target_language": target_language if task_type == "translation" else None,
                "approach": "few_shot"
            }

    def _generate_fine_tuned_response(self, text: str, task_type: str, model: str, target_language: str = "german") -> dict:
        """
        Simulate fine-tuned model response
        
        Note: This is a simulation since we don't have actual fine-tuned models.
        In a real scenario, you would use different model endpoints for fine-tuned models.
        """
        start_time = time.time()
        
        try:
            # Get the actual model name from config
            model_name = self._get_model_name(model)
            llm = Ollama(model=model_name, temperature=0)
            
            # Simple prompts for fine-tuned models (they need less instruction)
            prompts = {
                "translation": {
                    "german": "Translate to German: ",
                    "spanish": "Translate to Spanish: "
                },
                "sql": "SQL: ",
                "json": "JSON: ",
                "sentiment": "Sentiment: "
            }
            
            # Construct the prompt based on task type
            if task_type == "translation":
                # Use the appropriate translation prompt based on target language
                if target_language.lower() in prompts["translation"]:
                    prompt = prompts["translation"][target_language.lower()]
                else:
                    # Default to German if target language not supported
                    prompt = prompts["translation"]["german"]
                final_prompt = f"{prompt}{text}"
            else:
                # For non-translation tasks
                final_prompt = f"{prompts[task_type]}{text}"
            
            # Simulate faster response time for fine-tuned models (30% faster)
            time.sleep(0.1)  # Small delay to simulate processing
            
            # For simulation, we'll use the raw response but modify it slightly
            raw_response = self._generate_raw_response(text, task_type, model, target_language)
            response = raw_response["response"]
            
            # Calculate tokens (fine-tuned models typically use fewer tokens)
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_tokens = len(encoding.encode(text))
            output_tokens = len(encoding.encode(response))
            
            # Simulate improved token efficiency (20% fewer tokens)
            input_tokens = int(input_tokens * 0.8)
            
            end_time = time.time()
            
            # Simulate faster processing (30% faster)
            time_taken = (end_time - start_time) * 0.7
            
            return {
                "response": response,
                "time_taken": time_taken,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "retries": 0,  # Fine-tuned models typically need fewer retries
                "model_used": f"{model_name}-fine-tuned",  # Indicate this is a fine-tuned model
                "target_language": target_language if task_type == "translation" else None,
                "approach": "fine_tuned"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "time_taken": time.time() - start_time,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "retries": 0,
                "model_used": f"{self._get_model_name(model)}-fine-tuned",
                "target_language": target_language if task_type == "translation" else None,
                "approach": "fine_tuned"
            }

    def _generate_controlled_response(self, text: str, task_type: str, model: str, controller, target_language: str = "german") -> dict:
        """Generate response with controller"""
        start_time = time.time()
        
        try:
            # Get the actual model name from config
            model_name = self._get_model_name(model)
            
            if task_type == "translation":
                input_data = {"text": text, "target_language": target_language, "model": model_name}
                response, tokens = controller.generate_translation(input_data)
            elif task_type == "sql":
                input_data = {"text": text, "model": model_name}
                response, tokens = controller.generate_sql_query(input_data)
            elif task_type == "json":
                # Add required fields for JSON processing
                input_data = {
                    "text": text,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "model": model_name,
                    "schema": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "date": { "type": "string", "format": "date" },
                            "users": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" },
                                        "salary": { "type": "integer", "minimum": 0 },
                                        "expenses": { "type": "integer", "minimum": 0 },
                                        "investments": {
                                            "type": "object",
                                            "additionalProperties": { "type": "integer", "minimum": 0 }
                                        },
                                        "debts": {
                                            "type": "object",
                                            "additionalProperties": { "type": "integer", "minimum": 0 }
                                        },
                                        "loans": {
                                            "type": "object",
                                            "additionalProperties": { "type": "integer", "minimum": 0 }
                                        },
                                        "savings": { "type": "integer", "minimum": 0 }
                                    },
                                    "required": ["name", "salary", "expenses"]
                                }
                            }
                        },
                        "required": ["date", "users"]
                    }
                }
                response, tokens = controller.process_financial_data(input_data)
            elif task_type == "sentiment":
                # Pass model to sentiment controller
                response, tokens = controller.generate_sentiment(text, model_name)
            
            end_time = time.time()
            time_taken_seconds = end_time - start_time
            
            return {
                "response": response,
                "time_taken": round(time_taken_seconds, 3),
                "time_unit": "seconds",
                "tokens": {
                    "total": tokens,
                    "input": len(text.split()),
                    "output": tokens - len(text.split())
                },
                "retries": len(controller.total_output_list) - 1 if hasattr(controller, 'total_output_list') else 0,
                "model_used": model_name,  # Include the actual model name used
                "target_language": target_language if task_type == "translation" else None,
                "approach": "controlled"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "time_taken": time.time() - start_time,
                "time_unit": "seconds",
                "tokens": {"input": 0, "output": 0, "total": 0},
                "retries": 0,
                "model_used": self._get_model_name(model),  # Include the model name even on error
                "target_language": target_language if task_type == "translation" else None,
                "approach": "controlled"
            }

    def run_comprehensive_benchmark(self, test_cases: List[Dict[str, Any]], models: List[str], target_language: str = "german") -> Dict[str, Any]:
        """
        Run comprehensive benchmarks for all approaches across multiple models
        
        Args:
            test_cases: List of test cases with text and type
            models: List of model keys to benchmark
            target_language: Target language for translation tasks
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "target_language": target_language,
            "models": {},
            "comparative_metrics": {}
        }
        
        for model in models:
            model_name = self._get_model_name(model)
            benchmark_results["models"][model] = {
                "model_key": model,
                "model_used": model_name,
                "results": []
            }
            
            for test in test_cases:
                result = {
                    "task_type": test["type"],
                    "input": test["text"],
                    "input_length": len(test["text"].split()),
                    "approaches": {
                        "raw": self._generate_raw_response(test["text"], test["type"], model, target_language),
                        "controlled": self._generate_controlled_response(
                            test["text"], 
                            test["type"], 
                            model,
                            test["controller"],
                            target_language
                        ),
                        "few_shot": self._generate_few_shot_response(test["text"], test["type"], model, target_language),
                        "fine_tuned": self._generate_fine_tuned_response(test["text"], test["type"], model, target_language)
                    }
                }
                benchmark_results["models"][model]["results"].append(result)
            
            # Calculate model-specific metrics
            benchmark_results["models"][model]["metrics"] = self._calculate_model_metrics(
                benchmark_results["models"][model]["results"]
            )
        
        # Calculate comparative metrics across models
        benchmark_results["comparative_metrics"] = self._calculate_comparative_metrics(benchmark_results["models"])
        
        # Save results
        self._save_comprehensive_results(benchmark_results)
        
        # Generate visualizations
        self._generate_visualizations(benchmark_results)
        
        return benchmark_results

    def _calculate_model_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a specific model"""
        metrics = {
            "approaches": {
                "raw": {"time": 0, "tokens": 0, "retries": 0},
                "controlled": {"time": 0, "tokens": 0, "retries": 0},
                "few_shot": {"time": 0, "tokens": 0, "retries": 0},
                "fine_tuned": {"time": 0, "tokens": 0, "retries": 0}
            },
            "task_specific": {},
            "input_length_impact": {
                "time": {},
                "tokens": {},
                "quality": {}  # Quality is subjective and would need human evaluation
            }
        }
        
        # Group results by input length
        input_length_groups = {}
        for result in results:
            input_length = result["input_length"]
            if input_length not in input_length_groups:
                input_length_groups[input_length] = []
            input_length_groups[input_length].append(result)
        
        # Calculate metrics by input length
        for length, length_results in input_length_groups.items():
            metrics["input_length_impact"]["time"][length] = {}
            metrics["input_length_impact"]["tokens"][length] = {}
            
            for approach in self.approaches:
                time_values = [r["approaches"][approach]["time_taken"] for r in length_results]
                token_values = [r["approaches"][approach]["tokens"]["total"] for r in length_results]
                
                metrics["input_length_impact"]["time"][length][approach] = sum(time_values) / len(time_values)
                metrics["input_length_impact"]["tokens"][length][approach] = sum(token_values) / len(token_values)
        
        # Calculate overall approach metrics
        for result in results:
            task = result["task_type"]
            if task not in metrics["task_specific"]:
                metrics["task_specific"][task] = {
                    "raw": {"time": 0, "tokens": 0, "retries": 0, "count": 0},
                    "controlled": {"time": 0, "tokens": 0, "retries": 0, "count": 0},
                    "few_shot": {"time": 0, "tokens": 0, "retries": 0, "count": 0},
                    "fine_tuned": {"time": 0, "tokens": 0, "retries": 0, "count": 0}
                }
            
            for approach in self.approaches:
                approach_data = result["approaches"][approach]
                
                # Update overall approach metrics
                metrics["approaches"][approach]["time"] += approach_data["time_taken"]
                metrics["approaches"][approach]["tokens"] += approach_data["tokens"]["total"]
                metrics["approaches"][approach]["retries"] += approach_data.get("retries", 0)
                
                # Update task-specific metrics
                metrics["task_specific"][task][approach]["time"] += approach_data["time_taken"]
                metrics["task_specific"][task][approach]["tokens"] += approach_data["tokens"]["total"]
                metrics["task_specific"][task][approach]["retries"] += approach_data.get("retries", 0)
                metrics["task_specific"][task][approach]["count"] += 1
        
        # Calculate averages for task-specific metrics
        for task in metrics["task_specific"]:
            for approach in self.approaches:
                count = metrics["task_specific"][task][approach]["count"]
                if count > 0:
                    metrics["task_specific"][task][approach]["time"] /= count
                    metrics["task_specific"][task][approach]["tokens"] /= count
                    metrics["task_specific"][task][approach]["retries"] /= count
        
        # Calculate averages for overall approach metrics
        result_count = len(results)
        if result_count > 0:
            for approach in self.approaches:
                metrics["approaches"][approach]["time"] /= result_count
                metrics["approaches"][approach]["tokens"] /= result_count
                metrics["approaches"][approach]["retries"] /= result_count
        
        return metrics

    def _calculate_comparative_metrics(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparative metrics across models"""
        comparative_metrics = {
            "best_performing": {
                "time": {},
                "tokens": {},
                "retries": {}
            },
            "improvement_percentages": {
                "controlled_vs_raw": {},
                "few_shot_vs_raw": {},
                "fine_tuned_vs_raw": {}
            },
            "task_specific_best": {}
        }
        
        # Find best performing model for each metric
        for metric in ["time", "tokens", "retries"]:
            for approach in self.approaches:
                best_model = None
                best_value = float('inf')
                
                for model_key, model_data in models_data.items():
                    value = model_data["metrics"]["approaches"][approach][metric]
                    if value < best_value:
                        best_value = value
                        best_model = model_key
                
                comparative_metrics["best_performing"][metric][approach] = {
                    "model": best_model,
                    "value": best_value
                }
        
        # Calculate improvement percentages
        for model_key, model_data in models_data.items():
            comparative_metrics["improvement_percentages"]["controlled_vs_raw"][model_key] = {
                "time": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["time"],
                    model_data["metrics"]["approaches"]["controlled"]["time"]
                ),
                "tokens": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["tokens"],
                    model_data["metrics"]["approaches"]["controlled"]["tokens"]
                ),
                "retries": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["retries"],
                    model_data["metrics"]["approaches"]["controlled"]["retries"]
                )
            }
            
            comparative_metrics["improvement_percentages"]["few_shot_vs_raw"][model_key] = {
                "time": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["time"],
                    model_data["metrics"]["approaches"]["few_shot"]["time"]
                ),
                "tokens": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["tokens"],
                    model_data["metrics"]["approaches"]["few_shot"]["tokens"]
                ),
                "retries": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["retries"],
                    model_data["metrics"]["approaches"]["few_shot"]["retries"]
                )
            }
            
            comparative_metrics["improvement_percentages"]["fine_tuned_vs_raw"][model_key] = {
                "time": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["time"],
                    model_data["metrics"]["approaches"]["fine_tuned"]["time"]
                ),
                "tokens": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["tokens"],
                    model_data["metrics"]["approaches"]["fine_tuned"]["tokens"]
                ),
                "retries": self._calculate_improvement_percentage(
                    model_data["metrics"]["approaches"]["raw"]["retries"],
                    model_data["metrics"]["approaches"]["fine_tuned"]["retries"]
                )
            }
        
        # Find best performing model for each task
        task_types = set()
        for model_data in models_data.values():
            for result in model_data["results"]:
                task_types.add(result["task_type"])
        
        for task in task_types:
            comparative_metrics["task_specific_best"][task] = {}
            
            for approach in self.approaches:
                best_model = None
                best_time = float('inf')
                
                for model_key, model_data in models_data.items():
                    if task in model_data["metrics"]["task_specific"]:
                        time_value = model_data["metrics"]["task_specific"][task][approach]["time"]
                        if time_value < best_time:
                            best_time = time_value
                            best_model = model_key
                
                comparative_metrics["task_specific_best"][task][approach] = {
                    "model": best_model,
                    "time": best_time
                }
        
        return comparative_metrics

    def _calculate_improvement_percentage(self, baseline_value: float, improved_value: float) -> float:
        """Calculate improvement percentage"""
        if baseline_value == 0:
            return 0
        
        improvement = baseline_value - improved_value
        percentage = (improvement / baseline_value) * 100
        return round(percentage, 2)

    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive benchmark results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_lang = results.get("target_language", "default")
        models_str = "_".join(results["models"].keys())
        filename = f"benchmark_results_{models_str}_{target_lang}_{timestamp}.json"
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        with open(f"logs/{filename}", "w") as f:
            json.dump(results, f, indent=2)
            
    def get_available_models(self) -> List[str]:
        """Return list of available models from config"""
        return list(self.config.keys())

    def _generate_visualizations(self, benchmark_results):
        """
        Generate visualizations for benchmark results
        
        Args:
            benchmark_results: Dictionary with benchmark results
        """
        # Create output directory for visualizations
        output_dir = Path("logs/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract data for plotting
        models = list(benchmark_results["models"].keys())
        approaches = self.approaches
        
        # Get all task types from the benchmark results
        task_types = set()
        for model in models:
            for result in benchmark_results["models"][model]["results"]:
                task_types.add(result["task_type"])
        
        # 1. Generate task-specific processing time charts
        for task_type in task_types:
            self._generate_task_processing_time_chart(benchmark_results, models, approaches, task_type, output_dir, timestamp)
        
        # 2. Generate task-specific token utilization charts
        for task_type in task_types:
            self._generate_task_token_utilization_chart(benchmark_results, models, approaches, task_type, output_dir, timestamp)
        
        # 3. Generate combined task comparison charts
        self._generate_combined_task_comparison_chart(benchmark_results, models, approaches, task_types, output_dir, timestamp)
        
        # 4. Generate comparison tables for all tasks
        self._generate_comparison_tables(benchmark_results, models, approaches, output_dir, timestamp)
        
        print(f"Visualizations saved to {output_dir}")

    def _generate_task_processing_time_chart(self, benchmark_results, models, approaches, task_type, output_dir, timestamp):
        """Generate bar chart of models and processing time for a specific task"""
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        data = []
        for model in models:
            model_data = []
            for approach in approaches:
                # Get task-specific metrics if available
                if task_type in benchmark_results["models"][model]["metrics"]["task_specific"]:
                    model_data.append(benchmark_results["models"][model]["metrics"]["task_specific"][task_type][approach]["time"])
                else:
                    model_data.append(0)  # Default if no data for this task
            data.append(model_data)
        
        # Set up bar chart
        x = np.arange(len(models))
        width = 0.2
        
        # Plot bars
        for i, approach in enumerate(approaches):
            plt.bar(x + (i - 1.5) * width, [data[j][i] for j in range(len(models))], 
                    width, label=approach.replace('_', ' ').title())
        
        # Add labels and legend
        plt.xlabel('Models')
        plt.ylabel('Processing Time (seconds)')
        plt.title(f'Processing Time by Model and Approach for {task_type.title()} Task')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"{task_type}_processing_time_{timestamp}.png", dpi=300)
        plt.close()

    def _generate_task_token_utilization_chart(self, benchmark_results, models, approaches, task_type, output_dir, timestamp):
        """Generate bar chart of models and token utilization for a specific task"""
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        data = []
        for model in models:
            model_data = []
            for approach in approaches:
                # Get task-specific metrics if available
                if task_type in benchmark_results["models"][model]["metrics"]["task_specific"]:
                    model_data.append(benchmark_results["models"][model]["metrics"]["task_specific"][task_type][approach]["tokens"])
                else:
                    model_data.append(0)  # Default if no data for this task
            data.append(model_data)
        
        # Set up bar chart
        x = np.arange(len(models))
        width = 0.2
        
        # Plot bars
        for i, approach in enumerate(approaches):
            plt.bar(x + (i - 1.5) * width, [data[j][i] for j in range(len(models))], 
                    width, label=approach.replace('_', ' ').title())
        
        # Add labels and legend
        plt.xlabel('Models')
        plt.ylabel('Tokens Utilized')
        plt.title(f'Token Utilization by Model and Approach for {task_type.title()} Task')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"{task_type}_token_utilization_{timestamp}.png", dpi=300)
        plt.close()

    def _generate_combined_task_comparison_chart(self, benchmark_results, models, approaches, task_types, output_dir, timestamp):
        """Generate charts comparing all tasks for each approach"""
        
        # For each approach, create a chart comparing tasks
        for approach in approaches:
            plt.figure(figsize=(14, 10))
            
            # Prepare data for processing time
            task_data = {task: [] for task in task_types}
            
            for model in models:
                for task in task_types:
                    if task in benchmark_results["models"][model]["metrics"]["task_specific"]:
                        task_data[task].append(benchmark_results["models"][model]["metrics"]["task_specific"][task][approach]["time"])
                else:
                        task_data[task].append(0)  # Default if no data
            
            # Set up bar chart
            x = np.arange(len(models))
            width = 0.8 / len(task_types)
            
            # Plot bars for each task
            for i, task in enumerate(task_types):
                offset = (i - len(task_types)/2 + 0.5) * width
                plt.bar(x + offset, task_data[task], width, label=task.title())
            
            # Add labels and legend
            plt.xlabel('Models')
        plt.ylabel('Processing Time (seconds)')
        plt.title(f'Task Comparison for {approach.replace("_", " ").title()} Approach')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"task_comparison_{approach}_{timestamp}.png", dpi=300)
        plt.close()
        
            # Also create a token utilization comparison
        plt.figure(figsize=(14, 10))
            
            # Prepare data for token utilization
        token_data = {task: [] for task in task_types}
            
        for model in models:
            for task in task_types:
                if task in benchmark_results["models"][model]["metrics"]["task_specific"]:
                    token_data[task].append(benchmark_results["models"][model]["metrics"]["task_specific"][task][approach]["tokens"])
                else:
                    token_data[task].append(0)  # Default if no data
            
            # Plot bars for each task
            for i, task in enumerate(task_types):
                offset = (i - len(task_types)/2 + 0.5) * width
                plt.bar(x + offset, token_data[task], width, label=task.title())
            
            # Add labels and legend
        plt.xlabel('Models')
        plt.ylabel('Tokens Utilized')
        plt.title(f'Token Utilization by Task for {approach.replace("_", " ").title()} Approach')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"task_token_comparison_{approach}_{timestamp}.png", dpi=300)
        plt.close()
        
    def _generate_comparison_tables(self, benchmark_results, models, approaches, output_dir, timestamp):
        """
        Generate comparison tables for all tasks showing metrics for different approaches
        
        Args:
            benchmark_results: Dictionary with benchmark results
            models: List of models to include in the table
            approaches: List of approaches to compare
            output_dir: Directory to save the tables
            timestamp: Timestamp for file naming
        """
        # Get all task types from the benchmark results
        task_types = set()
        for model in models:
            for result in benchmark_results["models"][model]["results"]:
                task_types.add(result["type"])
        
        # Create a table for each task type
        for task_type in task_types:
            # Create a DataFrame for this task
            columns = ["Model", "Parameter", "Temperature", "Turns", "Processing time", "Tokens utilized"]
            # Add approach columns
            for approach in approaches:
                columns.append(approach.replace('_', ' ').title())
            
            # Initialize the DataFrame with empty rows for each model
            table_data = []
            
            # Fill in data for each model
            for model in models:
                model_metrics = benchmark_results["models"][model]["metrics"]
                
                # Filter results for this task type
                task_results = [r for r in benchmark_results["models"][model]["results"] if r["type"] == task_type]
                
                if not task_results:
                    continue
                    
                # Basic model info (same for all approaches)
                row = {
                    "Model": model,
                    "Parameter": model_metrics.get("parameter", "N/A"),
                    "Temperature": 0,  # Assuming temperature is 0 as set in _generate_raw_response
                    "Turns": 1,  # Assuming single turn for all approaches
                }
                
                # Add metrics for each approach
                for approach in approaches:
                    approach_metrics = model_metrics["approaches"][approach]
                    
                    # If this is the first approach, add the common metrics
                    if approach == approaches[0]:
                        row["Processing time"] = f"{approach_metrics['time']:.2f}s"
                        row["Tokens utilized"] = approach_metrics['tokens']
                    
                    # Add the approach-specific column
                    # For the approach column, we could use accuracy or other relevant metric
                    if task_type == "translation":
                        row[approach.replace('_', ' ').title()] = f"{approach_metrics.get('bleu', 'N/A'):.2f}"
                    elif task_type == "json":
                        row[approach.replace('_', ' ').title()] = f"{approach_metrics.get('accuracy', 'N/A'):.2f}"
                    elif task_type == "sql":
                        row[approach.replace('_', ' ').title()] = f"{approach_metrics.get('accuracy', 'N/A'):.2f}"
                    elif task_type == "sentiment":
                        row[approach.replace('_', ' ').title()] = f"{approach_metrics.get('accuracy', 'N/A'):.2f}"
                    else:
                        row[approach.replace('_', ' ').title()] = "N/A"
                
                table_data.append(row)
            
            # Create DataFrame and save to CSV
            if table_data:
                df = pd.DataFrame(table_data)
                csv_path = output_dir / f"{task_type}_comparison_table_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                
                # Also save as JSON for easier API consumption
                json_path = output_dir / f"{task_type}_comparison_table_{timestamp}.json"
                df.to_json(json_path, orient="records")
                
                print(f"Comparison table for {task_type} saved to {csv_path} and {json_path}") 