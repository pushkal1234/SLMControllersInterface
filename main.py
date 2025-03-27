from flask import Flask, request, jsonify, send_file
import pandas as pd
from datetime import datetime
import threading
from controllers.sentiment_controller import SentimentController
from controllers.translation_controller import TranslationController
from controllers.poem_controller import PoemController
from controllers.json_controller import JSONController
from controllers.sql_controller import SQLController
from utils import load_config, class_factory
from controllers.benchmark_controller import BenchmarkController
from controllers.SQL_benchmark_controller import SQLBenchmarkController
import os

app = Flask(__name__)

# Load configuration
CONFIG = load_config()

logs_csv = pd.read_csv('logs/input_output.csv', index_col=0)
lock = threading.Lock()

# Create a global dictionary to store benchmark jobs
benchmark_jobs = {}

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Create input data dictionary
        input_data = {
            'text': data['text'],
            'target_language': data.get('target_language', 'german')  # Default to German
        }
        
        model = data.get('model', 'phi3')  # Default to phi3 if not specified
        controller_name = 'TranslationController'
        
        translated_text, total_token = generate_response(input_data, model, controller_name)
        
        if isinstance(translated_text, str) and not translated_text.startswith('Error'):
            return jsonify({
                'translation': translated_text,
                'target_language': input_data['target_language'],
                'tokens_used': total_token,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'error': translated_text,
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data['text']
        model = data.get('model', 'phi3')  # Use a default model if not provided
        controller_name = 'SentimentController'
        sentiment_result, total_token = generate_response(text, model, controller_name)
        return jsonify({'response': sentiment_result, "total_token" : total_token})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/poem', methods=['POST'])
def generate_poem():
    try:
        data = request.get_json()
        text = data['text']      
        model = data.get('model', 'phi3')  # Use a default model if not provided
        controller_name = 'PoemController'
        poem_result, total_token = generate_response(text, model, controller_name)
        return jsonify({'response': poem_result, "total_token" : total_token})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/process-json', methods=['POST'])
def process_json():
    try:
        data = request.get_json()
        
        # Validate required fields in request
        required_fields = ['text', 'date', 'schema']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 400

        model_name = data.get('model', 'phi3')
        
        json_output, total_tokens = generate_response(
            data,  # Passing the entire data object including schema
            model_name,
            "JSONController"
        )

        if json_output:
            if "error" in json_output:
                return jsonify(json_output), json_output.get("code", 500)
                
            response = {
                'result': json_output,
                'tokens_used': total_tokens,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(response), 200
        else:
            return jsonify({
                'error': 'Failed to process data',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/generate-sql', methods=['POST'])
def generate_sql():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required fields',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 400

        model = data.get('model', 'phi3')
        controller_name = 'SQLController'
        
        sql_output, total_tokens = generate_response(data, model, controller_name)
        
        if isinstance(sql_output, dict) and sql_output.get('status') == 'success':
            return jsonify({
                'result': sql_output,
                'tokens_used': total_tokens,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify(sql_output), sql_output.get('code', 500)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    data = request.json
    
    # Create controllers for each task type
    translation_controller = TranslationController(data.get('model', 'phi3'))
    sql_controller = SQLController(data.get('model', 'phi3'))
    json_controller = JSONController(data.get('model', 'phi3'))
    sentiment_controller = SentimentController(data.get('model', 'phi3'))
    
    # Prepare test cases with controllers
    test_cases = []
    for test in data.get('test_cases', []):
        test_case = test.copy()
        if test['type'] == 'translation':
            test_case['controller'] = translation_controller
        elif test['type'] == 'sql':
            test_case['controller'] = sql_controller
        elif test['type'] == 'json':
            test_case['controller'] = json_controller
        elif test['type'] == 'sentiment':
            test_case['controller'] = sentiment_controller
        test_cases.append(test_case)
    
    benchmark_controller = BenchmarkController()
    
    results = benchmark_controller.run_comprehensive_benchmark(
        test_cases=test_cases,
        models=data.get('models', ['phi3']),
        target_language=data.get('target_language', 'german')
    )
    
    return jsonify(results)

@app.route('/sql-benchmark', methods=['POST'])
def sql_benchmark():
    """Start a SQL benchmark job"""
    try:
        data = request.json
        
        # Extract parameters with defaults
        models = data.get('models', ['phi3'])
        num_samples = data.get('num_samples', 20)
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize the SQL benchmark controller
        controller = SQLBenchmarkController()
        
        # Start benchmark in a background thread to avoid blocking
        def run_benchmark():
            try:
                results = controller.run_sql_benchmark(models=models, num_samples=num_samples)
                benchmark_jobs[job_id]['status'] = 'completed'
                benchmark_jobs[job_id]['results'] = results
                benchmark_jobs[job_id]['timestamp'] = datetime.now().isoformat()
            except Exception as e:
                benchmark_jobs[job_id]['status'] = 'failed'
                benchmark_jobs[job_id]['error'] = str(e)
                print(f"Benchmark error: {str(e)}")
        
        # Store job details
        benchmark_jobs[job_id] = {
            'status': 'running',
            'models': models,
            'num_samples': num_samples,
            'start_time': datetime.now().isoformat()
        }
        
        # Start the benchmark thread
        thread = threading.Thread(target=run_benchmark)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'running',
            'message': f'Benchmark started for models: {models}',
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/sql-benchmark/status/<job_id>', methods=['GET'])
def sql_benchmark_status(job_id):
    """Get the status of a benchmark job"""
    if job_id not in benchmark_jobs:
        return jsonify({
            'error': 'Job not found',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    job = benchmark_jobs[job_id]
    return jsonify(job), 200

@app.route('/sql-benchmark/results/<job_id>', methods=['GET'])
def sql_benchmark_results(job_id):
    """Get the results of a completed benchmark job"""
    if job_id not in benchmark_jobs:
        return jsonify({
            'error': 'Job not found',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    job = benchmark_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Benchmark not yet completed',
            'status': job['status'],
            'timestamp': datetime.now().isoformat()
        }), 400
    
    # Return tabular results as JSON
    try:
        # Find the latest comparison table CSV file
        log_dir = os.path.join('logs', 'sql_benchmark')
        files = [f for f in os.listdir(log_dir) if f.startswith('comparison_table_')]
        if not files:
            return jsonify({
                'error': 'No comparison table found',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        table_path = os.path.join(log_dir, latest_file)
        
        # Load and return the table as JSON
        table_df = pd.read_csv(table_path)
        return jsonify({
            'table': table_df.to_dict(orient='records'),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/sql-benchmark/visualizations/<job_id>/<viz_type>/<model>', methods=['GET'])
def sql_benchmark_visualizations(job_id, viz_type, model):
    """Get visualizations from a completed benchmark job"""
    if job_id not in benchmark_jobs:
        return jsonify({
            'error': 'Job not found',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    job = benchmark_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Benchmark not yet completed',
            'status': job['status'],
            'timestamp': datetime.now().isoformat()
        }), 400
    
    # Return the requested visualization
    try:
        # Find the latest visualization file matching the request
        viz_dir = os.path.join('logs', 'sql_benchmark', 'visualizations')
        
        # Map viz_type to filename pattern
        viz_patterns = {
            'accuracy': f'accuracy_by_complexity_{model}_',
            'tokens': f'token_efficiency_{model}_',
            'time': f'processing_time_{model}_',
            'errors': f'error_analysis_{model}_',
            'radar': f'radar_chart_{model}_'
        }
        
        if viz_type not in viz_patterns:
            return jsonify({
                'error': f'Invalid visualization type: {viz_type}',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        pattern = viz_patterns[viz_type]
        files = [f for f in os.listdir(viz_dir) if f.startswith(pattern)]
        
        if not files:
            return jsonify({
                'error': f'No {viz_type} visualization found for model {model}',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(viz_dir, x)))
        viz_path = os.path.join(viz_dir, latest_file)
        
        # Return the visualization file
        return send_file(viz_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/sql-benchmark/quick-run', methods=['POST'])
def sql_benchmark_quick_run():
    """Run a limited SQL benchmark for quick results"""
    try:
        data = request.json
        
        # Extract parameters with more limited defaults for faster execution
        models = data.get('models', ['phi3'])
        num_samples = data.get('num_samples', 5)  # Limited samples for speed
        complexity = data.get('complexity', 'simple')  # Focus on simple queries for speed
        
        # Initialize the SQL benchmark controller
        controller = SQLBenchmarkController()
        
        # Modify the benchmark to run only on specified complexity
        controller.complexity_levels = [complexity]
        
        # Run benchmark directly (blocking call for quick results)
        results = controller.run_sql_benchmark(models=models, num_samples=num_samples)
        
        # Find the latest comparison table CSV file
        log_dir = os.path.join('logs', 'sql_benchmark')
        files = [f for f in os.listdir(log_dir) if f.startswith('comparison_table_')]
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        table_path = os.path.join(log_dir, latest_file)
        
        # Load the table
        table_df = pd.read_csv(table_path)
        
        return jsonify({
            'table': table_df.to_dict(orient='records'),
            'message': 'Quick benchmark completed',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/sql-benchmark/analysis/<job_id>/<analysis_type>', methods=['GET'])
def get_benchmark_analysis(job_id, analysis_type):
    """Get analysis reports for a completed benchmark job"""
    if job_id not in benchmark_jobs:
        return jsonify({
            'error': 'Job not found',
            'status': 'error'
        }), 404
    
    job = benchmark_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Benchmark not yet completed',
            'status': job['status']
        }), 400
    
    try:
        # Initialize the controller
        controller = SQLBenchmarkController()
        
        # Set results from the job
        controller.results = job['results']
        
        # Generate the requested analysis
        if analysis_type == 'task_type':
            report = controller.analyze_by_task_type()
        elif analysis_type == 'domain':
            report = controller.analyze_by_domain()
        elif analysis_type == 'complexity':
            report = controller.analyze_by_original_complexity()
        else:
            return jsonify({
                'error': f'Invalid analysis type: {analysis_type}',
                'status': 'error'
            }), 400
        
        # Return the report
        return jsonify({
            'report': report,
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def generate_response(text, model, controller_name):
    try:
        model = CONFIG[model]
        controller = class_factory(controller_name, model)
        current_time = datetime.now()

        if isinstance(controller, TranslationController):
            translated_text, total_token = controller.generate_translation(text)
            new_row = [current_time, text, translated_text]
            return translated_text, total_token
        
        elif isinstance(controller, SentimentController):
            sentiment_result, total_token = controller.generate_sentiment(text)
            new_row = [current_time, text, sentiment_result]
            return sentiment_result, total_token
        
        elif isinstance(controller, PoemController):
            poem_result, total_token = controller.generate_poem(text)
            new_row = [current_time, text, poem_result]
            return poem_result, total_token
            
        elif isinstance(controller, JSONController):
            json_output, total_token = controller.process_financial_data(text)
            new_row = [current_time, str(text), str(json_output)]
            return json_output, total_token

        elif isinstance(controller, SQLController):
            sql_output, total_token = controller.generate_sql_query(text)
            new_row = [current_time, str(text), str(sql_output)]
            return sql_output, total_token

        else:
            raise ValueError(f"Unsupported controller type: {controller_name}")

    except Exception as e:
        print(f"\033[91mAn error occurred: {e}\033[0m")  # Print in red
        return None, 0  # Return tuple with None and 0 tokens
    
    finally:
        # Use a lock to ensure thread safety
        with lock:
            try:
                logs_csv.loc[len(logs_csv)] = new_row
                logs_csv.to_csv('logs/input_output.csv')
                print(f"\033[92mLog saved successfully.\033[0m")  
                
            except Exception as save_error:
                print(f"\033[91mFailed to save log: {save_error}\033[0m")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000)

