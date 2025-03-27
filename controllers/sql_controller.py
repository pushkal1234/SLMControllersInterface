import re
import tiktoken
from langchain_community.llms import Ollama
from typing import Dict, Tuple, Any
from datetime import datetime  # Add this import for the error handler

class SQLController:
    def __init__(self, model):
        self.model = model
        self.total_output_list = []
        self.supported_operations = {
            "select": "Generate a SELECT query to retrieve data",
            "insert": "Generate an INSERT query to add data",
            "update": "Generate an UPDATE query to modify data",
            "delete": "Generate a DELETE query to remove data",
            "create": "Generate a CREATE TABLE query",
            "alter": "Generate an ALTER TABLE query"
        }

    def validate_sql(self, query: str) -> bool:
        """Basic SQL validation"""
        # Check for basic SQL syntax
        basic_checks = [
            query.strip().endswith(';'),
            any(op.upper() in query.upper() for op in self.supported_operations.keys()),
            '(' in query if 'CREATE TABLE' in query.upper() else True,
            'WHERE' in query.upper() if any(op in query.upper() for op in ['UPDATE', 'DELETE']) else True
        ]
        return all(basic_checks)
    
    def clean_sql_output(self, sql_query: str) -> str:
        """
        Clean the SQL query output by removing markdown code block formatting
        and any other unnecessary formatting.
        """
        # Remove markdown code block formatting
        sql_query = re.sub(r'^```sql\n', '', sql_query)
        sql_query = re.sub(r'^```\n', '', sql_query)
        sql_query = re.sub(r'\n```$', '', sql_query)
        sql_query = re.sub(r'\n', ' ', sql_query)
        
        # Remove any leading/trailing whitespace
        sql_query = sql_query.strip()
        
        return sql_query

    def generate_sql_query(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        try:
            text = input_data['text']  # Natural language description
            operation = input_data.get('operation', 'select').lower()
            table_info = input_data.get('table_info', '')  # Optional table structure info

            if operation not in self.supported_operations:
                raise ValueError(f"Unsupported operation: {operation}. Supported operations: {list(self.supported_operations.keys())}")

            # Construct the prompt
            prompt = f"""
            {self.supported_operations[operation]}:
            Table Information: {table_info}
            Requirements: {text}
            
            Generate only the SQL query without any explanation.
            Ensure the query:
            1. Ends with a semicolon
            2. Uses proper SQL syntax
            3. Includes appropriate WHERE clauses for UPDATE/DELETE
            4. Uses proper data types for CREATE TABLE
            """

            # Generate SQL query
            llm = Ollama(model=self.model, temperature=0)
            sql_query = llm.invoke(prompt).strip()

            # Validate the generated query
            if not self.validate_sql(sql_query):
                # Retry with more specific prompt
                retry_prompt = f"{prompt}\nPrevious attempt was invalid. Please ensure proper SQL syntax."
                sql_query = llm.invoke(retry_prompt).strip()

            # Clean the SQL query output
            sql_query = self.clean_sql_output(sql_query)
            
            self.total_output_list.append(sql_query)

            # Calculate tokens
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            query_token = len(encoding.encode(text))
            response_token = len(encoding.encode(sql_query))
            total_token = query_token + response_token

            response_data = {
                "query": sql_query,
                "operation": operation,
                "status": "success"
            }

            return response_data, total_token

        except Exception as e:
            error_response = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "code": 500
            }
            return error_response, 0 