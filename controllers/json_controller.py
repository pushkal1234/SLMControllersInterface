import re
from datetime import datetime
from typing import Dict, Any, Tuple
import json
from jsonschema import validate
import jsonschema

class JSONController:
    def __init__(self, model):
        self.model = model
        self.total_output_list = []

    def _extract_amount(self, amount_str: str) -> int:
        # Remove ₹ symbol and convert to integer
        return int(amount_str.replace('₹', '').replace(',', ''))

    def _extract_user_data(self, text: str) -> list[dict]:
        # Extract user information using regex patterns
        users = []
        
        # Pattern to match user data blocks
        user_pattern = r'([A-Za-z\s]+) earns ₹([\d,]+) .+?(?=\. [A-Za-z]|$)'
        matches = re.finditer(user_pattern, text)

        for match in matches:
            user_block = match.group(0)
            name = match.group(1).strip()
            
            # Extract financial details
            salary = self._extract_amount(re.search(r'earns ₹([\d,]+)', user_block).group(1))
            expenses = self._extract_amount(re.search(r'spends ₹([\d,]+)', user_block).group(1))
            
            # Initialize dictionaries for financial categories
            investments = {}
            debts = {}
            loans = {}
            savings = 0

            # Extract investments
            investment_patterns = [
                (r'invested ₹([\d,]+) in ([^,\.]+)', 'investments'),
                (r'savings of ₹([\d,]+) in ([^,\.]+)', 'savings'),
                (r'([\w\s]+) loan of ₹([\d,]+)', 'loans'),
                (r'([\w\s]+) debt of ₹([\d,]+)', 'debts')
            ]

            for pattern, category in investment_patterns:
                for investment_match in re.finditer(pattern, user_block):
                    amount = self._extract_amount(investment_match.group(1))
                    item = investment_match.group(2).strip().lower()
                    
                    if category == 'investments':
                        investments[item] = amount
                    elif category == 'savings':
                        savings = amount
                    elif category == 'loans':
                        loans[item] = amount
                    elif category == 'debts':
                        debts[item] = amount

            users.append({
                "name": name,
                "salary": salary,
                "expenses": expenses,
                "investments": investments,
                "debts": debts,
                "loans": loans,
                "savings": savings
            })

        return users

    def process_financial_data(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        try:
            # Extract date, text and schema from input
            date_str = input_data['date']
            text = input_data['text']
            schema = input_data['schema']
            
            # Process the text to extract user data
            users_data = self._extract_user_data(text)
            
            # Create the response structure
            response_data = {
                "date": date_str,
                "users": users_data
            }
            
            # Validate against the provided schema
            try:
                validate(instance=response_data, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                return {
                    "error": f"Schema validation failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "code": 400
                }, 0
            
            # Calculate tokens
            input_tokens = len(text.split())
            output_tokens = len(str(response_data).split())
            total_tokens = input_tokens + output_tokens

            return response_data, total_tokens

        except Exception as e:
            error_response = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "code": 500
            }
            return error_response, 0