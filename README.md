# SLMControllersInterface

Designing a sophisticated system of specialised controllers to refine outputs from Small Language Models (SLMs). This approach is designed to improve efficiency and quality for specific task types. More precisely reducing tokens and processing time.

# Translation and Sentiment Analysis API

This project provides a Flask-based API for performing text translation and sentiment analysis using custom controllers. The API supports two main functionalities:
1. Translating text to German.
2. Analyzing the sentiment of a given text.

## Project Structure

```
project/
│
├── config.json
├── controllers/
│   ├── __init__.py
│   ├── translation_controller.py
│   └── sentiment_controller.py
├── utils.py
└── main.py
```

### Files Description
- **config.json:** Configuration file for the project.
- **controllers/translation_controller.py:** Contains the `TranslationController` class for text translation.
- **controllers/sentiment_controller.py:** Contains the `SentimentController` class for sentiment analysis.
- **controllers/__init__.py:** Initializes the controllers package.
- **utils.py:** Utility functions including `load_config` and `class_factory`.
- **main.py:** Flask application setup and endpoint definitions.

## Setup

- Install Ollama locally to run models like phi3, llama3.2
for installtion MAC users use link: https://ollama.com/download/mac

### Prerequisites

- Python 3.x
- pip (Python package installer)
- pip install langid
- pip install langchain-community
- pip install langchain-core

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Ensure the `config.json` file is correctly set up in the project root.

### Running the Flask API

Run the Flask application using the following command:

```sh
python main.py
```

The API will be available at `http://127.0.0.1:5000`.

## API Endpoints

### 1. Translate Text

- **URL:** `/translate`
- **Method:** `POST`
- **Description:** Translates the given text to German or Spanish
- **Request Body:**
  ```json
  {
    "text": "Text to be translated",
    "target_language": "german|spanish", // optional, defaults to "german"
    "model": "model_name"  // optional, default is 'phi3'
  }
  ```
- **Response:**
  ```json
  {
    "translation": "Translated text",
    "target_language": "german|spanish",
    "tokens_used": number,
    "status": "success",
    "timestamp": "ISO datetime"
  }
  ```

### 2. Analyze Sentiment

- **URL:** `/sentiment`
- **Method:** `POST`
- **Description:** Analyzes the sentiment of the given text.
- **Request Body:**
  ```json
  {
    "text": "Text for sentiment analysis",
    "model": "model_name"  // optional, default is 'phi3'
  }
  ```
- **Response:**
  ```json
  {
    "positive": count,
    "negative": count,
    "neutral": count
  }
  ```

### 3. Generate complex JSON using schema

- **URL:** `/process-json`
- **Method:** `POST`
- **Description:** Generates complex JSON based on schema provided.
- **Request Body:**
  ```json
  {
          "date": "2025-02-22",
          "text": "On 22nd February 2025, the financial summary for multiple users was generated. Rahul Sharma earns ₹12,00,000 annually, spends ₹6,00,000, and has invested ₹3,00,000 in mutual funds and ₹1,50,000 in stocks. He also has a credit card debt of ₹2,00,000. Priya Verma has an annual salary of ₹18,50,000 with expenses of ₹9,75,000. She has invested ₹2,50,000 in real estate and ₹1,00,000 in gold, and has a home loan of ₹5,00,000. Anil Mehta earns ₹9,50,000 per year and spends ₹4,00,000. He has a personal loan debt of ₹1,00,000 and a car loan of ₹3,50,000, along with savings of ₹1,75,000 in fixed deposits. Sunita Kapoor has a yearly salary of ₹22,00,000 and spends ₹10,00,000. She has invested ₹5,00,000 in government bonds, has business loan debt of ₹3,00,000, and has ₹3,00,000 in liquid savings. Vikram Singh earns ₹15,00,000, spends ₹8,00,000, and has invested ₹4,50,000 in ETFs. He has an education loan of ₹2,00,000 and a business loan of ₹4,50,000.",
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
          },
          "model": "llama2"
  }
  ```
- **Response:**
  ```json
  {
        "result": {
        "date": "2025-02-22",
        "users": [
            {
                "debts": {},
                "expenses": 600000,
                "investments": {
                    "mutual funds and ₹1": 300000
                },
                "loans": {},
                "name": "Rahul Sharma",
                "salary": 1200000,
                "savings": 0
            },
            {
                "debts": {},
                "expenses": 400000,
                "investments": {},
                "loans": {},
                "name": "Anil Mehta",
                "salary": 950000,
                "savings": 0
            },
            {
                "debts": {},
                "expenses": 800000,
                "investments": {
                    "etfs": 450000
                },
                "loans": {},
                "name": "Vikram Singh",
                "salary": 1500000,
                "savings": 0
            }
        ]
    },
    "status": "success",
    "timestamp": "2025-02-25T00:35:50.320605",
    "tokens_used": 208
  }
  ```

### 4. Generate SQL Queries

- **URL:** `/generate-sql`
- **Method:** `POST`
- **Description:** Generates SQL queries based on natural language description
- **Request Body:**
  ```json
  {
    "text": "Natural language description of the query",
    "operation": "select|insert|update|delete|create|alter",  // optional, defaults to "select"
    "table_info": "Description of table structure",  // optional
    "model": "model_name"  // optional, default is 'phi3'
  }
  ```
- **Response:**
  ```json
  {
    "result": {
      "query": "Generated SQL query;",
      "operation": "select|insert|update|delete|create|alter",
      "status": "success"
    },
    "tokens_used": number,
    "status": "success",
    "timestamp": "ISO datetime"
  }
  ```

### SQL Query Generation Example
```sh
curl -X POST http://127.0.0.1:5000/generate-sql \
-H "Content-Type: application/json" \
-d '{
    "text": "Find all users who have made purchases over $1000",
    "operation": "select",
    "table_info": "users(id, name, email), orders(id, user_id, amount, date)",
    "model": "phi3"
}'
```

Example Response:
```json
{
    "result": {
        "query": "SELECT u.name, u.email, o.amount, o.date FROM users u JOIN orders o ON u.id = o.user_id WHERE o.amount > 1000;",
        "operation": "select",
        "status": "success"
    },
    "tokens_used": 42,
    "status": "success",
    "timestamp": "2024-02-25T12:34:56.789Z"
}
```

Supported SQL Operations:
- **SELECT**: Retrieve data from tables
- **INSERT**: Add new records
- **UPDATE**: Modify existing records
- **DELETE**: Remove records
- **CREATE**: Create new tables
- **ALTER**: Modify table structure

## Example Usage

### Translation Example
```sh
curl -X POST http://127.0.0.1:5000/translate -H "Content-Type: application/json" -d '{"text": "I am good", "model": "phi3"}'
```

### Spanish Translation Example

```sh
curl --location --request POST 'http://127.0.0.1:5000/translate' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "Quantum machine learning uses quantum arithms on quantum computers to speed up machine learning",
    "target_language": "spanish",
    "model": "phi3"
}'
```

### Sentiment Analysis Example
```sh
curl -X POST http://127.0.0.1:5000/sentiment -H "Content-Type: application/json" -d '{"text": "I am very happy. I am very sad", "model": "phi3"}'
```

### Complex JSON Generation Example
```sh
curl --location --request POST 'http://127.0.0.1:5000/process-json' \
--header 'Content-Type: application/json' \
--data-raw '{
          "date": "2025-02-22",
          "text": "On 22nd February 2025, the financial summary for multiple users was generated. Rahul Sharma earns ₹12,00,000 annually, spends ₹6,00,000, and has invested ₹3,00,000 in mutual funds and ₹1,50,000 in stocks. He also has a credit card debt of ₹2,00,000. Priya Verma has an annual salary of ₹18,50,000 with expenses of ₹9,75,000. She has invested ₹2,50,000 in real estate and ₹1,00,000 in gold, and has a home loan of ₹5,00,000. Anil Mehta earns ₹9,50,000 per year and spends ₹4,00,000. He has a personal loan debt of ₹1,00,000 and a car loan of ₹3,50,000, along with savings of ₹1,75,000 in fixed deposits. Sunita Kapoor has a yearly salary of ₹22,00,000 and spends ₹10,00,000. She has invested ₹5,00,000 in government bonds, has business loan debt of ₹3,00,000, and has ₹3,00,000 in liquid savings. Vikram Singh earns ₹15,00,000, spends ₹8,00,000, and has invested ₹4,50,000 in ETFs. He has an education loan of ₹2,00,000 and a business loan of ₹4,50,000.",
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
          },
          "model": "llama2"  
        }'
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [LangID](https://github.com/saffsd/langid.py)
- [Langchain Community](https://github.com/langchain-community)
