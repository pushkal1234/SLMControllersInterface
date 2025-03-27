import re
import tiktoken
from langchain_community.llms import Ollama


class SentimentController:
    def __init__(self, model):
        self.model = model
        self.total_output_list = []

    def filter_sentiment(self, output, input_text):
        output_lower = output.lower()
        sentiments = ['positive', 'negative', 'neutral']

        output_sentiment = ''

        for sentiment in sentiments:
            if output_lower.find(sentiment) != -1:
                output_sentiment = sentiment
                return output_sentiment
        return None

    def get_sentiment(self, input_text):
        llm = Ollama(model=self.model, temperature=0)
        initial_prompt = "sentiment of this sentence is"
        final_prompt = f"{initial_prompt} '{input_text}'"

        print("final_prompt", final_prompt)

        output = llm.invoke(final_prompt, stop=['.'])
        print("output:", output)
        output_sentiment = self.filter_sentiment(output, input_text)
        
        print("output_sentiment:", output_sentiment)

        if output_sentiment is None:
            counter = 0 
            while output_sentiment is None:
                output = llm.invoke(final_prompt + ' in positive, negative and neutral is', stop=['.'])
                self.total_output_list.append(output)
                print(output)
                output_sentiment = self.filter_sentiment(output, input_text)
                counter = counter + 1
                if counter > 5:
                    break
            
            counter = 0
            while output_sentiment is None:
                output = llm.invoke(final_prompt)
                print(output)
                output_sentiment = self.filter_sentiment(output, input_text)
                counter = counter + 1
                if counter > 5:
                    break
        
        if output_sentiment is None:
            return "neutral"
        
        else:
            return output_sentiment

    def input_preprocess(self, input_text):
        sentence_list = re.split(r'(?<=[.!?]) +', input_text)
        return sentence_list

    def generate_sentiment(self, input_text):
        sentence_list = self.input_preprocess(input_text)
        print(sentence_list)

        sentiment_dict = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

        for sentence in sentence_list:
            sentiment_type = self.get_sentiment(sentence)
            if sentiment_type:
                sentiment_dict[sentiment_type] += 1
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        query_token = len(encoding.encode(''.join(sentence_list)))
        response_token = len(encoding.encode(''.join(sentiment_dict)))
        total_token = query_token + response_token

        return sentiment_dict, total_token


