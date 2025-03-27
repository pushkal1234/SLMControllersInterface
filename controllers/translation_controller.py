import re
import nltk
from nltk.corpus import words
import tiktoken
from langchain.llms import Ollama

class TranslationController:
    def __init__(self, model):
        self.model = model
        nltk.download('words')
        self.english_words = set(words.words())
        self.total_output_list = []
        self.supported_languages = {
            "german": {
                "prompt": "Translate the following English text to German. Return ONLY the translated text without any explanations, notes, or the original text: "
            },
            "spanish": {
                "prompt": "Translate the following English text to Spanish. Return ONLY the translated text without any explanations, notes, or the original text: "
            }
        }

    def remove_extra(self, text):
        chars_to_remove = ['"', "'", ':']
        for char in chars_to_remove:
            text = text.replace(char, '')
        return text
    
    def clean_translation_output(self, translation: str, target_language: str) -> str:
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

    def check_multiline_and_german(self, output_translation):
        """
        Checks english content at line level
        """
        output_multiline_length = len(output_translation.split('\n'))
        if output_multiline_length > 1:
            multiline_split = output_translation.split('\n')
            for line in multiline_split:
                language, _ = langid.classify(line)
                if language == 'de':
                    cleaned_line = self.remove_extra(line)
                    return cleaned_line
        else:
            output_translation = self.remove_extra(output_translation)
            return output_translation
    
    def check_english_content(self, output_translation):
        """
        Checks english content at word level
        """
        output_translation = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', output_translation)
        print("I am here")
        words = output_translation.split()
        print(words)
        german_filtered_words = [word for word in words if word.lower() not in self.english_words]
        return ' '.join(german_filtered_words)

    def input_preprocess(self, input_text):
        sentence_list = re.split(r'(?<=[.!?]) +', input_text)
        return sentence_list

    def get_translation_from_LLM(self, sentence: str, target_language: str = "german"):
        """
        Get translation from LLM for a given sentence
        Args:
            sentence: Sentence to translate
            target_language: Language to translate to (german or spanish)
        """
        if target_language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language: {target_language}. Supported languages: {list(self.supported_languages.keys())}")

        language_config = self.supported_languages[target_language.lower()]
        prompt = f"{language_config['prompt']}{sentence}"
        
        llm = Ollama(model=self.model, temperature=0)
        translation = llm.invoke(prompt)
        
        # Clean the translation output
        translation = self.clean_translation_output(translation, target_language)
        
        self.total_output_list.append(translation)
        return translation

    def generate_translation(self, input_data: dict) -> tuple:
        """
        Generate translation based on input text and target language
        Args:
            input_data: Dictionary containing 'text' and 'target_language'
        Returns:
            tuple: (translated_text, total_tokens)
        """
        try:
            input_text = input_data['text']
            target_language = input_data.get('target_language', 'german').lower()  # Default to German if not specified
            
            if target_language not in self.supported_languages:
                raise ValueError(f"Unsupported language: {target_language}. Supported languages: {list(self.supported_languages.keys())}")

            sentence_list = self.input_preprocess(input_text)
            print(f"Translating to {target_language}...")
            print(sentence_list)

            translation_list = []
            for index in range(len(sentence_list)):
                print("-----------")
                translation = self.get_translation_from_LLM(sentence_list[index], target_language)
                print(translation)
                translation_list.append(translation)
                if not translation.endswith(('.', '!', '?', '...', '"', "'", ')', ';', ':')):
                    translation_list.append('.')
            
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            query_token = len(encoding.encode(''.join(sentence_list)))
            response_token = len(encoding.encode(''.join(translation_list)))
            total_token = query_token + response_token

            return ''.join(translation_list), total_token

        except Exception as e:
            print(f"\033[91mTranslation error: {str(e)}\033[0m")
            return str(e), 0
    






                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               