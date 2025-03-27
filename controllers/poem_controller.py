import re
import tiktoken
from langchain_community.llms import Ollama


class PoemController:
    def __init__(self, model):
        self.model = model
        self.total_output_list = []
    
    def check_output(self, output, input_text_split):
        line_boolean = None
        word_boolean = None
        
        lines = output.split('\n')
    
        # Filter out any empty strings that might be caused by trailing newlines
        lines = [line for line in lines if line.strip()]
        
        if len(lines) == 5:
            line_boolean = True
        
        else:
            line_boolean = False
        
        output_split = [i.lower() for i in output.split(' ')]
        counter = 0
        for words in input_text_split:
            if words in input_text_split:
                counter = counter + 1
        
        if counter == len(input_text_split):
            word_boolean = True
        
        else:
            word_boolean = False
            
        return line_boolean, word_boolean
            
    def maintain_lines(self, output):  
        n_lines = 5
        lines = output.split('\n')
        lines = [line for line in lines if line.strip()]
        previous_line = lines[-1]

        if len(lines) > n_lines:
            return '\n'.join(lines[:5])
        
        if len(lines) < n_lines:
            diff_line = n_lines - len(lines)
            prompt =  f"generate me {diff_line} line poem whose previous line is {previous_line}"
            output = self.generate_output_from_llm(prompt, '\n')
            lines.append(output)
            
            return '\n'.join(lines[:5])   
    
    def generate_output_from_llm(self, final_prompt, stop = None):
        if stop:
            llm = Ollama(model=self.model, temperature = 0)
            output = llm.invoke(final_prompt, stop = ['\n'])
        else:
            print("I am not stop")
            llm = Ollama(model=self.model, temperature = 0)
            output = llm.invoke(final_prompt)
        
        return output
    
    def get_poem(self, input_text, input_text_split):  
          
        initial_prompt = "generate me a five line poem with words : "
        final_prompt = f"{initial_prompt} '{input_text}'"
        
        output = self.generate_output_from_llm(final_prompt)
        line_boolean, word_boolean = self.check_output(output, input_text_split)
        
        print(output)
        print(line_boolean, word_boolean)
        
        if line_boolean == False:
            final_prompt = final_prompt + '. You did wrong. The total number of lines must be five'
            
            counter = 0
            while line_boolean is False:
                print("-------------")
                print(final_prompt)
                output = self.generate_output_from_llm(final_prompt)
                print(output)
                line_boolean, word_boolean = self.check_output(output, input_text_split)
                print(line_boolean, word_boolean)
                                
                if counter > 1:
                    output = self.maintain_lines(output)
                    print(output)
                    line_boolean, word_boolean = self.check_output(output, input_text_split)
                    print(line_boolean, word_boolean)
                
                counter = counter + 1
                
                if counter > 5:
                    return 'please try again with next LLM'

        
        if word_boolean == False:
            final_prompt = final_prompt + '. Poem must contains defined words'
            while word_boolean is False:
                output = self.generate_output_from_llm(final_prompt)
                line_boolean, word_boolean = self.check_output(output, input_text_split)
        
        self.total_output_list.append(output)
        
        return output
                
    def input_preprocess(self, input_text):
        sentence_list = [i.lower() for i in input_text.split(',')]
        return sentence_list

    def generate_poem(self, input_text):
        input_text_split = self.input_preprocess(input_text)
        poem = self.get_poem(input_text, input_text_split)
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        query_token = len(encoding.encode(''.join(input_text_split)))
        response_token = len(encoding.encode(''.join(self.total_output_list)))
        total_token = query_token + response_token
        
        
        return poem, total_token
    


