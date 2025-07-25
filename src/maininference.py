from hf_model.model_cards.deepseek_coder import (
    DeepSeek_Coder_6_7B_Instruct
)

import hf_model.cli as cli

@cli.parse_one_argument()
def start():
    model = DeepSeek_Coder_6_7B_Instruct()
    while True: 
        message=input(">")
        if message == "quit":
            return
        print(model.generate(message))
