from hf_model.model_cards.deepseek_coder import (
    DeepSeek_Coder_6_7B_Instruct,
    DeepSeek_R1_QWEN3_8b
)
from hf_model.model_cards.language_cards import (
    Solidity_6b_LLM
)

import hf_model.cli as cli

@cli.parse_one_argument()
def start():
    model = DeepSeek_R1_QWEN3_8b()
    while True: 
        message=input(">")
        if message == "quit":
            return
        print(model.generate(message))

print(cli.parser.parse_args())
