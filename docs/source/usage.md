# Usage

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 src/maininference.py start
```

## How to run different models

Running models is as simple as importing models from huggingface and passing them 
same as DeepSeek models in `hf_model/model_cards/deepseek_coder.py`

running model is easy in maininference which is demo file for running LLMs.

```python
from hf_model.model_cards.deepseek_coder import DeepSeek_Coder_6_7B_Instruct
model = DeepSeek_Coder_6_7B_Instruct()
ouptut = model.generate(message)
```

yes that's it
