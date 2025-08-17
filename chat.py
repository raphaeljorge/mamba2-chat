import torch
from transformers import AutoTokenizer
from mamba_ssm.modules.mamba2 import Mamba2

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat") # You may need a different tokenizer for a v2 model
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

# Mamba v2 requires a configuration dictionary
config = {
    "d_model": 2560,  # Example value, adjust to your model's config
    "n_layer": 64,    # Example value, adjust to your model's config
    "vocab_size": 50277, # Example value, adjust to your model's config
    "d_state": 16, 
    "expand": 2, 
    "dt_rank": "auto", 
    "d_conv": 4, 
    "pad_vocab_size_multiple": 8, 
    "conv_bias": True, 
    "bias": False,
}

model = Mamba2(config)
# You will need to load the weights of a trained Mamba v2 model
# For example: model.load_state_dict(torch.load("path/to/your/mamba2/model.pt"))
model.to(device)


messages = []
while True:
    user_message = input("\nYour message: ")
    messages.append(dict(
        role="user",
        content=user_message
    ))

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    out = model.generate(input_ids=input_ids, max_length=2000, temperature=0.9, top_p=0.7, eos_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.batch_decode(out)
    messages.append(dict(
        role="assistant",
        content=decoded[0].split("<|assistant|>\n")[-1])
    )

    print("Model:", decoded[0].split("<|assistant|>\n")[-1])