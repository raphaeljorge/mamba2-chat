import torch
import argparse

# Import MambaConfig
from mamba_ssm.models.config_mamba import MambaConfig 
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from trainer.data import ChatDataModule
from trainer.mamba_trainer import MambaTrainer


def run(args):
    # Define the model configuration
    config = MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=50277,  # Make sure this matches your tokenizer's vocab size
    )
        
    # Initialize the model with the configuration
    model = MambaLMHeadModel(config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    # Ensure the model's vocab size matches the tokenizer's
    if config.vocab_size != len(tokenizer):
        print("Warning: vocab size in config and tokenizer mismatch. Resizing model embeddings.")
        model.resize_token_embeddings(len(tokenizer))


    data_module = ChatDataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
        conversation_template=tokenizer.chat_template,
        max_tokens=2048
    )


    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir="mamba-chat-from-scratch",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments for model configuration
    parser.add_argument("--d_model", type=int, default=2560, help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=64, help="Number of layers")

    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./data/ultrachat_small.jsonl")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)