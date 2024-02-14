import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and config when the script starts
config = PeftConfig.from_pretrained("% MODEL_NAME %")
model = AutoModelForCausalLM.from_pretrained("PRETRAINED_MODEL_NAME", config=config)
model = PeftModel.from_pretrained(model, "% MODEL_NAME %")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("PRETRAINED_MODEL_NAME", add_eos_token=True)


def greet(text):
    batch = tokenizer(f"'{text}' ->: ", return_tensors='pt')

    # Use torch.no_grad to disable gradient calculation
    with torch.no_grad():
        output_tokens = model.generate(**batch, do_sample=True, max_new_tokens=20)

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
