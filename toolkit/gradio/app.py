import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Device configuration (prioritize GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "%MODEL_ID%"

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch
.bfloat16
        )

# Load models and tokenizer efficiently
config = PeftConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config)

# Load the Lora model
model = PeftModel.from_pretrained(model, model_id)

def greet(text):
    with torch.no_grad():  # Disable gradient calculation for inference
        batch = tokenizer(f'"{text}" ->:', return_tensors='pt')  # Move tensors to device
        with torch.cuda.amp.autocast():  # Enable mixed-precision if available
            output_tokens = model.generate(**batch
, max_new_tokens=15)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

iface = gr.Interface(fn=greet, inputs="text", outputs="text", title="PEFT Model for Big Brain")
iface.launch()  # Share directly to Gradio Space