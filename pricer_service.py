import os
import modal
from modal import Image, Volume

# Modal app definition
app = modal.App("pricer-service")

# Container image for the remote inference service
image = Image.debian_slim().pip_install(
    "huggingface_hub",
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "peft",
)

# Modal secret must include key: HF_TOKEN
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"

PROJECT_NAME = "price"
HF_USER = "ed-donner"
RUN_NAME = "2025-11-28_18.47.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

CACHE_DIR = "/cache"
MIN_CONTAINERS = 0

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    ),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found. Set HF_TOKEN in Modal secret 'huggingface-secret'."
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            FINETUNED_MODEL,
            revision=REVISION,
            token=hf_token,
        )
        self.model.eval()

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Handle cases where the expected prefix is not strictly present
        contents = decoded.split(PREFIX, 1)[1] if PREFIX in decoded else decoded

        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0
