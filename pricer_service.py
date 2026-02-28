import os
import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!
app = modal.App("pricer-service")

image = (
    Image.debian_slim()
    .pip_install(
        "huggingface_hub",  # ✅ 用 hub 正式包
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
    )
)

# This collects the secret from Modal.
# Make sure your Modal secret has key: HF_TOKEN
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"

PROJECT_NAME = "price"
# your HF name here! Or use mine if you just want to reproduce my results.
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
            "TRANSFORMERS_CACHE": CACHE_DIR,  # ✅ transformers 也走同一个 cache
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
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # ✅ 从 Modal Secret 读 token（你的 secret 里 key 就是 HF_TOKEN）
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found. Please set HF_TOKEN in Modal secret 'huggingface-secret'."
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # ✅ 显式传 token，避免 gated repo 403
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
        )

        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION,
            token=hf_token,
        )

        # ✅ 推理模式
        self.fine_tuned_model.eval()

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(
                inputs,
                max_new_tokens=5,
                do_sample=False,  # ✅ 稳定一点
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 更稳：有些输出可能没有严格包含 "Price is $"
        if "Price is $" in result:
            contents = result.split("Price is $", 1)[1]
        else:
            contents = result

        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0
