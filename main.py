import time
import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm
import argparse
from datetime import datetime

import random
import string

def main():
    parser = argparse.ArgumentParser(description='プログラムの説明')

    # 引数の定義
    parser.add_argument('--prompt', type=str, help='Prompt')
    parser.add_argument('--quality', type=str, choices=['low', 'high'], default='low', 
                        help='Quality setting: low for Janus-Pro-1B, high for Janus-Pro-7B')

    # 引数の解析
    args = parser.parse_args()

    print(f"Prompt: {args.prompt}")
    print(f"Quality: {args.quality}")

    start_time = time.time()

    print(f"Starting the generation process... Start time: {start_time} seconds")

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    # specify the path to the model based on quality
    if args.quality == 'low':
        model_path = "deepseek-ai/Janus-Pro-1B"
        print("Using low quality model (1B)...")
    else:
        model_path = "deepseek-ai/Janus-Pro-7B"
        print("Using high quality model (7B)...")
        
    print("Loading processor and tokenizer...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the model
    print("Loading model...")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()  # bfloat16 for memory efficiency
    print("Model loaded and ready!")

    # Set up the conversation
    conversation = [
        {
            "role": "<|User|>",
            "content": args.prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Prepare the prompt
    print("Preparing prompt...")
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    @torch.inference_mode()
    def generate(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        temperature: float = 0.8,
        parallel_size: int = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        print("Initializing generation process...")
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(device)

        print("Setting up token generation...")
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

        print("Beginning token generation...")
        for i in tqdm(range(image_token_num_per_image), desc="Generating image tokens"):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        print("Decoding generated tokens into image...")
        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        print("Saving generated image...")
        # Create a base directory for all generated samples
        os.makedirs('generated_images', exist_ok=True)


        # このジェネレーション用にタイムスタンプ+ランダム文字列の新しいディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        generation_id = f"{timestamp}_{random_str}"
        generation_dir = os.path.join('generated_images', generation_id)
        os.makedirs(generation_dir, exist_ok=True)

        for i in range(parallel_size):
            save_path = os.path.join(generation_dir, f"{i + 1}.jpg")
            PIL.Image.fromarray(visual_img[i]).save(save_path)

        print(f"Generation complete! Check the '{generation_dir}' directory for your images.")

        elapsed_time = time.time() - start_time
        print(f"End time: {elapsed_time} seconds")

    # Generate images
    generate(vl_gpt, vl_chat_processor, prompt, parallel_size=5)

if __name__ == "__main__":
    main()
