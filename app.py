from flask import Flask, request, jsonify, send_from_directory
import time
import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from datetime import datetime
import random
import string
from threading import Thread

app = Flask(__name__)

# 基本設定
app.config['JSON_AS_ASCII'] = False  # 日本語などのUnicode文字を正しく処理

# グローバル変数として必要なモデルとプロセッサを読み込む
device = None
vl_chat_processor_1b = None
vl_gpt_1b = None
vl_chat_processor_7b = None
vl_gpt_7b = None

# タスク状態を管理するグローバル辞書
# ステータス: 'pending', 'processing', 'completed', 'failed'
tasks = {}

def load_model(model_size="1b"):
    global device, vl_chat_processor_1b, vl_gpt_1b, vl_chat_processor_7b, vl_gpt_7b
    
    # デバイスを確認
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    if model_size == "1b" and vl_gpt_1b is None:
        # 1Bモデルのパスを指定
        model_path = "deepseek-ai/Janus-Pro-1B"
        print(f"Loading 1B processor and tokenizer from {model_path}...")
        vl_chat_processor_1b = VLChatProcessor.from_pretrained(model_path)
        
        # 1Bモデルをロード
        print("Loading 1B model...")
        vl_gpt_1b = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        vl_gpt_1b = vl_gpt_1b.to(torch.bfloat16).to(device).eval()  # bfloat16 for memory efficiency
        print("1B model loaded and ready!")
    
    elif model_size == "7b" and vl_gpt_7b is None:
        # 7Bモデルのパスを指定
        model_path = "deepseek-ai/Janus-Pro-7B"
        print(f"Loading 7B processor and tokenizer from {model_path}...")
        vl_chat_processor_7b = VLChatProcessor.from_pretrained(model_path)
        
        # 7Bモデルをロード
        print("Loading 7B model...")
        vl_gpt_7b = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        vl_gpt_7b = vl_gpt_7b.to(torch.bfloat16).to(device).eval()  # bfloat16 for memory efficiency
        print("7B model loaded and ready!")

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    generation_id: str,
    temperature: float = 0.8,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    start_time = time.time()
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
    for i in range(image_token_num_per_image):
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
    # 生成したサンプルのベースディレクトリを作成
    os.makedirs('generated_images', exist_ok=True)

    # 指定されたgeneration_idで保存ディレクトリを使用
    generation_dir = os.path.join('generated_images', generation_id)
    # すでに存在するはずのディレクトリなので作成しない

    image_paths = []
    for i in range(parallel_size):
        save_path = os.path.join(generation_dir, f"{i + 1}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)
        image_paths.append(save_path)

    print(f"Generation complete! Check the '{generation_dir}' directory for your images.")

    elapsed_time = time.time() - start_time
    print(f"End time: {elapsed_time} seconds")
    
    return {
        "generation_id": generation_id,
        "image_paths": image_paths,
        "elapsed_time": elapsed_time
    }

def generate_in_background(generation_id, mmgpt, processor, formatted_prompt, temperature, parallel_size, cfg_weight):
    """バックグラウンドスレッドで画像生成を実行する関数"""
    global tasks
    try:
        tasks[generation_id]['status'] = 'processing'
        tasks[generation_id]['start_time'] = time.time()
        
        # 画像生成実行 - generation_idを渡す
        result = generate(
            mmgpt, 
            processor, 
            formatted_prompt, 
            generation_id=generation_id,  # generation_idを渡す
            temperature=temperature,
            parallel_size=parallel_size,
            cfg_weight=cfg_weight
        )
        
        # 成功時の処理
        tasks[generation_id]['status'] = 'completed'
        tasks[generation_id]['result'] = {
            'image_count': len(result['image_paths']),
            'elapsed_time': result['elapsed_time'],
            'image_paths': result['image_paths']
        }
    except Exception as e:
        # エラー時の処理
        tasks[generation_id]['status'] = 'failed'
        tasks[generation_id]['error'] = str(e)
        print(f"Error in background generation: {str(e)}")

    if generation_id in tasks:
        del tasks[generation_id]

@app.route('/generate', methods=['POST'])
def generate_image_api():
    global vl_gpt_1b, vl_chat_processor_1b, vl_gpt_7b, vl_chat_processor_7b, tasks
    
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required.'}), 400
    
    prompt = data['prompt']
    
    # オプションのパラメータ
    temperature = data.get('temperature', 0.8)
    parallel_size = data.get('parallel_size', 1)
    cfg_weight = data.get('cfg_weight', 5)
    
    # 品質パラメータを取得し、対応するモデルを選択
    quality = data.get('quality', 'low')  # デフォルトは 'low'
    
    if quality == 'high':
        # Janus-Pro-7Bを使用
        model_size = "7b"
        if vl_gpt_7b is None or vl_chat_processor_7b is None:
            load_model(model_size)
        mmgpt = vl_gpt_7b
        processor = vl_chat_processor_7b
        print("Using high quality model (Janus-Pro-7B)")
    else:
        # Janus-Pro-1Bを使用 (デフォルト、または明示的に 'low' が指定された場合)
        model_size = "1b"
        if vl_gpt_1b is None or vl_chat_processor_1b is None:
            load_model(model_size)
        mmgpt = vl_gpt_1b
        processor = vl_chat_processor_1b
        print("Using low quality model (Janus-Pro-1B)")
    
    # 会話形式でプロンプトを準備
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # プロンプトを準備
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    formatted_prompt = sft_format + processor.image_start_tag
    
    # このジェネレーション用にタイムスタンプ+ランダム文字列の新しいディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    generation_id = f"{timestamp}_{random_str}"
    
    # 生成したサンプルのベースディレクトリを作成
    os.makedirs('generated_images', exist_ok=True)
    generation_dir = os.path.join('generated_images', generation_id)
    os.makedirs(generation_dir, exist_ok=True)
    
    # タスク状態を初期化
    tasks[generation_id] = {
        'status': 'pending',
        'prompt': prompt,
        'quality': quality,
        'temperature': temperature,
        'parallel_size': parallel_size,
        'cfg_weight': cfg_weight,
        'created_at': time.time()
    }
    
    # バックグラウンドで画像生成を開始
    thread = Thread(
        target=generate_in_background,
        args=(generation_id, mmgpt, processor, formatted_prompt, temperature, parallel_size, cfg_weight)
    )
    thread.daemon = True  # メインプロセスが終了したらスレッドも終了
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Image generation task has started.',
        'generation_id': generation_id,
        'status': 'pending',
        'quality': quality
    })

@app.route('/images/<generation_id>/<filename>', methods=['GET'])
def get_image(generation_id, filename):
    """生成された画像を取得するエンドポイント"""
    directory = os.path.join('generated_images', generation_id)
    return send_from_directory(directory, filename)

@app.route('/images/<generation_id>', methods=['GET'])
def list_images(generation_id):
    """特定の生成IDに関連する画像のリストを取得するエンドポイント"""
    directory = os.path.join('generated_images', generation_id)
    try:
        files = os.listdir(directory)
        image_files = [f for f in files if f.endswith('.jpg')]
        return jsonify({
            'generation_id': generation_id,
            'images': image_files,
            'count': len(image_files)
        })
    except FileNotFoundError:
        return jsonify({'error': f'Generation ID {generation_id} not found'}), 404

@app.route('/status/<generation_id>', methods=['GET'])
def get_generation_status(generation_id):
    """タスクのステータスを確認するエンドポイント"""
    global tasks
    
    if generation_id not in tasks:
        return jsonify({'error': f'Generation ID {generation_id} not found. If it may have already been generated, try requesting /images/{generation_id}.'}), 404
    
    task_info = tasks[generation_id]
    response = {
        'generation_id': generation_id,
        'status': task_info['status'],
        'quality': task_info['quality'],
        'created_at': task_info['created_at']
    }
    
    # 処理完了時の情報を追加
    if task_info['status'] == 'completed':
        response['result'] = task_info['result']
    
    # エラー時の情報を追加
    if task_info['status'] == 'failed':
        response['error'] = task_info['error']
    
    # 処理中の場合は進行中の時間を追加
    if task_info['status'] == 'processing' and 'start_time' in task_info:
        response['processing_time'] = time.time() - task_info['start_time']
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
