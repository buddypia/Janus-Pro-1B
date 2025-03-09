# Janus-Pro API
## environment
python version Python 3.12.8
```bash
pip install -r requirements.txt
```

## test
main.py
```bash
python main.py --prompt "A warm, cozy illustration of a Japanese countryside kitchen where a smiling grandmother and her small grandchild are making onigiri (rice balls) together. Their hands are forming a triangle-shaped rice ball that glows with a soft magical light. Through the window, a beautiful autumn landscape can be seen with red maple leaves falling. The kitchen has traditional Japanese elements with modern touches. The scene is bathed in golden afternoon light, creating a heartwarming atmosphere. The grandmother is teaching the child with patience and love, and there's a small cat watching them curiously from the corner."
```

app.py
```bash
python app.py
```

## API
generate images
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A detailed futuristic cityscape at night",
    "temperature": 0.7,
    "parallel_size": 3,
    "cfg_weight": 5,
    "quality": "low"
  }'
{
  "elapsed_time": 65.84789705276489,
  "generation_id": "20250309_160200_OiuTvb",
  "image_count": 3,
  "quality": "low",
  "success": true
}
```

get images
```bash
curl -X GET http://localhost:5000/images/20250309_160200_OiuTvb
{
  "generation_id": "20250309_160200_OiuTvb",
  "images": [
    "2.jpg",
    "3.jpg",
    "1.jpg"
  ],
  "urls": [
    "/images/20250309_160200_OiuTvb/2.jpg",
    "/images/20250309_160200_OiuTvb/3.jpg",
    "/images/20250309_160200_OiuTvb/1.jpg"
  ]
}
```

get image
```bash
curl -X GET http://localhost:5000/images/20250309_160200_OiuTvb/1.jpg -o downloaded_image1.jpg
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 26883  100 26883    0     0  5000k      0 --:--:-- --:--:-- --:--:-- 5250k
```