import base64
import requests

api_key = "YOUR_API_KEY"

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def get_response_with_image(image_paths, text_prompt, try_time=0, encoded_images=None, max_tokens=2000):
    if try_time > 5:
        return ""
    # base64_image = encode_image(image_path) if encoded_image is None else encoded_image
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_image}"
                    #     }
                    # },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }
    if encoded_images is None:
        encoded_images = [encode_image(image_path) for image_path in image_paths]
    for encoded_image in encoded_images:
        payload['messages'][0]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })

    try:
        response = requests.post("https://api.claudeshop.top/v1/chat/completions", headers=headers, json=payload)
    except Exception as e:
        print(e)
        # breakpoint()
        return get_response_with_image(image_paths, text_prompt, try_time+1, encoded_images)

    return response.json()['choices'][0]['message']['content']


def get_response(text_prompt, try_time=0, max_tokens=2000):
    if try_time > 5:
        return ""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    try:
        response = requests.post("https://api.claudeshop.top/v1/chat/completions", headers=headers, json=payload)
    except Exception as e:
        print(e)
        return get_response(text_prompt, try_time+1)

    # 检查响应状态码
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        return ""  # 返回空字符串
    
    return response.json()['choices'][0]['message']['content']