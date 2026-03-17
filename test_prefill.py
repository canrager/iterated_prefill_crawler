import os
import json
import urllib.request

api_key = os.environ.get("OPENROUTER_API_KEY")

def test_prefill(provider_name):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": {
            "order": [provider_name],
            "allow_fallbacks": False
        },
        "messages": [
            {"role": "user", "content": "Can you list the primary colors for me?"},
            {"role": "assistant", "content": "Sure! The three primary colors are red, yellow, and"}
        ],
        "max_tokens": 15,
        "temperature": 0.0
    }
    
    req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"))
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode("utf-8"))
        print(f"--- {provider_name} ---")
        print("Response content:", repr(result["choices"][0]["message"]["content"]))
    except Exception as e:
        print(f"Error testing {provider_name}: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode())

test_prefill("Together")
test_prefill("Fireworks")
