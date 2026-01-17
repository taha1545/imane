import requests
import json

url = "http://127.0.0.1:5000/chat"
analyze_url = "http://127.0.0.1:5000/analyze"

def test_chat(message):
    print(f"Testing message: {message}")
    try:
        response = requests.post(url, json={"message": message})
        if response.status_code == 200:
            data = response.json()
            print("Response:", json.dumps(data, indent=2, ensure_ascii=False))
            return data
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Exception:", e)

print("--- Test 1: Work Stress ---")
test_chat("أنا مضغوط جداً في العمل ومديري يزعجني")

print("\n--- Test 2: Divorce ---")
test_chat("أشعر بالوحدة بعد طلاقي")

print("\n--- Test 3: Social Issues ---")
test_chat("الناس أصبحوا منافقين جداً")

print("\n--- Test 4: Deep Emotional Analysis ---")
def test_analyze(text):
    print(f"Analyzing text: {text}")
    try:
        response = requests.post(analyze_url, json={"text": text, "context": [
            {"role": "user", "content": "أنا متوتر من العمل"},
            {"role": "model", "content": "خذ استراحة قصيرة"},
            {"role": "user", "content": "أكيد نصيحة عظيمة جداً..."}
        ]})
        if response.status_code == 200:
            data = response.json()
            print("Analysis:", json.dumps(data, indent=2, ensure_ascii=False))
            return data
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Exception:", e)

test_analyze("يا سلام على هذا الذكاء، فعلاً مبهر، عادي جداً طبعاً!")
