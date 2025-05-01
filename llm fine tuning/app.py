from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

model_id = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = response[len(user_input):].strip()
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
