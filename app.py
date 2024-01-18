from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Import the functions from your existing code
from model import (api)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chat_bot():
    
    user_question = request.json["question"]
    user_question += ' summarize in 60 words'
    chatbot_answer = api(user_question)
    print(chatbot_answer)
    return jsonify({"error": False, "message": chatbot_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



