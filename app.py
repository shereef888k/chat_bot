from flask import Flask, request, jsonify
from rag_query import get_answer

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    reply = get_answer(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="127.0.0.1", port=5000, debug=True)