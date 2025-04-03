from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    url = data.get("url", "")
    return jsonify({"response": f"Analysis for {url}"})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    return jsonify({"response": f"Response to query: {user_query}"})

if __name__ == "__main__":
    app.run(debug=True)