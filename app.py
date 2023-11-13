from flask import Flask, request, jsonify, make_response
# Use a pipeline as a high-level helper
from transformers import pipeline

food_classification = pipeline("image-classification", model="Kaludi/food-category-classification-v2.0")

app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classify the image and return the result
    :return:
    """
    body = request.get_json()
    url = body.get("url")
    food_class = food_classification(url)
    return make_response(jsonify(food_class), 200)


if __name__ == "__main__":
    app.run()
