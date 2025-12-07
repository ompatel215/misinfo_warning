from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)


# load trained model

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "misinfo_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    warning = None
    user_text = ""
    probabilities = None

    if request.method == "POST":
        user_text = request.form.get("user_text", "")
        cleaned_text = clean_text(user_text)

        # predict label and probabilities
        prediction = model.predict([cleaned_text])[0]
        proba = model.predict_proba([cleaned_text])[0]
        probabilities = dict(zip(model.classes_, [round(p*100, 2) for p in proba]))

        if prediction == "misinfo":
            warning = "⚠️ Warning: This text may contain misinformation."
        else:
            warning = "✅ This text appears to be trustworthy."

    return render_template(
        "index.html",
        warning=warning,
        user_text=user_text,
        probabilities=probabilities
    )

# run app
if __name__ == "__main__":
    app.run(debug=True)
