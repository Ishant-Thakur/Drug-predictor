from flask import Flask, render_template, request
import sys
import os

# Allow Flask to find src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.predict import predict_properties


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    smiles = ""

    if request.method == "POST":

        smiles = request.form.get("smiles", "").strip()

        if not smiles:
            error = "Please enter a SMILES string."

        else:
            try:

                result = predict_properties(smiles)
                result["predicted_logS"] = round(result["predicted_logS"], 4)
                bbb_conf = result["bbb_probability"] if result["bbb_label"] == "Permeant" else 1 - result["bbb_probability"]
                tox_conf = result["toxicity_probability"] if result["toxicity_label"] == "Likely toxic" else 1 - result["toxicity_probability"]
                result["bbb_probability"] = round(bbb_conf * 100, 1)
                result["toxicity_probability"] = round(tox_conf * 100, 1)

            except ValueError as e:

                error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        smiles=smiles
    )


if __name__ == "__main__":
    app.run(debug=True)
