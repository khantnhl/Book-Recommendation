from flask import Flask, request, render_template
from groq import Groq
import os
import joblib


#load model
tfidf = joblib.load('models/tfidf.pkl')
scaler = joblib.load('models/maxscaler.pkl')
svd = joblib.load('models/svd.pkl')
birchmodel = joblib.load('models/birchmodel.pkl')


# Initialize Flask app
app = Flask(__name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":

        # Get inputs from form
        title_input = request.form.get("title")
        subject_input = request.form.get("subject")
        synopsis_input = request.form.get("synopsis")
        
        merged_input = f"{title_input} {subject_input} {synopsis_input}"
        
         # Process and predict
        tfidf_features = tfidf.transform([merged_input])
        scaled_features = scaler.transform(tfidf_features)
        predicted_cluster = birchmodel.predict(scaled_features)[0]


        # Call the Groq API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant in literature."},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=8000,
            top_p=1,
            stop=None,
        )

        # Render the response
        return render_template("response.html", response=response["choices"][0]["message"]["content"])

    return render_template("input_form.html")


if __name__ == "__main__":
    app.run(debug=True)
