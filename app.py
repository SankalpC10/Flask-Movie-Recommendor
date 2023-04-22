from flask import Flask, render_template, request
from recommendation_model import RecommendationModel

app = Flask(__name__)
model = RecommendationModel()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommendations = model.get_recommendations(movie_name)
    if recommendations is not None:
        return render_template('recommendations.html', movie_name=movie_name, recommendations=recommendations)
    else:
        return render_template('no_recommendations.html', movie_name=movie_name)

if __name__ == '__main__':
    app.run(debug=True)
