# Netflix Movie Recommendation App 
## Project Team 
Group 9

## Project Description 
A movie recommendation web application that suggests Netflix-style movie recommendations based on user select movie and associated similarity analysis.
This project demonstrates data processing, recommendation logic, and an interactive UI built for experimentation and deployment.

## Team Members 
Radhika Chaklasiya,
Nolan Scott,
Katherine Lonergan

## Features
- Interactive **Streamlit** web interface
- Movie recommendations based on similarity
- Clean, user-friendly UI
- Lightweight and easy to run

## User Functions
1. A movie is selected by the user
2. The app computes similarity scores between movies
3. The most relevant recommendations are returned and displayed along with their content rating, genre, and a brief description

> Recommendation based on the movie description similarity, critic reviews, cast, directors, and similar metadata.

## Prerequisites 
- Python 3.9+
- Git
- pip

## Installation & Setup
</> Bash
1. Clone the existing repository. 
    `git clone https://github.com/nbscott98-riv/netflix-movie-rec.git`. 
    `cd netflix-movie-recommendation system`  
    `pip install -r requirements.txt`  
2. Install dependencies from the terminal
    `pip install -r requirements.txt`  
3. Run the streamlit app from the terminal
    `streamlit run app.py`  
4. In your browser, navigate to: http://localhost:8501

## Future Improvements
- Hosted interface
- Connection to additional streaming platforms to provide recommendations across services
- Direct connection to Rotten Tomato and the Streaming Services to get live listings and reviews
- Enhanced training focusing on specific words within critic reviews
- Included overall audience/critic score in output UI


## License
The MIT License (MIT) 2017. Please have a look at the LICENSE.md for more details.
