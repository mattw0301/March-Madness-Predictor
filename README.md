# 🏀 March Madness Predictor

A Python-based machine learning project that predicts which NCAA Division I men’s basketball teams will make the tournament, using historical team stats.

Built using `scikit-learn` and a Random Forest classifier, this tool allows you to:

- Train a model on historical data (`cbb.csv`)
- Predict tournament qualification for the current season (`this_year_cbb.csv`)

Note: Provided data is from 2013-2024.

## Requirements
`pandas
numpy
matplotlib
scikit-learn`
## Data Format
### Historical Data (`cbb.csv`)
Same columns as provided data:
TEAM, CONF, YEAR, ADJOE, ADJDE, BARTHAG, etc
### Current Season (`this_year_cbb.csv`)
Same as historical but without POSTSEASON.
## Output
* Model accuracy metrics on historical data
* Feature importance visualization
* Top 20 teams most likely to make the tournament
* Full predictions in 'this_year_predictions.csv'
## Model Performance
The Random Forest model evaluates its performance on a 20% test split of historical data, providing:
* Accuracy score
* Detailed classification report
* Visual feature importance analysis
## License
MIT License
