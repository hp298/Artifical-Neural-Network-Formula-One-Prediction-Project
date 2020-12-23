# Artifical-Neural-Network-Formula-One-Prediction-Project
Using the PyTorch library, can we accurately predict the finishing position of a Formula One driver of a race?

Historical Formula One Race Data (CSVs in the self named file) are provided by: http://ergast.com/mrd/
Weather Data from each race used by calling Visual Crossing  Weather API from: https://rapidapi.com/visual-crossing-corporation-visual-crossing-corporation-default/api/visual-crossing-weather/endpoints

FDP-Data.py : Using CSV data and historical weather API, created ~25000 data samples to train and test the ANN
TraingData.txt : locally saved data constructed from FDP-Data.py
FDP-ML.py : ANN model used to predict finishing position, using data from TrainingData.txt
