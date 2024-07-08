# Importieren der notwendigen Bibliotheken
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Initialisieren der Flask-Anwendung
app = Flask(__name__)

# Definition der Immobilienpreisdaten für verschiedene Städte und Jahre
data_list = [
    ("Berlin", {
        2024: 23.75, 2023: 21.6, 2022: 23.57, 2021: 15.14, 2020: 15.11, 2019: 13.41, 2018: 12.65,
        2017: 11.4, 2016: 10.46, 2015: 11.65, 2014: 9.03, 2013: 8.47, 2012: 7.66, 2011: 6.94, 2010: 6.42
    }),
    ("München", {
        2024: 22.86, 2023: 22.04, 2022: 20.88, 2021: 20.79, 2020: 23.03, 2019: 21.25, 2018: 19.29,
        2017: 18.56, 2016: 18.38, 2015: 19.48, 2014: 15.5, 2013: 14.61, 2012: 14.34, 2011: 12.9, 2010: 12.8
    }),
    ("Hamburg", {
        2024: 16.29, 2023: 15.29, 2022: 14.87, 2021: 14.15, 2020: 13.47, 2019: 12.95, 2018: 12.99,
        2017: 12.71, 2016: 11.88, 2015: 12.6, 2014: 11.14, 2013: 10.99, 2012: 10.56, 2011: 10.17, 2010: 10.11
    }),
    ("Freiburg", {
        2024: 13.70, 2023: 13.20, 2022: 14.25, 2021: 14.91, 2020: 15.17, 2019: 13.37, 2018: 13.56,
        2017: 13.38, 2016: 10.80, 2015: 11.96, 2014: 9.90, 2013: 10.83, 2012: 9.73, 2011: 9.38, 2010: 10.28
    }),
    ("Frankfurt am Main", {
        2024: 19.94, 2023: 19.29, 2022: 17.14, 2021: 16.76, 2020: 21.99, 2019: 15.52, 2018: 16.78,
        2017: 14.53, 2016: 14.10, 2015: 15.43, 2014: 13.43, 2013: 12.19, 2012: 11.92, 2011: 10.95, 2010: 11.49
    }),
    ("Stuttgart", {
        2024: 19.94, 2023: 19.29, 2022: 17.14, 2021: 16.76, 2020: 21.99, 2019: 15.52, 2018: 16.78,
        2017: 14.53, 2016: 14.10, 2015: 15.43, 2014: 13.43, 2013: 12.19, 2012: 11.92, 2011: 10.95, 2010: 11.49
    }),
    ("Heidelberg", {
        2024: 16.26, 2023: 14.44, 2022: 14.74, 2021: 14.51, 2020: 14.01, 2019: 12.74, 2018: 12.87,
        2017: 11.96, 2016: 11.24, 2015: 11.88, 2014: 10.74, 2013: 10.37, 2012: 10.18, 2011: 9.72, 2010: 9.91
    }),
    ("Mainz", {
        2024: 14.53, 2023: 13.93, 2022: 13.77, 2021: 13.24, 2020: 13.08, 2019: 11.79, 2018: 12.27,
        2017: 12.10, 2016: 10.76, 2015: 11.40, 2014: 10.21, 2013: 10.42, 2012: 10.03, 2011: 9.17, 2010: 9.98
    }),
    ("Potsdam", {
        2024: 13.48, 2023: 13.07, 2022: 12.88, 2021: 13.04, 2020: 11.81, 2019: 10.99, 2018: 10.65,
        2017: 10.29, 2016: 9.81, 2015: 9.59, 2014: 8.71, 2013: 8.63, 2012: 8.12, 2011: 8.08, 2010: 7.50
    }),
    ("Köln", {
        2024: 15.07, 2023: 14.56, 2022: 14.24, 2021: 13.50, 2020: 13.39, 2019: 12.37, 2018: 12.48,
        2017: 11.71, 2016: 11.05, 2015: 11.97, 2014: 10.09, 2013: 9.67, 2012: 9.28, 2011: 8.51, 2010: 8.53
    })
]

# Funktion zur Vorhersage der Immobilienpreise für eine bestimmte Stadt
def forecast_for_city(city, prices, forecast_year):
    # Vorbereitung der Daten
    data = [[year, price] for year, price in prices.items()]
    df = pd.DataFrame(data, columns=["Jahr", "Preis"])

    # Entfernen des Vorhersagejahres, falls es in den Daten enthalten ist
    if forecast_year in df["Jahr"].values:
        df = df[df["Jahr"] != forecast_year]

    # Definieren der Merkmale (X) und Zielvariablen (y)
    X = df[["Jahr"]]
    y = df["Preis"]

    # Aufteilen der Daten in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisierung der Vorhersagen und Modelle
    predictions = {}
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }

    # Lineare Regression
    model_lr = models["Linear Regression"]
    model_lr.fit(X_train, y_train)
    predictions["Linear Regression"] = model_lr.predict([[forecast_year]])[0]

    # Polynomiale Regression (Grad 2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    model_pr = models["Polynomial Regression"]
    model_pr.fit(poly_features.transform(X_train), y_train)
    predictions["Polynomial Regression"] = model_pr.predict(poly_features.transform([[forecast_year]]))[0]

    # Entscheidungsbaum
    model_dt = models["Decision Tree"]
    model_dt.fit(X_train, y_train)
    predictions["Decision Tree"] = model_dt.predict([[forecast_year]])[0]

    # Random Forest
    model_rf = models["Random Forest"]
    model_rf.fit(X_train, y_train)
    predictions["Random Forest"] = model_rf.predict([[forecast_year]])[0]

    return predictions

# Route für die Startseite
@app.route('/', methods=['GET', 'POST'])
def index():
    # Liste der Städte und Jahre für das Dropdown-Menü
    cities = [city for city, _ in data_list]
    years = list(range(2010, 2035))

    # Verarbeitung der Formulardaten
    if request.method == 'POST':
        # Eingaben des Benutzers
        square_meters = float(request.form['square_meters'])
        city = request.form['city']
        year = int(request.form['year'])

        # Preise der gewählten Stadt finden
        prices = next((prices for city_name, prices in data_list if city_name == city), None)
        if prices is None:
            return "Stadt nicht gefunden", 400
        
        # Vorhersagen für die gewählte Stadt und das Jahr berechnen
        predictions = forecast_for_city(city, prices, year)

        # Gesamtpreise basierend auf der Wohnfläche berechnen
        total_prices = {model: price * square_meters for model, price in predictions.items()}
        
        # Ergebnisse rendern
        return render_template('result.html', city=city, year=year, square_meters=square_meters, total_prices=total_prices)
    
    # Startseite rendern
    return render_template('index.html', cities=cities, years=years)

# Flask-Anwendung starten
if __name__ == '__main__':
    app.run(debug=True)
