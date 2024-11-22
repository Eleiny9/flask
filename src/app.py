from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Inicializar 
app = Flask(__name__)

# Cargar el modelo guardado
model = joblib.load("tree_default.pkl")

# Página principal
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para realizar predicciones
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos del formulario
        features = [
            float(request.form["alcohol"]),
            float(request.form["ash"]),
            float(request.form["flavanoids"]),
            float(request.form["color_intensity"]),
            float(request.form["proline"])
        ]
        
        # Convertir a un array numpy y realizar predicción
        prediction = model.predict([features])[0]
        
        # Mapear la predicción a la clase de vino
        wine_classes = ["Clase 0", "Clase 1", "Clase 2"]
        result = wine_classes[prediction]

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)

