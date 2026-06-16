import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Load your trained model
MODEL_PATH = "decision_pkl.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# 2. Define the HTML and CSS template directly as a Python string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Predictor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 font-sans antialiased min-h-screen flex flex-col justify-between">

    <header class="bg-white border-b border-slate-200 py-4 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 class="text-xl font-bold text-slate-800 tracking-tight flex items-center gap-2">
                🚀 Smart Estimator Portal
            </h1>
            <span class="px-2.5 py-1 text-xs font-semibold text-emerald-700 bg-emerald-50 rounded-full border border-emerald-200">
                Model Active
            </span>
        </div>
    </header>

    <main class="max-w-4xl mx-auto w-full px-4 py-10 flex-grow">
        <div class="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden grid grid-cols-1 md:grid-cols-5">
            
            <form action="/" method="POST" class="p-6 sm:p-8 md:col-span-3 space-y-4">
                <h2 class="text-lg font-semibold text-slate-700 mb-2">Input Specifications</h2>
                
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Make ID</label>
                        <input type="number" name="make" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="e.g. 1">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Model ID</label>
                        <input type="number" name="model" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="e.g. 5">
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Year</label>
                        <input type="number" name="year" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" value="2022">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Engine Size (L)</label>
                        <input type="number" name="engine_size" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="e.g. 2.0">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Mileage</label>
                    <input type="number" name="mileage" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="e.g. 45000">
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Fuel Type ID</label>
                        <input type="number" name="fuel_type" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="0 or 1">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Transmission ID</label>
                        <input type="number" name="transmission" step="any" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition" placeholder="0 or 1">
                    </div>
                </div>

                <button type="submit" class="w-full mt-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg shadow-md transition duration-150 ease-in-out cursor-pointer text-center">
                    Generate Prediction
                </button>
            </form>

            <div class="bg-slate-900 p-6 sm:p-8 md:col-span-2 flex flex-col justify-center text-white text-center border-t md:border-t-0 md:border-l border-slate-800">
                {% if prediction is not none %}
                    <p class="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Estimated Value</p>
                    <p class="text-4xl sm:text-5xl font-extrabold text-blue-400 drop-shadow-sm">{{ prediction }}</p>
                    <p class="text-xs text-slate-400 mt-4">Prediction calculated instantly based on your model's Decision Tree logic.</p>
                {% elif error %}
                    <div class="text-red-400 bg-red-950/50 p-4 border border-red-900 rounded-lg text-sm">
                        ⚠️ {{ error }}
                    </div>
                {% else %}
                    <div class="space-y-2">
                        <div class="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto text-slate-400 text-xl animate-pulse">📊</div>
                        <p class="text-slate-300 font-medium">Awaiting Input</p>
                        <p class="text-xs text-slate-500 max-w-xs mx-auto">Fill out the features and click submit to generate a prediction.</p>
                    </div>
                {% endif %}
            </div>
        </div>
    </main>

    <footer class="text-center py-4 text-xs text-slate-400 border-t border-slate-200 bg-white">
        Powered by Flask & Render Web Services
    </footer>

</body>
</html>
"""

# 3. Route handling the form rendering and backend calculations
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            # Extract numbers from form submission
            make = float(request.form.get("make", 0))
            model_type = float(request.form.get("model", 0))
            year = float(request.form.get("year", 2020))
            engine_size = float(request.form.get("engine_size", 2.0))
            mileage = float(request.form.get("mileage", 0))
            fuel_type = float(request.form.get("fuel_type", 0))
            transmission = float(request.form.get("transmission", 0))
            
            # Formulate array structure to match model inputs
            features = np.array([[make, model_type, year, engine_size, mileage, fuel_type, transmission]])
            
            # Model inference execution
            pred_output = model.predict(features)
            prediction = round(pred_output[0], 2)
            
        except Exception as e:
            error = f"An error occurred with your inputs: {str(e)}"
            
    # Inject backend data parameters dynamically inside the string template
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    # Render maps to specific dynamic host ports automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
