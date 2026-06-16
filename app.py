import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Load your trained model
MODEL_PATH = "decision_pkl.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# 2. Define standard lookup mappings based on categorical index positions
MAKE_OPTIONS = [
    {"id": 0, "name": "Audi"},
    {"id": 1, "name": "BMW"},
    {"id": 2, "name": "Ford"},
    {"id": 3, "name": "Mercedes-Benz"},
    {"id": 4, "name": "Toyota"},
    {"id": 5, "name": "Volkswagen"},
    {"id": 6, "name": "Hyundai"}
]

MODEL_OPTIONS = [
    {"id": 0, "name": "A3"}, {"id": 1, "name": "A4"}, {"id": 2, "name": "A6"},
    {"id": 3, "name": "3 Series"}, {"id": 4, "name": "5 Series"}, {"id": 5, "name": "X5"},
    {"id": 6, "name": "Fiesta"}, {"id": 7, "name": "Focus"}, {"id": 8, "name": "Mustang"},
    {"id": 9, "name": "C-Class"}, {"id": 10, "name": "E-Class"}, {"id": 11, "name": "Corolla"},
    {"id": 12, "name": "Yaris"}, {"id": 13, "name": "Golf"}, {"id": 14, "name": "Polo"},
    {"id": 15, "name": "Tucson"}, {"id": 16, "name": "i30"}
]

FUEL_OPTIONS = [
    {"id": 0, "name": "Petrol"},
    {"id": 1, "name": "Diesel"},
    {"id": 2, "name": "Hybrid"},
    {"id": 3, "name": "Electric"}
]

TRANSMISSION_OPTIONS = [
    {"id": 0, "name": "Manual"},
    {"id": 1, "name": "Automatic"},
    {"id": 2, "name": "Semi-Automatic"}
]

# 3. HTML Layout Template with interactive drop-downs and responsive styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Value Estimator Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 font-sans antialiased min-h-screen flex flex-col justify-between">

    <header class="bg-white border-b border-slate-200 py-4 shadow-xs">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 class="text-xl font-bold text-slate-800 tracking-tight flex items-center gap-2">
                🚗 Automated Valuation Engine
            </h1>
            <span class="px-2.5 py-1 text-xs font-semibold text-emerald-700 bg-emerald-50 rounded-full border border-emerald-200">
                Model Instance Online
            </span>
        </div>
    </header>

    <main class="max-w-4xl mx-auto w-full px-4 py-8 flex-grow">
        <div class="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden grid grid-cols-1 md:grid-cols-5">
            
            <form action="/" method="POST" class="p-6 sm:p-8 md:col-span-3 space-y-4">
                <div>
                    <h2 class="text-lg font-bold text-slate-800">Vehicle Specifications</h2>
                    <p class="text-xs text-slate-400 mt-0.5">Provide specifications to compute estimated predictive output matrix metrics.</p>
                </div>
                
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Brand Make</label>
                        <select name="make" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm">
                            {% for option in make_opts %}
                                <option value="{{ option.id }}">{{ option.name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Variant Model</label>
                        <select name="model" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm">
                            {% for option in model_opts %}
                                <option value="{{ option.id }}">{{ option.name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Production Year</label>
                        <input type="number" name="year" min="1980" max="2030" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm" value="2022">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Engine displacement (Litre)</label>
                        <input type="number" name="engine_size" step="0.1" min="0.0" max="10.0" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm" value="2.0">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Total Distance (Odometer Mileage)</label>
                    <input type="number" name="mileage" step="any" min="0" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm" placeholder="e.g. 35000">
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Fuel Core Profile</label>
                        <select name="fuel_type" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm">
                            {% for option in fuel_opts %}
                                <option value="{{ option.id }}">{{ option.name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Transmission Assembly</label>
                        <select name="transmission" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition text-sm">
                            {% for option in trans_opts %}
                                <option value="{{ option.id }}">{{ option.name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <button type="submit" class="w-full mt-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg shadow-md transition duration-150 ease-in-out cursor-pointer text-center">
                    Compute Evaluation Output
                </button>
            </form>

            <div class="bg-slate-900 p-6 sm:p-8 md:col-span-2 flex flex-col justify-center text-white text-center border-t md:border-t-0 md:border-l border-slate-800">
                {% if prediction is not none %}
                    <p class="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Evaluated Decision Output</p>
                    <p class="text-4xl sm:text-5xl font-extrabold text-blue-400 tracking-tight drop-shadow-xs">${{ prediction }}</p>
                    <p class="text-xs text-slate-400 mt-4 leading-relaxed">Regression algorithm inference mapped against user parameters completed successfully.</p>
                {% elif error %}
                    <div class="text-red-400 bg-red-950/40 p-4 border border-red-900/50 rounded-lg text-xs leading-relaxed text-left">
                        <strong class="block font-bold mb-1">⚠️ Computation Exception</strong>
                        {{ error }}
                    </div>
                {% else %}
                    <div class="space-y-2">
                        <div class="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto text-slate-400 text-xl animate-pulse">📊</div>
                        <p class="text-slate-300 font-medium">Awaiting Matrix Feed</p>
                        <p class="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">Select criteria values within variables parameters structure to fetch decision matrix estimations.</p>
                    </div>
                {% endif %}
            </div>
        </div>
    </main>

    <footer class="text-center py-4 text-xs text-slate-400 border-t border-slate-200 bg-white">
        Deploy Target Core Engine Stack: Flask Micro-Framework | Provider Platform: Render Services
    </footer>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            # Parse parameters transmitted through interactive select forms
            make = float(request.form.get("make", 0))
            model_type = float(request.form.get("model", 0))
            year = float(request.form.get("year", 2022))
            engine_size = float(request.form.get("engine_size", 2.0))
            mileage = float(request.form.get("mileage", 0))
            fuel_type = float(request.form.get("fuel_type", 0))
            transmission = float(request.form.get("transmission", 0))
            
            # Pack values precisely following model feature inputs mapping specification
            features = np.array([[make, model_type, year, engine_size, mileage, fuel_type, transmission]])
            
            # Execute data model runtime logic calculations
            pred_output = model.predict(features)
            prediction = round(pred_output[0], 2)
            
        except Exception as e:
            error = f"Inference execution engine error: {str(e)}"
            
    # Inject variables parameters and lookup arrays onto the integrated layout string template
    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction, 
        error=error,
        make_opts=MAKE_OPTIONS,
        model_opts=MODEL_OPTIONS,
        fuel_opts=FUEL_OPTIONS,
        trans_opts=TRANSMISSION_OPTIONS
    )

if __name__ == "__main__":
    # Render runtime micro-service infrastructure configurations hook binding
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
