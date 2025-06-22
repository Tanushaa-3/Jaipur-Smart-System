#!/usr/bin/env python3
"""
Jaipur Smart System - Gradio Deployment (Fixed)
Interactive web interface using Gradio for the Jaipur Smart System
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import pytz
import requests
import joblib
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class JaipurSmartSystem:
    def __init__(self):
        self.ist = pytz.timezone("Asia/Kolkata")
        self.weather_api_key = "169fcce991bed1970d27eb53aeac8b0b"
        self.holiday_api_key = "RUGDUZpRzrzweMgB9ivBMCWtCGRpNG4F"
        self.jaipur_locations = [
            'C-Scheme', 'Malviya Nagar', 'Vaishali Nagar', 'Mansarovar',
            'Jagatpura', 'Tonk Road', 'Ajmer Road', 'Sikar Road',
            'Jhotwara', 'Sanganer', 'Bani Park', 'Raja Park',
            'Shyam Nagar', 'Sodala', 'Civil Lines'
        ]
        self.location_model = None
        self.label_encoder = None
        self.load_location_model()

    def load_location_model(self):
        try:
            if os.path.exists('jaipur_location_predictor.pkl'):
                self.location_model = joblib.load('jaipur_location_predictor.pkl')
                self.label_encoder = joblib.load('location_label_encoder.pkl')
                print("✅ Location prediction model loaded successfully!")
                return True
            else:
                print("⚠️ Location model files not found.")
                return False
        except Exception as e:
            print(f"⚠️ Error loading location model: {e}")
            return False

    def get_coordinates(self, location):
        try:
            geolocator = Nominatim(user_agent="jaipur_smart_system")
            location_data = geolocator.geocode(location + ", Jaipur, India")
            if location_data:
                return location_data.latitude, location_data.longitude
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error getting coordinates: {e}")
            return None, None

    def get_distance_km(self, source, destination):
        try:
            source_lat, source_lon = self.get_coordinates(source)
            dest_lat, dest_lon = self.get_coordinates(destination)

            if source_lat is None or dest_lat is None:
                return None

            source_coords = (source_lat, source_lon)
            dest_coords = (dest_lat, dest_lon)
            distance = geodesic(source_coords, dest_coords).kilometers
            return round(distance, 2)
        except Exception as e:
            print(f"⚠️ Error calculating distance: {e}")
            return 8.5  # Return default distance if calculation fails

    def get_weather_forecast(self, city="Jaipur"):
        try:
            current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.weather_api_key}&units=metric"

            current_response = requests.get(current_url, timeout=10)
            forecast_response = requests.get(forecast_url, timeout=10)

            weather_data = {}

            if current_response.status_code == 200:
                current_data = current_response.json()
                weather_data['current'] = {
                    'condition': current_data["weather"][0]["main"].lower(),
                    'description': current_data["weather"][0]["description"],
                    'temp': current_data["main"]["temp"],
                    'humidity': current_data["main"]["humidity"],
                    'pressure': current_data["main"]["pressure"]
                }
            else:
                # Default weather data if API fails
                weather_data['current'] = {
                    'condition': 'clear',
                    'description': 'Clear sky',
                    'temp': 28,
                    'humidity': 65,
                    'pressure': 1013
                }

            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                weather_data['forecast'] = []
                for item in forecast_data['list'][:8]:
                    utc_time = datetime.datetime.fromtimestamp(item['dt'], tz=pytz.UTC)
                    ist_time = utc_time.astimezone(self.ist)
                    weather_data['forecast'].append({
                        'datetime': ist_time,
                        'condition': item['weather'][0]['main'].lower(),
                        'temp': item['main']['temp']
                    })

            return weather_data
        except Exception as e:
            print(f"⚠️ Error fetching weather: {e}")
            return {
                'current': {'condition': 'clear', 'temp': 28, 'description': 'Clear sky', 'humidity': 65, 'pressure': 1013},
                'forecast': []
            }

    def is_today_holiday(self):
        """Check if today is a holiday"""
        try:
            today = datetime.datetime.today().strftime('%Y-%m-%d')
            year = today[:4]
            url = f"https://calendarific.com/api/v2/holidays?api_key={self.holiday_api_key}&country=IN&year={year}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                holidays = data.get("response", {}).get("holidays", [])

                for holiday in holidays:
                    if holiday["date"]["iso"] == today:
                        return True, holiday['name']
                return False, None
            else:
                return False, None

        except Exception as e:
            print(f"⚠️ Error checking holidays: {e}")
            return False, None

    def get_current_time_features(self):
        """Extract time-based features from current time"""
        current_time = datetime.datetime.now(self.ist)

        features = {
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),  # 0=Monday, 6=Sunday
            'day_of_month': current_time.day,
            'month': current_time.month,
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            'is_rush_hour': 1 if current_time.hour in [7, 8, 9, 17, 18, 19, 20] else 0,
            'is_late_night': 1 if current_time.hour >= 23 or current_time.hour <= 5 else 0,
            'is_peak_evening': 1 if current_time.hour in [18, 19, 20] else 0
        }

        return features, current_time

    def analyze_weather_conditions(self, weather_data):
        """Analyze weather conditions and return relevant flags"""
        current_weather = weather_data.get('current', {})
        condition = current_weather.get('condition', 'clear').lower()
        temp = current_weather.get('temp', 25)

        # Weather condition mapping for model
        weather_mapping = {
            'clear': 'clear',
            'clouds': 'clouds',
            'rain': 'rain',
            'drizzle': 'rain',
            'thunderstorm': 'thunderstorm',
            'snow': 'rain',      # fallback
            'mist': 'clouds',
            'fog': 'clouds',
            'haze': 'clouds'
        }

        mapped_weather = weather_mapping.get(condition, 'clear')

        # Determine extreme weather
        is_extreme = 1 if (temp > 40 or temp < 5 or condition in ['thunderstorm', 'snow']) else 0

        return mapped_weather, temp, is_extreme

    def predict_favorable_location(self, weather_data, is_festival=False, is_special_event=False):
        """Predict favorable location based on current conditions with proper preprocessing"""

        if self.location_model is None or self.label_encoder is None:
            print("⚠️ Location model not loaded. Using fallback prediction...")
            # Enhanced fallback logic
            return self._fallback_location_prediction(weather_data, is_festival, is_special_event)

        try:
            # Get current time features
            time_features, current_time = self.get_current_time_features()

            # Get weather features with proper preprocessing
            weather_condition, temp, is_extreme_weather = self.analyze_weather_conditions(weather_data)

            # Check if today is a holiday
            is_holiday_today, _ = self.is_today_holiday()

            # Create input data for prediction with proper feature names and preprocessing
            input_data = pd.DataFrame({
                'hour': [time_features['hour']],
                'day_of_week': [time_features['day_of_week']],
                'day_of_month': [time_features['day_of_month']],
                'month': [time_features['month']],
                'is_rush_hour': [time_features['is_rush_hour']],
                'is_weekend': [time_features['is_weekend']],
                'weather': [weather_condition],
                'is_late_night': [time_features['is_late_night']],
                'is_peak_evening': [time_features['is_peak_evening']],
                'temperature': [temp],
                'is_extreme_weather': [is_extreme_weather],
                'is_festival': [1 if is_festival else 0],
                'is_special_event': [1 if is_special_event else 0]
            })

            print(f"🔍 Prediction input features:")
            print(f"   Time: {current_time.strftime('%H:%M, %A')}")
            print(f"   Weather: {weather_condition}, {temp}°C")
            print(f"   Special conditions: Festival={is_festival}, Event={is_special_event}")

            # Make prediction
            prediction = self.location_model.predict(input_data)[0]
            probabilities = self.location_model.predict_proba(input_data)[0]

            # Get location name
            predicted_location = self.label_encoder.inverse_transform([prediction])[0]

            # Get confidence scores for all locations
            confidence_scores = dict(zip(self.label_encoder.classes_, probabilities))
            confidence_scores = {k: v for k, v in sorted(confidence_scores.items(),
                                                        key=lambda x: x[1], reverse=True)}

            print(f"✅ ML Model prediction: {predicted_location} (confidence: {confidence_scores[predicted_location]:.3f})")

            return predicted_location, confidence_scores

        except Exception as e:
            print(f"⚠️ Error in ML prediction: {e}")
            print("   Falling back to rule-based prediction...")
            return self._fallback_location_prediction(weather_data, is_festival, is_special_event)

    def _fallback_location_prediction(self, weather_data, is_festival=False, is_special_event=False):
        """Enhanced fallback prediction when ML model is not available"""
        current_time = datetime.datetime.now(self.ist)
        hour = current_time.hour
        day = current_time.weekday()
        weather_condition = weather_data['current']['condition']
        temp = weather_data['current']['temp']

        # Initialize location scores
        location_scores = {}

        # Base scoring for each location with realistic weights
        base_scores = {
            'C-Scheme': 0.75,      # Commercial hub
            'Malviya Nagar': 0.70,  # Mixed residential/commercial
            'Vaishali Nagar': 0.65, # Residential with good connectivity
            'Mansarovar': 0.60,     # Developing area
            'Civil Lines': 0.68,    # Government area
            'Raja Park': 0.58,      # Residential
            'Bani Park': 0.55,      # Traditional area
            'Jagatpura': 0.50,      # Suburban
            'Tonk Road': 0.52,      # Highway area
            'Ajmer Road': 0.54,     # Industrial/commercial
            'Sikar Road': 0.48,     # Suburban
            'Jhotwara': 0.45,       # Industrial
            'Sanganer': 0.47,       # Airport area, industrial
            'Shyam Nagar': 0.42,    # Residential
            'Sodala': 0.40          # Developing area
        }

        for location in self.jaipur_locations:
            score = base_scores.get(location, 0.5)

            # Time-based adjustments
            if 9 <= hour <= 18:  # Business hours
                if location in ['C-Scheme', 'Civil Lines', 'Malviya Nagar']:
                    score += 0.15  # Business districts
                elif location in ['Vaishali Nagar', 'Raja Park']:
                    score += 0.08  # Mixed areas
            elif 18 <= hour <= 22:  # Evening hours
                if location in ['C-Scheme', 'Malviya Nagar', 'Vaishali Nagar']:
                    score += 0.12  # Shopping and dining areas
                elif location in ['Raja Park', 'Bani Park']:
                    score += 0.10  # Residential areas with evening activity
            elif 22 <= hour or hour <= 6:  # Late night/early morning
                if location in ['C-Scheme']:
                    score += 0.05  # 24-hour activity
                else:
                    score -= 0.10  # Most areas less active

            # Day-based adjustments
            if day >= 5:  # Weekends
                if location in ['C-Scheme', 'Malviya Nagar']:
                    score += 0.10  # Weekend shopping/leisure
                elif location in ['Civil Lines']:
                    score -= 0.05  # Government area less active

            # Weather-based adjustments
            if weather_condition in ['rain', 'thunderstorm']:
                if location in ['C-Scheme', 'Malviya Nagar']:
                    score += 0.08  # Better infrastructure for bad weather
                else:
                    score -= 0.05  # Other areas may have drainage issues
            elif temp > 35:  # Hot weather
                if location in ['C-Scheme', 'Malviya Nagar', 'Vaishali Nagar']:
                    score += 0.05  # Areas with AC establishments

            # Special event adjustments
            if is_festival:
                if location in ['C-Scheme', 'Bani Park', 'Civil Lines']:
                    score += 0.12  # Areas with cultural significance

            if is_special_event:
                if location in ['C-Scheme', 'Malviya Nagar']:
                    score += 0.08  # Event hosting areas

            # Ensure score is within bounds
            location_scores[location] = min(max(score, 0.1), 1.0)

        # Sort by score
        sorted_locations = dict(sorted(location_scores.items(), key=lambda x: x[1], reverse=True))
        top_location = list(sorted_locations.keys())[0]

        print(f"✅ Rule-based prediction: {top_location} (confidence: {sorted_locations[top_location]:.3f})")

        return top_location, sorted_locations

    def calculate_surge_multiplier_for_time(self, target_time, weather_condition, is_holiday=False):
        hour = target_time.hour
        day = target_time.weekday()
        is_weekend = day >= 5

        rush_hour_multipliers = {7: 0.15, 8: 0.35, 9: 0.30, 10: 0.10, 17: 0.15, 18: 0.40, 19: 0.35, 20: 0.25, 21: 0.15}
        late_night_hours = {23: 0.1, 0: 0.2, 1: 0.2, 2: 0.25, 3: 0.25, 4: 0.2, 5: 0.1}

        surge = 1.0
        surge_reasons = []

        if hour in rush_hour_multipliers:
            surge += rush_hour_multipliers[hour]
            surge_reasons.append(f"Rush hour (+{rush_hour_multipliers[hour]})")

        if hour in late_night_hours:
            surge += late_night_hours[hour]
            surge_reasons.append(f"Late night (+{late_night_hours[hour]})")

        if is_weekend:
            if day == 5 and hour >= 20:
                surge += 0.25
                surge_reasons.append("Friday night (+0.25)")
            elif day == 6 and hour >= 20:
                surge += 0.3
                surge_reasons.append("Saturday night (+0.3)")
            elif day == 6:
                surge += 0.2
                surge_reasons.append("Weekend (+0.2)")
            else:
                surge += 0.15
                surge_reasons.append("Sunday (+0.15)")

        if is_holiday:
            surge += 0.2
            surge_reasons.append("Holiday (+0.2)")

        if weather_condition in ['rain', 'rainy']:
            surge += 0.3
            surge_reasons.append(f"Bad weather (+0.3)")
        elif weather_condition in ['thunderstorm']:
            surge += 0.5
            surge_reasons.append(f"Thunderstorm (+0.5)")

        surge = min(surge, 3.5)
        return round(surge, 2), surge_reasons

    def get_base_rate(self, ride_type):
        rates = {"auto": 12, "e_rickshaw": 8, "ac_car": 25, "bike": 6, "cab": 20}
        return rates.get(ride_type.lower(), 20)

    def calculate_final_price(self, ride_type, distance_km, surge_multiplier):
        base_rate = self.get_base_rate(ride_type)
        base_price = base_rate * distance_km
        final_price = base_price * surge_multiplier

        min_fares = {"auto": 25, "e_rickshaw": 20, "ac_car": 75, "bike": 15, "cab": 60}
        min_fare = min_fares.get(ride_type.lower(), 25)
        final_price = max(final_price, min_fare)

        return round(final_price, 2)

    def predict_prices_next_2_hours(self, ride_type, distance_km, weather_data, is_holiday_today=False):
        current_time = datetime.datetime.now(self.ist)
        predictions = []

        for i in range(5):
            future_time = current_time + datetime.timedelta(minutes=i*30)
            weather_condition = weather_data['current']['condition']

            surge, reasons = self.calculate_surge_multiplier_for_time(
                future_time, weather_condition, is_holiday_today
            )

            price = self.calculate_final_price(ride_type, distance_km, surge)

            predictions.append({
                'time': future_time,
                'surge': surge,
                'price': price,
                'weather': weather_condition,
                'reasons': reasons
            })

        return predictions

# Initialize the system
system = JaipurSmartSystem()

def create_price_prediction_chart(predictions):
    """Create an interactive price prediction chart"""
    times = [pred['time'].strftime('%H:%M') for pred in predictions]
    prices = [pred['price'] for pred in predictions]
    surges = [pred['surge'] for pred in predictions]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Forecast', 'Surge Multiplier'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    # Price line chart
    fig.add_trace(
        go.Scatter(
            x=times, y=prices,
            mode='lines+markers',
            name='Price (₹)',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            hovertemplate='<b>Time:</b> %{x}<br><b>Price:</b> ₹%{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # Surge bar chart
    fig.add_trace(
        go.Bar(
            x=times, y=surges,
            name='Surge Multiplier',
            marker_color='#ff6b6b',
            hovertemplate='<b>Time:</b> %{x}<br><b>Surge:</b> %{y}x<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title_text="Price & Surge Predictions - Next 2 Hours",
        showlegend=False,
        height=500,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Surge", row=2, col=1)

    return fig

def create_location_confidence_chart(location_scores):
    """Create location confidence visualization"""
    locations = list(location_scores.keys())[:8]  # Top 8 locations
    confidences = [location_scores[loc] * 100 for loc in locations] # Corrected indexing

    fig = go.Figure(data=[
        go.Bar(
            x=locations,
            y=confidences,
            marker_color=px.colors.qualitative.Set3[:len(locations)],
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Location Favorability Ranking",
        xaxis_title="Locations",
        yaxis_title="Confidence (%)",
        template="plotly_white",
        height=400
    )

    fig.update_xaxes(tickangle=45)

    return fig

def predict_ride_prices(source, destination, ride_type):
    """Main function for ride price prediction"""
    if not source or not destination:
        return "Please enter both source and destination locations.", None, ""

    try:
        # Get distance
        distance = system.get_distance_km(source, destination)
        if distance is None:
            distance = 8.5  # Default distance

        # Get weather data
        weather_data = system.get_weather_forecast()

        # Check for holidays
        is_holiday_today, _ = system.is_today_holiday()

        # Generate predictions
        predictions = system.predict_prices_next_2_hours(ride_type, distance, weather_data, is_holiday_today)

        # Create chart
        chart = create_price_prediction_chart(predictions)

        # Generate summary text
        current_price = predictions[0]['price']
        min_price = min(pred['price'] for pred in predictions)
        max_price = max(pred['price'] for pred in predictions)
        best_time = next(pred for pred in predictions if pred['price'] == min_price)

        summary = f"""
         🚗 Ride Price Analysis: {source} → {destination}

        📊 Key Metrics:
        - 📍 Distance: {distance} km
        - 💰 Current Price: ₹{current_price}
        - 🟢 Best Price: ₹{min_price} at {best_time['time'].strftime('%H:%M')}
        - 💸 Potential Savings: ₹{max_price - min_price}

        🌤️ Current Conditions:
        - Weather: {weather_data['current']['description'].title()}
        - Temperature: {weather_data['current']['temp']}°C

        💡 Recommendation:
        {'📈 Prices expected to rise - book now!' if current_price == min_price else f'⏰ Wait until {best_time["time"].strftime("%H:%M")} for best price!'}
        """

        return summary, chart, f"Ride type: {ride_type.title()} | Base rate: ₹{system.get_base_rate(ride_type)}/km"

    except Exception as e:
        return f"Error: {str(e)}", None, ""

def predict_favorable_locations(is_festival, is_special_event):
    """Main function for location prediction with enhanced processing"""
    try:
        # Get weather data
        weather_data = system.get_weather_forecast()

        # Make prediction with proper preprocessing
        top_location, location_scores = system.predict_favorable_location(
            weather_data, is_festival, is_special_event
        )

        # Create chart
        chart = create_location_confidence_chart(location_scores)

        # Generate enhanced summary
        current_time = datetime.datetime.now(system.ist)
        weather = weather_data['current']
        time_features, _ = system.get_current_time_features()

        # Get top 5 locations
        top_5 = list(location_scores.items())[:5]
        top_5_text = "\n".join([f"{i+1}. **{loc}** - {score*100:.1f}% confidence"
                               for i, (loc, score) in enumerate(top_5)])

        # Context-based insights
        insights = []
        if time_features['is_rush_hour']:
            insights.append("🚦 Rush hour detected - expect heavy traffic in recommended areas")
        if time_features['is_weekend']:
            insights.append("🎉 Weekend activity - leisure and shopping areas prioritized")
        if weather['condition'] in ['rain', 'thunderstorm']:
            insights.append("🌧️ Bad weather - indoor/covered locations recommended")
        elif weather['temp'] > 35:
            insights.append("🌡️ Hot weather - air-conditioned venues preferred")
        if time_features['is_peak_evening']:
            insights.append("🌆 Peak evening - entertainment and dining hotspots")
        if time_features['is_late_night']:
            insights.append("🌙 Late hours - 24-hour and safe areas prioritized")

        insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "- Standard conditions apply"

        summary = f"""
         📍 Favorable Location Recommendations

        🎯 Most Favorable Location: {top_location}

        ⏰ Current Conditions:
        - Time: {current_time.strftime('%H:%M, %A')}
        - Weather: {weather['description'].title()}
        - Temperature: {weather['temp']}°C

        🏆 Top 5 Recommendations:
        {top_5_text}

        🎭 Special Events:
        {'✅ Festival mode active' if is_festival else '❌ No festival'}
        {'✅ Special event mode active' if is_special_event else '❌ No special event'}

        💡 Context Insights:
        {insights_text}
        """

        return summary, chart

    except Exception as e:
        return f"Error: {str(e)}", None

# Create Gradio Interface
with gr.Blocks(title="Jaipur Smart System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌟 Jaipur Smart System
    ## AI-Powered Ride Pricing & Location Intelligence

    Welcome to the intelligent transportation system for Jaipur! Get real-time price predictions and discover the most favorable locations in the city.
    """)

    with gr.Tabs():
        # Tab 1: Ride Price Prediction
        with gr.TabItem("🚗 Ride Price Predictor"):
            gr.Markdown("### Predict ride prices for the next 2 hours")

            with gr.Row():
                with gr.Column():
                    source_input = gr.Textbox(
                        label="📍 Source Location",
                        placeholder="Enter pickup location (e.g., C-Scheme, Jaipur)",
                        value="C-Scheme"
                    )
                    destination_input = gr.Textbox(
                        label="📍 Destination",
                        placeholder="Enter destination (e.g., Malviya Nagar, Jaipur)",
                        value="Malviya Nagar"
                    )
                    ride_type_input = gr.Dropdown(
                        label="🚗 Ride Type",
                        choices=[
                            ("Auto-rickshaw (₹12/km)", "auto"),
                            ("E-rickshaw (₹8/km)", "e_rickshaw"),
                            ("Bike taxi (₹6/km)", "bike"),
                            ("Regular cab (₹20/km)", "cab"),
                            ("AC car (₹25/km)", "ac_car")
                        ],
                        value="auto"
                    )
                    predict_price_btn = gr.Button("🔮 Predict Prices", variant="primary")

                with gr.Column():
                    price_summary = gr.Markdown("Select locations and click predict to see analysis.")
                    price_info = gr.Textbox(label="ℹ️ Ride Information", interactive=False)

            price_chart = gr.Plot(label="📊 Price Forecast Chart")

            predict_price_btn.click(
                predict_ride_prices,
                inputs=[source_input, destination_input, ride_type_input],
                outputs=[price_summary, price_chart, price_info]
            )

        # Tab 2: Location Recommendation
        with gr.TabItem("📍 Location Recommender"):
            gr.Markdown("### Find the most favorable locations in Jaipur")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎭 Special Events (Optional)")
                    festival_checkbox = gr.Checkbox(label="🎪 Festival today", value=False)
                    event_checkbox = gr.Checkbox(label="🎉 Special event today", value=False)
                    predict_location_btn = gr.Button("📍 Find Best Locations", variant="primary")

                with gr.Column():
                    location_summary = gr.Markdown("Click the button to get location recommendations.")

            location_chart = gr.Plot(label="📊 Location Confidence Chart")

            predict_location_btn.click(
                predict_favorable_locations,
                inputs=[festival_checkbox, event_checkbox],
                outputs=[location_summary, location_chart]
            )

    # Footer
    gr.Markdown("""
    ---
    **🔧 System Features:**
    - Real-time weather integration
    - Dynamic surge pricing calculation
    - AI-powered location recommendations
    - Interactive visualizations

    **📱 Built with:** Python • Gradio • Plotly • Machine Learning
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow access from any IP
        # server_port=7860,       # Default Gradio port - Removed to allow automatic port selection
        share=True,             # Create public link
        show_error=True         # Show errors in interface
    )