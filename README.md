# Food Calorie Predictor Web App

A Django-based web application that predicts calories from food images using machine learning. The app features a beautiful, modern UI with drag-and-drop functionality and real-time predictions.

## Features

- 🍽️ **Food Image Upload**: Drag & drop or click to upload food images
- 🔮 **AI-Powered Predictions**: Machine learning model for calorie prediction
- 📊 **Real-time Results**: Instant calorie predictions with confidence scores
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 📈 **Prediction History**: View recent predictions with images
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations

## Technology Stack

- **Backend**: Django 4.2.7
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Machine Learning**: TensorFlow 2.15.0
- **Image Processing**: OpenCV, Pillow
- **Database**: SQLite (can be easily changed to PostgreSQL/MySQL)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create a superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

7. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

## Usage

### Making Predictions

1. **Upload an Image**:
   - Drag and drop a food image onto the upload area
   - Or click the upload area to browse and select an image
   - Supported formats: JPG, PNG, GIF, etc.

2. **Get Predictions**:
   - Click the "Predict Calories" button
   - Wait for the AI model to analyze the image
   - View the predicted food type and calorie count

3. **View History**:
   - Scroll down to see recent predictions
   - Each prediction shows the image, food type, calories, and timestamp

### API Endpoints

- `GET /` - Main web interface
- `POST /api/predict/` - Upload image and get calorie prediction
- `GET /api/predictions/` - Get recent prediction history

## Machine Learning Model

The application uses a custom machine learning model for food classification and calorie prediction:

### Model Features:
- **Food Classification**: Identifies common food types
- **Calorie Estimation**: Provides calorie estimates based on food type
- **Confidence Scoring**: Shows prediction confidence levels

### Supported Food Types:
- Fruits: Apple, Banana, Orange, Grapes, Strawberries, Blueberries
- Vegetables: Carrots, Broccoli, Tomatoes, Potatoes, Corn, Peas
- Proteins: Chicken, Beef, Fish, Eggs
- Grains: Rice, Bread, Pasta
- Dairy: Milk, Cheese, Yogurt
- Snacks: Pizza, Burgers, Hotdogs, Sandwiches
- Desserts: Ice Cream, Cake, Cookies, Chocolate
- Beverages: Coffee, Tea, Juice, Soda, Beer, Wine

## Project Structure

```
calorie_predictor/
├── calorie_predictor/          # Main Django project
│   ├── __init__.py
│   ├── settings.py             # Django settings
│   ├── urls.py                 # Main URL configuration
│   └── wsgi.py                 # WSGI configuration
├── food_calorie_predictor/     # Main Django app
│   ├── __init__.py
│   ├── apps.py                 # App configuration
│   ├── models.py               # Database models
│   ├── views.py                # View functions
│   ├── urls.py                 # App URL patterns
│   └── ml_model.py             # Machine learning model
├── templates/                  # HTML templates
│   └── food_calorie_predictor/
│       └── index.html          # Main web interface
├── media/                      # Uploaded images (created automatically)
├── static/                     # Static files (CSS, JS, images)
├── manage.py                   # Django management script
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Customization

### Adding New Food Types

To add support for new food types, edit the `food_calories` dictionary in `food_calorie_predictor/ml_model.py`:

```python
self.food_calories = {
    'new_food': 150,  # Add new food with calorie count
    # ... existing foods
}
```

### Improving the ML Model

For production use, consider:

1. **Using a pre-trained model** like ResNet or EfficientNet
2. **Training on a larger dataset** of food images
3. **Implementing portion size detection**
4. **Adding nutritional information** (protein, carbs, fat)

### Database Configuration

To use a different database (e.g., PostgreSQL):

1. Install the database adapter:
   ```bash
   pip install psycopg2-binary
   ```

2. Update `settings.py`:
   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': 'your_db_name',
           'USER': 'your_username',
           'PASSWORD': 'your_password',
           'HOST': 'localhost',
           'PORT': '5432',
       }
   }
   ```

## Deployment

### Production Settings

Before deploying to production:

1. **Update settings.py**:
   ```python
   DEBUG = False
   SECRET_KEY = 'your-secure-secret-key'
   ALLOWED_HOSTS = ['your-domain.com']
   ```

2. **Set up static files**:
   ```bash
   python manage.py collectstatic
   ```

3. **Use a production server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn calorie_predictor.wsgi:application
   ```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "calorie_predictor.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'tensorflow'**
   - Ensure you're in the virtual environment
   - Run `pip install -r requirements.txt`

2. **Database errors**
   - Run `python manage.py migrate`

3. **Static files not loading**
   - Run `python manage.py collectstatic`

4. **Image upload errors**
   - Ensure the `media/` directory exists
   - Check file permissions

### Performance Tips

- Use a CDN for static files in production
- Implement image compression
- Consider using Redis for caching
- Use a production-grade database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review Django and TensorFlow documentation
- Create an issue in the project repository

---

**Note**: This is a demonstration project. For production use, consider using more sophisticated ML models and implementing proper security measures. 