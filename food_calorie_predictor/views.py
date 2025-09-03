from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from .models import FoodImage
from .ml_model import FoodCaloriePredictor

# Initialize the ML model
predictor = FoodCaloriePredictor()

def index(request):
    """Main page view."""
    return render(request, 'food_calorie_predictor/index.html')

@csrf_exempt
def predict_calories(request):
    """API endpoint for calorie prediction."""
    if request.method == 'POST':
        try:
            
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                
                
                food_image = FoodImage(image=image_file)
                food_image.save()
                
                
                food_type, calories, confidence = predictor.predict_food(food_image.image.path)
                
                
                food_image.predicted_food = food_type
                food_image.predicted_calories = calories
                food_image.confidence_score = confidence
                food_image.save()
                
                return JsonResponse({
                    'success': True,
                    'food_type': food_type,
                    'calories': calories,
                    'confidence': confidence,
                    'image_url': food_image.image.url
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'No image provided'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Only POST method allowed'
    })

def get_predictions(request):
    """Get all previous predictions."""
    predictions = FoodImage.objects.all()[:10]  # Last 10 predictions
    data = []
    
    for pred in predictions:
        data.append({
            'food_type': pred.predicted_food,
            'calories': pred.predicted_calories,
            'confidence': pred.confidence_score,
            'image_url': pred.image.url,
            'uploaded_at': pred.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return JsonResponse({
        'success': True,
        'predictions': data
    }) 