from django.db import models
from django.utils import timezone


class FoodImage(models.Model):
    """Model to store uploaded food images and their predictions."""
    image = models.ImageField(upload_to='food_images/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    predicted_food = models.CharField(max_length=200, blank=True, null=True)
    predicted_calories = models.FloatField(blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.predicted_food} - {self.predicted_calories} calories"
    
    class Meta:
        ordering = ['-uploaded_at'] 