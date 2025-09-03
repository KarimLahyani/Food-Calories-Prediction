from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class FoodCaloriePredictor:
    def __init__(self):
        self.model = load_model("food_calorie_predictor/food101_model.keras", compile=False)
        with open("food_calorie_predictor/class_names.txt", "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.food_calories = {
            'apple_pie': 320, 'baby_back_ribs': 300, 'baklava': 300, 'beef_carpaccio': 300, 'beef_tartare': 180,
            'beet_salad': 150, 'beignets': 300, 'bibimbap': 300, 'bread_pudding': 250, 'breakfast_burrito': 350,
            'bruschetta': 300, 'caesar_salad': 150, 'cannoli': 300, 'caprese_salad': 150, 'carrot_cake': 300,
            'ceviche': 300, 'cheesecake': 300, 'cheese_plate': 330, 'chicken_curry': 450, 'chicken_quesadilla': 300,
            'chicken_wings': 300, 'chocolate_cake': 300, 'chocolate_mousse': 260, 'churros': 300, 'clam_chowder': 300,
            'club_sandwich': 350, 'crab_cakes': 300, 'creme_brulee': 300, 'croque_madame': 300, 'cup_cakes': 300,
            'deviled_eggs': 160, 'donuts': 280, 'dumplings': 300, 'edamame': 300, 'eggs_benedict': 160,
            'escargots': 300, 'falafel': 300, 'filet_mignon': 300, 'fish_and_chips': 300, 'foie_gras': 300,
            'french_fries': 365, 'french_onion_soup': 250, 'french_toast': 150, 'fried_calamari': 300,
            'fried_rice': 250, 'frozen_yogurt': 300, 'garlic_bread': 250, 'gnocchi': 300, 'greek_salad': 150,
            'grilled_cheese_sandwich': 350, 'grilled_salmon': 300, 'guacamole': 300, 'gyoza': 300, 'hamburger': 500,
            'hot_and_sour_soup': 250, 'hot_dog': 300, 'huevos_rancheros': 300, 'hummus': 300, 'ice_cream': 200,
            'lasagna': 300, 'lobster_bisque': 300, 'lobster_roll_sandwich': 350, 'macaroni_and_cheese': 330,
            'macarons': 300, 'miso_soup': 250, 'mussels': 300, 'nachos': 300, 'omelette': 300, 'onion_rings': 300,
            'oysters': 300, 'pad_thai': 300, 'paella': 300, 'pancakes': 300, 'panna_cotta': 300, 'peking_duck': 400,
            'pho': 300, 'pizza': 300, 'pork_chop': 300, 'poutine': 300, 'prime_rib': 300, 'pulled_pork_sandwich': 350,
            'ramen': 300, 'ravioli': 300, 'red_velvet_cake': 300, 'risotto': 300, 'samosa': 300, 'sashimi': 300,
            'scallops': 300, 'seaweed_salad': 150, 'shrimp_and_grits': 320, 'spaghetti_bolognese': 300,
            'spaghetti_carbonara': 420, 'spring_rolls': 350, 'steak': 450, 'strawberry_shortcake': 300,
            'sushi': 300, 'tacos': 300, 'takoyaki': 300, 'tiramisu': 300, 'tuna_tartare': 180, 'waffles': 300
        }

    def predict_food(self, image_path):
        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array)
            food_index = np.argmax(predictions, axis=-1)[0]
            confidence = float(np.max(predictions))
            if confidence < 0.7:
                return None, None, None

            food_type = self.class_names[food_index]
            calories = self.food_calories.get(food_type, None)

            return food_type, calories, confidence
        except Exception as e:
            print(f"Error predicting food with TensorFlow: {e}")
            return None, None, None
