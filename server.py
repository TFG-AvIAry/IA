from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import torch
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import PIL.Image as Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Rutas de los modelos y clases
model_path = 'C:/Users/Josemari/Desktop/Ingeniería del Software/IA/best_model.pth'
clases_path = 'C:/Users/Josemari/Desktop/Ingeniería del Software/IA/pajaros2/birds/birds'
tests_path = 'C:/Users/Josemari/Desktop/Ingeniería del Software/IA/pajaros2/submission_test/submission_test2'

# Cargar el modelo
model = torch.load(model_path)

# Cargar las clases
classes = os.listdir(clases_path)

# Transformaciones de imagen
mean = [0.4704, 0.4669, 0.3898]
std = [0.2037, 0.2002, 0.2051]

def load_and_preprocess_image(image_path):
    # Cargar la imagen
    image = Image.open(image_path)
    
    # Convertir la imagen a RGB (si es RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Aplicar transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    image = transform(image).float()
    image = image.unsqueeze(0)
    
    return image

def classify_image(image, model, classes, topk=5):
    model = model.eval()
    output = model(image)
    probabilities, predicted_indices = torch.topk(output, topk)
    
    results = []
    for i in range(topk):
        class_index = predicted_indices[0][i].item()
        class_label = classes[class_index]
        confidence = probabilities[0][i].item()
        results.append({"class": class_label, "confidence": confidence})
    
    return results

# Ruta para cargar una imagen y obtener la clasificación
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No se recibió una imagen válida'})
    
    file = request.files['image']
    basepath = os.path.dirname(__file__)
    filename = secure_filename(file.filename)
    upload_path = os.path.join(basepath, 'static/archivos', filename)
    file.save(upload_path)

    image = load_and_preprocess_image(os.path.join('static/archivos', filename))
    resultados = classify_image(image, model, classes)
    
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
