from flask import Flask, request, send_file, render_template
from PIL import Image, ImageDraw, ImageFont
import io
import torch
from torchvision.transforms import ToTensor
from transformers import DetrImageProcessor, DetrForObjectDetection

app = Flask(__name__)

# Load the pre-trained DETR model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")

# Get the COCO class labels
coco_classes = model.config.id2label

@app.route('/')
def index():
    return render_template('index_DETR.html')

@app.route('/annotate', methods=['POST'])
def annotate_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    
    # Transform the image and prepare it for the model
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)  # Set the font size to 20
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{coco_classes[label.item()]}: {score:.2f}", fill="red", font=font)

    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
