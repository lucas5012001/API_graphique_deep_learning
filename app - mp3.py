from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

app = Flask(__name__)

# Load pre-trained Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

@app.route('/')
def index():
    return render_template('index_mp3.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    audio = AudioSegment.from_file(file, format='mp3')
    
    # Convert audio to mono and sample rate to 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = audio.get_array_of_samples()
    input_values = processor(torch.tensor(samples), return_tensors="pt", sampling_rate=16000).input_values

    # Perform speech-to-text
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)