from flask import Blueprint, request, jsonify, render_template
from services.pdf_service import process_pdf

api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def home():
    return render_template('index.html')

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    pdf_file = request.files['pdf']
    raw_questions = request.form.get('questions', '')

    if not raw_questions.strip():
        return jsonify({"error": "No questions provided."}), 400

    # Split questions by new line
    questions = [q.strip() for q in raw_questions.splitlines() if q.strip()]

    # Process the PDF and return results
    try:
        answers = process_pdf(pdf_file, questions)
        return jsonify(answers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

