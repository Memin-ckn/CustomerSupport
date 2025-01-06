# CustomerSupport
 Customer Support AI with TTS and STT

python -m venv venv

venv/Scripts/activate

pip install -r requirements.txt

for backend ::
 uvicorn backend.main:app --reload

for frontend ::
 streamlit run frontend\app.py