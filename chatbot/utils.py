import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def detect_symptoms(symptoms):
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
    x = pipe(symptoms)
    output_symptoms = []
    for i in x:
        if i["score"] > 0.8:
            output_symptoms.append(i["word"])

    return output_symptoms