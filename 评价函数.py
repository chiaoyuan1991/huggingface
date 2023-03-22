#coding = utf-8
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier(input())[0]
print(result)