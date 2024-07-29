# sentiment-analysis does not take the keywork argument 'temperature or top_p'
from transformers import pipeline

temp_value = 0.8
top_p_value = 0.2
max_length_value = 1
pipe = pipeline('sentiment-analysis')

def sentiment_exercise(string):
    generate = pipe(string,max_length=max_length_value)
    print('************************************')
    print(generate[0])

sentiment_exercise('today is beautiful')
sentiment_exercise('could today get any better')

