from transformers import pipeline

temp_value = 0.5
top_p_value = 0.5
max_length_value = 10
pipe = pipeline('translation_en_to_fr')

# looks like temp and top_p don't matter when doing translations
# cannot set max_length to 1
def eng_to_frn_exercise(string):
    generate = pipe(string,max_length=max_length_value, temperature = temp_value, top_p = top_p_value)
    print('************************************')
    print(generate[0])

eng_to_frn_exercise('today is a beautiful day')

#{'translation_text': "aujourd'hui est une belle journée"}