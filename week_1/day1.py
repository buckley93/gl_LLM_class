from transformers import pipeline

temp_value = 0.8
top_p_value = 0.2
max_length_value = 200

generation = pipeline("text-generation")
print(generation("Write a poem about Robots",max_length=max_length_value,temperature=temp_value, top_p=top_p_value))