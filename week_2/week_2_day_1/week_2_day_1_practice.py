from openai import OpenAI

client = OpenAI(api_key="sk-proj-YPUWGGA25k1hkcbYpjblT3BlbkFJiF44K6iS1XPhpEFvHOAH")

images = client.images.generate(
    model="dall-e-3", prompt="chimpanzees playing vollyball", n=1, size="1024x1024"
)

print(images)
