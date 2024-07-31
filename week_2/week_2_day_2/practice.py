# from transformers import pipeline

# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# image_to_text("./tech_picture.jpg")

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st
from pathlib import Path

path = "models/vit-gpt2-image-captioning"

model = VisionEncoderDecoderModel.from_pretrained(path)
feature_extractor = ViTImageProcessor.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

st.title("Image Whisper")
st.subheader("POC Created by Talent Engine Team")
image=st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
if image is not None:
    save_folder = './images/'
    save_path = Path(save_folder, image.name)
   
    with open(save_path, mode='wb') as w:
        w.write(image.getvalue())
    img_output=Image.open("./images/"+image.name)
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(img_output,width=200, caption=None)
    with st.spinner('Wait for it...'):
        response=predict_step([image.name])
        st.balloons()
        st.text_area("Image Description",response[0])