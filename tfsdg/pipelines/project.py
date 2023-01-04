from tfsdg.pipelines import TFSDGPipeline

pipe = TFSDGPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to("cuda")

prompt = "A red car and a white sheep"
image = pipe(prompt, struct_attention="align_seq").images[0]
image.save('a_red_car_and_a_white_sheep.png')