import torch
from diffusers import DiffusionPipeline

model_id = "google/ncsnpp-ffhq-256"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sde_ve = DiffusionPipeline.from_pretrained(model_id)
sde_ve.to(device)  

image = sde_ve().images[0]
image.save("sde_ve_generated_image.png")