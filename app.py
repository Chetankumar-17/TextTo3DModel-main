import os
import torch
import trimesh
import gradio as gr

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# Set output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)
print(f" Outputs will be saved to: {output_dir}")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

# Load Shap-E models
print("⏳ Loading Shap-E models...")
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
print(" Models loaded successfully!")

def generate_mesh(prompt, guidance_scale, steps, format):
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Decode the latent into mesh
    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)

    # Save to file
    filename = f"{prompt.replace(' ', '_')}.{format}"
    path = os.path.join(output_dir, filename)
    mesh.export(path, file_type=format)

    return path, path  # First for preview, second for download

# Gradio UI
gr.Interface(
    fn=generate_mesh,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., a flying elephant"),
        gr.Slider(5, 20, value=16.5, step=0.5, label="Guidance Scale"),
        gr.Slider(20, 100, value=64, step=1, label="Steps"),
        gr.Radio(["glb", "obj", "ply"], label="Export Format", value="glb")
    ],
    outputs=[
        gr.Model3D(label="3D Preview"),
        gr.File(label="Download 3D Model")
    ],
    title="🧠 Text-to-3D Generator (No Blender)",
    description="Generate a 3D model from text and preview it interactively, plus export without Blender."
).launch(share=True)