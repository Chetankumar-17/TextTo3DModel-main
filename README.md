# Text-to-3D Model Generator using Shap-E and AWS S3

This project is a _Text-to-3D Model Generation_ tool built using _OpenAI's Shap-E, \*\*Gradio_ for the web interface, and _AWS S3_ for cloud-based storage and sharing. Users can enter a natural language prompt, and the application generates a 3D model (in .glb, .obj, or .ply format) which can be downloaded or shared via a public S3 URL.

## 📦 Features

- 🔡 _Text Prompt Input_: Enter any text like "a flying elephant" or "a futuristic car".
- 🧊 _3D Mesh Generation_: Uses OpenAI’s Shap-E model to convert text into 3D latent mesh.
- ☁ _AWS S3 Upload_: Automatically uploads the generated mesh to your S3 bucket.
- 📤 _Download & Share_: Get a downloadable file and a public URL to view or share.
- 🧑‍💻 _Gradio UI_: Lightweight, no frontend coding needed.

---

## 🧰 Technologies Used

- [OpenAI Shap-E](https://github.com/openai/shap-e)
- [Gradio](https://gradio.app/)
- [PyTorch](https://pytorch.org/)
- [Trimesh](https://github.com/mikedh/trimesh)
- [AWS S3 (via Boto3)](https://boto3.amazonaws.com/)
- EC2 (Ubuntu) for deployment

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/text-to-3d-generator.git
cd text-to-3d-generator
```
