![image](https://github.com/user-attachments/assets/4c19f18b-2543-41f4-9c07-f1fe69db06c1)# Edge AI Research Guide

## Description
This is a research tool aimed at reseachers going on field work in areas with very limited or absent internet access. The user can capture or upload an image and get it identified as the correct flora and fauna and receive an overview of it. Further they can continue chatting with the tool about the observed species. The user can also set up a live video feed which can be  monitored by the tool for any endangered species. 

![image](https://github.com/user-attachments/assets/ce5ffb2b-f862-4261-b26e-d6a1ae12d41c)


## Team Members
- Rohith Prakash - rohithprakash15@gmail.com
- ⁠Dhakkshin S R - srdhakkshin04@gmail.com    
- ⁠Aravinth Cheran K S -  cheran411@gmail.com    
- ⁠⁠Kishoreadhith V - kishoreadhithvijayakumaar@gmail.com    
- ⁠Karthik Srinivas S - karthikdp2005@gmail.com      


## Setup Instructions

1. Download the model from [google drive](https://drive.google.com/file/d/1f3AkEWWUSQCLbPQVh3DS2iCRYn-ZUjxO/view?usp=sharing) to the root directory.
2. Install and setup [AnythingLLM](https://anythingllm.com/).
    1. Choose Qualcomm QNN when prompted to choose an LLM provider to target the NPU
    2. Choose a model of your choice when prompted. This sample uses Llama 3.1 8B Chat with 8K context
2. Create a workspace by clicking "+ New Workspace"
3. Generate an API key
    1. Click the settings button on the bottom of the left panel
    2. Open the "Tools" dropdown
    3. Click "Developer API"
    4. Click "Generate New API Key"
4. Clone the repository
   ```bash
   git clone https://github.com/kishoreadhith-v/ffmrag.git
   cd ffmrag
   ```

5. Install dependencies
   ```bash
   cd ai-survival-guide-frontend
   npm install
   npm start
   cd ..
   
   # requirements for backend
   python -m venv venv
   pip install -r requirements.txt 
   ```

6. Create your `config.yaml` file with the following variables
    ```
    api_key: "your-key-here"
    model_server_base_url: "http://localhost:3001/api/v1"
    workspace_slug: "your-slug-here"
    stream: true
    stream_timeout: 60
    ```
7. Test the model server auth to verify the API key
    ```
    python src/auth.py
    ```
8. Get your workspace slug using the workspaces tool
    1. Run ```python src/workspaces.py``` in your command line console
    2. Find your workspace and its slug from the output
    3. Add the slug to the `workspace_slug` variable in config.yaml
9. Run the python scripts:
   ```bash
   # use a separate terminal for each python file execution
   python image_analyzer.py # desktop python app for live video classification
   python api_server.py     # image classification
   python fauna-rag/app.py   # RAG
    ```
---
## Tests
1. API endpoint to test the uptime of the RAG: GET `http://localhost:6000/health`
---
## Notes

## References
- [Classification model](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21)
- [Dataset](https://www.kaggle.com/c/inaturalist-2021)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/models)
- [AnythingLLM](https://anythingllm.com/)
- [ONYX](https://onnx.ai/onnx/)

## License

MIT License

Copyright (c) 2025 [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
