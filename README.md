### **Is it tourism in Cambodia?**

In this project, we will train an **Out-of-Distribution (OOD) Detector** or an **Anomaly Detector**.

### **Definitions**
- **In-Distribution (ID)**: `tourism_cambodia` (we have data for this).
- **Out-of-Distribution (OOD)**: `general` (we lack data for this).  
  The model's task is to learn what *"tourism in Cambodia"* looks like so well that it can identify when a question doesn't fit this category.

---

### **Approach**
We plan to use **pre-trained models** from Hugging Face (e.g., BERT, DistilBERT, RoBERTa). These models are trained on vast text corpora (e.g., Wikipedia, books, web crawls) and understand grammar, context, and common sense. We will fine-tune them for our specific task ("tourism in Cambodia" vs. "general"), which requires less data and computational resources.

---

### **To-Do List**

- [x] Collect data for `Tourism in Cambodia` dataset.
- [x] Create a `General` dataset.
- [ ] Test models:
  - Zero-shot classification:
    - [x] `facebook/bart-large-mnli`
  - Fine-tuning:
    - [ ] `BERT`
    - [ ] `DistilBERT`
    - [ ] `RoBERTa`
- [ ] Evaluate model performance.
- [x] Deploy the model using FastAPI.

---

### **For `uv` Users**

If you're using the `uv` environment, you can install the required packages directly into your environment.

To install `uv`, follow the instructions on the [official uv website](https://uv.dev).

Once you have `uv` installed, you can proceed to create a new environment and install the necessary packages.

To create a new environment, use the following command:

```bash
uv venv .venv --python=3.12
```

To install the required packages, use the following commands:

```bash
uv pip install --link-mode=copy -r requirements.txt
uv pip install --link-mode=copy <package_name>
```

**Note**: The `--link-mode=copy` option ensures compatibility with the `uv` environment by copying packages into the environment instead of linking them, avoiding dependency issues.

To run the server, use the following command:

```bash
uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API `http://localhost:8000/api/v1/predict` to interact with the model.

`POST` request body example:

```json
{
  "prompt": "I want to see you in Siem Reap where wwe wear the traditional wedding costume",
  "candidate_labels": ["love", "tourism"]
}
```

---

### **System Properties**

To check your system properties, you can use the following command in your command prompt or terminal:
```cmd
nvidia-smi
```
Below is an example output of that in my machine from the `nvidia-smi` command:

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.76                 Driver Version: 560.76         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1650      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   49C    P8             10W /   60W |       0MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Note that the CUDA version displayed by `nvidia-smi` indicates the highest version supported by your GPU driver, not necessarily the version installed on your system. To check the installed CUDA version, you can use the following command in your terminal or command prompt:
```bash
nvcc --version
```
Below is an example output of that in my machine from the `nvcc --version` command:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.20
```
---

Built with ❤️ by `Deepseek`, `Kimi`, `GitHub Copilot`, and `ChatGPT` - You Are All I Need.
