### **Is it tourism in Cambodia?**

In this project, will train a **"Out-of-Distribution (OOD) Detector"** or an **"Anomaly Detector"**.

**In-Distribution (ID)**: tourism_cambodia (you have data for this).

**Out-of-Distribution (OOD)**: general (we have no data for this). The model's job is to learn what *"tourism in Cambodia"* looks like so well that it can identify when a question doesn't look like it.

---

We think of using **models** on Hugging Face (like BERT, DistilBERT, RoBERTa) that are already pre-trained on enormous text corpora (Wikipedia, books, web crawls). They understand grammar, context, and common sense, so we just need to fine-tune them for our specific task ("tourism in Cambodia" vs. "general"), which requires far less data and computing power.


### **To-Do List:**

- [x] Finding data for `Tourism in Cambodia` dataset.
- [x] Create `General` dataset.
- [ ] Test models:
  - Zero-shot classification
    - [x] facebook/bart-large-mnli
  - Fine-tuning
    - [ ] BERT
    - [ ] DistilBERT
    - [ ] RoBERTa
- [ ] Evaluate model performance


## For *uv* users 

To install the required packages, you can use the following command:

```bash
uv pip install --link-mode=copy -r requirements.txt

uv pip install --link-mode=copy <package_name>
```

**Note**: The `--link-mode=copy` option is used to ensure that the packages are installed in a way that is compatible with the `uv` environment. This option helps to avoid issues related to package linking and ensures that the packages are copied into the environment rather than linked, which can sometimes cause problems with dependencies.