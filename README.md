# **GPT-2 Domain Adaptation & Parameter-Efficient Fine-Tuning (LoRA)**

This project demonstrates end-to-end development of a domain-adapted generative language model using GPT-2 Small. It evaluates baseline behaviour, performs full fine-tuning on a literary corpus (Jane Austen), and implements parameter-efficient adaptation with LoRA.

It’s designed to showcase practical Generative AI skills relevant to industry roles: model evaluation, LLM fine-tuning, dataset preparation, PEFT techniques, and comparative analysis.


## **Project Highlights**

### **LLM Evaluation**

* Baseline GPT-2 behaviour analysed using perplexity, tokenisation patterns, and qualitative outputs.
* Explored BPE tokenisation to understand subword representation for archaic English text.

### **Full Fine-Tuning**

* Two experimental setups:

  * **7-sample dataset**
  * **21-sample dataset**
* Achieved measurable improvements in stylistic coherence and perplexity.
* Trained all **124M parameters**, demonstrating ability to perform full-scale adaptation.

### **Parameter-Efficient Fine-Tuning (LoRA)**

* Applied LoRA adapters to GPT-2 attention and feed-forward layers.
* Only **2.36M trainable parameters** (under 2% of the model).
* Achieved strong stylistic alignment at a fraction of the compute cost.

### ** Quantitative Results**

| Method                   | Trainable Params | Avg Perplexity | Notes                         |
| ------------------------ | ---------------- | -------------- | ----------------------------- |
| **Baseline GPT-2**       | 124M             | ~35.1          | Fluent, inconsistent style    |
| **Full FT (7 samples)**  | 124M             | ~31.9          | Better structure & rhythm     |
| **Full FT (21 samples)** | 124M             | ~30.7          | Best performance              |
| **LoRA (21 samples)**    | 2.36M            | ~37.1          | Most efficient, lower compute |


## **Technical Stack**

* **PyTorch**
* **Hugging Face Transformers**
* **PEFT (LoRA)**
* **Datasets**
* **Matplotlib / WordCloud**
* Google Colab for GPU-backed training


## **Repository Structure**

```
├── baseline/
│   └── baseline_evaluation.ipynb
├── finetuning_full/
│   ├── finetune_7_samples.ipynb
│   └── finetune_21_samples.ipynb
├── finetuning_lora/
│   └── lora_finetuning.ipynb
├── results/
│   ├── perplexity_reports/
│   └── generated_samples/
└── README.md
```

---

## **Project Summary**

This work examines how GPT-2 adapts to stylistically distinctive text through three training strategies:

### ** 1 Baseline GPT-2**

* Produces fluent text but diverges from Austen’s tone.
* Higher perplexity indicates poor domain familiarity.

### **2 Full Fine-Tuning**

* Strongest results in coherence, narrative flow, and stylistic accuracy.
* Larger dataset (21 samples) yields notably improved outputs.
* Cost: long training cycles and full parameter updates.

### **3 LoRA Fine-Tuning**

* Efficient, lightweight training with minimal parameter updates.
* Maintains stylistic formality and structure.
* Ideal for production environments with compute constraints.

---

## **How to Run**

### **Install Dependencies**

```bash
pip install torch transformers datasets peft matplotlib wordcloud
```

### **Run Any Notebook**

* `baseline/` for model evaluation
* `finetuning_full/` for complete fine-tuning
* `finetuning_lora/` for PEFT training

Each notebook includes:

* Data preparation
* Tokenisation
* Training
* Perplexity evaluation
* Generated text samples

---

## **Future Extensions**

* QLoRA or Prefix-Tuning comparisons
* RLHF-style preference optimisation
* Larger models (GPT-Neo, GPT-J)
* Perplexity-free evaluation metrics (MAUVE, style classifiers)
* RAG + fine-tuning hybrid pipeline


