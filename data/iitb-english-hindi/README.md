# IITB English–Hindi Parallel Corpus (cfilt/iitb-english-hindi)

### Dataset Overview

The **IIT Bombay English-Hindi Parallel Corpus** is a large-scale bilingual
dataset created by the **Center for Indian Language Technology (CFILT)** at IIT
Bombay. It contains **1.66 million English–Hindi sentence pairs** collected
from multiple open sources and curated over several years for **machine
translation and linguistic research**.

| Field                 | Value                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Dataset name**      | `cfilt/iitb-english-hindi`                                                                                              |
| **Languages**         | English (`en`), Hindi (`hi`)                                                                                            |
| **Modality**          | Text (parallel corpus)                                                                                                  |
| **Format**            | Parquet                                                                                                                 |
| **Size**              | ~190 MB (≈ 1.66 M rows)                                                                                                 |
| **Splits**            | `train`, `validation`, `test`                                                                                           |
| **License**           | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)                                                         |
| **Hugging Face page** | 🔗 [https://huggingface.co/datasets/cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) |
| **Official site**     | [http://www.cfilt.iitb.ac.in/iitb_parallel](http://www.cfilt.iitb.ac.in/iitb_parallel)                                  |

---

### 🧠 Example Record

```json
{
  "en": "Give your application an accessibility workout",
  "hi": "अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें"
}
```

---

🔗 [IITB-English-Hindi-PC GitHub](https://github.com/cfiltnlp/IITB-English-Hindi-PC)

---

### 🧩 Typical Uses

* English↔Hindi machine translation
* Bilingual lexicon extraction
* Cross-lingual representation learning
* Evaluation of translation quality metrics (BLEU, chrF, etc.)

---

### 🧾 Citation

If you use this dataset, please cite:

> **Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya**
> *The IIT Bombay English–Hindi Parallel Corpus*
> *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan.

```bibtex
@inproceedings{kunchukuttan-etal-2018-iit,
  title     = {The IIT Bombay English-Hindi Parallel Corpus},
  author    = {Kunchukuttan, Anoop and Mehta, Pratik and Bhattacharyya, Pushpak},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year      = {2018},
  address   = {Miyazaki, Japan},
  publisher = {European Language Resources Association (ELRA)},
  url       = {https://aclanthology.org/L18-1548}
}
```

