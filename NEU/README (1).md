---
license: apache-2.0
task_categories:
- text-classification
language:
- vi
tags:
  - sentiment-analysis
  - topic-classification
  - education
  - NLP
---


# NEU-ESC: NEU Dataset for Educational Sentiment Analysis and Topic Classification

## Overview
NEU-ESC is a dataset designed for Educational Sentiment Analysis and Topic Classification. Each sentence in the dataset is labeled with two attributes:
- **Sentiment Analysis**: Indicates the sentiment of the text.
- **Topic Classification**: Categorizes the content into predefined educational topics.

The dataset is collected from online forums, educational social network fan pages, and groups. It has been preprocessed and cleaned to ensure high-quality textual data.

## Dataset Statistics

### Sentiment Analysis Labels
| Label | Sentiment | Count  | Percentage | Mean Length |
|-------|-----------|--------|------------|-------------|
| 0     | Neutral   | 22,773 | 69.08%     | 23.21       |
| 1     | Positive  | 4,148  | 12.58%     | 24.29       |
| 2     | Negative  | 5,250  | 15.77%     | 34.30       |
| 3     | Toxic     | 845    | 2.56%      | 22.78       |

### Topic Classification Labels
| Label | Classification      | Count  | Percentage | Mean Length |
|-------|---------------------|--------|------------|-------------|
| 0     | Spam               | 405    | 1.23%      | 22.71       |
| 1     | News               | 902    | 2.74%      | 59.55       |
| 2     | Academic           | 10,512 | 31.89%     | 29.62       |
| 3     | Other              | 14,402 | 43.69%     | 11.40       |
| 4     | Service            | 2,358  | 7.15%      | 30.94       |
| 5     | Jobs & Recruitment | 808    | 2.45%      | 55.14       |
| 6     | Personal Affairs   | 1,478  | 4.48%      | 33.17       |
| 7     | Social Affairs     | 769    | 2.33%      | 67.11       |
| 8     | Help & Share       | 670    | 2.03%      | 37.03       |
| 9     | Club & Events      | 662    | 2.01%      | 68.82       |

## Dataset Format
Each sample in the dataset contains:
- **Text**: The input sentence.
- **Sentiment**: One of the four sentiment classes (Neutral, Positive, Negative, Toxic).
- **Classification**: One of the ten topic categories.

## Usage
The dataset is useful for:
- Training and evaluating sentiment analysis models in the educational domain.
- Building topic classification models for educational discussions.
- Understanding user engagement in online educational communities.

## License
This dataset is released under an open-source license for research and educational purposes. Please ensure proper citation when using it in your work.

## Citation
If you use NEU-ESC in your research, please cite:
```
@misc{neu_esc,
  author = {Nguyen Quang Hung, Mai Phan Quoc Hung, Nguyen Thi Hong Hanh, Duong Phuong Giang},
  title = {NEU-ESC: NEU Dataset for Educational Sentiment Analysis and Topic Classification},
  year = {2024},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/datasets/hung20gg/NEU-ESC}
}
```

## Contact
For any inquiries or issues regarding the dataset, please reach out via Hugging Face discussions or GitHub Issues.
