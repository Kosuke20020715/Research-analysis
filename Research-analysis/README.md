# Research-analysis
## 1. Compare the accuracy of LLM (OpenAI API) and BERT (NLP)<p>
I've used "Wrime Dataset" (https://github.com/ids-cv/wrime).<p>
This dataset includes the following descriptions.<p>
- 35,000 Japanese posts from 60 crowdsourced workers
- subjective/objective label (8 emotions)
- use 4-point emotion intensity scale (0:no, 1:weak, 2:medium, and 3:strong)

the example of this dataset is as follows<p>
![image](https://github.com/user-attachments/assets/37bccd8c-5771-479a-ac85-9b34e6e8d07c)

I've compared the accuracy of prediction Bert with LLM.

### Results:
- Accuracy
Bert: 0.673 / LLM (Chatgpt-4o): 0.670
- Learning-Curve (Bert)
<img width="536" alt="Accuracy" src="https://github.com/user-attachments/assets/522c62a0-424b-4a11-91c0-b2a634c35956" />

- Loss-Curve (Bert)
<img width="532" alt="eval-loss" src="https://github.com/user-attachments/assets/9673289a-2dcf-4b3c-aac3-9e89539f12c8" />

### Conclusion 
- Accuracy is roughly the same
- In terms of Flexibility, time efficiency, and resource, LLM is superior.
