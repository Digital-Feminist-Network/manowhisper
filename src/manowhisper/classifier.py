from alive_progress import alive_bar
from transformers import pipeline


def classify(sentences, model_pipeline, model_name):
    """
    Classifies each sentence using a given model pipeline.
    Handles multiple label schemes.
    """
    scores = []
    labels = []
    with alive_bar(len(sentences), title="Classifying Sentences") as bar:
        for sentence in sentences:
            result = model_pipeline(sentence)
            # Result can sometimes be a list of dicts or sometimes a dict.
            if isinstance(result, list):
                result = result[0]
            label = result["label"]
            score = result["score"]
            # Normalize labels based on model.
            if model_name == "bitsanlp/Homophobia-Transphobia-v2-mBERT-EDA":
                if label == "LABEL_1":
                    label = "non-transphobic"
                elif label == "LABEL_0":
                    label = "transphobic"
            elif model_name == "annahaz/xlm-roberta-base-finetuned-misogyny-sexism":
                if label == "0":
                    label = "non-sexist"
                elif label == "1":
                    label = "sexist"
            elif model_name == "bitsanlp/Homophobia-Transphobia-v2-mBERT-EDA":
                if label == "LABEL_1":
                    label = "non-homophobic"
                elif label == "LABEL_0":
                    label = "homophobic"
            elif model_name == "facebook/roberta-hate-speech-dynabench-r4-target":
                pass
            elif model_name == "MilaNLProc/bert-base-uncased-ear-misogyny":
                pass
            elif model_name == "s-nlp/roberta_toxicity_classifier":
                pass
            scores.append(score)
            labels.append(label)
            bar()
    return scores, labels
