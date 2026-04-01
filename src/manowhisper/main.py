import csv

import click
from transformers import pipeline

from manowhisper import (classifier, emotions, parser, stats, summarizer,
                         transcriber)

model_map = {
    "misogyny": "MilaNLProc/bert-base-uncased-ear-misogyny",
    "hate": "facebook/roberta-hate-speech-dynabench-r4-target",
    "transphobia": "bitsanlp/Homophobia-Transphobia-v2-mBERT-EDA",
    "sexism": "annahaz/xlm-roberta-base-finetuned-misogyny-sexism",
    "emotion": "j-hartmann/emotion-english-distilroberta-base",
    "homophobia": "bitsanlp/Homophobia-Transphobia-v2-mBERT-EDA",
    "toxicity": "s-nlp/roberta_toxicity_classifier",
}


@click.group()
def cli():
    """manowhisper - transcribe, summarize, classify, and create
    visualizations of WebVTT files."""
    pass


@cli.command()
@click.option(
    "--model",
    type=click.Choice(
        ["transphobia", "sexism", "hate", "misogyny", "homophobia", "toxicity"]
    ),
    required=True,
    help="Model to use for classification.",
)
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to WebVTT transcripts.",
)
@click.argument("output", type=click.Path())
def classify(model, input, output):
    """Classify WebVTT files."""
    model_name = model_map[model]
    model_pipeline = pipeline("text-classification", model=model_name)

    sentences, filenames, timestamps = parser.extract_sentences_timestamps(input)

    show_name = parser.extract_show_name(input)

    scores, labels = classifier.classify(sentences, model_pipeline, model_name)

    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["show_name", "filename", "timestamp", "sentence", "label", "score"]
        )
        for filename, timestamp, sentence, label, score in zip(
            filenames, timestamps, sentences, labels, scores
        ):
            writer.writerow([show_name, filename, timestamp, sentence, label, score])


@cli.command()
@click.option("--threads", type=int, required=True, help="Number of threads to use.")
@click.option(
    "--model",
    default="turbo",
    help="Model to use for transcription. Default is 'turbo'.",
)
@click.option(
    "--fp16",
    type=bool,
    default=False,
    help="Whether to use fp16 precision. Default is False.",
)
@click.option(
    "--language", default="en", help="Language of the media. Default is 'en'."
)
@click.option("--output_format", default="vtt", help="Output format. Default is 'vtt'.")
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to input files (directory or single file).",
)
def transcribe(threads, model, fp16, language, output_format, input):
    """Transcribe files."""
    transcriber.transcribe(
        input_path=input,
        threads=threads,
        model=model,
        fp16=fp16,
        language=language,
        output_format=output_format,
    )


@cli.command(name="emotions")
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to WebVTT transcripts.",
)
@click.argument("output", type=click.Path())
def emotions_cmd(input, output):
    """Classify emotions in WebVTT files."""
    model_pipeline = pipeline("text-classification", model=model_map["emotion"])

    sentences, filenames, timestamps = parser.extract_sentences_timestamps(input)

    show_name = parser.extract_show_name(input)

    emotion_results = emotions.classify_emotions(sentences, model_pipeline)

    with open(output, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "show_name",
                "filename",
                "timestamp",
                "sentence",
                "anger",
                "disgust",
                "fear",
                "joy",
                "neutral",
                "sadness",
                "surprise",
            ],
        )
        writer.writeheader()
        for filename, sentence, timestamp, emotion in zip(
            filenames, sentences, timestamps, emotion_results
        ):
            writer.writerow(
                {
                    "show_name": show_name,
                    "filename": filename,
                    "timestamp": timestamp,
                    "sentence": sentence,
                    "anger": emotion["anger"],
                    "disgust": emotion["disgust"],
                    "fear": emotion["fear"],
                    "joy": emotion["joy"],
                    "neutral": emotion["neutral"],
                    "sadness": emotion["sadness"],
                    "surprise": emotion["surprise"],
                }
            )


@cli.command()
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to WebVTT transcripts.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory to save summaries.",
)
def summarize(input, output_dir):
    """Summarize WebVTT files."""
    summarizer.summarize_and_write(input, output_dir)


@cli.command()
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to WebVTT transcripts.",
)
@click.argument("output_csv", type=click.Path())
def statistics(input, output_csv):
    """Calculate various metrics for a podcast."""
    show_name = parser.extract_show_name(input)
    results = stats.process_podcast(input, show_name)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    cli()


if __name__ == "__main__":
    main()
