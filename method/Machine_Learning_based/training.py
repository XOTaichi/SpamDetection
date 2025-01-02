"""Functions related to the training and evaluation pipeline"""

from transformers import (
    AutoModelForSequenceClassification,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer,
    AutoTokenizer,
    Trainer,
)
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import evaluate
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate

import copy
import time
import pandas as pd
import pickle
import torch

from src.spamdetection.preprocessing import get_dataset, train_val_test_split
from src.spamdetection.utils import (
    SCORING,
    set_seed,
    plot_loss,
    plot_scores,
    save_scores,
)
from src.spamdetection.transforms import transform_df, encode_df, tokenize, init_nltk


MODELS = {
    "NB": (MultinomialNB(), 1000),
    "LR": (LogisticRegression(), 500),
    "KNN": (KNeighborsClassifier(n_neighbors=1), 150),
    "SVM": (SVC(kernel="sigmoid", gamma=1.0), 3000),
    "XGBoost": (XGBClassifier(learning_rate=0.01, n_estimators=150), 2000),
    "LightGBM": (LGBMClassifier(learning_rate=0.1, num_leaves=20), 3000),
}


LLMS = {
    "RoBERTa": (
        AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        ),
        AutoTokenizer.from_pretrained("roberta-base"),
    ),
    "SetFit-mpnet": (
        SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        None,
    ),
    "FLAN-T5-base": (
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base"),
        AutoTokenizer.from_pretrained("google/flan-t5-base"),
    ),
}

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path="roc_curve.png"):
    """
    Plot and save the ROC curve.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted probabilities or scores for the positive class.
        title (str): Title of the plot.
        save_path (str): Path to save the ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

class EvalOnTrainCallback(TrainerCallback):
    """Custom callback to evaluate on the training set during training."""

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_train = copy.deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_train


def get_trainer(model, dataset, tokenizer=None):
    """Return a trainer object for transformer models."""

    def compute_metrics(y_pred):
        """Computer metrics during training."""
        logits, labels = y_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("f1").compute(
            predictions=predictions, references=labels, average="macro"
        )

    if type(model).__name__ == "SetFitModel":
        trainer = SetFitTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            loss_class=CosineSimilarityLoss,
            metric="f1",
            batch_size=16,
            num_iterations=20,
            num_epochs=3,
        )
        return trainer

    elif "T5" in type(model).__name__ or "FLAN" in type(model).__name__:

        def compute_metrics_t5(y_pred, verbose=0):
            """Computer metrics during training for T5-like models."""
            predictions, labels = y_pred

            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            # Replace -100 with pad_token_id to decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [
                1 if "spam" in predictions[i] else 0 for i in range(len(predictions))
            ]
            labels = [1 if "spam" in labels[i] else 0 for i in range(len(labels))]

            result = evaluate.load("f1").compute(
                predictions=predictions, references=labels, average="macro"
            )
            return result

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="experiments_custom_enron_with_cross_test",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=5,
            predict_with_generate=True,
            # fp16=False,
            fp16=True, 
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=5,
            no_cuda=False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_t5,
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer

    else:
        training_args = TrainingArguments(
            output_dir="experiments_custom_enron_with_cross_test",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=10,
            fp16=True,     # Mixed precision
            no_cuda=False, # Make sure GPU is used
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer


# def predict(trainer, model, dataset, tokenizer=None):
    """Convert the predict function to specific classes to unify the API."""
    if type(model).__name__ == "SetFitModel":
        return model(dataset["text"])

    elif "T5" in type(model).__name__:
        predictions = trainer.predict(dataset)
        predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        predictions = [
            1 if "spam" in predictions[i] else 0 for i in range(len(predictions))
        ]

        return predictions

    else:
        return trainer.predict(dataset).predictions.argmax(axis=-1)

def predict(trainer, model, dataset, tokenizer=None):
    """Convert the predict function to specific classes to unify the API."""
    if type(model).__name__ == "SetFitModel":
        predictions = model(dataset["text"])
        probabilities = model.predict_proba(dataset["text"])[:, 1]
        return predictions, probabilities

    elif "T5" in type(model).__name__:
        predictions = trainer.predict(dataset)
        probabilities = predictions.predictions[:, 1]
        predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        predictions = [
            1 if "spam" in predictions[i] else 0 for i in range(len(predictions))
        ]
        return predictions, probabilities

    else:
        predictions = trainer.predict(dataset)
        probabilities = torch.softmax(
            torch.tensor(predictions.predictions), dim=1
        ).numpy()[:, 1]
        return predictions.predictions.argmax(axis=-1), probabilities

def train_llms(seeds, datasets, train_sizes, test_set="test"):
    """Train all the large language models."""
    for seed in list(seeds):
        set_seed(seed)

        for dataset_name in list(datasets):

            for train_size in train_sizes:
                # Get metrics
                scores = pd.DataFrame(
                    index=list(LLMS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                # Main loop
                df = get_dataset(dataset_name)
                _, dataset = train_val_test_split(
                    df, train_size=train_size, has_val=True
                )

                # Name experiment
                experiment = (
                    f"llm_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"
                )

                # Train, evaluate, test
                for model_name, (model, tokenizer) in LLMS.items():
                    tokenized_dataset = tokenize(dataset, tokenizer)
                    trainer = get_trainer(model, tokenized_dataset, tokenizer)

                    # Train model
                    start = time.time()
                    train_result = trainer.train()
                    end = time.time()
                    scores.loc[model_name]["training_time"] = end - start
                    if "SetFit" not in model_name:
                        log = pd.DataFrame(trainer.state.log_history)
                        log.to_csv(f"outputs/csv/loss_{model_name}_{experiment}.csv")
                        plot_loss(experiment, dataset_name, model_name)

                    # Test model
                    start = time.time()
                    predictions = predict(
                        trainer, model, tokenized_dataset[test_set], tokenizer
                    )
                    end = time.time()

                    for score_name, score_fn in SCORING.items():
                        scores.loc[model_name][score_name] = score_fn(
                            dataset[test_set]["label"], predictions
                        )

                    scores.loc[model_name]["inference_time"] = end - start
                    save_scores(
                        experiment, model_name, scores.loc[model_name].to_dict()
                    )

                # Display scores
                plot_scores(experiment, dataset_name)
                print(scores)


# def train_baselines(seeds, datasets, train_sizes, test_set="test"):
#     """Train all the baseline models."""
#     init_nltk()

#     for seed in list(seeds):
#         set_seed(seed)

#         for dataset_name in list(datasets):

#             for train_size in train_sizes:
#                 # Create list of metrics
#                 scores = pd.DataFrame(
#                     index=list(MODELS.keys()),
#                     columns=list(SCORING.keys()) + ["training_time", "inference_time"],
#                 )

#                 # Main loop
#                 df = get_dataset(dataset_name)
#                 df = transform_df(df)
#                 (df_train, df_val, df_test), _ = train_val_test_split(
#                     df, train_size=train_size, has_val=True
#                 )

#                 # Name experiment
#                 experiment = (
#                     f"ml_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"
#                 )

#                 # Cross-validate and test every model
#                 for model_name, (model, max_iter) in MODELS.items():
#                     # Encode the dataset
#                     encoder = TfidfVectorizer(max_features=max_iter)
#                     X_train, y_train, encoder = encode_df(df_train, encoder)
#                     X_test, y_test, encoder = encode_df(df_test, encoder)

#                     # Evaluate model with cross-validation
#                     if test_set == "val":
#                         cv = cross_validate(
#                             model,
#                             X_train,
#                             y_train,
#                             scoring=list(SCORING.keys()),
#                             cv=5,
#                             n_jobs=-1,
#                         )
#                         for score_name, score_fn in SCORING.items():
#                             scores.loc[model_name][score_name] = cv[
#                                 f"test_{score_name}"
#                             ].mean()

#                     # Evaluate model on test set
#                     if test_set == "test":
#                         start = time.time()
#                         model.fit(X_train, y_train)
#                         end = time.time()
#                         scores.loc[model_name]["training_time"] = end - start

#                         start = time.time()
#                         y_pred = model.predict(X_test)
#                         end = time.time()

#                         scores.loc[model_name]["inference_time"] = end - start
#                         for score_name, score_fn in SCORING.items():
#                             scores.loc[model_name][score_name] = score_fn(
#                                 y_pred, y_test
#                             )

#                     save_scores(
#                         experiment, model_name, scores.loc[model_name].to_dict()
#                     )

#                 # Display scores
#                 plot_scores(experiment, dataset_name)
#                 print(scores)
def train_baselines(seeds, datasets, train_sizes, test_set="test"):
    """Train all the baseline models and save ROC curves."""
    init_nltk()

    for seed in list(seeds):
        set_seed(seed)

        for dataset_name in list(datasets):

            for train_size in train_sizes:
                # Create list of metrics
                scores = pd.DataFrame(
                    index=list(MODELS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                # Main loop
                df = get_dataset(dataset_name)
                df = transform_df(df)
                (df_train, df_val, df_test), _ = train_val_test_split(
                    df, train_size=train_size, has_val=True
                )

                # Name experiment
                experiment = (
                    f"ml_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"
                )

                # Cross-validate and test every model
                for model_name, (model, max_iter) in MODELS.items():
                    # Encode the dataset
                    encoder = TfidfVectorizer(max_features=max_iter)
                    X_train, y_train, encoder = encode_df(df_train, encoder)
                    X_test, y_test, encoder = encode_df(df_test, encoder)

                    # Evaluate model with cross-validation
                    if test_set == "val":
                        cv = cross_validate(
                            model,
                            X_train,
                            y_train,
                            scoring=list(SCORING.keys()),
                            cv=5,
                            n_jobs=-1,
                        )
                        for score_name, score_fn in SCORING.items():
                            scores.loc[model_name][score_name] = cv[
                                f"test_{score_name}"
                            ].mean()

                    # Evaluate model on test set
                    if test_set == "test":
                        start = time.time()
                        model.fit(X_train, y_train)
                        end = time.time()
                        scores.loc[model_name]["training_time"] = end - start

                        start = time.time()
                        y_pred = model.predict(X_test)
                        y_pred_proba = (
                            model.predict_proba(X_test)[:, 1]
                            if hasattr(model, "predict_proba")
                            else None
                        )
                        end = time.time()

                        scores.loc[model_name]["inference_time"] = end - start
                        for score_name, score_fn in SCORING.items():
                            scores.loc[model_name][score_name] = score_fn(
                                y_pred, y_test
                            )

                        # Plot and save ROC curve if probabilities are available
                        if y_pred_proba is not None:
                            roc_save_path = f"outputs/roc_curves/{experiment}_{model_name}.png"
                            plot_roc_curve(
                                y_test,
                                y_pred_proba,
                                title=f"ROC Curve for {model_name}",
                                save_path=roc_save_path,
                            )

                    save_scores(
                        experiment, model_name, scores.loc[model_name].to_dict()
                    )

                # Display scores
                plot_scores(experiment, dataset_name)
                print(scores)

def train_and_test(seeds, datasets, train_sizes, fixed_test_set_path, test_set="test"):
    """
    Train models and test them on a fixed dataset.

    Args:
        seeds (list): List of seeds for reproducibility.
        datasets (list): List of datasets to train on.
        train_sizes (list): List of training sizes (e.g., 0.8 for 80% of the data).
        fixed_test_set_path (str): Path to the fixed dataset for testing.
        test_set (str): Test set to evaluate on during training (default: 'test').
    """
    # Load the fixed test set
    fixed_test_df = pd.read_csv(fixed_test_set_path)
    if "text" not in fixed_test_df.columns or "label" not in fixed_test_df.columns:
        raise ValueError("The fixed test dataset must have 'text' and 'label' columns.")
    
    # Preprocess the fixed test dataset
    fixed_test_df = transform_df(fixed_test_df)

    # Train and test baseline models
    init_nltk()

    for seed in seeds:
        set_seed(seed)

        for dataset_name in datasets:
            for train_size in train_sizes:
                # Training baselines
                print(f"Training baseline models on {dataset_name} with train_size={train_size} and seed={seed}...")
                scores_baselines = pd.DataFrame(
                    index=list(MODELS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                df = get_dataset(dataset_name)
                df = transform_df(df)
                (df_train, df_val, df_test), _ = train_val_test_split(
                    df, train_size=train_size, has_val=True
                )

                for model_name, (model, max_iter) in MODELS.items():
                    # Encode the dataset
                    encoder = TfidfVectorizer(max_features=max_iter)
                    X_train, y_train, encoder = encode_df(df_train, encoder)
                    X_fixed_test, y_fixed_test, encoder = encode_df(fixed_test_df, encoder)

                    # Train and evaluate on the fixed test set
                    start = time.time()
                    model.fit(X_train, y_train)
                    end = time.time()
                    scores_baselines.loc[model_name]["training_time"] = end - start

                    start = time.time()
                    y_pred_fixed = model.predict(X_fixed_test)
                    end = time.time()
                    scores_baselines.loc[model_name]["inference_time"] = end - start

                    for score_name, score_fn in SCORING.items():
                        scores_baselines.loc[model_name][score_name] = score_fn(
                            y_fixed_test, y_pred_fixed
                        )

                    save_scores(
                        f"baseline_fixed_test_{dataset_name}_{train_size}_seed_{seed}",
                        model_name,
                        scores_baselines.loc[model_name].to_dict(),
                    )

                print("Baseline Scores on Fixed Test Set:")
                print(scores_baselines)

                # Training LLMs
                print(f"Training LLMs on {dataset_name} with train_size={train_size} and seed={seed}...")
                scores_llms = pd.DataFrame(
                    index=list(LLMS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                _, dataset = train_val_test_split(df, train_size=train_size, has_val=True)
                for model_name, (model, tokenizer) in LLMS.items():
                    tokenized_dataset = tokenize({"train": dataset["train"], "test": fixed_test_df}, tokenizer)
                    trainer = get_trainer(model, tokenized_dataset, tokenizer)

                    start = time.time()
                    trainer.train()
                    end = time.time()
                    scores_llms.loc[model_name]["training_time"] = end - start

                    start = time.time()
                    predictions = predict(
                        trainer, model, tokenized_dataset["test"], tokenizer
                    )
                    end = time.time()
                    scores_llms.loc[model_name]["inference_time"] = end - start

                    for score_name, score_fn in SCORING.items():
                        scores_llms.loc[model_name][score_name] = score_fn(
                            fixed_test_df["label"], predictions
                        )

                    save_scores(
                        f"llms_fixed_test_{dataset_name}_{train_size}_seed_{seed}",
                        model_name,
                        scores_llms.loc[model_name].to_dict(),
                    )

                print("LLM Scores on Fixed Test Set:")
                print(scores_llms)
