# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

from src.spamdetection.training import train_llms, train_baselines, train_and_test
from src.spamdetection.preprocessing import init_datasets

if __name__ == "__main__":

    # Download and process datasets
    # if os.path.exists("data") == False:
    #     init_datasets()

    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/png").mkdir(parents=True, exist_ok=True)
    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    from matplotlib import rc
    rc('text', usetex=False)
    # Train baseline models
    train_baselines(
        list(range(3)),
        # ["ling", "sms", "spamassassin", "enron"],
        ["custom_filtered_body"],
        [0.8],
        "test",
    )
    # Path to the fixed test dataset
    # fixed_test_set_path = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/processed/custom_enron/testdata.csv"

    # Train and test models
    # train_and_test(
    #     seeds=[2],
    #     datasets=["custom_enron_with_test"],
    #     train_sizes=[0.8],
    #     fixed_test_set_path=fixed_test_set_path,
    #     test_set="test",
    # )
    # Train LLMs
    # train_llms(
    #     # list(range(5)),
    #     list(range(1)),
    #     # ["ling", "sms", "spamassassin", "enron"],
    #     # ["spamassassin", "enron"],
    #     ["custom_filtered_body"],
    #     [0.8],
    #     "test",
    # )
