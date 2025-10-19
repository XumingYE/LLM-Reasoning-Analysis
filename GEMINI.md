# Gemini Agent Project Guide

This document provides a guide for the Gemini agent on how to interact with and manage this project.

## Project Overview

[**TODO**]: Briefly describe the project's purpose and goals. Explain what the project does and what kind of tasks are expected.

## Key Files

*   `README.md`: The main project README file. Contains a high-level overview of the project.
*   `scripts/`: This directory contains all the scripts for data processing, analysis, and model training.
    *   `scripts/data_processing/`: Scripts for processing data.
    *   `scripts/analysis/`: Scripts for analyzing data.
    *   `scripts/training/`: Scripts for training models.
*   `data/`: This directory contains the datasets used in the project.
*   `predictor_model/`: This directory contains the trained predictor model.
*   `predictor_classifier_model/`: This directory contains the trained classifier model.

## Available Tools

This section describes the tools available to the Gemini agent and how to use them in the context of this project.

*   **`run_shell_command`**: Use this tool to execute shell commands. For example, you can use it to run Python scripts, manage files, and run tests.
*   **`read_file`**: Use this to read the contents of a file. This is useful for understanding the code, data, and documentation.
*   **`write_file`**: Use this to create or modify files. This is useful for creating new scripts, updating documentation, and saving analysis results.
*   **`replace`**: Use this to perform a search and replace in a file. This is useful for refactoring code.
*   **`glob`**: Use this to find files and directories. This is useful for exploring the project structure.

## Common Tasks

This section provides examples of common tasks that the Gemini agent might be asked to perform.

### Running analysis scripts

To run an analysis script, use the `run_shell_command` tool. For example, to run the `analyze_data.py` script, you would use the following command:

```bash
python scripts/analysis/analyze_data.py
```

### Training a model

To train a model, use the `run_shell_command` tool to execute the appropriate training script. For example, to train the predictor model, you would use the following command:

```bash
python scripts/training/train_predictor.py
```

### Adding a new script

To add a new script, use the `write_file` tool to create a new file in the `scripts/` directory. Make sure to follow the existing coding style and conventions.
