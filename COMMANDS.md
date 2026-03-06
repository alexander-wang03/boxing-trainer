# Load Splits

python -m src.data.load_boxingvi


# Train Punch Classifier:

python -m src.training.train --model punch

# Evaluate:

python -m src.training.evaluate --model punch
