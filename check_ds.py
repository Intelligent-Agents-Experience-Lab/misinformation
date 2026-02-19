import os
from langsmith import Client
from dotenv import load_dotenv

load_dotenv(override=True)
client = Client()

dataset_name = "Health_Misinformation_Eval_Subset"
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"Example count: {dataset.example_count}")
except Exception as e:
    print(f"Error reading dataset: {e}")
