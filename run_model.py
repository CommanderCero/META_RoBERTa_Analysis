import math
import pandas as pd
import numpy as np
import models as huggingface_models

if __name__ == "__main__":
    # Change parameters if necessary
    seed = 1
    num_samples = 100000
    batch_size = 10000
    model_fns = {
        "twitter": huggingface_models.create_twitter_model,
        "english": huggingface_models.create_large_english_model,
        "financial": huggingface_models.create_financial_model
    }
    results_folder = "./results"
    
    import argparse
    parser = argparse.ArgumentParser(description='Sexy Dominik.')
    parser.add_argument('-model_name', type=str, help='Name of the model to be executed')
    parser.add_argument('-batch_index', default=0, type=int, help='Batch index from which to resume executing')
    args = parser.parse_args()
    
    print("Loading data...")
    data = pd.read_csv("./results/sample_data.csv")
    
    print(f"Loading model {args.model_name}...")
    model = model_fns[args.model_name]()
    
    print(f"Running {args.model_name}...")
    batches = np.array_split(data, math.ceil(num_samples / batch_size))
    for i, batch in enumerate(batches[args.batch_index:], args.batch_index):
        res = model.predict(batch["review_text"].fillna(""))
        res.index = batch.index
        res.to_csv(f"results/{args.model_name}_{i}_results.csv")
        print(f"Processed {(i + 1) * batch_size} rows...")
    print("Done!")