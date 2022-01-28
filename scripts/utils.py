import os
import pandas as pd


# saving results like losses as a csv files
def save_csv(data, output_path, flag="Train", name="Probabilities"):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, flag + "_" + name + ".csv"), index=False)
