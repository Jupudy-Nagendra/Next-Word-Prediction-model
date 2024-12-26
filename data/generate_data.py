import pandas as pd

def generate_dataset(file_path):
    """
    Generate a synthetic dataset for next-word prediction.

    Args:
        file_path (str): Path to save the generated dataset.
    """
    sentences = [
        "I love machine learning",
        "I enjoy programming",
        "This is a great tutorial",
        "FastAPI makes API development easier",
        "Python is a versatile language"
    ]
    
    data = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words) - 1):
            data.append({"Context": " ".join(words[:i+1]), "NextWord": words[i+1]})
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Dataset generated and saved at {file_path}")

if __name__ == "__main__":
    generate_dataset("data/next_word_dataset.csv")

