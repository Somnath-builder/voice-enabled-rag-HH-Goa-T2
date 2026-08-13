from datasets import load_dataset

def main():
    print("Loading dataset 'ai4bharat/MSMARCO-XI' (subset 'en' if applicable, else default)...")
    try:
        ds = load_dataset('ai4bharat/MSMARCO-XI', 'en', split='train', streaming=True)
    except Exception as e:
        print(f"Error loading 'en', trying default: {e}")
        ds = load_dataset('ai4bharat/MSMARCO-XI', split='train', streaming=True)
        
    it = iter(ds)
    first_row = next(it)
    print("Columns:", first_row.keys())
    print("First row:", first_row)

if __name__ == "__main__":
    main()
