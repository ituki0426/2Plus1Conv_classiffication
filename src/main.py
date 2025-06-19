from data.make_dataset import make_dataset
import tqdm
def main():
    print("Starting dataset creation...")
    result = make_dataset()
    print("Dataset creation completed.")

if __name__ == "__main__":
    main()