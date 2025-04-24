import subprocess
import os

def run_preprocessing():
    """Run the preprocessing script."""
    print("Running preprocessing...")
    subprocess.run(["python", "preprocessing.py"], check=True)
    print("Preprocessing completed.")

def run_model():
    """Run the model script and collect results."""
    print("Running model.py...")
    result = subprocess.run(["python", "model.py"], capture_output=True, text=True, check=True)
    print("Model execution completed.")
    return result.stdout

def main():
    """Main function to orchestrate preprocessing, model execution, and result display."""
    try:
        run_preprocessing()
        results = run_model()
        
        print("\nResults:")
        print(results)
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {e.cmd}: {e.stderr}")

if __name__ == "__main__":
    main()
