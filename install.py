import os

# Define the dependencies required for the project
dependencies = [
    "flask",
    "pandas",
    "scikit-learn",
    "joblib"
]

def install_dependencies():
    print("Installing dependencies...")
    for package in dependencies:
        os.system(f"pip install {package}")
    print("All dependencies installed successfully.")

if __name__ == "__main__":
    install_dependencies()
