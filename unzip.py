import zipfile

# Paths to zip files
train_zip_path = 'data/WIDER_train.zip'
val_zip_path = 'data/WIDER_val.zip'
test_zip_path = 'data/WIDER_test.zip'

# Directory to extract files
data_dir = 'data'

# Function to extract zip files
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Extract the datasets
extract_zip(train_zip_path, data_dir)
extract_zip(val_zip_path, data_dir)
extract_zip(test_zip_path, data_dir)
