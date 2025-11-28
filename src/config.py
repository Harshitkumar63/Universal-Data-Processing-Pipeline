import os

# Base output directory for saved models, reports, images
BASE_OUTPUT = os.environ.get("UDPP_OUTPUT", os.path.join(os.getcwd(), "udpp_output"))
os.makedirs(BASE_OUTPUT, exist_ok=True)

# Default test split and random seed
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Top cardinality for categorical trimming
TOP_CATEGORY_K = 30

