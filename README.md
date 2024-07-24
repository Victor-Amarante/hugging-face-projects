# Fill-Mask Model with Hugging Face

## Table of Contents
+ [About](#about)
+ [Getting Started](#getting_started)


## About <a name = "about"></a>
This project utilizes the Hugging Face transformers library to demonstrate the use of a fill-mask language model. The script predicts the most likely words to fill in a masked token within a given sentence, providing both the predicted word and the associated confidence score.

## Getting Started <a name = "getting_started"></a>
These instructions will help you set up the project on your local machine for development and testing purposes. See the deployment section for notes on how to deploy the project in a live environment.

### Prerequisites

Ensure you have the following installed:

```
- Python 3.6 or higher
- Pip (Python package installer)
```

### Installing

Follow these steps to get your development environment running:

1. Navigate to the project directory:
```
cd path/to/hugging-face-project_n
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages (reminder to install PyTorch library):
```
pip install transformers
```

End with an example of getting some data out of the system or using it for a little demo.

4. Run the script:
```
python project_n.py
```
