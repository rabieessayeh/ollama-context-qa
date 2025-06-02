# Contextual Question Answering with Ollama and FAISS

Installation
============

1. Install **Anaconda** and **git** if not already available on your system.

2. Clone the project repository:
```bash
git clone https://github.com/rabieessayeh/ollama-context-qa.git
cd ollama-context-qa
 ```

3. Create a new Anaconda environment named ollama:

```bash
conda env create -f environment.yml
conda activate ollama
 ```
4. Install the required Python packages:
```bash
pip install -r requirements.txt
 ```
5. Install Ollama to run LLMs locally:
```bash
curl -fsSL https://ollama.com/install.sh | sh
 ```
6. Pull and run the LLaMA 3 model:
```bash
ollama run llama3
 ```

# Usage
Start the assistant by running the main script:
```bash
python main.py
 ```
You can then ask questions based on the predefined document base, for example:

```bash
Your question: What is the impact of vaccines?
 ```

