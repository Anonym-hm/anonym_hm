# HistoiresMorales

The code is structured as follows.

# Usage

1. `gpt-3-5-translation.ipynb` contains code for data translation.
2. `data-quality-lt.ipynb` and `data-quality-translation-QE.ipynb` contain code for grammaticality evaluation and estimation of the translation quality, respectively.
3. `ppl.py` contains code for computing perplexity of moral and immoral text.
4. `lm-harness-results.ipynb` contains code for integrating the data to the framework of the same name.
5. `declarative_prompt.py` contains code for experiments with declarative prompt.
6. `dpo.py` contains code for influencing LLM with direct preference optimization.

Requirements for running the code are mentioned in ```requirements.txt```. Seeds and other parameters if any are listed in the scripts.

We provide more details below.

## 1 Dataset Translation
To translate the data, we use GPT-3.5-turbo with a 16k context window size which was released in Nov 2023: https://openai.com/index/new-models-and-developer-products-announced-at-devday/. 
We use [OpenAI API](https://platform.openai.com/docs/quickstart?context=python). Running the code does not require a GPU. 
## 2 Data Quality Evaluation
`data-quality-lt.ipynb` and `data-quality-translation-QE.ipynb` contain code for grammaticality evaluation and estimation of the translation quality.
To evaluate grammatical correctness, we use LanguageTool: https://languagetool.org/. It is a rule-based error detector with numerous rules for French that can be found here: https://community.languagetool.org/rule/list?lang=fr.
To evaluate the translation quality, we use the reference-free CometKiwi model by Unbabel for direct assessment, published in 2022: https://huggingface.co/Unbabel/wmt22-comet-da.

## 3 Likelihood evaluation
Setting: Norm + Context + Intention + Action, Action $\in \{moral, immoral\}$.
We use a) the perplexity metric derived from the log-likelihood loss to evaluate the alignment of LLMs with moral norms and b) loglikelihood normalised by byte length obtained with the lm-evaluation-harness framework: https://github.com/EleutherAI/lm-evaluation-harness.
We provide the code for integrating the data in the framework in `lm-harness-results.ipynb`
## 4 Action selection with declarative prompt
We prompt the model in a declarative manner to choose an action between two choices based on a scenario. 
Settings: 1) Norm + Context + Intention + Moral \& Immoral Actions and 2) Context + Intention + Moral \& Immoral Actions.
We use the prompts mentioned in `declarative_prompt.py`. 
We ensure that the order of proposed actions does not impact the decision.

## 5 Influencing LLM with Direct Preference Optimization

We evaluate the robustness of LLM's moral alignment. 
Using Direct Preference Optimization (DPO): https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf. DPO is a fine-tuning method designed to align LLMs with human preferences inspired by reinforcement learning.
We aim to influence the model to prefer either moral or immoral actions. 
