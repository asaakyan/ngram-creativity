# Raw data

To obtain the raw data, please navigate to this google form to get the password to the encrypted zip file: https://forms.gle/U58ahCPBbNuXykUJ6

Inside the raw data folder, you will find two folders: frontier_study and main_study. The frontier study is the exploratory study with frontier models, while the main study is the study with OLMo models.

Each data folder contains the following files:
- expr_data.csv: the expressions with their prec-computed n-gram novelty scores (see data_generation folder). Columns: 
  - seed_passage_id: ID of the seed human passage used for llm passage generation
  - gen_source: generation source (human or model name)
  - gen_passage_id: seed_passage_id + gen_source
  - expression: the expression from the passage
  - part: this was used for data splitting during the study
  - batch: this was used for data splitting during the study
  - ppl: n-gram perplexity of the expression computed as described in the paper
  - rest of the columns: other n-gram novelty scores (not used for the analysis)
- ratings.csv: the human ratings of the pre-highlighted expressions collected during the study. Columns:
  - expression, seed_passage_id, gen_source, gen_passage_id, part, batch: same as above
  - annotator_id: ID of the annotator
  - meaningful, pragmatic, novel: the three creativity dimensions rated by the annotators
  - novelJustification: required comment justifying novelty annotation
  - generalComment: optional comment, e.g. for pragmaticality annotation
- highlights.csv: creative expression hihglights (in addition to the pre-highlighted expression ratings). Columns:
  - seed_passage_id, gen_source, gen_passage_id, part, batch, annotator_id: same as above
  - assignment: human, ai, or frontier model
  - novel_expr: the highlighted expression by the annotator
  - note: required comment justifying the creativity annotation
- hlt_scores.csv: n-gram novelty scores for the highlighted expressions (computed as described in the paper). Columns:
  - same as expr_data.csv, but for highlighted expressions
  - in addition, novelty annotation is assumed as 1, meaningful, pragmatic columns are assumed as NA (since we do not use the highlighted expressions for the pragmaticality study)
- passages: mapping of IDs to actual passages with New Yorker links and human authors
  - seed_passage_id, gen_passage_id: same as above
  - passage: the actual passage text
  - seed_nyer_link: link to the New Yorker story from which the seed passage was taken
  - seed_human_author: author of the New Yorker story from which the seed passage was taken

In addition, the main study also contains the following files:
- all_novel_expressions_main.csv: all pre-highlighted expressions marked as novel or all highlighted expressions (with their novelty justification notes) from the main study. This will be used for benchmarking novel expression identification. Columns:
  - annotation_type: whether it was a highlight or a pre-highlighted expression
- all_nonprag_expressions_main.csv: all pre-highlighted expressions marked as non-pragmatic (with their pragmaticality justification notes) from the main study. This will be used for benchmarking non-pragmatic expression identification. Columns:
  -  note: the optional comment typically justifying the non-pragmatic annotation
-  wqrm_ail_scores.csv: scores from writing quality reward models and Pangram AI likelihood scores for each of the passage.

# Data for linear models

The for_linear_models folder contains the data processed for fitting the mixed-effects models (see linear_models folder for the analysis code). If you need the actual expressions to be present in the data, you need to request access to the raw data. The folder contains the following files:
- prehlt_and_hlt.csv: data with both pre-highlighted expression ratings and highlighted expressions. We slightly modified how to incorporate highlighted expressions to make the data release simpler. Specifically, we consider the highlighted expressions as pragmatic and do not remove pre-highlighted expression ratings if they overlap with the highlights. This does not substantially change the results or the claims. See the preprocess_raw_data.ipynb notebook for details.
  - prehlt_only.csv: data with only pre-highlighted expression ratings (no highlighted expressions).

TBD: for_models folder contains the data processed for fine-tuning and few-shot evaluation.