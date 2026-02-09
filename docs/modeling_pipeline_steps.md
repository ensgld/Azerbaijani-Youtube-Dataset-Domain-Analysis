# Part-1 Domain Attach + ExpA v2 Commands

# Build source inventory + robust attach + mapping + splits_with_domain_v2
python code/13_rebuild_part1_domain_v2.py

# Experiment A using v2 domain-aware test split
python code/08_evaluate_domainwise_A.py \
  --model_path runs/ft_tuned/model.keras \
  --test_path data/part1/processed/splits_with_domain_v2/test.csv
