# Pattern_Locator_KG:
This repo is built to find pattern in Knowledge Graph. The file format is in triples (subject, predicate and Object). 

# Temporal_pattern_lookout:
- This script is to find pattern in (Temporal) Knowledge Graph. The file format is in quaternion (subject, predicate, object, time) or triples ((subject, predicate, object).
- Unzip the data.zip in the same folder as folder 'temporal_pattern_lookout'
How to run:
  two hyperparameters needed to be determined, i.e. dataset and threshold for pattern lookup table. 
  example: 
  - python run.py --dataset icews15 --threshold 0.5
