token_gen.py:  generate the 4 types of tokens carl did, without 5% frequency to complement what might be missed. Downsampled and only included the positive/negative ratio > 0.05 (will change to 0.01 to see if improves).

feature_ge.py:  generate features csv file using the tokens from token_gen.py

model.py:  cv and full model.