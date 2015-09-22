token_gen.py:  generate the 4 types of tokens carl did, without 5% frequency to complement what might be missed. Downsampled and only included the positive/negative ratio > 0.05 (will change to 0.01 to see if improves).

token_gen_v2.py: 're.findall('://([A-Za-z0-9][A-Za-z0-9\.]*)',line)' and 're.findall(r"[\S]+",line)' tokens.

token_gen_v3.py: tokens from the last 5% lines of files.

xz_com_model.py:  cv and full model.

random_model_dato.py: randomly generate models for ensemble