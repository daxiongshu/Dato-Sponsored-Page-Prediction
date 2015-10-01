# repair original html files
repair_html: repair_html.py
	python $<

# extract javascript from html files
extract_script: extract_data.py
	python $< -o json/script -f extract_script --repair

# extract css related data from html files
extract_style: extract_data.py
	python $< -o json/style -f extract_style --repair

# extract visible text from html files
extract_text: extract_data.py
	python $< -o json/text -f extract_text --repair

# extract url and filename features
extract_url_fn: extract_data.py
	python $< -o mapping/url_filename -f extract_url_fn --repair

# extract special character count features
extract_count_feat: extract_data.py
	python $< -o mapping/count_feature -f extract_count_feat --repair

# extract whitespace character count features
extract_count_whitespace: extract_data.py
	python $< -o mapping/count_whitespace -f extract_count_whitespace --no-repair

# extract token size features
extract_count_word_len: extract_data.py
	python $< -o mapping/count_wordlen_raw -f extract_count_word_len --no-repair
	python $< -o mapping/count_wordlen_repair -f extract_count_word_len --repair

# combine mapping files
combine_url: combine_mapping.py
	python $< -o mapping/id_url_count_map.pkl -p "mapping/url_filename/id_url_partial*_map.pkl"

combine_fn: combine_mapping.py
	python $< -o mapping/id_fn_count_map.pkl -p "mapping/url_filename/id_fn_partial*_map.pkl"

combine_count_feat: combine_mapping.py
	python $< -o mapping/id_feat_count_map.pkl -p "mapping/count_feature/*.pkl"

combine_count_whitespace: combine_mapping.py
	python $< -o mapping/id_whitespace_count_map.pkl -p "mapping/count_whitespace/*.pkl"

combine_count_word_len: combine_mapping.py
	python $< -o mapping/id_wordlen_raw_count_map.pkl -p "mapping/count_wordlen_raw/*.pkl"
	python $< -o mapping/id_wordlen_repair_count_map.pkl -p "mapping/count_wordlen_repair/*.pkl"

# convert url feature to sparse matrix
url_matrix: cm2sp_converter.py
	python $< -i mapping/id_url_count_map.pkl -o matrix/url_count

# convert filename feature to sparse matrix
fn_matrix: cm2sp_converter.py
	python $< -i mapping/id_fn_count_map.pkl -o matrix/fn_count

# convert special character count features to sparse matrix
count_feat_matrix: cm2sp_converter.py
	python $< -i mapping/id_feat_count_map.pkl -o matrix/feat_count

# convert whitespace character count feature to sparse matrix
count_whitespace_matrix: cm2sp_converter.py
	python $< -i mapping/id_whitespace_count_map.pkl -o matrix/whitespace_count

# convert token size features to sparse matrix
count_wordlen_raw_matrix: cm2sp_converter.py
	python $< -i mapping/id_wordlen_raw_count_map.pkl -o matrix/wordlen_raw_count

# extract urls
derive_url_mapping: derive_mapping.py mapping/id_url_count_map.pkl
	python $< -f derive_url_map

# convert domain features to sparse matrix
domain_matrix: cm2sp_converter.py
	python $< -i mapping/id_domain_count_map.pkl -o matrix/domain_count

# convert url parameters to sparse matrix
url_param_matrix: cm2sp_converter.py
	python $< -i mapping/id_url_param_count_map.pkl -o matrix/url_param_count

# extract filename, file extension related features
derive_fn_mapping: derive_mapping.py mapping/id_fn_count_map.pkl
	python $< -f derive_fn_map

# convert special character count feature to sparse matrix
fext_matrix: cm2sp_converter.py
	python $< -i mapping/id_file_extension_count_map.pkl -o matrix/fext_count

# convert javascript token feature to sparse matrix
js_token_matrix: js2sp_converter.py json/script
	python $< -o matrix/js_token -f js_token_matrix

# convert css token feature to sparse matrix
css_token_matrix: js2sp_converter.py json/style
	python $< -o matrix/css_token -f css_token_matrix

# convert visible text token feature to sparse matrix
text_token_matrix: js2sp_converter.py json/text
	python $< -o matrix/text_token -f text_token_matrix

# extract tag-attribute-value related features
extract_feature_matrix: generate_matrix.py
	python $< -p parse_tav -o matrix/tag_attr_val_token --repair -l 0.0005
	python $< -p parse_ta -o matrix/tag_attr_token --repair -l 0.0005
	python $< -p parse_t -o matrix/tag_token --repair
	python $< -p parse_t -o matrix/tag_ngram_token --repair -a 2 -b 3

# compute maximum and sum feature
max_sum_matrix: max_sum_feat.py
	python $<

# feature reduction based on the feature frequency
feature_reduction: feat_reduce.py
	python $< -i matrix/space_count -o matrix/space_count -l 0.00005
	python $< -i matrix/url_count -o matrix/url_count -l 0.00005
	python $< -i matrix/domain_count -o matrix/domain_count -l 0.00005
	python $< -i matrix/url_param_count -o matrix/url_param_count -l 0.00005
	python $< -i matrix/fn_count -o matrix/fn_count -l 0.00005
	python $< -i matrix/js_token -o matrix/js_token -l 0.00005
	python $< -i matrix/css_token -o matrix/css_token -l 0.00005
	python $< -i matrix/text_token -o matrix/text_token -l 0.00005

# perform feature reduction using xgboost feature importance
feature_reduction_xgb: feat_reduce_xgb.py
	python $< -i matrix/css_token -o matrix/css_token
	python $< -i matrix/js_token -o matrix/js_token
	python $< -i matrix/tag_attr_val_token -o matrix/tag_attr_val_token
	python $< -i matrix/url_count -o matrix/url_count
	python $< -i matrix/url_param_count -o matrix/url_param_count
	python $< -i matrix/tag_ngram_token -o matrix/tag_ngram_token
	python $< -i matrix/tag_attr_token -o matrix/tag_attr_token

# combine all matrices
combine_matrix: combine_matrix.py
	python $< --reduce

# xgboost feature extraction
xgb_feature: xgb_feature.py
	python $<
