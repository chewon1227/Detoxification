python3 1_evaluate_with_category.py input_base.json
python3 1_evaluate_with_category.py input_detox.json

#output detox.json
#output base.json

python3 2_evaluate_implicit_bias.py input_base.json implicit_bias_base.json
python3 2_evaluate_implicit_bias.py input_detox.json implicit_bias_detox.json


python3 3_analyze.py detox.json
python3 3_analyze.py detox.json

#output : base_stats.json
#output : detox_stats.json

python3 4_tern_ter.py
# output: ter_comparison_result.json

python3 5_stor.py base.json detox.json stor_result.json

python3 6_total.py
