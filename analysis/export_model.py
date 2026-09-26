"""Export the final CatBoost model (EC2-informed) to model.js for the browser."""
import json, sys
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'model.js'
m = json.load(open(src))
trees = []
for t in m['oblivious_trees']:
    s = [[sp['float_feature_index'], sp['border']] for sp in t['splits']]
    trees.append({'s': s, 'v': t['leaf_values']})
sb = m.get('scale_and_bias', [1, [0]])
model = {'scale': sb[0], 'bias': sb[1][0] if isinstance(sb[1], list) else sb[1], 'trees': trees}
open(out, 'w').write('window.OPENPUNCH_MODEL=' + json.dumps(model, separators=(',', ':')) + ';')
print('trees', len(trees), 'bias', model['bias'])
