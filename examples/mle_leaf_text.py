"""CPU training adapter example for Leaf and Spooky; uses public training data only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=Path, required=True)
args = parser.parse_args()
task = json.loads(args.task.read_text())
data, output = Path(task['public_data_dir']), Path(task['output_dir'])
started = time.monotonic()
train, test = pd.read_csv(data/'train.csv'), pd.read_csv(data/'test.csv')
sample = pd.read_csv(data/'sample_submission.csv')
if task['competition_id'] == 'spooky-author-identification':
    x, xtest, y = train['text'], test['text'], train['author']
    def model():
        return make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=30000),
                             LogisticRegression(max_iter=300, random_state=task['seed']))
elif task['competition_id'] == 'leaf-classification':
    x, xtest, y = train.drop(columns=['id', 'species']), test.drop(columns=['id']), train['species']
    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=task['seed']))
else:
    raise ValueError('Unsupported fixed test adapter')
fit, val = train_test_split(np.arange(len(train)), test_size=.25, stratify=y, random_state=task['seed'])
classifier = model().fit(x.iloc[fit], y.iloc[fit])
score = log_loss(y.iloc[val], classifier.predict_proba(x.iloc[val]), labels=classifier.classes_)
classifier = model().fit(x, y)
predictions = classifier.predict_proba(xtest)
joblib.dump(classifier, output/'model.joblib')
assert np.allclose(predictions, joblib.load(output/'model.joblib').predict_proba(xtest))
prediction_frame = pd.DataFrame(predictions, columns=classifier.classes_, index=test['id'])
for column in sample.columns[1:]:
    sample[column] = prediction_frame.loc[sample['id'], column].to_numpy()
sample.to_csv(output/'submission.csv', index=False)
result = {'schema_version': 1, 'task_id': task['task_id'], 'competition_id': task['competition_id'],
          'seed': task['seed'], 'valid': True, 'validation_metric': 'log_loss', 'validation_score': score,
          'higher_is_better': False, 'submission_path': 'submission.csv', 'checkpoint_path': 'model.joblib',
          'checkpoint_reload_matches': True, 'wall_seconds': time.monotonic()-started,
          'llm_calls': 0, 'tokens': 0, 'api_cost': 0, 'gpu_used': False,
          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(output/'candidate-result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
