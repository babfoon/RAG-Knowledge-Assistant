import json
import csv
import os
from ragas import evaluate, metrics

# Path to input and result files 
DATA_DIR = './data'
RESULTS_DIR = './results'
INPUTS_FILE = f'{DATA_DIR}/inputs.json'
GT_ANSWERS_FILE = f'{DATA_DIR}/gt_answers.json' 
BEFORE_RESULTS_FILE = f'{DATA_DIR}/before_eval_results.csv' 
AFTER_RESULTS_FILE = f'{RESULTS_DIR}/after_eval_results.csv'
COMPARISON_FILE = f'{RESULTS_DIR}/comparison_summary.csv'

# Define the metrics to evaluate
METRICS = [
  metrics.FactualConsistency(),
  metrics.AnswerRelevance(),
  metrics.ContextPrecision(),
  metrics.ContextRecall(),
]

def load_json(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'File not found: {file_path}')
  with open(file_path, 'r') as file:
    return json.load(file)

def save_csv(result_dict, output_file):
  with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(result_dicr.keys())
    writer.writerow(result_dict.values())

def evaluate_rag_system(inputs, ground_truth, generated_answers):
  if len(inputs) != len(ground_truth) or len(inputs) != len(generated_answers):
    raise ValueError('Inputs, ground truths, and generated answers must have the same length')

  results = evaluate(generated_answers, ground_truth, inputs, evaluator = METRICS)
  return results

def calculate_comparison(before_reults, after_results):
  comparison = {
    'Metric': [],
    'Before': [],
    'After': [],
    'Change (%)': []
  }
  for metric in before_results:
    before_value = before_results.get(metric, 0)
    after_value = after_results.get(metric, 0)
    precent_change = ((after_value - before_value) / vefore_value * 100) if before_value else None

    comparison['Metric'].append(metric)
    comparison['Before'].append(before_value)
    comparison['After'].append(after_value)
    comparison['Change (%)'].append(round(percent_change, 2) if percent_change else 'N/A')

  return comparison

def save_comparison(comparison, output_file):
  with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Metric','Before','After','Change (%)'])
    for i in range(len(comparison['Metric'])):
      writer.writerow([
        comparison['Metric'][i],
        comparison['Before'][i],
        comparison['After'][i],
        comparison['Change (%)'][i]
      ])
def main():
  os.makedirs(RESULTS_DIR, exist_ok=True)

  inputs = load_json(INPUTS_FILE)
  ground_truth = load_json(GT_ANSWERS_FILE)

  print('Running evaluation for "Before" configuration...')
  generated_answers_before = [input['Generated_answer_after'] for input in inputs]
  after_results = evaluate_rag_system(
    [input['question'] for input in inputs],
    ground_truth,
    generated_answers_before.
  )
  print(f'Before evaluation results: {before_results}')
  save_csv(before_results, BEFORE_RESULTS_FILE)

  print('Running evaluation for "After" configuration...')
  generated_answers_after = [input['generated_answer_after'] for input in inputs]
  after_results = evaluate_rag_system(
    [input['question'] for input in inputs],
    ground_truth,
    generated_answers_after,
  )
  print(f'After evaluation results: {after_results}')
  save_csv(after_results, AFTER_RESULTS_FILE)

  print('Generating comparison summary...')
  comparison = calculate_comparison(before_results, after_results)
  save_comparison(comparison, COMPARISON_FILE)
  print(f'Comparison summary saved to {COMPARISON_FILE}')

if __name__ == '__main__':
  main()


