from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import json
import csv
import os

# Path to input and result files 
DATA_DIR = '/home/babfoon/projects/RAG-Knowledge-Assistant/data'
RESULTS_DIR = '/home/babfoon/projects/RAG-Knowledge-Assistant/results'
INPUTS_FILE = f'{DATA_DIR}/inputs.json'
GT_ANSWERS_FILE = f'{DATA_DIR}/gt_answers.json'
BEFORE_RESULTS_FILE = f'{DATA_DIR}/before_eval_results.csv' 
AFTER_RESULTS_FILE = f'{RESULTS_DIR}/after_eval_results.csv'
COMPARISON_FILE = f'{RESULTS_DIR}/comparison_summary.csv'

# Load a pre-trained Sentence Transformer model for semantic similarity
MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_json(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'File not found: {file_path}')
  with open(file_path, 'r') as file:
    return json.load(file)

  def save_csv(results, output_file):
    with open(output_file, mode='w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['metric_name','score'])
      for metric_name, score in results.items():
        writer.writerow([metric_name, score])

def calculate_rouge(generated_answers, references):
  scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
  scores = {'rouge1':0,'rouge2':0,'rouge:':0}
  for generateed, reference in zip(generated_answers, references):
    rouge_results = scorer.score(generated, reference)
    for key, value in rouge_results.items():
      scores[key] += value.fmeasure

  return {k: v/len(generated_answers) for k, v in scores.items()}

def calculate_semantic_similarity(generated_answers, questions):
  question_embeddings = MODEL.encode(questions)
  answer_embeddings = MODEL.encode(generated_answers)
  similarities = cosine_similarity(question_embeddings, answer_embeddings)
  avg_similarity = similarities.diagonal().mean()
  return avg_similarity

def calculate_precision_recall(retrieved_contexts, ground_truths):
  total_precision, total_recall = 0,0
  for context, ground_truth in zip(retrieved_contexts, ground_truths):
    context_set, gt_set = set(context), set(ground_truth)
    truth_positives = len(context_set & gt_set)
    total_precision += true_positives / len(context_set) if context_set else 0
    total_recall += true_positives / len(gt_set) if gt_set else 0
  return {
    'context_precision': total_precision / len(retrieved_contexts),
    'context_recall': total_recall / len(retrieved_contexts),
    }

def evaluate_rag_system(inputs, ground_truths, generated_answers):
  rouge_scores = calculate_rouge(generated_answers, ground_truths)
  relevancy = calculate_semantic_similiarity(generated_answers, inputs)
  precision_recall = calculate_precision_recall(inputs, ground_truths)

  results = {
    **rouge_scores,
    'answer_relevance': relevancy,
    **precision_recall,
    }
  return results

def main():
  os.makedirs(RESULTS_DIR, exist_ok=True)
  inputs = load_json(INPUTS_FILE)
  ground_truths = load_json(GT_ANSWERS_FILE)
  input_questions = [item['question'] for item in inputs]
  before_answers = [item['generated_answer_before'] for item in inputs]
  after_answers = [item['generated_answer_after'] for item in inputs]

  print('Evaluating before optimization...')
  before_results = evaluate_rag_system(input_questions, ground_truths, before_answrs)
  print(f'Before results: {before_results}')
  save_csv(after_results, BEFORE_RESULTS_FILE)

  print('Evaluating after optimization...')
  after_results=evaluate_rag_system(input_questions, ground_truths, after_answers)
  print(f'After results: {after_results}')
  save_csv(after_results, AFTER_RESULTS_FILE)

  comparison_data = {
      metric: {
          "before": before_results[metric],
          "after": after_results.get(metric, 0),
          "change (%)": round(((after_results.get(metric, 0) - before_results[metric]) / before_results[metric]) * 100, 2) if before_results[metric] else "N/A",
      }
      for metric in before_results
  }

  print('Saving comparison results to file...')
  save_csv(comparison_data, COMPARISON_FILE)
  print(f'Comparison saved to {COMPARISON_FILE}.')

if __name__ == '__main__':
  main()


