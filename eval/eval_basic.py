import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

LOCAL_EVAL_UNANSWERABLE_PROMPT = """--- Role ---
You are a fact-level evaluation assistant.

--- Goal ---
This question is UNANSWERABLE from the available knowledge.
Evaluate whether the model behaved correctly by refusing, and whether it hallucinated facts.

--- Task ---
Given:
1) Question
2) Model prediction

Do the following:

1) Determine whether the model clearly REFUSED to answer.
A refusal is present if the model explicitly says it cannot answer due to missing/insufficient information
(e.g., "I don't know", "not provided", "insufficient evidence", "cannot be determined").

2) Identify all distinct factual elements explicitly asserted in the prediction
(e.g., numbers, named entities, dates, specific events, definitive claims).

3) Count how many asserted factual elements are INVALID for an unanswerable question.
- If the model asserts any specific answer or concrete facts as if it knows them, count them as invalid.
- Generic statements like "insufficient information" are not factual elements.

--- Output (strict JSON) ---
{
  "refused": <0 or 1>,
  "invalid_factual_claims": <int>
}

Notes:
- If refused = 1 and invalid_factual_claims = 0, that is ideal behavior.
- If refused = 0, the model did not properly refuse.
- Any invalid_factual_claims > 0 indicates hallucination or over-assertion.
 """


USR_TEMPLATE_UNANSWERABLE = """
 --- Input ---
Question:
{question}

Model Prediction:
{prediction}

Return the result in **strict JSON format**

--- Output ---"""

LOCAL_EVAL_ANSWERABLE_PROMPT = """
You are a strict evaluation judge.

--- Goal ---
Evaluate a model answer for an ANSWERABLE question by counting:
- True Positives (TP): required facts correctly stated
- False Negatives (FN): required facts missing or refused
- False Positives (FP): incorrect or irrelevant facts stated

This evaluation is meant to measure whether the RETRIEVAL CONTEXT was sufficient and clean.
If the model refuses to answer, treat that as missing information (FN).

--- Definitions ---
A "required fact" is an atomic piece of information needed to fully answer the question,
as specified by the ground-truth answer. Examples: a number for a specific quarter,
a named entity, a date, a list item, a comparison result.

--- Procedure ---
1) Build a REQUIRED FACT CHECKLIST from the ground-truth answer.
   - List distinct atomic required facts.
   - Keep them minimal and countable (typically 1–10).
   - Do NOT add facts not explicitly in the ground truth.

2) Compare the model prediction to the checklist:
   For each required fact, mark it as:
   - TP if the model states it correctly (paraphrase allowed if meaning is identical).
   - FN if the model does not state it OR explicitly refuses / says insufficient info.
   - FP does NOT apply here (FP is for extra/incorrect facts, see step 3).

   IMPORTANT: If the model gives the wrong value/entity/time for a required fact,
   count it as FP (because it is an incorrect stated fact) AND also as FN
   (because the required fact was not correctly provided).

3) Count FALSE POSITIVES (FP):
   - Count each distinct factual element stated by the model that is incorrect.
   - This includes:
     a) wrong attempts at required facts (e.g., wrong quarter value),
     b) extra facts not in the checklist that are wrong/unsupported.
   - Do NOT count harmless restatements or purely qualitative filler as FP
     unless it asserts a concrete fact.

4) Be strict about temporal correctness:
   - Wrong year/quarter/month/order => incorrect.
   - Missing a required time qualifier => incorrect if it changes meaning.

--- Output ---
Return strict JSON only:

{
  "tp": <int>,
  "fn": <int>,
  "fp": <int>,
  "required_total": <int>
}

Constraints:
- required_total must equal the number of checklist items you created.
- tp + fn must equal required_total.
- fp can be any non-negative integer.

--- Mini Example ---
Question: "Give revenue for Q1–Q3 2023."
Ground Truth: "Q1: 3.7B, Q2: 3.1B, Q3: 2.8B."
Prediction: "Q1 was 3.7B and Q2 was 3.0B."

Checklist has 3 required facts.
TP: Q1 correct => 1
Q2 wrong => counts as FP (wrong stated fact) AND FN (required fact not correctly provided)
Q3 missing => FN

Output:
{
  "tp": 1,
  "fn": 2,
  "fp": 1,
  "required_total": 3
}
"""

USR_TEMPLATE_ANSWERABLE = """
--- Input ---
Question:
{question}

Ground-Truth Answer:
{answer}

Model Prediction:
{prediction}

--- Output ---"""

# --- Evaluation Logic ---


def evaluate_entry(client: OpenAI, entry: dict, model: str):
    question = entry["question"]
    gold_answer = entry["answer"]
    prediction = entry["predicted_answer"]

    if not prediction:
        prediction = ""

    if isinstance(prediction, list):
        prediction = prediction[0]

    # Determine type of evaluation: Answerable vs Unanswerable
    is_unanswerable = (str(gold_answer).lower().strip() == "unanswerable")

    if is_unanswerable:
        system_prompt = LOCAL_EVAL_UNANSWERABLE_PROMPT
        user_prompt = USR_TEMPLATE_UNANSWERABLE.format(
            question=question,
            prediction=prediction
        )
    else:
        system_prompt = LOCAL_EVAL_ANSWERABLE_PROMPT
        user_prompt = USR_TEMPLATE_ANSWERABLE.format(
            question=question,
            answer=gold_answer,
            prediction=prediction
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = response.choices[0].message.content
        result_json = json.loads(content)
        return result_json
    except Exception as e:
        print(f"Error evaluating entry: {e}")
        return None
    


def calc_stats_answerable(eval_metric_objs):
    f1_sum = 0.0
    for e in eval_metric_objs:
        tp = e.get("tp", 0)
        fp = e.get("fp", 0)
        fn = e.get("fn", 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        f1_sum += f1
    return f1_sum / len(eval_metric_objs)



def calc_stats_unanswerable(eval_metric_objs):
    total_refusals = sum(e["refused"] for e in eval_metric_objs)
    total_invalid_claims = sum(e["invalid_factual_claims"] for e in eval_metric_objs)
    total = len(eval_metric_objs)
    refusal_rate = total_refusals / total
    avg_invalid_claims = total_invalid_claims / total
    return refusal_rate, avg_invalid_claims
            

def eval_rag_output_dir(data_dir: Path, stems_to_ignore:set[str]={}):
    api_key = os.environ["MODEL_API_KEY"]
    base_url = os.environ["MODEL_BASE_URL"]
    model = os.environ["LLM_MODEL"]
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    jsonl_files = data_dir.glob("*.jsonl")

    for file_path in jsonl_files:
        if "evaluated" in file_path.name or file_path.stem in stems_to_ignore:
            continue
            
        print(f"Processing {file_path}...")
        results = []
        output_file_qa = file_path.with_name(f"evaluated_basic{file_path.stem}.jsonl")

        eval_unanswerable_metrics_objs = []
        eval_answerable_metrics_objs = []
        
        with open(file_path, "r", encoding="utf-8") as f_in, open(output_file_qa, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                entry = json.loads(line)

                eval_metrics = evaluate_entry(client, entry, model)
                
                if eval_metrics:
                    entry["evaluation"] = eval_metrics

                is_unanswerable = (entry["answer"].lower().strip() == "unanswerable")

                if is_unanswerable:
                    eval_unanswerable_metrics_objs.append(eval_metrics)
                else:
                    eval_answerable_metrics_objs.append(eval_metrics)

                    if eval_metrics["required_total"] != eval_metrics["fn"] + eval_metrics["tp"]:
                        print(f"WARNING fn + tp != total for llm as a judge eval for question: {entry.get('question', '')}")

                # Write back line with evaluation
                f_out.write(json.dumps(entry) + "\n")
                results.append(eval_metrics)

        print(f"Saved evaluated results to {output_file_qa}")

        save_f1 = False
        save_refusal_and_invalid = False
        # After processing all entries in the file, calculate stats
        if eval_answerable_metrics_objs:
            f1_score = calc_stats_answerable(eval_answerable_metrics_objs)
            print(f"Answerable Questions - F1 Score: {f1_score:.4f}")
            save_f1 = True

        if eval_unanswerable_metrics_objs:
            refusal_rate, avg_invalid_claims = calc_stats_unanswerable(eval_unanswerable_metrics_objs)
            print(f"Unanswerable Questions - Refusal Rate: {refusal_rate:.4f}, Avg Invalid Claims: {avg_invalid_claims:.4f}")
            save_refusal_and_invalid = True

        if save_f1 or save_refusal_and_invalid:
            output_file_stats = file_path.with_name(f"evaluation_stats_{file_path.stem}.txt")
            with open(output_file_stats, "a", encoding="utf-8") as f_out:
                if save_f1:
                    f_out.write(f"Answerable Questions - F1 Score: {f1_score:.4f}\n")
                if save_refusal_and_invalid:
                    f_out.write(f"Unanswerable Questions - Refusal Rate: {refusal_rate:.4f}, Avg Invalid Claims: {avg_invalid_claims:.4f}\n")
            print(f"Saved evaluation stats to {output_file_stats}")

if __name__ == "__main__":
    load_dotenv()
    start_ts = time.time()
    eval_rag_output_dir(Path("eval/rag_results_to_evaluate/daniel_diy_tkg_big"), stems_to_ignore={"vec_search_tkg_answers", "edge_search_tkg_answers", 
                                                                                              "cypher_agent_answers", "vec_and_edge_search", "cypher_agent_answers_no_limits"
                                                                                             }) #,
    #eval_rag_output_dir(Path("rag_results_to_evaluate/tkg_from_paper"))
    end_ts = time.time()
    print(f"Total evaluation time: {end_ts - start_ts:.2f} seconds")