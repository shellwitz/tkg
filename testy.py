import json

print("lol")
with open("ect-qa/questions/local_base.jsonl", "r") as f:
        questions_raw = f.readlines()

question_objs = []
for i, q in enumerate(questions_raw):
    question_obj = json.loads(q)
    print(len(question_obj["evidence_list"]))
    if len(question_obj["evidence_list"]) == 1:
          print(f"i: {question_obj["evidence_list"]}")


with open("ect-qa/extracted/corpus/base.jsonl", "r") as f:
       lines_raw = f.readlines()

for i, l in enumerate(lines_raw):
      l_obj =
      