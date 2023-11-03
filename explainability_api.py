import torch
import shap
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import preprocessing_functions as pf

app = FastAPI()

class InputExample(BaseModel):
    premise: str
    hypothesis: str
    label:str

def predict(model_path, premise, hypothesis, label):
    premise = pf.clean_text(premise)
    hypothesis = pf.clean_text(hypothesis)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    premise_id = tokenizer.encode(premise, add_special_tokens=False)
    hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
    segment_ids = torch.tensor([0] * (len(premise_id) + 2) + [1] * (len(hypothesis_id) + 1)).unsqueeze(0)  # sentence 0 and sentence 1
    attention_mask_ids = torch.tensor([1] * len(pair_token_ids)).unsqueeze(0)  # mask padded values

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(pair_token_ids).unsqueeze(0), token_type_ids=segment_ids, attention_mask=attention_mask_ids)[0]
        prediction = torch.argmax(logits).item()
    class_names = ['contradiction', 'Entailment', 'Neutral']
    output = class_names[prediction]
    return output, logits, output == label


@app.post("/shapley")
async def shapley_values(input_examples: InputExample):
    model_path = "saved_model.pth"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    with torch.no_grad():
        explainer = shap.Explainer(model, tokenizer)
        output_label, logits = predict(model_path, input_examples.premise, input_examples.hypothesis)
        shap_values = explainer(input_examples, predict, output_rank_order=output_label)

    return {"shapley_values": shap_values}

