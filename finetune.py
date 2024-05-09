import torch
import json
import evaluate
import jieba
# from transformers import NllbTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader

"""
Chinese(simplified) : zho_Hans
English             : eng_Latn
French              : fra_Latn
"""


def read_json(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class Config:
    def __init__(self):
        self.test_jsonl_path = "tiny_stories/en-zh/test.jsonl"
        self.train_jsonl_path = "tiny_stories/en-zh/train.jsonl"
        self.valid_jsonl_path = "tiny_stories/en-zh/valid.jsonl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-4
        self.n_epochs = 1
        self.batch_size = 4
        self.max_len = 256
        self.accumulate_step = 4

class TranslateDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=256):
        self.tokenizer = tokenizer
        self.src_text = [x["src_text"]["text"] for x in data]
        self.tgt_text = [x["tgt_text"] for x in data]
        self.max_len = max_len

    def __getitem__(self, idx):
        src = self.src_text[idx]
        tgt = self.tgt_text[idx]
        # print(src)
        inputs = self.tokenizer(
            src,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # print(f"Inputs shape: {inputs['input_ids'].shape}")
        # print(f"Attention mask shape: {inputs['attention_mask'].shape}")
        # print(f"Labels shape: {labels['input_ids'].shape}")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

    def __len__(self):
        return len(self.src_text)

class myModel:
    def __init__(self, tokenizer, model, config):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(config.device)
        self.device = config.device
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.lr)
    
    def finetune(self, train_dataLoader, valid_dataLoader, log_file=None):
        if log_file != None:
            log_file.write("---Starting fine-tuning.---\n")    
        print("---Starting fine-tuning.---")
        for ep in range(self.config.n_epochs):
            accumulate_steps = 0
            print(f"---Epoch {ep}.---")
            self.model.train()
            train_loss = 0
            print(f"---{len(train_dataLoader)} batches, with batch_size {self.config.batch_size}.---")
            num_batch = 0
            for batchData in train_dataLoader:
                self.optimizer.zero_grad()
                num_batch += 1
                print(f"---In batch {num_batch}---")
                inputs = batchData["input_ids"].to(self.device)
                masks = batchData["attention_mask"].to(self.device)
                labels = batchData["labels"].to(self.device)
                torch.cuda.empty_cache()
                # print(f"Inputs shape: {inputs.shape}")
                # print(f"Attention mask shape: {masks.shape}")
                # print(f"Labels shape: {labels.shape}")
                    
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=masks,
                    labels=labels
                )
                train_loss += outputs.loss.item()
                loss = outputs.loss
                loss.backward()
                
                self.optimizer.step()
                # if num_batch % 50 == 0:
                #     # do test ?
                
                # accumulate_steps += 1
                # if accumulate_steps == self.config.accumulate_step:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                #     accumulate_steps = 0
                #     print("---Optimizer steps.---")
                del inputs, masks, labels, loss
                torch.cuda.empty_cache()
                
            train_loss /= len(train_dataLoader)
            if log_file != None:
                log_file.write(f"Train loss: {train_loss}.\n")  
            print(f"Train loss: {train_loss}.")

            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                for batchData in valid_dataLoader:
                    inputs = batchData["input_ids"].to(self.device)
                    masks = batchData["attention_mask"].to(self.device)
                    labels = batchData["labels"].to(self.device)
                    outputs = self.model(
                        input_ids=inputs,
                        attention_mask=masks,
                        labels=labels
                    )
                    valid_loss += outputs.loss.item()
                    del inputs, masks, labels
                valid_loss /= len(valid_dataLoader)
                torch.cuda.empty_cache()
                if log_file != None:
                    log_file.write(f"Validation loss: {valid_loss}.\n")  
                print(f"Validation loss: {valid_loss}.")

    def evaluate(self, testData, tgt_id="zho_Hans", log_file=None):
        if log_file != None:
            log_file.write("---Starting evaluation.---\n")
        print("---Starting evaluation.---")
        ttl_score = 0
        # bleu = evaluate.load("evaluate_/metrics/bleu/bleu.py")
        cnt = 0
        weights = (0.25, 0.25, 0.25, 0.25)
        smooth = SmoothingFunction().method1
        for singleData in testData:
            cnt += 1
            if cnt % 100 == 0:
                print(f"---Evaluation step {cnt}.---")
            src = singleData["src_text"]["text"]
            tgt = jieba.lcut(singleData["tgt_text"])
            inputs = self.tokenizer(src, return_tensors="pt").to(self.device)
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_id],
                max_length=512
            )
            pred = jieba.lcut(self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
            # print(tgt)
            # print(pred)
            # tgt = ["This is a test.".split()]
            # pred = "This is test.".split()
            # results = bleu.compute(predictions=pred, references=tgt)
            # res_bleu = results["bleu"]
            res_bleu = sentence_bleu([tgt], pred, weights=weights, smoothing_function=smooth)
            if log_file != None:
                log_file.write(f"Test {cnt} BLEU score is {res_bleu}.\n")
            print(f"Test {cnt} BLEU score is {res_bleu}.")
            ttl_score += res_bleu
        avg_score = ttl_score/len(testData)
        if log_file != None:
            log_file.write(f"Test avg BLEU score is {avg_score}.\n")
        print(f"Test avg BLEU score is {avg_score}.")
        return avg_score
        

def testcase(model, tokenizer):
    print("---In a test case.---")
    article = "这是一个示例。"
    inputs = tokenizer(article, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=256
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


if __name__ == "__main__":
    # tgt = [jieba.lcut("中文分词测试")]
    # pred = jieba.lcut("中文分词")
    # print(tgt)
    # print(pred)
    # print(sentence_bleu(tgt, pred))
    
    
    print("---Starting program.---")
    config = Config()

    # get pre-trained models
    print("---Getting pre-trained model and tokenizer.---")
    pretrain_model = AutoModelForSeq2SeqLM.from_pretrained("./nllb-200-distilled-600M", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "./nllb-200-distilled-600M", 
        src_lang="eng_Latn", 
        tgt_lang="zho_Hans", 
        local_files_only=True
    )
    
    # ans = testcase(pretrain_model, tokenizer)
    # print("Translate output: "+ans)
    
    # get data
    print("---Reading data.---")
    test_data = read_json(config.test_jsonl_path)
    train_data = read_json(config.train_jsonl_path)
    valid_data = read_json(config.valid_jsonl_path)
    test_data = test_data[:int(len(test_data)/100)]
    train_data = train_data[:int(len(train_data)/100)]
    valid_data = valid_data[:int(len(valid_data)/100)]
    print(f"length of train data: {len(train_data)}")
    print(f"length of test data: {len(test_data)}")
    print(f"length of valid data: {len(valid_data)}")
    
    # create dataset
    print("---Creating datasets.---")
    # test_dataset = TranslateDataset(tokenizer, test_data)
    train_dataset = TranslateDataset(tokenizer, train_data)
    valid_dataset = TranslateDataset(tokenizer, valid_data)
    
    # create dateloader
    print("---Creating dataLoaders.---")
    # test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataLoader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataLoader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    del train_data, train_dataset, valid_data, valid_dataset
    
    log_file = open("./log.txt", "w")
    model = myModel(tokenizer, pretrain_model, config)
    model.evaluate(test_data, log_file=log_file)
    model.finetune(train_dataLoader, valid_dataLoader, log_file=log_file)
    model.evaluate(test_data, log_file=log_file)
    log_file.close()
    
