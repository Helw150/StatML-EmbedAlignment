from transformers import AutoTokenizer


def token_indices(list_of_words, tokenizer):
    return [
        str(lists[0])
        for lists in tokenizer(list_of_words, add_special_tokens=False)["input_ids"]
        if len(lists) == 1 and lists[0] > 10
    ]


en_tokenizer = AutoTokenizer.from_pretrained("WillHeld/en-bert-xnli")
en_tokenized = token_indices(
    open("most_common_words/english.txt", "r").readlines(), en_tokenizer
)
print(len(en_tokenized))
open("most_common_words/english_index.txt", mode="w").write("\n".join(en_tokenized))
es_tokenizer = AutoTokenizer.from_pretrained("WillHeld/es-bert-xnli")
es_tokenized = token_indices(
    open("most_common_words/spanish.txt", "r").readlines(), es_tokenizer
)
print(len(es_tokenized))
open("most_common_words/spanish_index.txt", mode="w").write("\n".join(es_tokenized))
