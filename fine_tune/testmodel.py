#https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a

# setup imports to use the model
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

#model = TFGPT2LMHeadModel.from_pretrained("gpt2", from_pt=True)
model = TFGPT2LMHeadModel.from_pretrained("C:/Users/tiffany/Desktop/CS175-Story-Generator/test/test-clm", from_pt=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_ids = tokenizer.encode("Once upon a time", return_tensors='tf')

generated_text_samples = model.generate(
    input_ids, 
    max_length=150,  
    num_return_sequences=5,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,
    top_p=0.92,
    temperature=.85,
    do_sample=True,
    top_k=125,
    early_stopping=True
)

#Print output for each sequence generated above
for i, beam in enumerate(generated_text_samples):
  print("{}: {}".format(i,tokenizer.decode(beam, skip_special_tokens=True)))
  print()