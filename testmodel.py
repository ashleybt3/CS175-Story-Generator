#https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a

# setup imports to use the model
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import pipeline
from collections import defaultdict
from random import random


model = TFGPT2LMHeadModel.from_pretrained("gpt2", from_pt=True)
# model = TFGPT2LMHeadModel.from_pretrained("/Users/henfong/desktop/CS175-story-generator/output_clm", from_pt=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

nlp = pipeline("ner")

context = dict()

full_text = ""
# print(full_text)
while(True):
    # Save the old length to calculate what to print
    old_len = len(full_text)
    user_input = input("> ")

    # words = user_input.split(" ")
    # for word in words:
    # 	ner = nlp(word)[0]
    # 	if ner.entity == 'I-PER':
    # 		print(ner.entity)
    # 		if ner.entity in context:
    # 			context[ner.word] += (1-context[ner.word])/3
    # 		else:
	   #  		context[ner.word] = .75

    # print(nlp(user_input))
    # type 'exit' to quit
    if(user_input.lower() == 'exit'):
        print("------------------------\n")
        print(full_text)
        print("The End.")
        break

    # Add the user_input to the full_text to keep context
    full_text += user_input
    
    # Generate text 5 times in hopes of building a longer text
    for i in range(1):
        in_len = len(full_text)
        input_ids = tokenizer.encode(full_text, return_tensors='tf')

        generated_text_samples = model.generate(
            input_ids, 
            max_length=50,     # add the in_len so that the story can keep generating
            # add the in_len so that the story can keep generating
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            top_p=0.92, #92,
            temperature=.85, #85,
            do_sample=True,
            top_k=150,
            early_stopping=True
        )

        # decode the generated text
        # generated_text_samples is a list, so you just have to get the first element
            #only one line is being produced anyway
        full_text = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
    
    output_words = full_text[old_len:].split(" ")

    # for i in range(len(output_words)):
    # 	word = output_words[i]
    # 	output_ner = nlp(word)[0]
    # 	if output_ner.entity == 'I-PER':
    # 		if output_ner.name not in context:
    # 			if random()>=.5:
    # 				context[output_ner.name] = .75
    # 			else:
    # 				print("not in context")


    # for k,v in context.items():
    # 	context[k] = context[k]/3
    # 	if context[k] <= .25:
    # 		del context[k]

    # print(context)
    print(full_text[old_len:],'\n')
    