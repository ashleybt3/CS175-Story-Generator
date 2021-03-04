import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time
import numpy
import random

import language_tool_python


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("C:/Users/tiffany/Desktop/CS175-Story-Generator/output_clm", from_pt = True)

user_input = "Once upon a time"
story = ""

def run():
	run = True
	while(run):
		global user_input
		print('-----INSTRUCTIONS:-----\n * Enter any text to continue the story\n * Avoid closing sentence(".", "!", "?") if possible\n * If you must close a sentence, start a new sentence with a few words\n--------------------------\n')
		#user_input =  input("> ")
		new_input = input("> ")
		user_input = user_input + " " + new_input
		if (new_input == "quit"):
		    run = False
		elif (new_input == "story"):
		    print(story)
		    print()
		elif (new_input == "\n"):
			print("invalid, please enter text\n")
		else:
		    exec_model()

def exec_model():
	global user_input
	print("\n---generating output in progress---")
	print("user_input: ", user_input)
	tool = language_tool_python.LanguageTool('en-US')

	start = time.time()

	input_ids = tokenizer.encode(user_input, return_tensors='tf')

	# input_ids = tokenizer(user, return_tensors="pt").input_ids
	# generate text until the output length (which includes the context length) reaches 50
	greedy_output = model.generate(input_ids, max_length=len(user_input))

	# set seed to reproduce results. Feel free to change the seed though to get different results
	# tf.random.uniform([1], seed=1)
	tf.random.set_seed(0)

	sample_outputs = model.generate(
		input_ids,
		# max_length=100,  
		# min_length = 100,
		# num_return_sequences=10,
		# no_repeat_ngram_size=2,
		# repetition_penalty=1.5,
		# top_p=0.92,
		# temperature=.85,
		# do_sample=True,
		# top_k=125,
		# early_stopping=True,

		do_sample = True,
		max_length = 100,
		top_p = 0.92,
		top_k = 125,
		no_repeat_ngram_size = 2,
		num_return_sequences = 10,
		repetition_penalty = 1.5,
		temperature = 0.7

	)

	# prints the random sequence
	next_phrase = random.choice(sample_outputs) #max(sample_outputs, key = len)
	output = tokenizer.decode(next_phrase, skip_special_tokens=True)
	global story
	story = story + " " + output

	#model cannot process too big of output, take only last 10 words
	#average words in sentence = 15-20 (10 words from original output + (~5-10) words from user_input)
	user_input = output#" ".join(output.split()[-10:]) 
	#print("updated user_input: ", user_input)
	end = time.time()

	print("\nthe story so far=========:")
	matches = tool.check(output)
	output = tool.correct(output)
	print(output)

	print("time: {:.2f} seconds".format(end - start))
	print()

if __name__ == "__main__":

	print(100 * "-")

	exec_model()
	run()