import tensorflow as tf
import nltk
from nltk.corpus import stopwords 
import time
import re
import numpy
import random
import sys
import language_tool_python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, pipeline


# source: https://huggingface.co/blog/how-to-generate

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id )
model = TFGPT2LMHeadModel.from_pretrained("./output_clm", from_pt=True, pad_token_id=tokenizer.eos_token_id)


key_words = set()
stop_words = set(stopwords.words('english')) 
pos = ['NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$' ]

iteration = 0
user_input = 'Once upon a time'
prev_output = ''
story = ''
def run():
    """ Run entire program to continuously take in user input and feed that into the model"""

    run = True
    while(run):
        global user_input
        user_input =  input("> ")
        # new_input = input("> ")
        # user_input = user_input + " " + new_input
        if (user_input == "quit"):
            run = False
        # elif(user_input == "story"):
        #     print(story)
        else:
            exec_model()

def part_of_speech(output):
    """Finds designated POS tags to add to key word list"""
    wordsList = nltk.word_tokenize(output) 
    wordsList = [w for w in wordsList if not w in stop_words]    
    tagged = nltk.pos_tag(wordsList) 
    for word, tag in tagged:
        if tag in pos:
            key_words.add(word.lower())


def best_sequence(sample_outputs):
    """Algorithm to find the best sequence based on number of key words and POS"""    
    total_words = 0
    total_pos = 0
    matched_words = 0
    matched_pos = 0
    best_output = None
    backup_output = None
    for i, sample_output in enumerate(sample_outputs):
        # for each output: decode output from model, tokenize, find POS tags 
        output = tokenizer.decode(sample_output, skip_special_tokens=True)
        wordsList = nltk.word_tokenize(output.lower()) 
        wordsList = [w for w in wordsList if not w in stop_words]    
        tagged = nltk.pos_tag(wordsList) 
        # increment for each matching key word and pos tag
        for word, tag in tagged:
            if word in key_words:
                matched_words += 1
            if tag in pos:
                matched_pos += 1
        # if this specific output has the most key words, it becomes the best output
        if matched_words > total_words:
            best_output = output
        # if this specific output has the most pos tags, it becomes a back up output
        if matched_pos > total_pos:
            backup_output = output
        matched_pos = 0
        matched_words = 0
    # if there was no best output found
    if not best_output:
        best_output = backup_output
    return best_output

def exec_model():
    global user_input
    global prev_output
    global story
    global iteration
    start = time.time()

    # feed new input into model that includes previous input and user input
    next_input = (prev_output + " " + user_input)
    split_text = re.split("[;.]+", next_input)
    if len(split_text) > 1:
        next_input = ".".join(split_text[-2:])
    input_ids = tokenizer.encode(next_input, return_tensors='tf')
    tool = language_tool_python.LanguageTool('en-US')
    sample_outputs = model.generate(
        input_ids,
        do_sample = True,
		max_length = 10000,
		min_length = 50,
		top_p = 0.92,
		top_k = 125,
		no_repeat_ngram_size = 2,
		num_return_sequences = 10,
		repetition_penalty = 1.5,
		temperature = 0.7
    )

    # find best output from all outputs and add key words from the best output found
    best_output = best_sequence(sample_outputs)
    part_of_speech(best_output)
   
    # print and add to story if the iteration is an even number
    if iteration % 2 == 0:
        story = story + best_output
        print("=========the story so far=========:")
        print(story)
    end = time.time()    

    # correct the grammar of best output
    matches = tool.check(best_output)
    best_output = tool.correct(best_output)
    prev_output = best_output
    if iteration % 2 != 0: 
        print(best_output)
    iteration += 1

    print("time: {:.2f} seconds".format(end - start))

if __name__ == "__main__":
    print(100 * '-')
    print('------INSTRUCTIONS:------\n * Enter any text to continue the story\n * Avoid closing sentence(".", "!", "?") if possible\n * If you must close a sentence, start a new sentence with a few words\n--------------------------\n')
    exec_model()
    run()