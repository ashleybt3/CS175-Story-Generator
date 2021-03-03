import tensorflow as tf
import nltk
from nltk.corpus import stopwords 
import time
import numpy
import random
import language_tool_python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, pipeline


# source: https://huggingface.co/blog/how-to-generate

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
# model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id )
model = TFGPT2LMHeadModel.from_pretrained("./output_clm", from_pt=True)

# tokenizer = AutoTokenizer.from_pretrained("pranavpsv/gpt2-genre-story-generator")
# model = AutoModelForCausalLM.from_pretrained("pranavpsv/gpt2-genre-story-generator")

key_words = set()
stop_words = set(stopwords.words('english')) 
pos = ['NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$' ]


user_input = 'Once upon a time'
story = ''
def run():
    run = True
    while(run):
        global user_input
        print("----------enter any text to continue the story, try to avoid periods\nif you must close a sentence, start the next with a few words----------\n")

        user_input =  input("> ")
        if (user_input == "quit"):
            run = False
        elif(user_input == "story"):
            print(story)
        else:
            exec_model()

def part_of_speech(output):
    wordsList = nltk.word_tokenize(output) 
    wordsList = [w for w in wordsList if not w in stop_words]    
    tagged = nltk.pos_tag(wordsList) 
    for word, tag in tagged:
        if tag in pos:
            key_words.add(word.lower())


def best_sequence(sample_outputs):
    print("Output:\n" + 100 * '-')
    
    total_words = 0
    total_pos = 0
    matched_words = 0
    matched_pos = 0
    best_output = None
    backup_output = None
    for i, sample_output in enumerate(sample_outputs):
        output = tokenizer.decode(sample_output, skip_special_tokens=True)
        wordsList = nltk.word_tokenize(output.lower()) 
        wordsList = [w for w in wordsList if not w in stop_words]    
        tagged = nltk.pos_tag(wordsList) 
        for word, tag in tagged:
            if word in key_words:
                matched_words += 1
            if tag in pos:
                matched_pos += 1
        if matched_words > total_words:
            best_output = output
        if matched_pos > total_pos:
            backup_output = output
        matched_pos = 0
        matched_words = 0
        print("{}: {}".format(i, output))
    if not best_output:
        print("finding back up:", backup_output)
        best_output = backup_output
    print()
    
    # print(best_output)
    return best_output

def exec_model():
    global user_input
    start = time.time()
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(user_input, return_tensors='tf')

    # input_ids = tokenizer(user, return_tensors="pt").input_ids
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)

    # set seed to reproduce results. Feel free to change the seed though to get different results
    # tf.random.uniform([1], seed=1)
    # tf.random.set_seed(1)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=100, 
        min_length = 15,
        # TopK sampling
        top_k=40, 
        # TopP sampling
        top_p=0.9, 
        # number of highest scoring beams
        num_return_sequences=5,
        temperature = 0.7,
        repetition_penalty=1.2,

        # Beam Search
        num_beams = 5,
        no_repeat_ngram_size=3,
        early_stopping = True
    )

    # prints the top n sequences
    # print("Output:\n" + 100 * '-')
    # for i, sample_output in enumerate(sample_outputs):
    #     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    # prints the top sequence
    
    # output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    best_output = best_sequence(sample_outputs)
    part_of_speech(best_output)

   
    global story
    story = story + " " + best_output
    end = time.time()    

    matches = tool.check(best_output)
	best_output = tool.correct(best_output)
    print(best_output)
    print("time: ", end - start)

if __name__ == "__main__":
    print(100 * '-')
    exec_model()
    run()