import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time

model = TFGPT2LMHeadModel.from_pretrained("C:/Users/tiffany/Desktop/CS175-Story-Generator/output_clm", from_pt = True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#input_ids = tokenizer.encode("Once upon a time", return_tensors='tf')

user_input = "Once upon a time"
story = ""

def run():
    run = True
    while(run):
        global user_input
        user_input =  input("> ")
        if (user_input == "quit"):
            run = False
        elif(user_input == "story"):
            print(story)
        else:
            exec_model()

def exec_model():
	# encode context the generation is conditioned on
    global user_input
    start = time.time()
    
    input_ids = tokenizer.encode(user_input, return_tensors='tf')

    # input_ids = tokenizer(user, return_tensors="pt").input_ids
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)

    # set seed to reproduce results. Feel free to change the seed though to get different results
    # tf.random.uniform([1], seed=1)
    tf.random.set_seed(1)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    sample_outputs = model.generate(
        input_ids,
        max_length=100,  
        min_length = 100,
	    num_return_sequences=5,
	    no_repeat_ngram_size=2,
	    repetition_penalty=1.5,
	    top_p=0.92,
	    temperature=.85,
	    do_sample=True,
	    top_k=125,
	    early_stopping=True
    )

    # prints the top n sequences
    # print("Output:\n" + 100 * '-')
    # for i, sample_output in enumerate(sample_outputs):
    #     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    # prints the top sequence
    next_phrase = max(sample_outputs, key = len)
    output = tokenizer.decode(next_phrase, skip_special_tokens=True)
    global story
    story = story + " " + output
    end = time.time()
    print(output)
    
    print("time: {:.2f} seconds".format(end - start))

if __name__ == "__main__":
	print(100 * "-")
	exec_model()
	run()