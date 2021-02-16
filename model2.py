import tensorflow as tf
import time
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


# source: https://huggingface.co/blog/how-to-generate

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
user_input = 'Once upon a time'
story = ''
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
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=50)


    # METHODS:
    # - Greedy Search
    # - Beam Search
    # - Sampling
    # - TopK Sampling
    # - TopP(nucleus) Sampling

    # set seed to reproduce results. Feel free to change the seed though to get different results
    # tf.random.uniform([1], seed=1)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=50, 
        # TopK sampling
        top_k=50, 
        # TopP sampling
        top_p=0.95, 
        # number of highest scoring beams
        num_return_sequences=3,

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
    
    output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    global story
    user_input += output
    story = story + " " + output
    end = time.time()
    print(output)
    print("time: ", end - start)

if __name__ == "__main__":
    print(100 * '-')
    exec_model()
    run()