import transformer as model

if __name__ == "__main__":

	print(100 * "-")
	print('-----INSTRUCTIONS:-----\n * Enter any text after the ">" prompt to continue the story\n * Avoid closing sentence(".", "!", "?") if possible\n * If you must close a sentence, start a new sentence with a few words\n * To exit the process, please enter "quit" at the next input\n-----------------------\n')
	
	model.exec_model()
	model.run()