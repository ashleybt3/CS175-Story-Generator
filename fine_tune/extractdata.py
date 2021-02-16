from sklearn.model_selection import train_test_split

with open('alice.txt', 'r', encoding = 'utf-8') as file:
	#strip trailing characters and white space
	dataset = ["<|title|>" + line.rstrip() for line in file.readlines()]

# split into training data and validation(test) data
# train_size = proportion of dtaset to include in training
# random_state = controls shuffling applied to data before split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
train, test = train_test_split(dataset, train_size = 0.9, random_state = 1000)

print("training data size: ", len(train))
print("validation data size: ", len(test))

with open("train_temp.txt", "w") as file_handle:
	file_handle.write("<|endoftext|>".join(train))
with open("valid_temp.txt", "w") as file_handle:
	file_handle.write("<|endoftext|>".join(test))

