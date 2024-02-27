import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # data.append(json.loads(line))
            data.append(line)
    return data

# Example usage:
file_path = 'notes2.jsonl'
jsonl_data = read_jsonl(file_path)
# print(jsonl_data)
print(type(jsonl_data))
print(len(jsonl_data))
print(type(jsonl_data[-1]))
print(len(jsonl_data[-1]))
print(jsonl_data[-1])
jsonl_data = list(set(jsonl_data))
print(len(jsonl_data))
print(type(jsonl_data[-1]))

print ("\n\n")
for element in jsonl_data:
    print (element)


def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            # file.write(json.dumps(item) + '\n')
            file.write(item)


new_file_path = 'notes.jsonl'
write_jsonl(new_file_path, jsonl_data[:61])

new_file_path = 'notes_validation.jsonl'
write_jsonl(new_file_path, jsonl_data[61:])
