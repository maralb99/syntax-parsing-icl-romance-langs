import time

def retry_with_backoff(func, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                raise e

def load_example_annotation(example_annotation_file):
    try:
        with open(example_annotation_file, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Example annotation file not found: {example_annotation_file}")
        return ""
    except Exception as e:
        print(f"Error loading example annotation: {e}")
        return ""

def generate_conll_u_annotation(tokenized_input, example_annotation=""):
    prompt = (
        "Please annotate the following sentence using the Universal Dependencies standard. "
        "Make sure the output strictly adheres to the CoNLL-U format (10 fields, tab-separated, with ID starting each line). "
        "The input is already tokenized, with the ID, FORM, and LEMMA columns provided. "
        "Complete the remaining columns (UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC).\n\n"
        f"Here is an example of the expected format:\n{example_annotation}\n\n"
        "Sentence to annotate:\n"
        f"{tokenized_input}"
    )

    response = retry_with_backoff(lambda: co.generate(
        model='c4ai-aya-expanse-32b',
        prompt=prompt,
        max_tokens=5000,
        temperature=0.3,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    ))

    return response.generations[0].text

def extract_conll_u_format(response_text, original_lines):
    response_lines = [
        line.strip() for line in response_text.strip().split("\n")
        if line.strip() not in ("conllu", "")
    ]

    output_lines = []
    response_dict = {line.split("\t")[0]: line for line in response_lines if "\t" in line}

    for original_line in original_lines:
        if original_line.strip() and not original_line.startswith("#"):
            original_columns = original_line.split("\t")
            token_id = original_columns[0]

            if token_id in response_dict:
                response_columns = response_dict[token_id].split("\t")
                output_line = "\t".join(original_columns[:3]) + "\t" + "\t".join(response_columns[3:])
                output_lines.append(output_line)
            else:
                output_line = "\t".join(original_columns[:3]) + "\t_\t_\t_\t_\t_\t_\t_"
                output_lines.append(output_line)
        else:
            output_lines.append(original_line.strip())

    return "\n".join(output_lines)

def process_sentences_from_conllu(input_file, output_file, raw_output_file, example_annotation_file):
    rate_limit = 40
    time_window = 60
    processed_count = 0
    start_time = time.time()

    example_annotation = load_example_annotation(example_annotation_file)

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile, \
             open(raw_output_file, 'w', encoding='utf-8') as rawfile:

            sentence_id = ""
            tokenized_sentence = ""
            first_sentence = True

            for line in infile:
                if line.startswith("#"):
                    continue

                if line.strip() == "":
                    if tokenized_sentence:
                        original_lines = tokenized_sentence.strip().split('\n')

                        if processed_count >= rate_limit:
                            elapsed_time = time.time() - start_time
                            if elapsed_time < time_window:
                                time.sleep(time_window - elapsed_time)
                            processed_count = 0
                            start_time = time.time()

                        try:
                            response = generate_conll_u_annotation(tokenized_sentence.strip(), example_annotation)

                            rawfile.write(f"Raw model output for sentence {sentence_id}:\n{response}\n")

                            conll_u_output = extract_conll_u_format(response, original_lines)

                            if conll_u_output:
                                if not first_sentence:
                                    outfile.write("\n")
                                outfile.write(conll_u_output + "\n")
                                first_sentence = False
                            else:
                                print(f"No valid CoNLL-U output for sentence {sentence_id}")

                            processed_count += 1

                        except Exception as e:
                            print(f"Error processing sentence {sentence_id}:\n{tokenized_sentence}\nException: {e}")

                    tokenized_sentence = ""
                else:
                    tokenized_sentence += line

            if tokenized_sentence.strip():
                if not first_sentence:
                    outfile.write("\n")
                outfile.write(tokenized_sentence.strip() + "\n")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except IOError as io_error:
        print(f"IO error: {io_error}")
    except Exception as general_error:
        print(f"An unexpected error occurred: {general_error}")

input_file_path = '/content/drive/MyDrive/eu_input.conllu'
output_file_path = '/content/drive/MyDrive/eu_0_unconstrained.conllu'
raw_output_file_path = '/content/drive/MyDrive/eu_0_unconstrained_raw.txt'
example_annotation_file_path = '/content/drive/MyDrive/example_annotation.conllu'

process_sentences_from_conllu(input_file_path, output_file_path, raw_output_file_path, example_annotation_file_path)