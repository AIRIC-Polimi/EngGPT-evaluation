def doc_to_target(doc):
    return chr(65 + doc['answer'])

import json
import os

FEWSHOT_PROMPT = None

def get_fewshot_prompt(fewshot_file: str = "fewshot.jsonl"):
    global FEWSHOT_PROMPT
    if FEWSHOT_PROMPT is not None:
        return FEWSHOT_PROMPT

    prompt_parts = []
    file_path = os.path.join(os.path.dirname(__file__), fewshot_file)
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # if i >= 3:
            #     break
            if not line.strip():
                continue
            ex = json.loads(line)

            # Extract options values
            options = [list(opt.values())[0] for opt in ex['options']]
            num_options = len(options)
            letters = [chr(65 + i) for i in range(num_options)]

            options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

            ex_text = f"Domanda: {ex['question'].strip()}\n{options_str}\nRisposta:\n{ex['example_answer']}"
            prompt_parts.append(ex_text)

    FEWSHOT_PROMPT = "\n\n".join(prompt_parts) + "\n\n"
    return FEWSHOT_PROMPT

def doc_to_text_mmlu_cot_fewshot(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    instructions = f"Data la seguente domanda e {num_options} possibili risposte ({letters_str}), scegli la risposta migliore.\n\n- Per problemi semplici:\nFornisci direttamente la risposta con una spiegazione minima.\n\n- Per problemi complessi:\nUsa questo formato passo dopo passo:\n## Passo 1: [Descrizione concisa]\n[Breve spiegazione]\n## Passo 2: [Descrizione concisa]\n[Breve spiegazione]\n\nIndipendentemente dall'approccio, concludi sempre con:\nLa risposta corretta è [lettera_della_risposta].\ndove la [lettera_della_risposta] è una tra {letters_str_or}.\n\nPensa passo dopo passo.\n\n"

    fewshot_examples = get_fewshot_prompt()
    current_question = f"Domanda: {doc['question'].strip()}\n{options_str}\nRisposta:\n"

    return instructions + fewshot_examples + current_question

def doc_to_text_cot(doc):
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(doc['options'])])
    return f"Data la seguente domanda e possibili risposte, scegli la risposta migliore.\n\nDomanda: {doc['question'].strip()}\n{options_str}\nPer problemi semplici:\nFornisci direttamente la risposta con una spiegazione minima.\n\n- Per problemi complessi:\nUsa questo formato passo dopo passo:\n## Passo 1: [Descrizione concisa]\n[Breve spiegazione]\n## Passo 2: [Descrizione concisa]\n[Breve spiegazione]\n\nIndipendentemente dall’approccio, concludi sempre con:\nLa risposta corretta è [lettera_della_risposta].\ndove [lettera_della_risposta] è una tra le possibili opzioni fornite.\n\nPensa passo dopo passo."

def doc_to_text_mmlu_cot(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    return f"Data la seguente domanda e {num_options} possibili risposte ({letters_str}), scegli la risposta migliore.\n\nDomanda: {doc['question'].strip()}\n{options_str}\n\n- Per problemi semplici:\nFornisci direttamente la risposta con una spiegazione minima.\n\n- Per problemi complessi:\nUsa questo formato passo dopo passo:\n## Passo 1: [Descrizione concisa]\n[Breve spiegazione]\n## Passo 2: [Descrizione concisa]\n[Breve spiegazione]\n\nIndipendentemente dall’approccio, concludi sempre con:\nLa risposta corretta è [lettera_della_risposta].\ndove la [lettera_della_risposta] è una tra {letters_str_or}.\n\nPensa passo dopo passo."

def doc_to_text_original_cot(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])
    return f"Rispondi alla seguente domanda a scelta multipla sull'argomento '{doc['category']}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {letters_str_or}. Ragiona brevemente prima di rispondere.\n{doc['question']}\n{options_str}"

def doc_to_text_original_cot_fewshot(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    instructions = f"Rispondi alla seguente domanda a scelta multipla sull'argomento '{doc['category']}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {letters_str_or}. Ragiona brevemente prima di rispondere.\n\n"

    fewshot_examples = get_fewshot_prompt(fewshot_file="fewshot_original.jsonl")
    current_question = f"{doc['question']}\n{options_str}\nRisposta:\n"

    return instructions + fewshot_examples + current_question


def doc_to_text_arc_chat(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    return f"Data la seguente domanda e {num_options} possibili risposte ({letters_str}), scegli la risposta migliore.\nDomanda: {doc['question'].strip()}\n{options_str}\nLa tua risposta deve terminare con \"La risposta corretta è [lettera_della_risposta]\" dove la [lettera_della_risposta] è una tra {letters_str_or}."

def doc_to_text_original_fast(doc):
    options = doc['options']
    num_options = len(options)
    letters = [chr(65 + i) for i in range(num_options)]

    if num_options == 1:
        letters_str = letters[0]
        letters_str_or = letters[0]
    elif num_options == 2:
        letters_str = f"{letters[0]} e {letters[1]}"
        letters_str_or = f"{letters[0]} o {letters[1]}"
    else:
        letters_str = ", ".join(letters[:-1]) + f" e {letters[-1]}"
        letters_str_or = ", ".join(letters[:-1]) + f" o {letters[-1]}"

    options_str = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    return f"Rispondi alla seguente domanda a scelta multipla sull'argomento '{doc['category']}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {letters_str_or}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.\n{doc['question']}\n{options_str}\nRisposta:\n"
