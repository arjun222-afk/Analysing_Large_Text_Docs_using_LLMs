import pdfplumber
pdf_path = "google_terms_of_service_en_in.pdf"
output_text_file = "extracted_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()

with open(output_text_file, "w") as text_file:
    text_file.write(extracted_text)

print(f"Text extracted and saved to {output_text_file}")


with open ("extracted_text.txt", "r") as file:
    document_text = file.read()
print(document_text[:500])


from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small")

summary = summarizer(document_text[:1000], max_length=150, min_length=30, do_sample=False)
print("Summary: ",summary[0]['summary_text'])



import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(document_text)

passages = []
current_passage = ""
for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < 200:
        current_passage += " " + sentence
    else:
        passages.append(current_passage.strip())
        current_passage = sentence

if current_passage:
    passages.append(current_passage.strip())

# Step 5
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def generate_questions_pipeline(passage,min_questions=3):
    input_text = f"generate questions: {passage}"
    results =qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')

    questions = [q.strip() for q in questions if q.strip()]


    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])
    return questions[:min_questions]

for idx, passage in enumerate(passages):
    questions = generate_questions_pipeline(passage)
    print(f"Passage {idx+1}:\n{passage}\n")
    print("Generated Questions: ")
    for q in questions:
         print(f"- {q}")
    print(f"\n{'-'*50}\n")

qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2")

def answer_unique_questions(passages,qa_pipeline):
    answered_questions = set()

    for idx, passage in enumerate(passages):
        questions = generate_questions_pipeline(passage)

        for question in questions:
            if question not in answered_questions:
                answer = qa_pipeline({'question':question, 'context': passage})
                print(f"Q: {question}")
                print(f"A: {answer}")
                answered_questions.add(question)
            print(f"{'='*50}\n")
answer_unique_questions(passages, qa_pipeline)