from transformers import pipeline

models = [
    {
     'name': 'FacebookAI/xlm-roberta-large',
     'token': '<mask>'
    },
    {
     'name': 'neuralmind/bert-base-portuguese-cased',
     'token': '[MASK]'
    },
    {
     'name': 'rufimelo/Legal-BERTimbau-large',
     'token': '[MASK]'
    },
]

for dict_model in models:
    model_name = dict_model['name']
    model_token = dict_model['token']
    print(f'Testing model {model_name}')
    
    base_model = pipeline(task='fill-mask', model=model_name)
    sentence = f'Este documento é essencial para a {model_token}'
    predictions = base_model(sentence)
    for predict in predictions:
        score = predict['score']
        score_adj = score * 100
        response_mask = predict['token_str']
        sentence_markered = predict['sequence']
        
        print(f'Predict "{response_mask.strip()}" with score {score_adj:.2f}% -> {sentence_markered}')
    input('Press "ENTER" to continue...')


# OUTPUTS for each model used:
'''  
For FacebookAI/xlm-roberta-large model
Predict "empresa" with score 4.02% -> Este documento é essencial para a empresa
Predict "organização" with score 3.93% -> Este documento é essencial para a organização
Predict "aprendizagem" with score 3.53% -> Este documento é essencial para a aprendizagem
Predict "comunidade" with score 3.33% -> Este documento é essencial para a comunidade
'''

'''  
For neuralmind/bert-base-portuguese-cased
Predict "[UNK]" with score 7.32% -> Este documento é essencial para a
Predict "sua" with score 3.85% -> Este documento é essencial para a sua
Predict "educação" with score 2.46% -> Este documento é essencial para a educação
Predict "saúde" with score 2.42% -> Este documento é essencial para a saúde
Predict "democracia" with score 2.26% -> Este documento é essencial para a democracia
'''

'''  
For rufimelo/Legal-BERTimbau-large
Predict "decisão" with score 80.93% -> Este documento é essencial para a decisão
Predict ":" with score 3.26% -> Este documento é essencial para a :
Predict "prova" with score 1.67% -> Este documento é essencial para a prova
Predict "sentença" with score 1.05% -> Este documento é essencial para a sentença
Predict "." with score 0.82% -> Este documento é essencial para a.
'''