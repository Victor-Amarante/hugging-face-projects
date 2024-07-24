from transformers import pipeline


base_model = pipeline('fill-mask', model='distilroberta-base')

sentence = 'The capital of <mask> is Brasilia.'

predictions = base_model(sentence)

for predict in predictions:
    score = predict['score']
    score_adj = score * 100
    response_mask = predict['token_str']
    sentence_markered = predict['sequence']
    
    print(f'Predict "{response_mask.strip()}" with score {score_adj:.2f}% -> {sentence_markered}')
    
