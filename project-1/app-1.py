from transformers import pipeline


base_model = pipeline('fill-mask', model='distilroberta-base')

phrase = 'The capital of <mask> is Brasilia.'

predictions = base_model(phrase)

for predict in predictions:
    score = predict['score']
    score_adj = score * 100
    response_mask = predict['token_str']
    phrase_markered = predict['sequence']
    
    print(f'Predict "{response_mask.strip()}" with score {score_adj:.2f}% -> {phrase_markered}')
    
