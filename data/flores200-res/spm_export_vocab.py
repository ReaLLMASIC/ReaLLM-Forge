import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="trained_spm_model.model")

with open("trained_spm_model.exported.vocab", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        f.write(f"{piece}\t{score}\n")

