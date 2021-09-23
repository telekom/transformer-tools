test_text = "wie läuft's?"

from augmentation import TextAugEmbedding
text_emd = TextAugEmbedding(nr_aug_per_sent=1,
        pos_model_path="",
        embedding_path="cc.de.300.bin",
        score_threshold=0.5,
        base_embedding="fasttext",
        swap_dice=0.2,
        language="de")