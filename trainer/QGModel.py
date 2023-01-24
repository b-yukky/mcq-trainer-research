from transformers import (
    AdamW,
    T5ForConditionalGeneration,
)
import pytorch_lightning as pl

class QGModel(pl.LightningModule):
    def __init__(self, model_name: str, tokenizer_len: float, learning_rate: float = 0.0001):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self.model.resize_token_embeddings(tokenizer_len) #resizing after adding new tokens to the tokenizer
        self.learning_rate = learning_rate
        print('model initialized!')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        print('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True)

        return loss
  
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
