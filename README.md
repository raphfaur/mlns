to reproduce our submission, run 

```
uv run python src/submit_and_predict.py \
  model.kind=pair_mlp \
  pair_mlp.feature_builder=handcrafted \
  pair_mlp.notebook_compat=true \
  pair_mlp.handcrafted_preset=best83 \
  pair_mlp.handcrafted_proj_dim=256 \
  pair_mlp.hidden_dim_1=128 \
  pair_mlp.hidden_dim_2=32 \
  pair_mlp.dropout_1=0.3 \
  pair_mlp.dropout_2=0.25 \
  pair_mlp.dropout_3=0.1 \
  pair_mlp.handcrafted_lr=0.0008527956198396431 \
  pair_mlp.handcrafted_wd=0.0052831819172649425 \
  pair_mlp.batch_size=256 \
  pair_mlp.handcrafted_patience=20 \
  pair_mlp.handcrafted_lr_patience=6 \
  pair_mlp.handcrafted_epochs=60 \
  inference.refit_on_full_train=true \
  inference.full_train_epochs=60 \
  inference.submit=true \
  inference.message="best83 handcrafted mlp full retrain"

```