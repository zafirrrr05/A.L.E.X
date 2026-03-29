from torch.utils.data import DataLoader
from src.training.ssl_dataset import TeamSequenceDataset
from src.training.ssl_collate import SSLTaskSampler

# python -m scripts.test_ssl_dataset
ds = TeamSequenceDataset("data/sequences")

loader = DataLoader(
    ds,
    batch_size=1,
    shuffle=True,
    collate_fn=SSLTaskSampler()
)

# Print the first batch to verify the collate function
for i, batch in enumerate(loader):
    print(batch["task"])
    for k in batch:
        if k != "task":
            print(k, type(batch[k]))
    break

# Print shapes of tensors in the batch
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape)