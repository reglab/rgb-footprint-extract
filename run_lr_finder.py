from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

desired_batch_size, real_batch_size = 32, 4
accumulation_steps = desired_batch_size // real_batch_size

dataset = ...

# Beware of the `batch_size` used by `DataLoader`
trainloader = DataLoader(dataset, batch_size=real_batch_size, shuffle=True)

model = ...
criterion = ...
optimizer = ...

# (Optional) With this setting, `amp.scale_loss()` will be adopted automatically.
# model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode="exp", accumulation_steps=accumulation_steps)
lr_finder.plot()
lr_finder.reset()