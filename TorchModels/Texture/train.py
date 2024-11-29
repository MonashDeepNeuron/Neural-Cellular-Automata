import torch
from model import SelfOrganisingTexture 

STEPS = 5000

model = SelfOrganisingTexture()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(STEPS):
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: {loss.item()}")
